"""Selected W3 operands against pristine WRF and their original frozen gates."""
import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics import mynn_pbl as P
from gpuwrf import runtime_parameters as RP

ROOT = Path(__file__).resolve().parents[2]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


INPUTS = load('w3_predict_inputs', 'tests/v025/b_phys/test_mynn_predict_saw_floor.py')
PHI = load('w3_phi_reference', 'proofs/w44_params/phim_oracle.py')
PHI_INPUT = load('w3_phi_input', 'tests/w44_params/test_phim_oracle.py')
PREDICT = load('w3_predict_reference', 'proofs/w44_params/sqfac_oracle.py')
LENGTH = load('w3_length_reference', 'proofs/w44_params/length_oracle.py')
LENGTH_INPUT = load('w3_length_input', 'tests/v025/b_phys/test_mynn_elb_mf.py')
ENT = load('w3_ent_reference', 'proofs/w44_params/entrainment_oracle.py')


@pytest.fixture(autouse=True)
def release_flags(monkeypatch):
    for name in ('GPUWRF_MYNN_FP32_COLUMNS', 'GPUWRF_MYNN_PREDICT_SAW_FLOOR', 'GPUWRF_MYNN_TRIDIAG_REAL'):
        monkeypatch.setenv(name, '1')
    jax.clear_caches()
    yield
    jax.clear_caches()


def byte_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert a.dtype == b.dtype and a.shape == b.shape
    assert a.tobytes() == b.tobytes(), (float(np.max(np.abs(a - b))), int(np.count_nonzero(a != b)))


def test_stability_coefficients_predict_qke_with_frozen_bp116_bound(tmp_path):
    phi, predict = PHI.build(tmp_path / 'phi'), PREDICT.build(tmp_path / 'predict')
    for row in INPUTS.DATA['rows']:
        state, flux, turb, saw = INPUTS.operands(row)
        traces = []

        def call(st, sf, tu, sa, coefficients):
            traces.append(1)
            return P._mym_predict_qke(st, 2 * st.tke, tu, row['dt'], sf.ustar, sf,
                                      s_aw=sa, **({} if coefficients is None else coefficients))[0]

        fn = jax.jit(call)
        with jax.enable_x64(False):
            static = fn(state, flux, turb, saw, None)
            for unst, stable in ((16., 6.1), (17.6, 6.1), (16., 6.71)):
                coeff = {'cphm_unst': jnp.float32(unst), 'am_st': jnp.float32(stable)}
                actual = np.asarray(fn(state, flux, turb, saw, coeff))
                zet = PHI_INPUT.real_zet(row)
                pmz = PHI.evaluate(phi['runtime'], zet, unst, stable) - zet
                truth = PREDICT.evaluate(predict['original'], state, flux, turb, saw, pmz, row['dt'], 3.)[:, 0]
                assert np.isfinite(actual).all() and np.isfinite(truth).all()
                scale = np.max(np.abs(truth), axis=-1, keepdims=True) + 1e-37
                assert np.max(np.abs(actual.astype(float) - truth.astype(float)) / scale) <= 64 * np.finfo(np.float32).eps
                if (unst, stable) == (16., 6.1):
                    byte_equal(static, actual)
        assert len(traces) == 2, 'values must not retrace the predictor'


def test_alp1_real_columns_default_and_changed_original_length(tmp_path):
    refs = LENGTH.build(tmp_path / 'length')
    traces = []

    def call(st, qke, dtv, fltv, ust, xland, pblh, psig, rmol, qmf, factor):
        traces.append(1)
        return P._mym_length_option1_with(st, qke, dtv, fltv, ust, 1000., xland,
            pblh=pblh, psig_bl=psig, rmol=rmol, qkw_mf=qmf,
            **({} if factor is None else {'alp1': factor}))

    fn = jax.jit(call)
    with jax.enable_x64(False):
        for row in LENGTH_INPUT.PACKET['rows']:
            state, flux = LENGTH_INPUT.batch(row)
            args = (state, 2 * state.tke, jnp.asarray(row['dtv'], jnp.float32), flux.fltv, flux.ustar,
                    flux.xland, jnp.asarray(row['pblh'], jnp.float32), jnp.asarray(row['psig_bl'], jnp.float32),
                    jnp.asarray(row['rmol'], jnp.float32), jnp.asarray(row['qkw_mf'], jnp.float32))
            static = fn(*args, None)
            for factor in (.23, .253):
                actual = fn(*args, jnp.float32(factor))
                truth = LENGTH.evaluate(refs['runtime'], row, factor)
                LENGTH_INPUT.gate({'qkw': np.asarray(actual[0]), 'el': np.asarray(actual[1])},
                                  {'qkw': truth[:, 0], 'el': truth[:, 1]}, {})
                if factor == .23:
                    for a, b in zip(static, actual, strict=True):
                        byte_equal(a, b)
    assert len(traces) == 2


def test_entrainment_native_default_and_changed_original_fluxes(tmp_path, monkeypatch):
    refs = ENT.build_and_run(tmp_path / 'entrainment')
    legacy = load('w3_original_dmp_input', 'tests/test_mynn_edmf_oracle.py')
    from gpuwrf.physics import mynn_edmf
    from gpuwrf.kernels.phys_mynn_plume import dmp_mf_columns_native
    captured = []
    with monkeypatch.context() as patch:
        patch.setattr(mynn_edmf, 'dmp_mf_columns', lambda *a, **kw: captured.append((a, kw)))
        legacy._dmp_result(json.loads(Path(legacy.COL).read_text()))
    args, kwargs = captured[0]
    traces = []

    def call(inputs, options, factor):
        traces.append(1)
        return dmp_mf_columns_native(*inputs, **options,
            **({} if factor is None else {'entrainment': factor}))

    fn = jax.jit(call)
    static = fn(args, kwargs, None)
    for label, factor in (('default', .33), ('changed', .363)):
        actual = fn(args, kwargs, jnp.float32(factor))
        assert all(np.isfinite(np.asarray(v)).all() for v in actual.values())
        for name in ('s_aw', 's_awqv'):  # Exactly the existing independent DMP gate fields/bound.
            truth = refs[label]['edmf_' + name]
            values = np.asarray(actual[name][0])[:len(truth)]
            assert np.max(np.abs(values - truth)) / max(np.max(np.abs(truth)), 1e-30) <= legacy.TOL_REL
        if label == 'default':
            for name in actual:
                byte_equal(static[name], actual[name])
    assert len(traces) == 2


def test_selected_family_values_share_key_and_names_do_not(monkeypatch):
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    from gpuwrf.runtime.aot_cheap_key import static_config_hash, carry_aval_hash, canonical_digest
    base = OperationalNamelist(GridSpec.canary_3km_template(), None, None, 18., run_physics=True)
    values = {name: spec[0] for name, spec in RP.MYNN.items()}
    a = RP.bind_mynn_parameters(base, values)
    b = RP.bind_mynn_parameters(base, {name: value * 1.1 for name, value in values.items()})
    assert static_config_hash(a) == static_config_hash(b) != static_config_hash(base)
    assert carry_aval_hash((None, a), {}) == carry_aval_hash((None, b), {})
    assert canonical_digest(RP.restart_static_namelist(a)) == canonical_digest(RP.restart_static_namelist(b))
    assert RP.restart_parameter_digests(a) != RP.restart_parameter_digests(b)
    assert set(RP.mynn_kwargs(a)['mynn_coefficients']) == set(values) - {'sqfac'}
    one = RP.bind_mynn_parameters(base, {'alp1': .23})
    other = RP.bind_mynn_parameters(base, {'am_st': 6.1})
    assert static_config_hash(one) != static_config_hash(other)
    for key in ('alp1', 'cphm_unst', 'am_st', 'entrainment'):
        for bad in (-1., float('nan'), float('inf'), True):
            with pytest.raises(ValueError):
                RP.bind_mynn_parameters(base, {key: bad})
    monkeypatch.setenv('GPUWRF_MYNN_FP32_COLUMNS', '0')
    with pytest.raises(ValueError, match='native FP32'):
        RP.bind_mynn_parameters(base, {'alp1': .23})
    RP.bind_mynn_parameters(base, {'sqfac': 3.})  # Existing retained-only SQFAC remains admitted.


def test_each_coefficient_reaches_native_pallas_operands():
    from jax.extend.core import Literal
    state, flux, _, _ = INPUTS.operands(INPUTS.DATA['rows'][0])
    values = {name: jnp.float32(RP.MYNN[name][0]) for name in ('alp1', 'am_st', 'cphm_unst', 'entrainment')}

    def call(st, sf, coefficients):
        return P.step_mynn_pbl_column_with_pblh(st, 18., surface=sf, edmf=True, dx=1000.,
                                               mynn_coefficients=coefficients)

    with jax.enable_x64(False):
        trace = jax.make_jaxpr(call)(state, flux, values)
    offset = len(jax.tree.leaves((state, flux)))

    def walk(closed, incoming, calls):
        jp = closed.jaxpr if hasattr(closed, 'jaxpr') else closed
        assert len(jp.invars) == len(incoming)
        live = {var for var, flag in zip(jp.invars, incoming) if flag}
        for eq in jp.eqns:
            flags = [not isinstance(var, Literal) and var in live for var in eq.invars]
            nested = eq.params.get('jaxpr')
            if eq.primitive.name == 'pallas_call':
                calls.append(any(flags))
                outputs = [any(flags)] * len(eq.outvars)
            elif nested is not None and hasattr(nested, 'jaxpr'):
                outputs = walk(nested, flags, calls)
            else:
                outputs = [any(flags)] * len(eq.outvars)
            live.update(var for var, flag in zip(eq.outvars, outputs) if flag)
        return [var in live for var in jp.outvars]

    for index, name in enumerate(sorted(values)):
        incoming = [False] * len(trace.jaxpr.invars)
        incoming[offset + index] = True
        calls = []
        assert any(walk(trace, incoming, calls)), name
        assert any(calls), f'{name} has no native Pallas operand dependency'


def test_first_call_initializer_uses_the_same_traced_alp1():
    state, flux, _, _ = INPUTS.operands(INPUTS.DATA['rows'][0])
    traces = []

    def call(st, ust, xland, factor):
        traces.append(1)
        return P.mynn_coldstart_init_columns(st, ust, 1000., xland,
            **({} if factor is None else {'alp1': factor}))

    fn = jax.jit(call)
    with jax.enable_x64(False):
        static = fn(state, flux.ustar, flux.xland, None)
        default = fn(state, flux.ustar, flux.xland, jnp.float32(.23))
        changed = fn(state, flux.ustar, flux.xland, jnp.float32(.253))
    for a, b in zip(static, default, strict=True):
        byte_equal(a, b)
    assert all(np.isfinite(np.asarray(a)).all() for a in changed)
    assert np.any(np.asarray(changed[0]) != np.asarray(default[0]))
    assert len(traces) == 2
