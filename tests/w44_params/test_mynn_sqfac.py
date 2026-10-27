"""SQFAC: pristine WRF predictor, compiled default identity and live operands."""
from dataclasses import replace
import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.extend.core import Literal
import numpy as np
import pytest

from gpuwrf.physics import mynn_pbl as P
from gpuwrf import runtime_parameters as rp

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


INPUTS = load("sqfac_real_inputs", ROOT / "tests/v025/b_phys/test_mynn_predict_saw_floor.py")
ORACLE = load("sqfac_original", ROOT / "proofs/w44_params/sqfac_oracle.py")


@pytest.fixture(autouse=True)
def native_flags(monkeypatch):
    monkeypatch.setenv("GPUWRF_MYNN_PREDICT_SAW_FLOOR", "1")
    monkeypatch.setenv("GPUWRF_MYNN_TRIDIAG_REAL", "1")
    jax.clear_caches()
    yield
    jax.clear_caches()


@pytest.fixture(scope="module")
def references(tmp_path_factory):
    return ORACLE.build(tmp_path_factory.mktemp("sqfac_wrf"))


@pytest.mark.parametrize("row", INPUTS.DATA["rows"], ids=lambda row: row["label"])
def test_pristine_predictor_default_and_changed_sqfac(row, references):
    state, flux, turb, saw = INPUTS.operands(row)
    with jax.enable_x64(False):
        pmz = P._pmz_surface(flux.fltv, flux.ustar, state.dz[:, 0])
        qke = 2 * state.tke
        traces = []

        def predict(state, flux, turb, saw, factor):
            traces.append(1)
            return P._mym_predict_qke(state, 2 * state.tke, turb, row["dt"], flux.ustar,
                                       flux, s_aw=saw, sqfac=factor)[0]

        compiled = jax.jit(predict)
        original = ORACLE.evaluate(references["original"], state, flux, turb, saw, pmz, row["dt"], 3.)
        extracted = ORACLE.evaluate(references["runtime"], state, flux, turb, saw, pmz, row["dt"], 3.)
        np.testing.assert_array_equal(extracted.view(np.uint32), original.view(np.uint32))
        actuals, expected = [], []
        for factor in (3., 6.):
            truth = ORACLE.evaluate(references["runtime"], state, flux, turb, saw, pmz, row["dt"], factor)[:, 0]
            actual = np.asarray(compiled(state, flux, turb, saw, jnp.float32(factor)))
            assert np.isfinite(actual).all() and np.isfinite(truth).all()
            # Existing BP56/BP116 profile-conditioning bound, unchanged.
            scale = np.max(np.abs(truth), axis=-1, keepdims=True) + 1e-37
            assert np.max(np.abs(actual.astype(float) - truth.astype(float)) / scale) <= 64 * np.finfo(np.float32).eps
            actuals.append(actual)
            expected.append(truth)
        assert traces == [1]
        assert not np.array_equal(*expected) and not np.array_equal(*actuals)
        static = jax.jit(lambda st, sf, tu, sa: P._mym_predict_qke(
            st, 2 * st.tke, tu, row["dt"], sf.ustar, sf, s_aw=sa)[0])(state, flux, turb, saw)
        np.testing.assert_array_equal(actuals[0].view(np.uint32), np.asarray(static).view(np.uint32))


def test_sqfac_is_a_pallas_input_dependency():
    state, flux, turb, saw = INPUTS.operands(INPUTS.DATA["rows"][0])
    with jax.enable_x64(False):
        trace = jax.make_jaxpr(lambda factor: P._mym_predict_qke(
            state, 2 * state.tke, turb, 18., flux.ustar, flux, s_aw=saw, sqfac=factor)[0])(jnp.float32(3))
    calls = []

    def walk(closed, incoming):
        jp = closed.jaxpr if hasattr(closed, "jaxpr") else closed
        assert len(jp.invars) == len(incoming)
        tainted = {v for v, flag in zip(jp.invars, incoming) if flag}
        for eq in jp.eqns:
            flags = [not isinstance(v, Literal) and v in tainted for v in eq.invars]
            nested = eq.params.get("jaxpr")
            if eq.primitive.name == "pallas_call":
                calls.append(any(flags))
                outputs = [any(flags)] * len(eq.outvars)
            elif nested is not None and hasattr(nested, "jaxpr"):
                outputs = walk(nested, flags)
            else:
                outputs = [any(flags)] * len(eq.outvars)
            tainted.update(v for v, flag in zip(eq.outvars, outputs) if flag)
        return [v in tainted for v in jp.outvars]

    assert walk(trace, [True]) == [True]
    assert calls and all(calls), "SQFAC does not reach the native tridiagonal operands"


def test_mynn_family_is_separate_and_value_independent():
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    from gpuwrf.runtime.aot_cheap_key import static_config_hash, carry_aval_hash, canonical_digest
    base = OperationalNamelist(GridSpec.canary_3km_template(), None, None, 18., run_physics=True)
    a = rp.bind_mynn_parameters(base, {"sqfac": 3.})
    b = rp.bind_mynn_parameters(base, {"sqfac": 6.})
    assert static_config_hash(a) == static_config_hash(b) != static_config_hash(base)
    assert carry_aval_hash((None, a), {}) == carry_aval_hash((None, b), {})
    assert rp.restart_parameter_digests(a) != rp.restart_parameter_digests(b)
    va, vb = rp.restart_static_namelist(a), rp.restart_static_namelist(b)
    assert canonical_digest(va) == canonical_digest(vb)
    roundtrip = jax.tree.unflatten(jax.tree.structure(a), jax.tree.leaves(a))
    assert rp.restart_parameter_digests(roundtrip) == rp.restart_parameter_digests(a)
    assert rp.mynn_kwargs(base) == {}
    assert float(rp.mynn_kwargs(a)["sqfac"]) == 3.
    for values in ({}, {"b1": 24.}, {"sqfac": -1.}, {"sqfac": np.nan}, {"sqfac": np.inf}, {"sqfac": True}):
        with pytest.raises(ValueError):
            rp.bind_mynn_parameters(base, values)
    with pytest.raises(ValueError, match="active MYNN"):
        rp.bind_mynn_parameters(replace(base, bl_pbl_physics=1), {"sqfac": 3.})
