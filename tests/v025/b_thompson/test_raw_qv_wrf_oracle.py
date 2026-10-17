"""Constructed UNIT counterexamples for WRF's raw qv1d / private qv split.

The pristine mp8 driver runs one step at itimestep=1. These fixtures do not
reproduce an observed F2 event, its call clock, dynamics, or a forecast.
Component bounds are the existing mixed-phase increment bounds; the raw/floor
pair and no_micro assertions discriminate the lost-baseline bug separately.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.kernels import phys_thompson_full as full
from gpuwrf.physics import thompson_column as tc

FIXTURE = Path(__file__).with_name('fixtures') / 'raw_qv_columns.npz'
MANIFEST = FIXTURE.with_suffix('.json')
FIELDS = ('qv', 'qc', 'qr', 'qi', 'qs', 'qg', 'ni', 'nr', 'th')
FLAGS = dict(
    GPUWRF_THOMPSON_FP32='0', GPUWRF_THOMPSON_NATIVE_REAL='1',
    GPUWRF_THOMPSON_COLUMN_SED='1', GPUWRF_THOMPSON_SED_FP32='1',
    GPUWRF_THOMPSON_COLUMN_LAYOUT='1', GPUWRF_THOMPSON_FULL_COLUMN='1',
    GPUWRF_THOMPSON_FULL_COLUMN_EARLY_EXIT='1', GPUWRF_THOMPSON_SED_PREP_FUSED='0',
    GPUWRF_THOMPSON_IMPLICIT_SED='0', GPUWRF_THOMPSON_MIXED_PHASE_WRF='1',
    GPUWRF_THOMPSON_WRF_CONSTANTS='1',
)


@pytest.fixture(scope='module')
def batch(tmp_path_factory):
    manifest = json.loads(MANIFEST.read_text())
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == manifest['fixture_sha256']
    z = dict(np.load(FIXTURE))
    assert all(np.isfinite(v).all() for k, v in z.items() if k != 'names')
    assert jax.devices()[0].platform == 'cpu'
    f = lambda name: jnp.asarray(z['in_' + name], jnp.float32)
    T, p, qv = f('th') * f('pii'), f('p'), f('qv')
    zero = jnp.zeros_like(qv)
    state = tc.ThompsonColumnState(
        qv=qv, qc=f('qc'), qr=f('qr'), qi=f('qi'), qs=f('qs'), qg=f('qg'),
        Ni=f('ni'), Nr=f('nr'), Ns=zero, Ng=zero, T=T, p=p,
        rho=tc.density_from_pressure_temperature(p, T, qv), dz=f('dz8w'), w=f('w'),
    )
    with pytest.MonkeyPatch.context() as mp:
        for k, v in FLAGS.items():
            mp.setenv(k, v)
        jax.clear_caches()
        out, ppt = full.full_column(state, 18., interpret=True)
    port = {k: np.asarray(getattr(out, {'ni': 'Ni', 'nr': 'Nr'}.get(k, k)))
            for k in FIELDS if k != 'th'}
    port['th'] = np.asarray(out.T, np.float32) / z['in_pii']
    port['rainnc'] = sum(np.asarray(ppt[k]) for k in ('rain', 'snow', 'graupel', 'ice'))
    port['snownc'], port['graupelnc'] = np.asarray(ppt['snow']), np.asarray(ppt['graupel'])
    port['precip'] = port['rainnc']
    assert all(np.isfinite(v).all() for v in port.values())
    np.savez_compressed(tmp_path_factory.getbasetemp() / 'raw_qv_result.npz', **port)
    yield z, manifest, port
    jax.clear_caches()


def test_constructed_columns_keep_frozen_pristine_increment_bounds(batch):
    z, _manifest, port = batch
    path = Path(__file__).with_name('test_mixed_phase_wrf_oracle.py')
    spec = importlib.util.spec_from_file_location('raw_qv_frozen_bounds', path)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    z = dict(z, wrf_rainncv=z['wrf_rainnc'])
    excess = gate._excess(port, z, np.arange(len(z['names'])), gate._scales(z))
    bad = {k: v for k, v in excess.items() if v > gate.TOL[k[1]]}
    assert not bad, bad


@pytest.mark.parametrize('index', range(3, 6))
def test_no_micro_returns_raw_vapor_including_negative_and_subfloor(batch, index):
    z, _manifest, port = batch
    expected = z['wrf_qv'][index]
    np.testing.assert_array_equal(expected, z['in_qv'][index])
    np.testing.assert_array_equal(port['qv'][index], expected,
                                  err_msg='WRF no_micro preserves raw qv1d')
    assert z['in_qc'][index, 8] > 0.
    assert port['qc'][index, 8] == z['wrf_qc'][index, 8] == 0.


@pytest.mark.parametrize('index', [2, *range(18, 27)])
def test_negative_no_micro_output_gets_original_driver_copyback_repair(batch, index):
    z, _manifest, port = batch
    assert np.any(z['in_qv'][index] < 0.)
    # Public mp_gt_driver applies :1435-1445 AFTER inner no_micro, even
    # with mp_zero_out=0. Preserve its exact neighbor order and end bounds.
    assert np.all(z['wrf_qv'][index] >= 0.)
    np.testing.assert_array_equal(port['qv'][index], z['wrf_qv'][index],
                                  err_msg='WRF driver returned-negative copyback differs')


@pytest.mark.parametrize('name', [
    'cloud_unspent_deficit_raw', 'rain_unspent_deficit_raw',
    'cloud_then_rain_unspent_deficit_raw',
])
def test_evaporation_does_not_erase_an_unspent_raw_vapor_deficit(batch, name):
    z, _manifest, port = batch
    index = list(z['names']).index(name)
    floor = np.float32(1e-10)
    assert z['in_qv'][index, 9] < 0.
    assert z['wrf_qv'][index, 9] == floor
    assert port['qv'][index, 9] == floor, 'private work floor must not become the tendency baseline'


@pytest.mark.parametrize('pair', [(0, 1), (8, 9), (14, 15), (16, 17)])
def test_raw_and_prefloored_inputs_retain_pristine_causal_difference(batch, pair):
    z, _manifest, port = batch
    raw, floored = pair
    wrf_delta = z['wrf_qv'][floored].astype(np.float64) - z['wrf_qv'][raw]
    port_delta = port['qv'][floored].astype(np.float64) - port['qv'][raw]
    scale = float(np.max(np.abs(wrf_delta)))
    assert scale > 0., 'fixture must exercise the raw/work distinction'
    magnitude = np.float32(max(np.abs(z['wrf_qv'][[raw, floored]]).max(), 1e-30))
    bound = 1e-2 * scale + 2. * float(np.spacing(magnitude))
    error = float(np.max(np.abs(port_delta - wrf_delta)))
    assert error <= bound, (str(z['names'][raw]), error, bound)


def test_existing_positive_control_outputs_remain_compatible(batch):
    """GPU-port vs GPU-port saved output is regression evidence only."""
    z, _manifest, port = batch
    for k in (*FIELDS, 'rainnc', 'snownc', 'graupelnc'):
        expected = z['old_port_' + k][1]
        assert port[k][1].dtype == expected.dtype
        assert port[k][1].tobytes() == expected.tobytes(), k


def _non_real_entry_state():
    """Small interface probe, with no pristine/physical parity claim."""
    f = lambda value: jnp.full((1, 4), value, jnp.float32)
    zero = f(0.)
    return tc.ThompsonColumnState(
        qv=f(1e-5), qc=zero, qr=zero, qi=zero, qs=zero, qg=zero,
        Ni=zero, Nr=zero, Ns=zero, Ng=zero, T=f(280.), p=f(1e5),
        rho=f(1.), dz=f(250.), w=zero,
    )


def test_native_real_off_entry_keeps_existing_mixed_phase_configuration_error(monkeypatch):
    for key, value in {**FLAGS, 'GPUWRF_THOMPSON_NATIVE_REAL': '0'}.items():
        monkeypatch.setenv(key, value)
    namespace, contexts = full._namespace, []

    def observed_namespace(*args):
        env = namespace(*args)
        body = env['_thompson_source_sink_body']

        def observed(*body_args, **kwargs):
            contexts.append(kwargs.get('raw_vapor', False))
            return body(*body_args, **kwargs)

        env['_thompson_source_sink_body'] = observed
        return env

    monkeypatch.setattr(full, '_namespace', observed_namespace)
    jax.clear_caches()
    try:
        # The retained mixed-phase equations still require REAL. This test
        # preserves that rejection and separately checks the new opt-in.
        with pytest.raises(ValueError, match='requires GPUWRF_THOMPSON_NATIVE_REAL=1'):
            full.full_column(_non_real_entry_state(), 18., interpret=True)
        assert contexts and not any(contexts), 'non-REAL entry enabled the raw-vapor tuple contract'
    finally:
        jax.clear_caches()


def test_native_real_off_entry_preserves_rain_helper_state_only_return(monkeypatch):
    for key, value in {**FLAGS, 'GPUWRF_THOMPSON_NATIVE_REAL': '0'}.items():
        monkeypatch.setenv(key, value)
    namespace = full._namespace

    def isolated_namespace(*args):
        env = namespace(*args)

        def no_source_rates(state, dt, tables, cold_tables, *, vapor=None):
            # Isolate the actual body/rain return seam from the separate
            # mixed-phase REAL requirement. No physics is qualified here.
            result = (state, jnp.zeros_like(state.qv), jnp.ones_like(state.qv))
            return result if vapor is None else (*result, jnp.zeros_like(vapor[0]))

        env['_wrf_mixed_phase_sources'] = no_source_rates
        return env

    monkeypatch.setattr(full, '_namespace', isolated_namespace)
    jax.clear_caches()
    try:
        try:
            out, _ppt = full.full_column(_non_real_entry_state(), 18., interpret=True)
        except TypeError as exc:
            if 'cannot unpack non-iterable ThompsonColumnState' not in str(exc):
                raise
            raise AssertionError('non-REAL rain helper state-only return was unpacked as a vapor tuple') from exc
        assert isinstance(out, tc.ThompsonColumnState)
    finally:
        jax.clear_caches()
