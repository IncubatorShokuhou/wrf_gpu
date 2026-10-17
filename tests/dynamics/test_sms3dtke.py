"""Strict pristine-WRF SMS coefficient component gates (full caller gates separate)."""
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from proofs.s1_km5 import oracle
from gpuwrf.dynamics import sms3dtke as sms

FIXTURE = ROOT / 'data/fixtures/sms3dtke-coefficients-wrf-v1.npz'
EPS = np.finfo(np.float32).eps


def assert_parity(actual, reference):
    for field in oracle.OUT_NAMES:
        a, r = np.asarray(getattr(actual, field)), np.asarray(reference[field])
        assert np.isfinite(a).all(), field
        assert np.isfinite(r).all(), field
        np.testing.assert_allclose(a, r, rtol=32*EPS, atol=32*EPS*float(np.abs(r).max()), err_msg=field)
        assert a.dtype == np.float32, field


@pytest.mark.parametrize('key', oracle.CONFIGS)
@pytest.mark.parametrize('jit', [False, True])
def test_sms_components_match_pristine(key, jit):
    cfg = oracle.CONFIGS[key]
    with np.load(FIXTURE, allow_pickle=False) as z:
        inp, config = oracle.input_from_fixture(z, key), oracle.config_from_case(cfg)
        fn = lambda x: sms.sms_tke_coefficients(x, config)
        actual = (jax.jit(fn) if jit else fn)(inp)
        reference = {f: z[key + '/' + f] for f in oracle.OUT_NAMES}
    assert_parity(actual, reference)


def test_required_upstream_operands_are_not_approximated():
    cfg = oracle.CONFIGS['grey_spec']
    with np.load(FIXTURE, allow_pickle=False) as z:
        inp, config = oracle.input_from_fixture(z, 'grey_spec'), oracle.config_from_case(cfg)
    for kw in ({'hpbl': None}, {'dlk': None}):
        with pytest.raises(ValueError, match='require all WRF'):
            sms.sms_tke_coefficients(inp._replace(**kw), config)
    with pytest.raises(ValueError, match='mass-grid shape'):
        sms.sms_tke_coefficients(inp._replace(hpbl=inp.hpbl[:1]), config)


def test_branch_census_is_active():
    with np.load(FIXTURE, allow_pickle=False) as z:
        # Both specified and periodic paths, true LES limit, mesoscale cap,
        # TKE-seeding epsilon=1e-10 edge, and all six independently computed fields.
        for key in oracle.CONFIGS:
            for field in oracle.OUT_NAMES:
                assert np.count_nonzero(z[key + '/' + field]) > 0, (key, field)
        for field in ('xkmv', 'xkhv'):
            assert np.count_nonzero(z['meso_cap/' + field] == 1000.) > 0
        assert not np.array_equal(z['grey_spec/xkmv'], z['grey_periodic/xkmv'])
        assert np.max(z['les_seed/xkmh']) > 0
        assert np.max(z['drag_seed_edge/xkmh']) > 0


def test_real_component_has_no_fp64_arithmetic():
    cfg = oracle.CONFIGS['grey_spec']
    with np.load(FIXTURE, allow_pickle=False) as z:
        inp, config = oracle.input_from_fixture(z, 'grey_spec'), oracle.config_from_case(cfg)
    jx = jax.make_jaxpr(lambda x: sms.sms_tke_coefficients(x, config))(inp)
    def visit(jpr):
        jpr = jpr.jaxpr if hasattr(jpr, 'jaxpr') else jpr
        for eq in jpr.eqns:
            for v in eq.outvars:
                assert getattr(v.aval, 'dtype', None) != np.dtype('float64'), eq.primitive
            for p in eq.params.values():
                for child in (p if isinstance(p, (tuple, list)) else (p,)):
                    if hasattr(child, 'jaxpr') or hasattr(child, 'eqns'):
                        visit(child)
    visit(jx)


@pytest.mark.parametrize('field', oracle.OUT_NAMES)
@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_numeric_gate_rejects_nonfinite_in_candidate_and_reference(field, bad):
    values = {f: np.ones((1,), np.float32) for f in oracle.OUT_NAMES}
    broken = dict(values, **{field: np.array([bad], np.float32)})
    with pytest.raises(AssertionError):
        assert_parity(sms.SmsCoefficients(**broken), values)
    with pytest.raises(AssertionError):
        assert_parity(sms.SmsCoefficients(**values), broken)
