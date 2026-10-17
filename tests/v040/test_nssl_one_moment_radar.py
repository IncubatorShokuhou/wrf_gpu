"""Legacy NSSL one-moment (mp=19/21) REFL_10CM: pristine radardd02 ipconc=0 branch and its caller.

Oracle: pristine WRF 4.7.1 module_mp_nssl_2mom.F (sha 29f42e76) column fixtures of s2-mpmore
(proofs/s2_mpmore/nssl1mom/fixtures), stage S5 (``an`` after smallvalues) -> S6 ``dbz2d``
(== DBZ_OUT).  fp32 is BYTE-exact; fp64 <= 1e-15 relative (XLA f64 log10 vs glibc, <= 1 ulp).
"""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'proofs/s2_mpmore/nssl1mom/fixtures'
CASES = range(1, 18)


def _load(variant, mode, case):
    return json.loads((DATA / f'v{variant}_{mode}_{case}.json').read_text())


def _canonical(d, stage):
    """Index remap only (original masses 1..8; mp=19 volume 9 / CCNA 10 -> 16 / 18)."""
    from gpuwrf.physics.nssl2mom.indices import LVH, LCCNA, NA
    native = np.asarray(d['stages'][stage]['an'])
    out = np.zeros((NA + 1,) + native.shape[1:], native.dtype)
    for il in range(1, min(len(native), 8) + 1):
        out[il] = native[il - 1]
    if len(native) == 10:
        out[LVH], out[LCCNA] = native[8], native[9]
    return out


def _prec(mode):
    from gpuwrf.physics.nssl2mom.indices import FP32, FP64
    return FP32 if mode == 'fp32' else FP64


def _radar(variant, mode, C=None):
    from gpuwrf.physics.nssl2mom.constants import get_one_moment_constants
    from gpuwrf.physics.nssl2mom.diagnostics import radardd02_one_moment
    C = C or get_one_moment_constants(mode, variant)
    prec = _prec(mode)
    fn = jax.jit(lambda a, d: radardd02_one_moment(a, d, C, prec))
    for case in CASES:
        d = _load(variant, mode, case)
        dt = prec.nR
        an = _canonical(d, 'S5').astype(dt)
        dn = np.asarray(d['stages']['S0']['dn1'], dt)
        yield case, an, np.asarray(fn(an, dn)), np.asarray(d['stages']['S6']['dbz2d'], dt), d


def _assert_parity(variant, mode, C=None):
    for case, _an, got, ref, d in _radar(variant, mode, C):
        assert got.dtype == ref.dtype and np.isfinite(got).all()
        if mode == 'fp32':
            assert np.asarray(d['columns']['DBZ_OUT'], np.float32).tobytes() == ref.tobytes()
            assert got.tobytes() == ref.tobytes(), (variant, case, np.flatnonzero(got != ref)[:10])
        else:
            np.testing.assert_allclose(got, ref, rtol=1e-15, atol=0, err_msg=f'{variant} {case}')


@pytest.mark.parametrize('variant', [19, 21])
@pytest.mark.parametrize('mode', ['fp32', 'fp64'])
def test_one_moment_radar_matches_pristine_s6(variant, mode):
    _assert_parity(variant, mode)


@pytest.mark.parametrize('variant', [19, 21])
def test_oracle_exercises_every_one_moment_reflectivity_branch(variant):
    """E154 census: rain-only, snow-only, graupel-only, mixed and (19) hail points are non-zero."""
    from gpuwrf.physics.nssl2mom.constants import get_one_moment_constants
    C = get_one_moment_constants('fp32', variant)
    seen = {}
    for _case, an, _got, ref, _d in _radar(variant, 'fp32'):
        rain, snow, grp = an[4] >= C.qxmin[4], an[6] >= C.qxmin[6], an[7] >= C.qxmin[7]
        hail = an[8] >= C.qxmin[8] if variant == 19 else np.zeros_like(rain)
        for key, mask in {'rain_only': rain & ~snow & ~grp & ~hail, 'snow_only': snow & ~rain & ~grp & ~hail,
                          'graupel': grp, 'mixed': rain & snow & grp,
                          'hail': hail, 'none': ~(rain | snow | grp | hail)}.items():
            seen[key] = seen.get(key, 0) + int(np.count_nonzero(mask & (ref > 0)))
    # Graupel-only levels stay below 0 dBZ in this set; graupel enters through mixed levels and
    # its intercept/density terms are pinned by the cnoh/rho_qh/volume mutants.
    for key in ('rain_only', 'snow_only', 'graupel', 'mixed') + (('hail',) if variant == 19 else ()):
        assert seen[key] > 0, (variant, key, seen)
    assert seen['none'] == 0


@pytest.mark.parametrize('field,variant', [('cnos', 21), ('cnoh', 21), ('cnor', 19), ('cnohl', 19),
                                           ('rho_qh', 21), ('lvh', 19)])
def test_intercept_and_density_mutants_break_parity(field, variant):
    """Deletion sensitivity (E39): each active constant of the one-moment branch is observed."""
    from gpuwrf.physics.nssl2mom.constants import get_one_moment_constants
    C = get_one_moment_constants('fp32', variant)
    mut = SimpleNamespace(**vars(C))
    index = {'cnor': 4, 'cnos': 6, 'cnoh': 7, 'cnohl': 8}
    if field in index:
        mut.cno = C.cno.copy()
        mut.cno[index[field]] *= np.float32(1.01)
    elif field == 'lvh':
        mut.lvh = 0  # mp=19 graupel density from rho_qh instead of the predicted volume
    else:
        setattr(mut, field, type(getattr(C, field))(getattr(C, field) * 1.3))
    with pytest.raises(AssertionError):
        _assert_parity(variant, 'fp32', mut)


def test_hail_branch_is_required_for_mp19():
    from gpuwrf.physics.nssl2mom.constants import get_one_moment_constants
    mut = SimpleNamespace(**vars(get_one_moment_constants('fp32', 19)))
    mut.lhl = 0  # drop the izieg/lhl>1 hail term
    with pytest.raises(AssertionError):
        _assert_parity(19, 'fp32', mut)


NAMES = {'th': 'TH', 'qv': 'QV', 'qc': 'QC', 'qr': 'QR', 'qi': 'QI', 'qs': 'QS', 'qg': 'QH', 'qh': 'QHL',
         'qndrop': 'CCW', 'qnr': 'CRW', 'qni': 'CCI', 'qns': 'CSW', 'qng': 'CHW', 'qnh': 'CHL', 'qnn': 'CN',
         'qvolg': 'VHW', 'qvolh': 'VHL'}
DRIVER = {}


@pytest.mark.parametrize('variant', [19, 21])
@pytest.mark.parametrize('depth', [0, 1, 40])
def test_driver_radar_depth_matches_original_and_keeps_prognostics(variant, depth):
    """WRF makediag/ke_diag selection on the actual column driver (fp64 original DBZ_OUT)."""
    from gpuwrf.physics.nssl2mom.constants import get_one_moment_constants
    from gpuwrf.physics.nssl2mom.driver import nssl2mom_column
    from gpuwrf.physics.nssl2mom.indices import FP64
    C = get_one_moment_constants('fp64', variant)
    for case in (4, 9, 16):
        d = _load(variant, 'fp64', case)
        f = {k: np.asarray(d['columns'][v + '_IN'], np.float64) for k, v in NAMES.items()}
        f.update({k: np.asarray(d['columns'][v], np.float64)
                  for k, v in (('pii', 'PII'), ('p', 'P'), ('rho', 'DN'), ('w', 'W'), ('dz', 'DZ'))})
        dt = float(d['scalars']['DT'])
        key = (variant, dt)
        if key not in DRIVER:
            DRIVER[key] = jax.jit(lambda fields, first, dep: nssl2mom_column(
                fields, dt, C, FP64, first_step=first, hail_enabled=variant == 19, radar_depth=dep))
        first = np.bool_(d['scalars']['ITIMESTEP'] == 1)
        nz = f['qv'].shape[-1]
        dep = nz if depth == 40 else depth
        out, pr, diag = DRIVER[key](f, first, jnp.int32(dep))
        ref = np.asarray(d['columns']['DBZ_OUT'])
        assert np.count_nonzero(ref) > 0
        dbz = np.asarray(diag['dbz'])
        if dep:
            np.testing.assert_allclose(dbz[:dep], ref[:dep], rtol=1e-9, atol=1e-12)
        assert np.count_nonzero(dbz[dep:]) == 0
        off = DRIVER[key](f, first, jnp.int32(0))
        for a, b in zip(jax.tree.leaves((out, pr)), jax.tree.leaves(off[:2])):
            assert np.asarray(a).tobytes() == np.asarray(b).tobytes()


@pytest.mark.parametrize('mp', [19, 21])
def test_wrf_forced_allocation_and_nssl_caller_cadence(mp):
    """check_a_mundo: 19/21 -> NSSL_2MOM -> compute_radar_ref=1; NSSL makediag lifetime."""
    from gpuwrf.diagnostics.online import configuration, radar_from_context, validate_options
    nl = SimpleNamespace(mp_physics=mp, do_radar_ref=0, nwp_diagnostics=0, run_physics=True,
                         dt_s=54., diagnostic_history_interval=60.)
    assert configuration(nl)['do_radar_ref'] == 1
    assert configuration({'physics': {'mp_physics': [mp]}})['do_radar_ref'] == 1
    validate_options(nl, specified=True, nested=False, periodic=False)
    context = np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4) + 1
    previous = np.full_like(context, 99.)
    call = jax.jit(lambda step: radar_from_context(jnp.asarray(previous), jnp.asarray(context), nl, step, 1))
    for step in (1, 66, 67, 68):
        got = np.asarray(call(jnp.int32(step)))
        want = previous.copy()
        if step == 1:
            want[0, 1:-1, 1:-1] = context[0, 1:-1, 1:-1]
        if step == 67:
            want[:, 1:-1, 1:-1] = context[:, 1:-1, 1:-1]
        np.testing.assert_array_equal(got, want)


def _fixture_module():
    spec = importlib.util.spec_from_file_location('classd_lw_wiring', ROOT / 'tests/test_rrtm_lw_operational_wiring.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize('mp', [19, 21])
def test_initial_carry_holds_real_radar_for_one_moment(mp, monkeypatch):
    from gpuwrf.contracts import state as sc
    from gpuwrf.runtime import operational_mode as op
    monkeypatch.setattr(sc, '_gpu_device', lambda: jax.devices('cpu')[0])
    f = _fixture_module()
    g = f._grid(nz=8, ny=5, nx=6)
    st = f._state(g).ensure_conditional_leaves(mp_physics=mp)
    nl = replace(f._namelist(g), mp_physics=mp)
    carry = op._initial_carry_for_run(st, nl)
    radar = carry.radar_reflectivity
    assert radar is not None and radar.dtype == jnp.float32 and radar.shape == st.qv.shape
    assert np.count_nonzero(np.asarray(radar)) == 0  # WRF allocation zero before the first makediag


@pytest.mark.parametrize('mp', [19, 21])
def test_honour_check_admits_one_moment_radar_selection(mp):
    from gpuwrf.io.namelist_binding import require_namelist_honoured
    require_namelist_honoured({'physics': {'mp_physics': [mp], 'cu_physics': [0], 'do_radar_ref': [1]}}, ('d01',))
    require_namelist_honoured({'physics': {'mp_physics': [mp], 'cu_physics': [0], 'do_radar_ref': [0]}}, ('d01',))


def test_real_log10_stays_real():
    """S2: WRF's Log10 of the REAL dtmp is REAL (module_mp_nssl_2mom.F:8811,9648); only the declared
    DOUBLE islands (dads/dadh/zsdryc/zhdryc and their accumulations) may be f64."""
    from gpuwrf.physics.nssl2mom.constants import get_one_moment_constants
    from gpuwrf.physics.nssl2mom.diagnostics import radardd02_one_moment
    from gpuwrf.physics.nssl2mom.indices import FP32, NA
    C = get_one_moment_constants('fp32', 19)
    hlo = jax.jit(lambda a, d: radardd02_one_moment(a, d, C, FP32)).lower(
        jnp.zeros((NA + 1, 40), jnp.float32), jnp.ones(40, jnp.float32)).as_text()
    f64_logs = [line for line in hlo.splitlines() if 'log' in line and 'f64' in line]
    assert not f64_logs, f64_logs
    assert 'f64' in hlo  # the declared DOUBLE islands remain DOUBLE


def test_real_log10_constants_key_nssl_programs(monkeypatch):
    import gpuwrf.physics.nssl2mom.diagnostics as Dm
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.runtime.aot_baked_tables import baked_table_hash
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    nl = OperationalNamelist(GridSpec.canary_3km_template(), None, None, 18., mp_physics=19,
                             ra_sw_physics=0, ra_lw_physics=0, sf_sfclay_physics=0, cu_physics=0)
    fn = jax.jit(lambda x, nl: x)
    before = baked_table_hash(fn, nl)
    monkeypatch.setattr(Dm, '_INV_LN10', (Dm._INV_LN10[0], np.float32(0.0)))
    assert baked_table_hash(fn, nl) != before
