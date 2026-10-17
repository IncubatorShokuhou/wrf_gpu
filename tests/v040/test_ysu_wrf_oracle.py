"""YSU (bl_pbl_physics=1) against pristine WRF 4.7.1 module_bl_ysu.F / bl_ysu.F90.

Truth: proofs/t_ysuqc (unmodified lane build copies, public call mirroring
module_pbl_driver.F:1208-1251: QC3D/QI3D, FLAG_QC/QI, PSIM=fm, PSIH=fhh, CTOPO,
RTHRATEN, YSU_TOPDOWN_PBLMIX; -O0 no-FMA REAL4, init zero == init inf on every
consumed output). Columns: real 44-level CPU-WRF columns (c02 Kessler qc-only,
c07 cloud ice, c40 clear, c85 marine cloud) in both top-down modes.
"""
import ctypes
import dataclasses
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / 'proofs/t_ysuqc/fixtures/ysu_wrf_oracle.npz'
META = ROOT / 'proofs/t_ysuqc/fixtures/ysu_wrf_oracle.json'
FIELDS = (('u', 0), ('v', 1), ('theta', 2), ('qv', 3), ('qc', 4), ('qi', 5), ('exch_h', 6), ('exch_m', 7))
SCALARS = (('pblh', 0), ('kpbl', 1), ('wstar', 2), ('delta', 3), ('u10', 4), ('v10', 5))
# XLA:CPU flushes subnormal operands/results (FINDINGS E117); WRF keeps them.
# Hydrometeor rates below this magnitude are compared with this absolute floor.
Q_SUBNORMAL_FLOOR = 1.0e-30
EPS32 = float(np.finfo(np.float32).eps)


def _load():
    meta = json.loads(META.read_text())
    assert hashlib.sha256(FIX.read_bytes()).hexdigest() == meta['fixture_sha256']
    assert meta['init_zero_vs_inf_consumed_outputs_byte'] is True
    assert meta['reference_build']['fma_instructions'] == 0
    return np.load(FIX), meta


def _groups(d):
    keys = {}
    for i in range(len(d['nz'])):
        keys.setdefault((int(d['topdown'][i]), bool(d['flag_qi'][i]), float(d['surface'][i, 16])), []).append(i)
    return keys.items()


def _run(d, idxs, flag_qi, topdown, eager):
    from gpuwrf.physics.pbl_ysu import ysu_columns
    n = int(d['nz'][idxs[0]])
    prof, surf = d['profiles'][idxs], d['surface'][idxs]
    with jax.enable_x64(False):
        real = lambda x: jnp.asarray(x, jnp.float32)
        c = lambda j: real(prof[:, :n, j])
        s = lambda j: real(surf[:, j])
        kw = dict(psfc=s(0), znt=s(1), ust=s(2), psim=s(3), psih=s(4), xland=s(5), hfx=s(6), qfx=s(7), wspd=s(8),
                  br=s(9), u10=s(10), v10=s(11), uoce=s(12), voce=s(13), ctopo=float(surf[0, 14]),
                  ctopo2=float(surf[0, 15]), dt=float(surf[0, 16]), qc=c(4), qi=c(5) if flag_qi else None,
                  rthraten=c(10), ysu_topdown_pblmix=topdown)
        args = (c(0), c(1), c(2), c(3), c(6), real(prof[:, :, 7]), c(8), c(9))
        call = lambda *a: ysu_columns(*a, **kw)
        if eager:
            with jax.disable_jit():
                out = call(*args)
        else:
            out = jax.jit(call)(*args)
    return n, {k: np.asarray(v) for k, v in out.items()}


@pytest.fixture
def glibc_transcendentals(monkeypatch):
    """Route the kernel's exp/tanh through glibc expf/tanhf (what gfortran -O0
    calls). XLA:CPU's own exp (tanh) differs from glibc in ~9 % (~58 %) of REAL4
    inputs by 1 ULP; pow is already bitwise glibc powf. This isolates operation
    order, which is what the literal gate certifies."""
    import gpuwrf.physics.pbl_ysu as ysu
    libm = ctypes.CDLL('libm.so.6')
    for name in ('expf', 'tanhf'):
        getattr(libm, name).restype = ctypes.c_float
        getattr(libm, name).argtypes = [ctypes.c_float]
    used = []

    def host(fn):
        vec = np.vectorize(lambda x: getattr(libm, fn)(float(x)), otypes=[np.float32])

        def call(x):
            assert x.dtype == jnp.float32
            used.append(fn)
            return jax.pure_callback(lambda a: vec(np.asarray(a)).astype(np.float32),
                                     jax.ShapeDtypeStruct(x.shape, x.dtype), x, vmap_method='broadcast_all')
        return call

    class GlibcJnp:
        exp = staticmethod(host('expf'))
        tanh = staticmethod(host('tanhf'))

        def __getattr__(self, name):
            return getattr(jnp, name)

    monkeypatch.setattr(ysu, 'jnp', GlibcJnp())
    return used


def test_literal_real4_ysu_is_byte_identical_to_pristine_wrf(glibc_transcendentals):
    """Op-by-op REAL4 evaluation reproduces the original public call bit for bit."""
    d, _ = _load()
    checked = 0
    for (topdown, flag_qi, _dt), idxs in _groups(d):
        n, out = _run(d, idxs, flag_qi, topdown, eager=True)
        assert ('qi' in out) == flag_qi
        for pos, i in enumerate(idxs):
            for name, j in FIELDS:
                if name == 'qi' and not flag_qi:
                    continue
                wrf = d['wrf'][i][:n, j]
                act = out[name][pos].astype(np.float32)
                assert np.isfinite(act).all(), (i, name)
                if name in ('qc', 'qi'):
                    tiny = np.abs(wrf) < Q_SUBNORMAL_FLOOR
                    assert np.all(np.abs(act[tiny] - wrf[tiny]) <= Q_SUBNORMAL_FLOOR), (i, name)
                    wrf, act = wrf[~tiny], act[~tiny]
                assert np.array_equal(act, wrf), (i, name, np.max(np.abs(act - wrf)))
                checked += 1
            for name, j in SCALARS:
                assert np.float32(out[name][pos]) == d['wrf'][i][j, 8], (i, name)
    assert checked > 300 and {'expf', 'tanhf'} <= set(glibc_transcendentals)


def test_compiled_ysu_inside_wrf_input_ulp_envelope():
    """Jitted XLA:CPU path inside WRF's own +-1 ULP input envelope (+2 output ULPs)."""
    d, meta = _load()
    assert meta['envelope']['npert'] >= 96
    for (topdown, flag_qi, _dt), idxs in _groups(d):
        n, out = _run(d, idxs, flag_qi, topdown, eager=False)
        for pos, i in enumerate(idxs):
            for name, j in FIELDS:
                if name == 'qi' and not flag_qi:
                    continue
                lo, hi = d['lower'][i][:n, j], d['upper'][i][:n, j]
                act = out[name][pos]
                widen = 2.0 * EPS32 * np.maximum(np.abs(lo), np.abs(hi))
                if name in ('qc', 'qi'):
                    widen = widen + Q_SUBNORMAL_FLOOR
                bad = (act < lo - widen) | (act > hi + widen) | ~np.isfinite(act)
                assert not bad.any(), (i, name, np.nonzero(bad)[0].tolist())
            assert int(out['kpbl'][pos]) == int(d['wrf'][i][1, 8]), i
            for name, j in (('pblh', 0), ('wstar', 2), ('delta', 3)):
                lo, hi = d['lower'][i][j, 8], d['upper'][i][j, 8]
                widen = 2.0 * EPS32 * max(abs(lo), abs(hi))
                assert lo - widen <= out[name][pos] <= hi + widen, (i, name)


def test_oracle_census_covers_the_ported_branches():
    """E154: the frozen columns exercise every branch this lane ported."""
    d, meta = _load()
    wrf = d['wrf']
    assert d['topdown'][0::2].tolist() == [1] * (len(d['nz']) // 2) and not d['topdown'][1::2].any()
    topdown_sensitive = sum(np.any(wrf[a] != wrf[a + 1]) for a in range(0, len(wrf), 2))
    assert topdown_sensitive >= 10 and meta['census']['topdown_sensitive'] == topdown_sensitive
    assert np.count_nonzero(np.abs(wrf[:, :-1, 5]).max(axis=1) > 1.e-12) >= 6        # RQIBLTEN
    assert np.count_nonzero(np.abs(wrf[:, :-1, 4]).max(axis=1) > 1.e-9) >= 20        # RQCBLTEN
    prof = d['profiles']
    cloud = (prof[:, :-1, 4] + prof[:, :-1, 5]) > 0.01e-3
    assert np.count_nonzero(np.any(cloud[:, :-1] & cloud[:, 1:], axis=1)) >= 10        # in-cloud Ri
    assert (~d['flag_qi']).sum() >= 4 and np.any(prof[:, :-1, 10] < 0)


def _state_and_forcing():
    import sys
    sys.path.insert(0, str(ROOT / 'tests'))
    from test_ysu_surface_contract import column_state
    state, grid, forcing = column_state()
    nz = state.theta.shape[0]
    k = jnp.arange(nz, dtype=jnp.float32).reshape(nz, 1, 1)
    state = state.replace(qc=jnp.where((k >= 3) & (k <= 6), 2.e-4, 0.).astype(jnp.float32),
                          qi=jnp.where((k >= 8) & (k <= 10), 5.e-5, 0.).astype(jnp.float32))
    return state, grid, forcing


def test_adapter_passes_wrf_caller_operands_and_applies_cloud_rates(monkeypatch):
    """module_pbl_driver.F:1208-1251 operands reach the kernel; RQC/RQIBLTEN applied."""
    from gpuwrf.coupling import scan_adapters as adapters
    from gpuwrf.physics.pbl_ysu import ysu_columns
    state, _, forcing = _state_and_forcing()
    nz = state.theta.shape[0]
    rth = jnp.linspace(-3.e-4, 1.e-5, nz, dtype=jnp.float32).reshape(nz, 1, 1)
    seen = {}

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return ysu_columns(*args, **kwargs)

    monkeypatch.setattr(adapters, 'ysu_columns', spy)
    result, rates = adapters.ysu_pbl_adapter(state, 60., surface_forcing=forcing, _with_sources=True,
                                             rthraten=rth, ysu_topdown_pblmix=0)
    col = lambda a: np.asarray(a, np.float32).reshape(nz)
    np.testing.assert_array_equal(np.asarray(seen['qc']).reshape(nz), col(state.qc))
    np.testing.assert_array_equal(np.asarray(seen['qi']).reshape(nz), col(state.qi))
    np.testing.assert_array_equal(np.asarray(seen['rthraten']).reshape(nz), col(rth))
    assert seen['ysu_topdown_pblmix'] == 0 and seen['qc'].dtype == jnp.float32
    for key in ('qc', 'qi'):
        rate = np.asarray(rates[key]).reshape(nz)
        assert np.abs(rate).max() > 0, key
        np.testing.assert_array_equal(col(getattr(result, key)),
                                      (col(getattr(state, key)) + np.float32(60.) * rate).astype(np.float32))


def test_operational_ysu_call_passes_rthraten_and_namelist_topdown(monkeypatch):
    from gpuwrf.runtime import operational_mode as om
    import sys
    sys.path.insert(0, str(ROOT / 'tests'))
    from test_v013_operational_smoke import _grid, _base_state, _namelist
    from gpuwrf.runtime.operational_state import initial_operational_carry
    grid = _grid(nz=12, ny=3, nx=3)
    namelist = dataclasses.replace(_namelist(grid, mp_physics=0, cu_physics=0, bl_pbl_physics=1,
        sf_sfclay_physics=1, sf_surface_physics=0, use_noahmp=False, ra_lw_physics=0, ra_sw_physics=0),
        ysu_topdown_pblmix=0)
    carry = initial_operational_carry(_base_state(grid))
    carry = carry.replace(rthraten=jnp.full_like(carry.rthraten, -2.e-5))
    seen = {}

    def pbl(st, dt, grid, **kwargs):
        seen.update(kwargs)
        zero = jnp.zeros((grid.ny * grid.nx, grid.nz), jnp.float32)
        return st, {key: zero for key in ('u', 'v', 'theta', 'qv', 'qc', 'qi')}

    monkeypatch.setitem(om.PBL_SCAN_ADAPTERS, 1, pbl)
    om._physics_step_forcing(carry, namelist, 0., run_radiation=False, first_timestep=True)
    assert seen.get('ysu_topdown_pblmix') == 0, sorted(seen)
    assert seen.get('rthraten') is not None, sorted(seen)
    np.testing.assert_array_equal(np.asarray(seen['rthraten']), np.asarray(carry.rthraten))


def test_namelist_binds_ysu_topdown_pblmix_and_keeps_default_treedef():
    from gpuwrf.io.namelist_binding import bind_namelist_options, namelist_honour_report
    import sys
    sys.path.insert(0, str(ROOT / 'tests'))
    from test_v013_operational_smoke import _grid, _namelist
    base = _namelist(_grid(nz=12, ny=3, nx=3), bl_pbl_physics=1)
    assert base.ysu_topdown_pblmix == 1
    off = bind_namelist_options(base, {'physics': {'ysu_topdown_pblmix': 0}}, 'd01')
    assert off.ysu_topdown_pblmix == 0
    assert bind_namelist_options(base, {'physics': {}}, 'd01').ysu_topdown_pblmix == 1
    errors, _ = namelist_honour_report({'physics': {'bl_pbl_physics': 1, 'ysu_topdown_pblmix': 0}}, ('d01',))
    assert not any('ysu_topdown_pblmix' in e for e in errors)
    # The Registry default rides no static aux (default programs/keys unchanged);
    # a nondefault value is a static compile constant and splits the treedef.
    _, tree_default = jax.tree_util.tree_flatten(base)
    off_direct = dataclasses.replace(base, ysu_topdown_pblmix=0)
    _, tree_off = jax.tree_util.tree_flatten(off_direct)
    _, tree_on = jax.tree_util.tree_flatten(dataclasses.replace(off_direct, ysu_topdown_pblmix=1))
    assert tree_on == tree_default and tree_off != tree_default
    assert jax.tree_util.tree_unflatten(tree_off, jax.tree_util.tree_leaves(off_direct)).ysu_topdown_pblmix == 0


def _get_pblh_literal(kts, kte, thetav, qke, zw, dz, landsea):
    """Literal do-while transcription of WRF bl_ysu.F90:1586-1693 (1-based,
    REAL4 scalars), independent of the vectorised port."""
    f = np.float32
    th = lambda k: thetav[k - 1]
    q = lambda k: qke[k - 1]
    z = lambda k: zw[k - 1]
    d = lambda k: dz[k - 1]
    k, kthv, ktke, maxqke, minthv = kts + 1, 1, 1, f(0.), f(9.e9)
    while z(k) <= f(500.):
        qtke = max(q(k), f(0.))
        if maxqke < qtke:
            maxqke, ktke = qtke, k
        if minthv > th(k):
            minthv, kthv = th(k), k
        k += 1
    tkeeps = min(max(maxqke / f(40.), f(0.025)), f(0.25))
    delt = f(0.75) if (landsea - f(1.5)) >= 0 else f(1.5)
    zi, k = f(0.), kthv + 1
    while zi == 0.:
        if th(k) >= minthv + delt:
            zi = z(k) - d(k - 1) * min((th(k) - (minthv + delt)) / max(th(k) - th(k - 1), f(1e-6)), f(1.0))
        k += 1
        if k == kte - 1:
            zi = z(kts + 1)
    pblh_tke, k = f(0.), ktke + 1
    while pblh_tke == 0.:
        qtke, qtkem1 = max(q(k) / f(2.), f(0.)), max(q(k - 1) / f(2.), f(0.))
        if qtke <= tkeeps:
            pblh_tke = max(z(k) - d(k - 1) * min((tkeeps - qtke) / max(qtkem1 - qtke, f(1e-6)), f(1.0)), z(kts + 1))
        k += 1
        if k == kte - 1:
            pblh_tke = z(kts + 1)
    wt = f(.5) * f(np.tanh(np.float32((zi - f(200.)) / f(400.)))) + f(.5)
    return pblh_tke * (f(1.) - wt) + zi * wt


def test_get_pblh_vectorised_matches_literal_wrf_loops():
    """The real columns never reach get_pblh's exit safeguards; synthetic ones do:
    theta-v that never exceeds the 500 m minimum + delta (safeguard), a hit
    exactly at k=kte-2 (WRF overwrites it), zero TKE, monotone TKE."""
    from gpuwrf.physics.pbl_ysu import _get_pblh
    rng = np.random.default_rng(7)
    n = 20
    cases = []
    for case in range(40):
        dz = np.float32(rng.uniform(40., 400., n))
        zw = np.concatenate([[0.], np.cumsum(dz)]).astype(np.float32)
        thv = np.float32(290. + np.cumsum(rng.uniform(-.3, .9, n)))
        tke = np.float32(np.maximum(rng.uniform(-.2, 2., n), 0.) * (case % 3 != 0))
        if case % 4 == 1:
            thv = np.float32(np.full(n, 290.) + 1.e-3 * np.arange(n))          # never exceeds min + delta
        if case % 4 == 2:
            thv = np.float32(np.full(n, 290.)); thv[n - 3:] = 295.             # first hit at k = kte-2
            tke = np.float32(np.full(n, 1.)); tke[n - 3:] = 0.
        cases.append((thv, tke, zw, dz, np.float32(1. + (case % 2))))
    with jax.enable_x64(False):
        for thv, tke, zw, dz, land in cases:
            want = _get_pblh_literal(1, n, thv, tke, zw, dz, land)
            got = np.float32(_get_pblh(jnp.asarray(thv), jnp.asarray(tke), jnp.asarray(zw), jnp.asarray(dz),
                                       jnp.asarray(land)))
            # tanh: XLA vs glibc may differ by one ULP of the blend weight.
            assert abs(got - want) <= 4 * EPS32 * max(abs(want), 1.), (got, want)
            assert (got == np.float32(zw[1])) == (want == np.float32(zw[1]))


def _radiation_case(monkeypatch):
    """Legacy LW1/SW1 radiation (no RRTMG driver carry) with YSU; sentinel
    radiation and PBL record call order and operands."""
    from gpuwrf.runtime import operational_mode as om
    import sys
    sys.path.insert(0, str(ROOT / 'tests'))
    from test_v013_operational_smoke import _grid, _base_state, _namelist
    from gpuwrf.runtime.operational_state import initial_operational_carry
    grid = _grid(nz=12, ny=3, nx=3)
    namelist = _namelist(grid, mp_physics=0, cu_physics=0, bl_pbl_physics=1, sf_sfclay_physics=1,
                         sf_surface_physics=0, use_noahmp=False, ra_lw_physics=1, ra_sw_physics=1, rad_rk_tendf=1)
    entry = _base_state(grid)
    carry = initial_operational_carry(entry)
    carry = carry.replace(rthraten=jnp.full_like(carry.rthraten, -2.e-5))
    assert carry.radiation_diagnostics is None
    log = {'calls': [], 'rad_inputs': []}

    def radiation(st, *args, **kwargs):
        log['calls'].append('radiation')
        log['rad_inputs'].append(st)
        return jnp.full_like(st.theta, -1.e-4)

    monkeypatch.setattr(om, 'rrtm_lw_theta_tendency', radiation)
    monkeypatch.setattr(om, 'dudhia_sw_theta_tendency', radiation)

    def pbl(st, dt, grid, **kwargs):
        log['calls'].append('pbl')
        log['pbl_rthraten'] = np.asarray(kwargs['rthraten'])
        zero = jnp.zeros((grid.ny * grid.nx, grid.nz), jnp.float32)
        return st, {key: zero for key in ('u', 'v', 'theta', 'qv', 'qc', 'qi')}

    monkeypatch.setitem(om.PBL_SCAN_ADAPTERS, 1, pbl)
    return om, namelist, entry, carry, log


def test_legacy_radiation_refresh_reaches_ysu_once_from_entry_state(monkeypatch):
    """WRF first_rk_step_part1: radiation_driver (:264) before surface/PBL (:1113);
    YSU reads this step's RTHRATEN (pbl_driver :1239). One SW + one LW solve,
    on the step-entry state, reused as the held rate."""
    om, namelist, entry, carry, log = _radiation_case(monkeypatch)
    forcing = om._physics_step_forcing(carry, namelist, 0., run_radiation=True, first_timestep=True)
    assert log['calls'] == ['radiation', 'radiation', 'pbl'], log['calls']
    current = np.asarray(forcing.carry.rthraten)
    np.testing.assert_allclose(current, -2.e-4, rtol=1.e-6, atol=0.)  # SW + LW sentinel, solved once each
    np.testing.assert_array_equal(log['pbl_rthraten'], current)
    for st in log['rad_inputs']:  # entry state, not the post-surface-layer state
        for name in ('theta', 'qv', 'ustar', 'hfx', 't_skin'):
            np.testing.assert_array_equal(np.asarray(getattr(st, name)), np.asarray(getattr(entry, name)), err_msg=name)


def test_legacy_radiation_held_step_feeds_ysu_the_held_rate(monkeypatch):
    om, namelist, entry, carry, log = _radiation_case(monkeypatch)
    forcing = om._physics_step_forcing(carry, namelist, 0., run_radiation=False, first_timestep=False)
    assert log['calls'] == ['pbl'], log['calls']
    np.testing.assert_array_equal(log['pbl_rthraten'], np.asarray(carry.rthraten))
    np.testing.assert_array_equal(np.asarray(forcing.carry.rthraten), np.asarray(carry.rthraten))


def test_ssib_ysu_rates_enter_rk_once_and_restore_entry_state(monkeypatch, tmp_path):
    """c40 SSiB+YSU: all six raw YSU rates reach the RK source lanes once
    (module_physics_addtendc.F:309,316); prognostics return to the PBL entry,
    the PBL diagnostics (PBLH) are retained; the PBL is solved once."""
    import sys
    from types import SimpleNamespace
    sys.path.insert(0, str(ROOT / 'tests'))
    from test_s2_ssib_wiring import _case
    from gpuwrf.runtime import operational_mode as om
    from gpuwrf.coupling import ssib_surface_hook as hook
    from gpuwrf.coupling import scan_adapters as adapters
    import gpuwrf.physics.pbl_ysu as ysu
    grid, state, ns, carry = _case(tmp_path)
    ns = dataclasses.replace(ns, rad_rk_tendf=1)
    z = jnp.zeros((2, 3), jnp.float32)
    # swdown: the SWDOWN SSiB consumes at slope_rad=0 (surface_driver.F:1918-1936).
    carry = carry.replace(radiation_diagnostics=SimpleNamespace(
        swdown=z, swnorm=z, glw=z, coszen=z, cloud_fraction=jnp.zeros_like(state.theta)))
    monkeypatch.setattr(om, '_refresh_rrtmg_driver', lambda c, *a, **k: c)
    monkeypatch.setattr(om, '_phys_tend_rk_wrf_enabled', lambda: True)
    monkeypatch.setattr(hook, 'ssib_day', lambda *a, **k: 1.)
    monkeypatch.setattr(hook, 'ssib_radiation_surface', lambda *a, **k: None)
    surface = {key: jnp.ones_like(z) for key in ('fm', 'fh', 'br', 'u10', 'v10')}
    monkeypatch.setattr(adapters, 'sfclay_revised_mm5_adapter', lambda st, *a, **k: (st, surface))
    history = carry.history_diagnostics._asdict()
    monkeypatch.setattr(hook, 'ssib_surface_step', lambda st, land, *a, **k: (st, land, history))
    values = dict(u=1.e-3, v=-2.e-3, theta=3.e-4, qv=4.e-7, qc=1.e-7, qi=2.e-7)
    solves, builder = [], []
    original_builder = om._source_leaf_dry_tendencies

    def spy_builder(*args, **kwargs):  # (mu, held_rthraten, rthblten, rqvblten, rublten, rvblten, ...)
        builder.append(args[2:6])
        return original_builder(*args, **kwargs)
    monkeypatch.setattr(om, '_source_leaf_dry_tendencies', spy_builder)

    def run(scale):
        rates = {key: jnp.full((6, 6), value * scale, jnp.float32) for key, value in values.items()}

        def columns(*args, **kwargs):
            solves.append(scale)
            return dict(rates, pblh=jnp.full((6,), 432., jnp.float32), exch_h=jnp.zeros((6, 6), jnp.float32))
        monkeypatch.setattr(ysu, 'ysu_columns', columns)
        return om._physics_step_forcing(carry, ns, 0., run_radiation=False, first_timestep=True)

    forcing, zero = run(1.0), run(0.0)
    assert solves == [1.0, 0.0]
    for key in ('u', 'v', 'theta', 'qv', 'qc', 'qi'):
        np.testing.assert_array_equal(np.asarray(getattr(forcing.state, key)), np.asarray(getattr(state, key)), err_msg=key)
    np.testing.assert_array_equal(np.asarray(forcing.state.pblh), np.full((2, 3), 432., np.float32))
    raw = forcing.dry_tendencies.moist_tendf
    mass = np.asarray(ns.metrics.c1h)[:, None, None] * np.asarray(state.mu_total)[None] + np.asarray(ns.metrics.c2h)[:, None, None]
    for key in ('qv', 'qc', 'qi'):
        np.testing.assert_allclose(np.asarray(raw[key]), mass * values[key], rtol=8 * EPS32, atol=0., err_msg=key)
    # Raw dry/A-grid RTHBLTEN, RQVBLTEN, RUBLTEN, RVBLTEN reach the RK source builder
    # exactly once per step (the 2x3 grid has no interior u/v face to inspect ru_tendf).
    assert len(builder) == 2 and all(b is not None for b in builder[0]) and builder[1][0] is not None
    for got, key in zip(builder[0], ('theta', 'qv', 'u', 'v')):
        np.testing.assert_array_equal(np.asarray(got), np.full(state.theta.shape, np.float32(values[key])), err_msg=key)
    a, b = np.asarray(forcing.dry_tendencies.t_tendf), np.asarray(zero.dry_tendencies.t_tendf)
    assert np.abs(a - b).max() > 0
