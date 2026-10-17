"""t-cfix-sfc F13 (rv-t-cfix FIX 22:54Z / SFC2 01:28Z): the sf3/bl3 GFS PBL is WRF's BL_GFS call.

WRF BL_GFS evaluates its pure-REAL right-hand sides (QV3D/(1.+QV3D), P3D*.001, Z differences
times G ...) in REAL and stores them into REAL(kind_phys) work arrays (module_bl_gfs.F:276-398);
MONINP runs in DOUBLE (module_gfs_machine.F:8) with default-REAL literal constants widened
(module_gfs_physcons.F, :663-672); the REAL tendencies, UST=SQRT(STRESS) and
WSPD=SQRT(U3D(kts)**2+V3D(kts)**2)+1.E-9 are stored back (:432-466); the next SF_GFS reads
that UST.  References: the UNMODIFIED BL_GFS public call (blgfs_oracle.f90) on real CPU-WRF c04
columns fed by the pristine SF_GFS and on the ACTUAL caller operands, and a BYTE-checked copy-only
capture of what that call hands to / gets back from MONINP (blgfs_capture.f90).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.coupling import scan_adapters as sa
from gpuwrf.physics import bl_gfs

ROOT = Path(__file__).resolve().parents[2]
FIX = np.load(ROOT / 'proofs/t_cfix_sfc/fixtures/blgfs_wrf_oracle.npz')
META = json.loads((ROOT / 'proofs/t_cfix_sfc/fixtures/blgfs_wrf_oracle.json').read_text())
PROF, SFC, TEND, DIAG = META['prof'], META['sfc'], META['tend'], META['diag']
CAP = np.load(ROOT / 'proofs/t_cfix_sfc/fixtures/blgfs_capture_oracle.npz')
CAPM = json.loads((ROOT / 'proofs/t_cfix_sfc/fixtures/blgfs_capture_oracle.json').read_text())
# Compiled REAL setup + DOUBLE MONINP vs the gfortran build: inside WRF's own +-1 ULP input
# envelope, no extra output slack.
SLACK_ULP = 0
# DOUBLE MONINP on the captured pristine operands, |port-WRF| <= tol * max|WRF| per field.
# [M CPU] measured <= 4.2e-13 (XLA:CPU contraction/constant-divisor rounding); exact-decimal
# constants instead of WRF's widened REAL literals miss by 1.2e-8 .. 2.8e-7.
MONINP_REL_TOL = 1.0e-11
TINY = np.finfo(np.float32).tiny


def _ulp(a, b):
    a = np.asarray(a, np.float32).view(np.int32).astype(np.int64)
    b = np.asarray(b, np.float32).view(np.int32).astype(np.int64)
    a = np.where(a < 0, -(a & 0x7fffffff), a)
    b = np.where(b < 0, -(b & 0x7fffffff), b)
    return np.abs(a - b)


def _ftz(a):  # E117: XLA:CPU flushes subnormal results, WRF/GPU keep them
    a = np.asarray(a, np.float32)
    return np.where(np.abs(a) < TINY, np.float32(0), a)


def _inside(value, lo, hi):
    value, lo, hi = _ftz(value), _ftz(lo), _ftz(hi)
    return np.maximum(_ulp(np.minimum(value, lo), lo), _ulp(np.maximum(value, hi), hi))


def _port_outputs(prof, sfc, has_qi, dt):
    nz = prof.shape[1] - 1
    col = lambda k: jnp.asarray(prof[:, :nz, PROF.index(k)])
    s = lambda k: jnp.asarray(sfc[:, SFC.index(k)])
    args = (col('u'), col('v'), col('t'), col('qv'), col('qc'), col('p'), col('pi'), col('dz'), col('z'))
    kw = dict(psfc=s('psfc'), ust=s('ust'), hfx=s('hfx'), qfx=s('qfx'), tsk=s('tsk'), gz1oz0=s('gz1oz0'),
              psim=s('psim'), psih=s('psih'), wspd=s('wspd'), br=s('br'), qi=col('qi') if has_qi else None)
    with jax.enable_x64(False):  # the operational caller's REAL scope
        return sa.gfs_bl_real_double(args, kw, dt)


def _assert_envelope(out, prefix, has_qi):
    checked = 0
    for j, name in enumerate(TEND):
        if name == 'rqiblten' and not has_qi:
            continue
        value = np.asarray(out[name])
        assert value.dtype == np.float32 and np.isfinite(value).all(), name
        bad = _inside(value, FIX[f'{prefix}_tend_lower'][..., j], FIX[f'{prefix}_tend_upper'][..., j])
        assert bad.max() <= SLACK_ULP, (prefix, name, int(bad.max()), np.unravel_index(int(bad.argmax()), bad.shape))
        checked += 1
    for j, (name, key) in enumerate((('ust', 'ust'), ('pbl', 'pblh'))):
        value = np.asarray(out[key])
        assert value.dtype == np.float32, key
        bad = _inside(value, FIX[f'{prefix}_diag_lower'][:, j], FIX[f'{prefix}_diag_upper'][:, j])
        assert bad.max() <= SLACK_ULP, (prefix, name, int(bad.max()))
        checked += 1
    np.testing.assert_array_equal(np.asarray(out['kpbl']), FIX[f'{prefix}_zero_diag'][:, DIAG.index('kpbl')])
    return checked


def test_wrf_bl_gfs_on_real_columns_fed_by_pristine_sf_gfs():
    for has_qi in (1, 0):
        out = _port_outputs(FIX['real_prof'], FIX['real_sfc'], has_qi, float(FIX['dt']))
        assert _assert_envelope(out, f'real_q{has_qi}', has_qi) == (8 if has_qi else 7)


def test_bl_gfs_constants_are_the_values_wrf_compiles():
    """gfortran receipt (proofs/t_cfix_sfc/blgfs_constants.f90 with the pristine .mod files)."""
    rows = {}
    for line in (ROOT / 'proofs/t_cfix_sfc/fixtures/blgfs_constants.txt').read_text().splitlines():
        rows[line.split()[0]] = [t for t in line.split() if re.fullmatch('[0-9A-F]{8}|[0-9A-F]{16}', t)]
    f32 = lambda x: format(int(np.asarray(x, np.float32).view(np.uint32)), '08X')  # noqa: E731
    f64 = lambda x: format(int(np.asarray(x, np.float64).view(np.uint64)), '016X')  # noqa: E731
    c = bl_gfs.WRF_MONINP
    assert rows['REAL'] == [f32(v) for v in (bl_gfs.WRF_R_D, bl_gfs.WRF_CP, bl_gfs.WRF_G, bl_gfs.WRF_RCP,
                                             bl_gfs.WRF_ROVG, bl_gfs.WRF_EP_1, bl_gfs.WRF_KARMAN)]
    assert rows['PHYS'] == [f64(v) for v in (bl_gfs.WRF_CON_G, bl_gfs.WRF_CON_RD, bl_gfs.WRF_CON_RV,
                                             bl_gfs.WRF_CON_CP, c.FV)]
    assert c.CON_G == bl_gfs.WRF_CON_G
    assert rows['MONINP'] == [f64(v) for v in (c.GOR, c.GOCP, c.VK, c.DW2MIN, c.CFAC, c.SFCFRAC, c.QMIN, c.ZFMIN)]
    assert rows['INLINE'] == [f64(v) for v in (c.PRINV_LIT, c.DKU_LIT, c.DKT_LIT, c.PRNUM_LIT)]


def _capture_case(case):
    prefix, has_qi = CAPM['cases'][case]
    prof, sfc = FIX[f'{prefix}_prof'], FIX[f'{prefix}_sfc']
    return prof, sfc, has_qi, prof.shape[1] - 1


def _port_stage(prof, sfc, has_qi, nz):
    col = lambda k: jnp.asarray(prof[:, :nz, PROF.index(k)])  # noqa: E731  REAL grid operands
    s = lambda k: jnp.asarray(sfc[:, SFC.index(k)])  # noqa: E731
    surface = {k: s(k) for k in ('psfc', 'hfx', 'qfx', 'tsk', 'gz1oz0', 'psim', 'psih', 'wspd', 'br')}
    with jax.enable_x64(True):  # compiled like the production step (XLA rewrites included)
        stage = jax.jit(lambda profiles, surface, qi: bl_gfs.bl_gfs_wrf_stage(*profiles, **surface, qi=qi))
        return stage(tuple(col(k) for k in ('u', 'v', 't', 'qv', 'qc', 'p', 'dz', 'z')), surface,
                     col('qi') if has_qi else None)


def test_bl_gfs_real_setup_is_byte_identical_to_the_pristine_moninp_operands():
    """module_bl_gfs.F:276-398 REAL right-hand sides then kind_phys stores == what WRF passes MONINP."""
    for case in CAPM['cases']:
        prof, sfc, has_qi, nz = _capture_case(case)
        work = _port_stage(prof, sfc, has_qi, nz)
        stage, scal = CAP[f'{case}_stage'], CAP[f'{case}_scal']
        checked = 0
        for j, name in enumerate(CAPM['stage']):
            if name == 'q1i' and not has_qi:
                assert work['q1i'] is None
                continue
            value = np.asarray(work[{'del': 'delp'}.get(name, name)])
            assert value.dtype == np.float64, name
            reference = stage[:, :nz + 1 if name in ('prsi', 'phii') else nz, j]
            np.testing.assert_array_equal(value.view(np.uint64), reference.view(np.uint64), err_msg=(case, name))
            checked += 1
        for j, name in enumerate(CAPM['scal'][:-1]):
            np.testing.assert_array_equal(np.asarray(work[name]).view(np.uint64), scal[:, j].view(np.uint64),
                                          err_msg=(case, name))
            checked += 1
        dt = float(FIX['caller_dt'] if CAPM['cases'][case][0] == 'caller' else FIX['dt'])
        assert float(np.float32(dt)) == scal[0, -1]  # DELTIM=DT (REAL dtbl)
        assert checked == (21 if has_qi else 20), (case, checked)


def _moninp_on_captured_operands(case, constants=None):
    prof, sfc, has_qi, nz = _capture_case(case)
    stage, scal = CAP[f'{case}_stage'], CAP[f'{case}_scal']
    work = {{'del': 'delp'}.get(name, name): jnp.asarray(stage[:, :nz + 1 if name in ('prsi', 'phii') else nz, j])
            for j, name in enumerate(CAPM['stage']) if name != 'q1i'}
    work['q1i'] = jnp.asarray(stage[:, :nz, CAPM['stage'].index('q1i')]) if has_qi else None
    work.update({name: jnp.asarray(scal[:, j]) for j, name in enumerate(CAPM['scal'][:-1])})
    work['qfx'] = jnp.zeros(prof.shape[0])  # QSS path only (dead in MONINP)
    pii = jnp.asarray(prof[:, :nz, PROF.index('pi')])
    with jax.enable_x64(True):  # compiled like the production step
        if constants is None:
            return jax.jit(lambda w, pi: bl_gfs.bl_gfs_wrf_moninp(w, pi, float(scal[0, -1])))(work, pii)
        out = jax.jit(jax.vmap(lambda w, pi: bl_gfs._moninp_and_store(
            w, jnp.asarray(pi, jnp.float64), float(scal[0, -1]), constants)))(work, pii)
        return dict(zip(('rublten', 'rvblten', 'rthblten', 'rqvblten', 'rqcblten', 'rqiblten', 'pblh', 'kpbl',
                         'ust'), out))


def _pristine_moninp(case):
    prof, _, has_qi, nz = _capture_case(case)
    stage, scal, mout, mdiag = (CAP[f'{case}_{k}'] for k in ('stage', 'scal', 'mout', 'mdiag'))
    pii = prof[:, :nz, PROF.index('pi')].astype(np.float64)
    q = lambda j: stage[:, :nz, j]  # noqa: E731
    expect = dict(rublten=mout[..., 0], rvblten=mout[..., 1], rthblten=mout[..., 2] / pii,  # :432-439
                  rqvblten=mout[..., 3] / ((1. - q(3)) * (1. - q(3))), rqcblten=mout[..., 4] / ((1. - q(4)) * (1. - q(4))),
                  pblh=mdiag[:, 0], ust=np.sqrt(scal[:, CAPM['scal'].index('stress')]))
    if has_qi:
        expect['rqiblten'] = mout[..., 5] / ((1. - q(5)) * (1. - q(5)))
    return expect, mdiag[:, 1]


def _moninp_misses(out, expect):
    misses = {}
    for name, reference in expect.items():
        value = np.asarray(out[name])
        assert value.dtype == np.float64 and np.isfinite(value).all() and np.isfinite(reference).all(), name
        misses[name] = float(np.max(np.abs(value - reference)) / max(float(np.max(np.abs(reference))), 1e-300))
    return misses


def test_double_moninp_matches_pristine_moninp_on_its_captured_operands():
    """Genuine DOUBLE MONINP, separately from the REAL setup (rv-t-cfix SFC2 series 1)."""
    for case in CAPM['cases']:
        expect, kpbl = _pristine_moninp(case)
        out = _moninp_on_captured_operands(case)
        np.testing.assert_array_equal(np.asarray(out['kpbl']), kpbl.astype(np.int64), err_msg=case)
        np.testing.assert_array_equal(np.asarray(out['ust']), expect['ust'], err_msg=case)
        misses = _moninp_misses(out, expect)
        assert max(misses.values()) <= MONINP_REL_TOL, (case, misses)


def test_double_moninp_gate_rejects_exact_decimal_constants():
    """The MONINP gate separates WRF's widened REAL literals from exact decimals (deletion check)."""
    for case in CAPM['cases']:
        expect, _ = _pristine_moninp(case)
        misses = _moninp_misses(_moninp_on_captured_operands(case, bl_gfs.LEGACY_MONINP), expect)
        assert max(misses.values()) > 100 * MONINP_REL_TOL, (case, misses)


def _caller(monkeypatch, steps=1, wspd_leaf=False):
    from gpuwrf.runtime import operational_mode as om
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from gpuwrf.coupling import sfclay_selected
    from test_t_cfix_sfc_selected_surface import _case, _forbid_resolves
    _, state, nl = _case(3, noahmp=False, bl=3)
    if wspd_leaf:  # optional grid WSPD leaf (present when a WSPD consumer is configured)
        # Calm lowest level (< 1 m/s): SF_GFS floors its WSPD at 1 m/s (PROGTM) while BL_GFS
        # stores the raw SQRT(U**2+V**2)+1.E-9, so the two writers are distinguishable.
        state = state.replace(_cast=False, sfc_wspd=jnp.full(state.t_skin.shape, 1.25, jnp.float32),
                              u=state.u.at[0].multiply(.05), v=state.v.at[0].multiply(.05))
    om._resolve_operational_suite(nl)
    _forbid_resolves(monkeypatch, om)
    seen = {'pbl': [], 'surface': []}
    real_pbl, real_surface = sa.gfs_bl_real_double, sfclay_selected.column_call
    real_stage, real_moninp = bl_gfs.bl_gfs_wrf_stage, bl_gfs.bl_gfs_wrf_moninp

    def pbl(args, kwargs, dt):
        out = real_pbl(args, kwargs, dt)
        seen['pbl'].append(dict(args=[np.asarray(a) for a in args], kwargs={k: None if v is None else np.asarray(v)
                                for k, v in kwargs.items()}, dt=dt, out={k: np.asarray(v) for k, v in out.items()}))
        return out

    def typed(name, jaxpr):
        # Typed arithmetic of the ACTUAL kernel calls (REAL setup, DOUBLE MONINP).
        counts = {'f32': 0, 'f64': 0}

        def visit(value):
            if hasattr(value, 'jaxpr'):
                visit(value.jaxpr)
            elif hasattr(value, 'eqns'):
                for eqn in value.eqns:
                    if eqn.primitive.name in ('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'exp', 'pow'):
                        for var in eqn.outvars:
                            dtype = var.aval.dtype
                            counts['f64' if dtype == np.float64 else 'f32' if dtype == np.float32 else 'other'] = \
                                counts.get('f64' if dtype == np.float64 else 'f32' if dtype == np.float32 else 'other', 0) + 1
                    for child in eqn.params.values():
                        visit(child)
            elif isinstance(value, (tuple, list)):
                for child in value:
                    visit(child)
        visit(jaxpr)
        seen.setdefault(name, []).append(counts)

    def stage(*args, **kwargs):
        typed('typed_stage', jax.make_jaxpr(lambda *values: real_stage(*values, **kwargs))(*args))
        return real_stage(*args, **kwargs)

    def moninp(work, pii, dt):
        typed('typed', jax.make_jaxpr(lambda w, p: real_moninp(w, p, dt))(work, pii))
        return real_moninp(work, pii, dt)

    def surface(sf, inputs, itimestep):
        seen['surface'].append({k: np.asarray(v) for k, v in inputs.items()})
        return real_surface(sf, inputs, itimestep)
    monkeypatch.setattr(sa, 'gfs_bl_real_double', pbl)
    monkeypatch.setattr(bl_gfs, 'bl_gfs_wrf_stage', stage)
    monkeypatch.setattr(bl_gfs, 'bl_gfs_wrf_moninp', moninp)
    monkeypatch.setattr(sfclay_selected, 'column_call', surface)
    carry = initial_operational_carry(state)
    results = []
    for step in range(steps):
        result = om._physics_step_forcing(carry, nl, 20. * step, run_radiation=False, first_timestep=step == 0)
        results.append(result)
        carry = result.carry.replace(state=result.state)
    return state, results, seen


def test_actual_sf3_bl3_caller_is_the_pristine_bl_gfs_call(monkeypatch):
    _, results, seen = _caller(monkeypatch)
    assert len(seen['pbl']) == 1, 'the sf3/bl3 caller did not make the WRF BL_GFS (REAL+DOUBLE) call'
    call = seen['pbl'][0]
    # The caller hands BL_GFS REAL grid operands (WRF REAL arrays) ...
    assert all(a.dtype == np.float32 for a in call['args'])
    assert all(v is None or v.dtype == np.float32 for v in call['kwargs'].values())
    # ... the setup evaluates REAL right-hand sides (module_bl_gfs.F:276-398) and MONINP is DOUBLE
    # (kind_phys, module_gfs_machine.F:8) ...
    assert seen['typed_stage'][0].get('f32', 0) > 0, seen['typed_stage'][0]
    assert seen['typed'][0].get('f64', 0) > 0 and seen['typed'][0].get('f32', 0) == 0, seen['typed'][0]
    # ... and those exact operands reproduce the frozen pristine-WRF caller oracle.
    nz = call['args'][0].shape[1]
    for k, name in enumerate(('u', 'v', 't', 'qv', 'qc', 'p', 'pi', 'dz', 'z')):
        np.testing.assert_array_equal(call['args'][k], FIX['caller_prof'][:, :nz, PROF.index(name)], err_msg=name)
    for name in SFC:
        if name != 'pbl':
            np.testing.assert_array_equal(call['kwargs'][name], FIX['caller_sfc'][:, SFC.index(name)], err_msg=name)
    assert float(call['dt']) == float(FIX['caller_dt']) and int(FIX['caller_has_qi']) == 0
    _assert_envelope(call['out'], 'caller', 0)


def test_bl_gfs_ust_and_wspd_inout_survive_into_the_next_surface_call(monkeypatch):
    entry, results, seen = _caller(monkeypatch, steps=2, wspd_leaf=True)
    out = seen['pbl'][0]['out']
    state1 = results[0].state
    np.testing.assert_array_equal(np.asarray(state1.ustar, np.float32), out['ust'].reshape(state1.ustar.shape))
    # module_bl_gfs.F:465-466 REAL WSPD store from the PBL's own lowest-level winds.
    u0, v0 = seen['pbl'][0]['args'][0][:, 0], seen['pbl'][0]['args'][1][:, 0]
    expected = (np.sqrt(u0 * u0 + v0 * v0) + np.float32(1.e-9)).astype(np.float32)
    assert expected.max() < 1.  # calm: distinct from SF_GFS's floored WSPD
    np.testing.assert_allclose(np.asarray(state1.sfc_wspd, np.float32).reshape(-1), expected,
                               rtol=2 * np.finfo(np.float32).eps, atol=0)
    # The selected SF_GFS UST was overwritten by BL_GFS and THAT value is the next UST INOUT.
    surface1 = seen['surface'][0]
    assert not np.array_equal(out['ust'].reshape(-1), np.asarray(surface1['ust']).reshape(-1))
    np.testing.assert_array_equal(np.asarray(seen['surface'][1]['ust']).reshape(-1), out['ust'].reshape(-1))
