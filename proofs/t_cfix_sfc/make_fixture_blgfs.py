"""Freeze the pristine BL_GFS gates (lane t-cfix-sfc): real c04 columns + ACTUAL operational caller operands.

usage: PYTHONPATH=src JAX_PLATFORMS=cpu python make_fixture_blgfs.py <lane_dir>
Part 'real': every column of <lane>/oracle/oracle_blgfs.npz (pristine SF_GFS -> BL_GFS chain on CPU-WRF
c04 profiles) with its +-1 ULP envelope.  Part 'caller': the operands the ACTUAL sf3/bl3 operational step
(_physics_step_forcing, tests/v040 _case) hands to the WRF BL_GFS call, captured at
scan_adapters.gfs_bl_real_double, and the UNMODIFIED BL_GFS public call (zero/fma builds, 64-draw
+-1 ULP envelope) on exactly those REAL operands.  Writes proofs/t_cfix_sfc/fixtures/blgfs_wrf_oracle.*
"""
from pathlib import Path
import ctypes, hashlib, json, sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / 'tests'), str(ROOT / 'tests/v040')]
LANE = Path(sys.argv[1])
from make_oracle_blgfs import PROF, SFC, TEND, DIAG, call, nudge, lib, A  # noqa: E402


def caller_operands():
    import pytest
    from gpuwrf.coupling import scan_adapters as sa
    from gpuwrf.runtime import operational_mode as om
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from test_t_cfix_sfc_selected_surface import _case
    _, state, nl = _case(3, noahmp=False, bl=3)
    om._resolve_operational_suite(nl)
    seen = {}
    real = sa.gfs_bl_real_double

    def spy(args, kwargs, dt):
        seen.update(args=[np.asarray(a) for a in args], kwargs={k: None if v is None else np.asarray(v)
                                                                 for k, v in kwargs.items()}, dt=dt)
        return real(args, kwargs, dt)
    mp = pytest.MonkeyPatch()
    mp.setattr(sa, 'gfs_bl_real_double', spy)
    try:
        om._physics_step_forcing(initial_operational_carry(state), nl, 0., run_radiation=False, first_timestep=True)
    finally:
        mp.undo()
    u, v, t, qv, qc, p, pii, dz, z = seen['args']
    n, nz = u.shape
    prof = np.zeros((n, nz + 1, len(PROF)), np.float32)
    kw = seen['kwargs']
    qi = kw['qi'] if kw['qi'] is not None else np.zeros_like(qv)
    for name, val in zip(PROF, (u, v, t / pii, t, qv, qc, qi, p, pii, dz, z)):
        prof[:, :nz, PROF.index(name)] = val; prof[:, nz, PROF.index(name)] = val[:, -1]
    sfc = np.zeros((n, len(SFC)), np.float32)
    for name in SFC:
        sfc[:, SFC.index(name)] = kw['tsk' if name == 'tsk' else name] if name != 'pbl' else 0.
    return prof, sfc, int(kw['qi'] is not None), float(seen['dt'])


def main():
    d = np.load(LANE / 'oracle' / 'oracle_blgfs.npz')
    rows = np.arange(0, d['prof'].shape[0], 2)  # every second column keeps all surfaces/frames
    arrays = {'real_rows': rows, 'real_prof': d['prof'][rows], 'real_sfc': d['sfc'][rows], 'dt': d['dt']}
    for q in (1, 0):
        for part in ('zero_tend', 'zero_diag', 'tend_lower', 'tend_upper', 'diag_lower', 'diag_upper'):
            arrays[f'real_q{q}_{part}'] = d[f'q{q}_{part}'][rows]
    prof, sfc, has_qi, dt = caller_operands()
    args = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, A, A, A, A]
    zero = lib(LANE / 'oracle' / 'pbl3' / 'zero' / 'oracle.so', args)
    fma = lib(LANE / 'oracle' / 'pbl3' / 'fma' / 'oracle.so', args)
    import make_oracle_blgfs as mo
    mo.DT = dt
    tends, diags = [], []
    for fn in (zero, fma):
        t, dg = call(fn, prof, sfc, has_qi); tends.append(t); diags.append(dg)
    rng = np.random.default_rng(20261018)
    for _ in range(64):
        t, dg = call(zero, nudge(prof, rng), nudge(sfc, rng), has_qi); tends.append(t); diags.append(dg)
    arrays.update(caller_prof=prof, caller_sfc=sfc, caller_has_qi=np.array(has_qi), caller_dt=np.array(dt),
                  caller_zero_tend=tends[0], caller_zero_diag=diags[0],
                  caller_tend_lower=np.min(tends, 0), caller_tend_upper=np.max(tends, 0),
                  caller_diag_lower=np.min(diags, 0), caller_diag_upper=np.max(diags, 0))
    path = HERE / 'fixtures' / 'blgfs_wrf_oracle.npz'
    np.savez_compressed(path, **arrays)
    manifest = json.loads((LANE / 'oracle' / 'manifest.json').read_text())
    meta = dict(fixture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source='WRF 4.7.1 phys/module_bl_gfs.F (+gfs machine/physcons, share/module_model_constants.F), unmodified',
                builds={k: v for k, v in manifest['builds'].items() if k.startswith('pbl3/')},
                wrapper_sha256=manifest['wrappers'].get('blgfs_oracle.f90'),
                real_oracle_npz_sha256=json.loads((LANE / 'oracle' / 'oracle_blgfs.json').read_text())['npz_sha256'],
                prof=PROF, sfc=SFC, tend=TEND, diag=DIAG, caller_has_qi=has_qi, caller_dt=dt,
                rule='compiled REAL wrapper + DOUBLE solver inside [lower, upper] (+-1 ULP input draws + FMA build) '
                     '+8 REAL4 output ULPs; subnormal results flushed (E117)',
                caller='tests/v040/test_t_cfix_sfc_selected_surface._case(3, noahmp=False, bl=3), first step')
    (HERE / 'fixtures' / 'blgfs_wrf_oracle.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(meta['fixture_sha256'], path.stat().st_size, prof.shape, has_qi, dt)


if __name__ == '__main__':
    main()
