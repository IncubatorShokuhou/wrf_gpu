"""Freeze what the pristine BL_GFS hands to / gets from MONINP (lane t-cfix-sfc F13 FIX, no JAX).

rv-t-cfix SFC2 series 1: the REAL setup statements and the DOUBLE MONINP must be gated
separately. The 'pbl3cap' build (build_oracle.py) is the UNMODIFIED module_bl_gfs.F plus two
copy-only CALLs around CALL MONINP (blgfs_capture.f90). Before anything is frozen, its BL_GFS
outputs must be BYTE-identical to (a) the unmodified 'pbl3' zero build called now and (b) the
frozen pristine outputs in fixtures/blgfs_wrf_oracle.npz, on the same operands: the frozen real
c04 columns (QI on/off) and the ACTUAL operational caller columns.
usage: python make_fixture_blgfs_capture.py <lane_dir>
"""
from pathlib import Path
import ctypes, hashlib, json, sys
import numpy as np

HERE = Path(__file__).resolve().parent
LANE = Path(sys.argv[1])
FIXTURE = HERE / 'fixtures' / 'blgfs_wrf_oracle.npz'
A32 = np.ctypeslib.ndpointer(dtype=np.float32, flags='F_CONTIGUOUS')
A64 = np.ctypeslib.ndpointer(dtype=np.float64, flags='F_CONTIGUOUS')
STAGE = 'u1 v1 t1 q1 q1c q1i prsl prslk del phil prsi phii'.split()
SCAL = 'psk rbsoil fm fh tsea heat evap stress spd1 deltim'.split()
MOUT = 'du dv tau rtg1 rtg2 rtg3'.split()
MDIAG = 'hpbl kpbl'.split()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    plain_so = LANE / 'oracle' / 'pbl3' / 'zero' / 'oracle.so'
    cap_so = LANE / 'oracle' / 'pbl3cap' / 'zero' / 'oracle.so'
    plain = ctypes.CDLL(str(plain_so)).probe
    plain.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, A32, A32, A32, A32]
    cap = ctypes.CDLL(str(cap_so)).probe_capture
    cap.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, A32, A32, A32, A32,
                    A64, A64, A64, A64]
    fix = np.load(FIXTURE)
    cases = {'real_q1': ('real', 1), 'real_q0': ('real', 0),
             'caller': ('caller', int(fix['caller_has_qi']))}
    out, receipts = {}, {}
    for case, (prefix, has_qi) in cases.items():
        prof, sfc = np.asfortranarray(fix[f'{prefix}_prof']), np.asfortranarray(fix[f'{prefix}_sfc'])
        dt = float(fix['caller_dt'] if prefix == 'caller' else fix['dt'])
        n, nz = prof.shape[0], prof.shape[1] - 1
        zeros = lambda *shape, dtype=np.float32: np.zeros(shape, dtype, order='F')  # noqa: E731
        t0, d0 = zeros(n, nz, 6), zeros(n, 4)
        plain(n, nz, has_qi, dt, prof, sfc, t0, d0)
        t1, d1 = zeros(n, nz, 6), zeros(n, 4)
        stage, scal = zeros(n, nz + 1, 12, dtype=np.float64), zeros(n, 10, dtype=np.float64)
        mout, mdiag = zeros(n, nz, 6, dtype=np.float64), zeros(n, 2, dtype=np.float64)
        cap(n, nz, has_qi, dt, prof, sfc, t1, d1, stage, scal, mout, mdiag)
        frozen = f'{prefix}_zero' if prefix == 'caller' else f'{prefix}_q{has_qi}_zero'
        receipts[case] = dict(
            capture_eq_unmodified=bool(np.array_equal(t0.view(np.uint32), t1.view(np.uint32))
                                       and np.array_equal(d0.view(np.uint32), d1.view(np.uint32))),
            capture_eq_frozen=bool(np.array_equal(t1.view(np.uint32), fix[f'{frozen}_tend'].view(np.uint32))
                                   and np.array_equal(d1.view(np.uint32), fix[f'{frozen}_diag'].view(np.uint32))),
            columns=n, levels=nz, has_qi=has_qi, dt=dt)
        assert receipts[case]['capture_eq_unmodified'] and receipts[case]['capture_eq_frozen'], receipts[case]
        assert np.isfinite(stage).all() and np.isfinite(mout).all(), case
        out.update({f'{case}_stage': stage, f'{case}_scal': scal, f'{case}_mout': mout, f'{case}_mdiag': mdiag,
                    f'{case}_tend': t1, f'{case}_diag': d1})
    path = HERE / 'fixtures' / 'blgfs_capture_oracle.npz'
    np.savez_compressed(path, **out)
    manifest = json.loads((LANE / 'oracle' / 'manifest.json').read_text())
    meta = dict(stage=STAGE, scal=SCAL, mout=MOUT, mdiag=MDIAG, cases={k: list(v) for k, v in cases.items()},
                receipts=receipts, source_fixture_sha256=sha(FIXTURE), npz_sha256=sha(path),
                builds={'pbl3/zero': sha(plain_so), 'pbl3cap/zero': sha(cap_so)},
                generated=manifest.get('generated'), compiler=manifest['compiler'],
                wrapper_sha256=sha(HERE / 'blgfs_capture.f90'))
    (HERE / 'fixtures' / 'blgfs_capture_oracle.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(json.dumps(receipts), meta['npz_sha256'])


if __name__ == '__main__':
    main()
