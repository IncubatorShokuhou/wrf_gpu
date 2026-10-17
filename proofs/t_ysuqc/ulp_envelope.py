"""WRF's own +-1 ULP input-sensitivity envelope of the pristine YSU build (lane t-ysuqc).

For every oracle case the unmodified zero build is re-run NPERT times with every
profile and surface input independently moved by -1, 0 or +1 REAL4 ULP (fixed
seed). Per cell, the envelope is [min, max] over the perturbed runs, the
unperturbed run and the FMA build. A compiled port result inside that range
(widened by 2 output ULPs) is as close to WRF as WRF is to itself.
usage: python ulp_envelope.py <lane_dir> [NPERT]
"""
from pathlib import Path
import ctypes, json, sys, time
import numpy as np

LANE = Path(sys.argv[1])
NPERT = int(sys.argv[2]) if len(sys.argv) > 2 else 24


def main():
    t0 = time.monotonic()
    d = np.load(LANE / 'oracle.npz')
    fn = ctypes.CDLL(str(LANE / 'oracle/zero/oracle.so')).probe
    arr = np.ctypeslib.ndpointer(dtype=np.float32, flags='F_CONTIGUOUS')
    fn.argtypes = [ctypes.c_int] * 4 + [arr, arr, arr]
    rng = np.random.default_rng(20261015)
    lower = np.minimum(d['out_zero'], d['out_fma']).copy()
    upper = np.maximum(d['out_zero'], d['out_fma']).copy()
    calls = 0
    for row in range(d['profiles'].shape[0]):
        n = int(d['nz'][row])
        prof0, surf0 = d['profiles'][row], d['surface'][row]
        for _ in range(NPERT):
            step = rng.integers(-1, 2, size=prof0.shape)
            prof = np.where(step > 0, np.nextafter(prof0, np.float32(np.inf)),
                            np.where(step < 0, np.nextafter(prof0, np.float32(-np.inf)), prof0)).astype(np.float32)
            prof[:, 4:6] = np.where(prof0[:, 4:6] == 0, 0, prof[:, 4:6])   # keep cloud-free cells cloud-free
            sstep = rng.integers(-1, 2, size=surf0.shape)
            surf = np.where(sstep > 0, np.nextafter(surf0, np.float32(np.inf)),
                            np.where(sstep < 0, np.nextafter(surf0, np.float32(-np.inf)), surf0)).astype(np.float32)
            surf[[5, 12, 13, 14, 15, 16]] = surf0[[5, 12, 13, 14, 15, 16]]  # xland, ocean currents, ctopo, dt exact
            res = np.zeros((n + 1, 9), np.float32, order='F')
            fn(n, int(d['topdown'][row]), 1, int(d['flag_qi'][row]), np.asfortranarray(prof), surf, res)
            calls += 1
            if not np.isfinite(res).all():
                continue
            lower[row] = np.minimum(lower[row], res)
            upper[row] = np.maximum(upper[row], res)
    np.savez_compressed(LANE / 'ulp_envelope.npz', lower=lower, upper=upper)
    meta = dict(npert=NPERT, calls=calls, seed=20261015, elapsed_s=time.monotonic() - t0,
                kpbl_varies_rows=int(np.count_nonzero(lower[:, 1, 8] != upper[:, 1, 8])))
    (LANE / 'ulp_envelope.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(meta)


if __name__ == '__main__':
    main()
