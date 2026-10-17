"""WRF's own REAL4 input-ULP envelope for the Shin-Hong public call (lane t-shinhong).

For every mode-1 row of oracle.npz: each REAL4 input cell (profiles incl. the
p8w top interface, surface scalars) is moved one ULP down and up, one at a time,
through the UNMODIFIED pristine build (-O0, no FMA, init zero); the per-cell
min/max over those calls, the base call and the -O2 -mfma build is the envelope.
Pre-registered (before looking at compiled-port residuals beyond the eager
byte proof): compiled port output must lie inside envelope widened outward by
2 REAL4 output ULPs; subnormal-range cells (|x| < FLT_MIN) are excluded (E117).
usage: python ulp_envelope.py <lane_dir>
"""
from pathlib import Path
import ctypes, hashlib, json, sys, time
import numpy as np

lane = Path(sys.argv[1])
d = np.load(lane / 'oracle.npz')
fn = ctypes.CDLL(str(lane / 'oracle/zero/oracle.so')).probe
arr = np.ctypeslib.ndpointer(dtype=np.float32, flags='F_CONTIGUOUS')
fn.argtypes = [ctypes.c_int, ctypes.c_int, arr, arr, arr]
t0 = time.monotonic()
rows = [i for i in range(len(d['mode'])) if int(d['mode'][i]) == 1]
lo = np.array(d['out_zero'], copy=True); hi = np.array(d['out_zero'], copy=True)
calls = 0
for i in rows:
    n = int(d['nz'][i])
    base_p = np.array(d['profiles'][i][:n + 1], np.float32, order='F')
    base_s = np.array(d['surface'][i], np.float32)
    for cand in (d['out_fma'][i],):
        lo[i] = np.minimum(lo[i], cand); hi[i] = np.maximum(hi[i], cand)
    cells = [('p', k, j) for j in range(11) for k in range(n + (1 if j == 5 else 0))] + [('s', k, None) for k in range(19)]
    for kind, k, j in cells:
        for direction in (-np.inf, np.inf):
            p = base_p.copy(order='F'); s = base_s.copy()
            if kind == 'p':
                p[k, j] = np.nextafter(p[k, j], np.float32(direction))
            else:
                if k in (9, 13, 14):  # dt, dx, dy are namelist constants, not field data
                    continue
                s[k] = np.nextafter(s[k], np.float32(direction))
            out = np.zeros((n + 1, 10), np.float32, order='F')
            fn(n, 1, p, s, out)
            calls += 1
            pad = np.zeros_like(lo[i]); pad[:n + 1] = out
            lo[i] = np.minimum(lo[i], pad); hi[i] = np.maximum(hi[i], pad)
np.savez(lane / 'ulp_envelope.npz', lower=lo, upper=hi, rows=np.array(rows))
summary = dict(rows=len(rows), original_calls=calls, elapsed_s=time.monotonic() - t0,
               oracle_npz_sha256=hashlib.sha256((lane / 'oracle.npz').read_bytes()).hexdigest(),
               envelope_sha256=hashlib.sha256((lane / 'ulp_envelope.npz').read_bytes()).hexdigest(),
               nonfinite=int(np.count_nonzero(~np.isfinite(lo)) + np.count_nonzero(~np.isfinite(hi))))
(lane / 'ulp_envelope.json').write_text(json.dumps(summary, indent=1) + '\n')
print(json.dumps(summary))
