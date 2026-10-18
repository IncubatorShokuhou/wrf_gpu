"""Scale-aware dx coverage for the Shin-Hong pristine oracle (lane w41-shinhongtke).

Reuses the t-shinhong pristine builds (zero/inf/fma .so, SHA-checked against its manifest;
no recompilation) and its column generators (4 synthetic + 32 real c07 CPU-WRF columns), and
re-runs every column/mode at dx=dy in DX (WN3/G2 nest resolutions 9/1 km; t-shinhong covered
3 km real + 0.5 km synthetic). Writes <out>/dx<dx>/{oracle.npz,oracle_manifest.json} in the
t-shinhong layout so proofs/t_shinhong/compare_port.py consumes it unchanged, plus a census of
the active scale-aware clamp branches (pu/pq/pthl/pthnl/ptke pmin/pmax, module_bl_shinhong.F:2297-2425).
usage: python build_dx_oracle.py <t_shinhong_lane> <out_dir>
"""
import ctypes, hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
TS_LANE, OUT = Path(sys.argv[1]), Path(sys.argv[2])
DX = (1000., 9000.)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest0 = json.loads((TS_LANE / 'oracle_manifest.json').read_text())
argv = sys.argv
sys.argv = [argv[0], str(TS_LANE)]
spec = importlib.util.spec_from_file_location('ts_build', HERE.parent / 't_shinhong' / 'build_oracle.py')
ts = importlib.util.module_from_spec(spec); spec.loader.exec_module(ts)
sys.argv = argv


def load(name):
    so = TS_LANE / 'oracle' / name / 'oracle.so'
    assert sha(so) == manifest0['builds'][name]['sha256'], name
    fn = ctypes.CDLL(str(so)).probe
    arr = np.ctypeslib.ndpointer(dtype=np.float32, flags='F_CONTIGUOUS')
    fn.argtypes = [ctypes.c_int, ctypes.c_int, arr, arr, arr]
    return fn


def scale_functions(doh):
    """REAL4 numpy copies of pu/pq/pthl/pthnl/ptke pre-clamp values (census only)."""
    f = np.float32; d = f(doh)
    pu = (d ** f(2.) + f(.07) * d ** f(.6666667)) / (d ** f(2.) + f(.142) * d ** f(.6666667) + f(.071))
    pq = f(.5) * (d ** f(2.) + f(-.098)) / (d ** f(2.) + f(.106)) + f(.5)
    pthnl = f(.243) * (d ** f(2.) + f(.936) * d ** f(.875) + f(-1.11)) / (d ** f(2.) + f(.312) * d ** f(.875) + f(.329)) + (f(1.) - f(.243))
    pthl = f(.28) * (d ** f(2.) + f(.87) * d ** f(.5) + f(-.913)) / (d ** f(2.) + f(.153) * d ** f(.5) + f(.278)) + (f(1.) - f(.28))
    return dict(pu=pu, pq=pq, pthnl=pthnl, pthl=pthl, ptke=pu)


t0 = time.monotonic()
fns = {name: load(name) for name in ('zero', 'inf', 'fma')}
columns = list(ts.synthetic()) + list(ts.real_columns())
summary = {}
for dx in DX:
    lane = OUT / f'dx{int(dx)}'
    lane.mkdir(parents=True, exist_ok=True)
    rows, profiles, surfaces, outs = [], [], [], {k: [] for k in fns}
    census = {f'{k}_{b}': 0 for k in ('pu', 'pq', 'pthnl', 'pthl', 'ptke') for b in ('pmin', 'pmax')}
    for meta, prof, surf, _ in columns:
        surf = surf.copy(); surf[13] = surf[14] = dx
        n = prof.shape[0] - 1
        cold = None
        for mode, label in ((0, 'mode0'), (1, 'mode1_cold'), (1, 'mode1_warm')):
            p = prof.copy(order='F')
            if label == 'mode1_warm':
                p[:-1, 8] = cold[:-1, 0]; p[-1, 8] = cold[-2, 0]
            for name, fn in fns.items():
                res = np.zeros((n + 1, 10), np.float32, order='F')
                fn(n, mode, p, surf.copy(), res)
                outs[name].append(res)
            if label == 'mode1_cold':
                cold = outs['zero'][-1]
                hpbl = float(cold[0, 9])
                if hpbl > 0:
                    for k, v in scale_functions(np.sqrt(np.float32(dx) * np.float32(dx)) / np.float32(hpbl)).items():
                        census[f'{k}_pmin'] += int(v < 0); census[f'{k}_pmax'] += int(v > 1)
            rows.append(dict(meta, mode=mode, label=label, nz=n, dx=dx))
            profiles.append(p); surfaces.append(surf.copy())
    nmax = max(p.shape[0] for p in profiles)
    pad = lambda a: np.pad(a, ((0, nmax - a.shape[0]), (0, 0)))
    stack = {k: np.stack([pad(o) for o in v]) for k, v in outs.items()}
    np.savez(lane / 'oracle.npz', profiles=np.stack([pad(p) for p in profiles]), surface=np.stack(surfaces),
             nz=np.array([r['nz'] for r in rows]), mode=np.array([r['mode'] for r in rows]),
             **{f'out_{k}': v for k, v in stack.items()})
    m = dict(source_sha256=manifest0['source_sha256'], wrapper_sha256=manifest0['wrapper_sha256'],
             builds=manifest0['builds'], dx=dx, rows=rows,
             init_sensitivity_byte_identical=bool(stack['zero'].tobytes() == stack['inf'].tobytes()),
             nonfinite={k: int(np.count_nonzero(~np.isfinite(v))) for k, v in stack.items()},
             clamp_census_cold_columns=census, oracle_npz_sha256=sha(lane / 'oracle.npz'), GPU=False)
    (lane / 'oracle_manifest.json').write_text(json.dumps(m, indent=1) + '\n')
    summary[int(dx)] = {k: v for k, v in m.items() if k not in ('rows', 'builds')}
summary['elapsed_s'] = time.monotonic() - t0
(OUT / 'dx_oracle_summary.json').write_text(json.dumps(summary, indent=1) + '\n')
print(json.dumps(summary, indent=1))
