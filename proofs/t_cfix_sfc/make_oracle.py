"""Pristine-WRF surface-layer oracle on real-structured columns (lane t-cfix-sfc, no JAX).

Columns: lowest-level atmosphere of the ORIGINAL CPU-WRF c04 composite run
(Swiss 2023-01-15, sf_sfclay_physics=3, all land) at 7 frames, picks spanning
max/min HFX and UST plus random columns; the same air over WATER variants
(xland=2, lake/sea, skin colder/warmer than the air, small z0) and calm variants
(winds x0.03) so every regime/branch is exercised.  Grid memory (ZNT, UST, MOL,
QSFC, HFX, QFX, PBLH, ZOL) comes from the CPU-WRF frame where it exists and from a
seeded generator otherwise (sign/zero mixes for MOL, QSFC<=0 recompute cases).
Each scheme (91 SFCLAY, 7 PXSFCLAY, 3 SF_GFS) is called TWICE: call 2 consumes
call 1's own INOUT outputs exactly like the WRF grid memory (PX itimestep 1 -> 2).
Builds: zero (reference), inf (uninitialised-read check), fma (optimiser spread).
usage: python make_oracle.py <lane_dir>   -> <lane>/oracle/oracle.npz
"""
from pathlib import Path
import ctypes, hashlib, json, sys
import numpy as np
import netCDF4

LANE = Path(sys.argv[1])
CPU = Path('<USER_HOME>/wrf_gpu2_lanes/v-prep/c04/cpu')
FRAMES = [f'2023-01-15_{h:02d}:00:00' for h in (3, 6, 9, 12, 15, 18, 21)]
R_D, CP, G = 287.0, 1004.5, 9.81
IN = ('u v t th qv p dz psfc tsk xland lakemask mavail pblh znt ust mol qsfc hfx qfx zol dx regime').split()
OUT = ('ust znt mol zol rmol regime psim psih fm fh qsfc qgh cpm hfx qfx lh flhc flqc chs chs2 cqs2 '
       'u10 v10 th2 t2 q2 gz1oz0 wspd br spare').split()
MEMORY = {'ust': 'ust', 'znt': 'znt', 'mol': 'mol', 'zol': 'zol', 'qsfc': 'qsfc', 'hfx': 'hfx', 'qfx': 'qfx',
          'regime': 'regime'}


def columns():
    rng = np.random.default_rng(20261015)
    rows, labels = [], []
    for frame in FRAMES:
        with netCDF4.Dataset(CPU / f'wrfout_d01_{frame}') as d:
            v = {k: np.asarray(d.variables[k][0], np.float64) for k in (
                'U', 'V', 'T', 'P', 'PB', 'PH', 'PHB', 'QVAPOR', 'PSFC', 'TSK', 'XLAND', 'LAKEMASK', 'UST',
                'PBLH', 'HFX', 'QFX')}
        ny, nx = v['TSK'].shape
        pick = {'max_hfx': np.argmax(v['HFX']), 'min_hfx': np.argmin(v['HFX']),
                'max_ust': np.argmax(v['UST']), 'min_ust': np.argmin(v['UST'])}
        for r in range(4):
            pick[f'random{r}'] = int(rng.integers(0, ny * nx))
        for tag, flat in pick.items():
            j, i = np.unravel_index(int(flat), (ny, nx))
            u = .5 * (v['U'][0, j, i] + v['U'][0, j, i + 1]); vv = .5 * (v['V'][0, j, i] + v['V'][0, j + 1, i])
            p = v['P'][0, j, i] + v['PB'][0, j, i]; qv = v['QVAPOR'][0, j, i]
            th = v['T'][0, j, i] + 300.; t = th * (p / 1.e5) ** (R_D / CP)
            dz = ((v['PH'] + v['PHB'])[1, j, i] - (v['PH'] + v['PHB'])[0, j, i]) / G
            rho = p / (R_D * t * (1. + .608 * qv)); ust = max(v['UST'][j, i], .01)
            mol = -v['HFX'][j, i] / (rho * CP * ust)
            sat = .622 * .6112e3 * np.exp(17.67 * (v['TSK'][j, i] - 273.15) / (v['TSK'][j, i] - 29.65)) / v['PSFC'][j, i]
            base = dict(u=u, v=vv, t=t, th=th, qv=qv, p=p, dz=dz, psfc=v['PSFC'][j, i], tsk=v['TSK'][j, i],
                        xland=1., lakemask=0., mavail=float(rng.uniform(.2, 1.)), pblh=v['PBLH'][j, i],
                        znt=float(rng.choice([.03, .1, .25, .5, .8])), ust=v['UST'][j, i],
                        mol=float(rng.choice([mol, 0., -abs(mol) - .01, abs(mol) + .01])),
                        qsfc=float(rng.choice([-1., 0., .8 * sat, sat])), hfx=v['HFX'][j, i], qfx=v['QFX'][j, i],
                        zol=float(rng.uniform(-2., 2.)), dx=3000., regime=0.)
            rows.append(base); labels.append(dict(frame=frame, pick=tag, j=int(j), i=int(i), surface='land'))
            if tag.startswith('random') or tag in ('max_hfx', 'min_hfx'):
                for k, dt in enumerate((-4., -.5, 1.5, 6.)):
                    w = dict(base, xland=2., lakemask=float(k == 1), mavail=1., tsk=t + dt,
                             znt=float(rng.choice([1.e-4, 4.e-4, 2.e-3])), qsfc=-1., ust=max(base['ust'], .05),
                             dx=float(rng.choice([3000., 9000.])))
                    rows.append(w); labels.append(dict(labels[-1] if k == 0 else labels[-1], surface=f'water{dt:+}'))
            if tag in ('random0', 'min_ust'):
                calm = dict(base, u=.03 * u, v=.03 * vv, ust=float(rng.choice([.005, .02])))
                rows.append(calm); labels.append(dict(labels[-1], surface='calm'))
    arr = np.array([[r[k] for k in IN] for r in rows], np.float32)
    return arr, labels


def load(scheme, variant):
    so = LANE / 'oracle' / f'sf{scheme}' / variant / 'oracle.so'
    fn = ctypes.CDLL(str(so)).probe
    a = np.ctypeslib.ndpointer(dtype=np.float32, flags='F_CONTIGUOUS')
    fn.argtypes = [ctypes.c_int, a, a] if scheme == 91 else [ctypes.c_int, ctypes.c_int, a, a]
    return fn


def call(fn, scheme, inp, itimestep):
    out = np.zeros((inp.shape[0], len(OUT)), np.float32, order='F')
    if scheme == 91:
        fn(inp.shape[0], np.asfortranarray(inp), out)
    elif scheme == 7:
        # PXSFCLAY takes the domain's scalar DX: one public call per distinct DX.
        for dx in np.unique(inp[:, IN.index('dx')]):
            rows = np.flatnonzero(inp[:, IN.index('dx')] == dx)
            part = np.zeros((rows.size, len(OUT)), np.float32, order='F')
            fn(rows.size, itimestep, np.asfortranarray(inp[rows]), part)
            out[rows] = part
    else:
        fn(inp.shape[0], itimestep, np.asfortranarray(inp), out)
    return out


def chain(fn, scheme, inp):
    out1 = call(fn, scheme, inp, 1)
    inp2 = inp.copy()
    for src, dst in MEMORY.items():
        if scheme == 3 and src not in ('ust', 'znt'):
            continue  # SF_GFS INOUT is UST/ZNT only; other grid memory untouched.
        if scheme == 7 and src in ('regime',):
            continue
        inp2[:, IN.index(dst)] = out1[:, OUT.index(src)]
    if scheme != 3:
        for k in ('hfx', 'qfx'):
            inp2[:, IN.index(k)] = out1[:, OUT.index(k)]
    out2 = call(fn, scheme, inp2, 2)
    return out1, inp2, out2


def main():
    inp, labels = columns()
    result = {'inputs': inp}
    for scheme in (91, 7, 3):
        for variant in ('zero', 'inf', 'fma'):
            o1, i2, o2 = chain(load(scheme, variant), scheme, inp)
            result[f'sf{scheme}_{variant}_out1'] = o1
            result[f'sf{scheme}_{variant}_out2'] = o2
            result[f'sf{scheme}_{variant}_inp2'] = i2
        same = all(np.array_equal(result[f'sf{scheme}_zero_out{c}'], result[f'sf{scheme}_inf_out{c}']) for c in (1, 2))
        result[f'sf{scheme}_init_byte_identical'] = np.array(same)
        print(scheme, 'zero==inf', same, 'finite', bool(np.isfinite(result[f'sf{scheme}_zero_out2']).all()))
    path = LANE / 'oracle' / 'oracle.npz'
    np.savez_compressed(path, **result)
    meta = dict(inputs=IN, outputs=OUT, labels=labels, npz_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                build_manifest=json.loads((LANE / 'oracle' / 'manifest.json').read_text()))
    (LANE / 'oracle' / 'oracle.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(inp.shape, meta['npz_sha256'])


if __name__ == '__main__':
    main()
