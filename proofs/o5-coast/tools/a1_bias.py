"""A1: signed GPU−CPU surface differences by coastal mask and lead block, WN3 0227 d03 (o5-coast).

usage: python a1_bias.py OUT.json
A = GPU v0.4.2 (w43-teacher te_gpu extract, 73 frames) and GPU v0.4.0 (F20 raw, 25 frames);
B = original CPU-WRF 4.7.1 dmpar twin (4x1, w43-twin0227, 73 frames). Identical input SHAs (notes).
Masks from the CPU twin statics: coast = within 2 cells (5x5 dilation) of the land/sea edge (as w43-coastwind);
land classes by IVGTYP (MODIS). Statistics per mask and lead block: mean of A, mean of B, mean(A−B) (bias),
pooled RMSE, and a 6-h moving-block bootstrap 95 % CI of the bias over leads (spatial mean per lead first).
WS10 = |(U10,V10)|; earth-relative components u10_east/v10_north. No JAX."""
import json
import sys

import numpy as np
from scipy.ndimage import binary_dilation

S = '<DATA_ROOT>/wrf_gpu2/scratch/o5-coast'
TE = '<USER_HOME>/wrf_gpu2_lanes/w43-teacher/te_gpu/20260227_gpu/d03_sfc.npz'
FIELDS = ('ws', 'ue', 'vn', 'T2', 'Q2', 'HFX', 'LH', 'PBLH', 'PSFC')
BLOCKS = {'t1_24': (1, 24), 't25_47': (25, 47), 't48_72': (48, 72), 't1_72': (1, 72), 't1_24only': (1, 24)}


def from_te(path):
    z = np.load(path)
    d = dict(ws=z['wind_speed'], ue=z['u10_east'], vn=z['v10_north'])
    for k in ('T2', 'Q2', 'HFX', 'LH', 'PBLH', 'PSFC'):
        d[k] = z[k]
    return d, dict(LANDMASK=z['LANDMASK'], HGT=z['HGT'])


def from_raw(path):
    z = np.load(path)
    d = dict(ws=np.hypot(z['U10'], z['V10']), ue=z['u10_east'], vn=z['v10_north'])
    for k in ('T2', 'Q2', 'HFX', 'LH', 'PBLH', 'PSFC'):
        d[k] = z[k]
    return d, {k[2:]: z[k] for k in z.files if k.startswith('s_')}


def masks(st):
    lm = st['LANDMASK'] > 0.5
    k = np.ones((5, 5), bool)
    edge = binary_dilation(lm, k) & binary_dilation(~lm, k)
    iv = st['IVGTYP']
    m = {'all': np.ones_like(lm), 'open_sea': ~lm & ~edge, 'coast_sea': edge & ~lm, 'coast_land': edge & lm,
         'interior_land': lm & ~edge}
    for nm, cls in (('urban13', (13,)), ('barren16', (16,)), ('grass10', (10,)), ('shrub6_7', (6, 7)),
                    ('savanna8_9', (8, 9)), ('forest1_5', (1, 2, 3, 4, 5)), ('crop12_14', (12, 14))):
        mm = lm & np.isin(iv, cls)
        if mm.any():
            m[f'land_{nm}'] = mm
            m[f'coast_{nm}'] = mm & edge
    return {k: v for k, v in m.items() if v.any()}


def boot_ci(series, nb=2000, blk=6, seed=1):
    rng = np.random.default_rng(seed)
    n = len(series)
    if n < blk:
        return [float('nan')] * 2
    starts = np.arange(n - blk + 1)
    nblk = int(np.ceil(n / blk))
    vals = []
    for _ in range(nb):
        idx = np.concatenate([np.arange(s, s + blk) for s in rng.choice(starts, nblk)])[:n]
        vals.append(series[idx].mean())
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def stats(A, B, m, leads, taus):
    out = {}
    for f in FIELDS:
        sel = [i for i, t in enumerate(taus) if leads[0] <= t <= leads[1]]
        if not sel:
            continue
        a = A[f][sel][:, m].astype(np.float64)
        b = B[f][sel][:, m].astype(np.float64)
        d = a - b
        per = d.mean(1)
        out[f] = dict(n=int(d.size), mean_A=float(a.mean()), mean_B=float(b.mean()), bias=float(d.mean()),
                      rmse=float(np.sqrt((d ** 2).mean())), bias_ci95=boot_ci(per),
                      frac_leads_pos=float((per > 0).mean()))
    return out


def run_pair(A, B, st, ntau):
    taus = list(range(ntau))
    res = {}
    for mk, m in masks(st).items():
        res[mk] = {'ncells': int(m.sum())}
        for bk, lb in BLOCKS.items():
            if lb[0] >= ntau:
                continue
            res[mk][bk] = stats(A, B, m, (lb[0], min(lb[1], ntau - 1)), taus)
    return res


def main():
    twin, st = from_raw(f'{S}/twin0227_d03.npz')
    g42, st42 = from_te(TE)
    g40, st40 = from_raw(f'{S}/f20gpu0227_d03.npz')
    for k in ('LANDMASK', 'HGT'):
        assert np.array_equal(st[k], st42[k]) and np.array_equal(st[k], st40[k]), k
    for d in (twin, g42, g40):
        for f in FIELDS:
            assert np.isfinite(d[f]).all(), f
    out = {'note': __doc__, 'pairs': {}}
    out['pairs']['gpu042_minus_cputwin'] = run_pair(g42, twin, st, 73)
    out['pairs']['gpu040_minus_cputwin'] = run_pair(g40, {k: v[:25] for k, v in twin.items()}, st, 25)
    out['pairs']['gpu042_minus_gpu040'] = run_pair({k: v[:25] for k, v in g42.items()}, g40, st, 25)
    json.dump(out, open(sys.argv[1], 'w'), indent=1)
    p = out['pairs']['gpu042_minus_cputwin']
    for mk in p:
        r = p[mk]
        line = [f"{mk:22s} n={r['ncells']:5d}"]
        for bk in ('t1_24', 't25_47', 't48_72'):
            s = r[bk]['ws']
            line.append(f"{bk} ws A {s['mean_A']:.2f} B {s['mean_B']:.2f} bias {s['bias']:+.3f} [{s['bias_ci95'][0]:+.3f},{s['bias_ci95'][1]:+.3f}] rmse {s['rmse']:.2f}")
        print(' | '.join(line))


if __name__ == '__main__':
    main()
