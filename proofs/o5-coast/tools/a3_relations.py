"""A3: chaos-insensitive surface-layer relationship test, GPU vs CPU-WRF on identical inputs (o5-coast).

usage: python a3_relations.py GPU.npz CPU.npz OUT.json [TAU_MAX]
Inputs = load_raw.py caches of the same case/domain (GPU v0.4.0 F20 raw vs original CPU-WRF twin, WN3 0227 d03).
For every cell and lead τ1..TAU_MAX we form column-local ratios that MYNN-SL determines from (z0, stability, za) only:
  r10  = WS10 / WS_L0            (10 m diagnostic vs lowest mass level)
  cdn  = (UST / WS_L0)^2         (drag at the lowest level)
  chx  = HFX / (rho cp WS_L0 (TSK*(1e5/PSFC)^0.2857 − θ_L0))   (bulk heat exchange; |Δθ| > 0.5 K only)
and a bulk Richardson number Rib = g z_L0 (θ_L0 − θ_sfc) / (θ_L0 WS_L0^2). Chaos moves samples ALONG these relations; a
port difference in z0, the 10 m diagnostic or the stability functions shifts the relation. Per surface class × Rib bin:
median and mean of each ratio for GPU and CPU, their difference, n, and a cell-block bootstrap CI (blocks = cells,
all leads of a cell together). No JAX."""
import json
import sys

import numpy as np
from scipy.ndimage import binary_dilation

RIB_BINS = [-np.inf, -0.5, -0.1, -0.02, 0.02, 0.1, 0.25, np.inf]
CLASSES = {'water': None, 'coast_water': None, 'urban13': (13,), 'barren16': (16,), 'grass10': (10,),
           'savanna8_9': (8, 9), 'forest1_5': (1, 2, 3, 4, 5), 'shrub6_7': (6, 7)}


def derive(z, t1, t2):
    sl = slice(t1, t2 + 1)
    ws10 = np.hypot(z['U10'][sl], z['V10'][sl]).astype(np.float64)
    wsl0 = np.hypot(z['ul0'][sl], z['vl0'][sl]).astype(np.float64)
    th0 = z['th0'][sl].astype(np.float64)
    ths = z['TSK'][sl].astype(np.float64) * (1e5 / z['PSFC'][sl].astype(np.float64)) ** 0.2857
    zl0 = z['z0'][sl].astype(np.float64)
    rib = 9.81 * zl0 * (th0 - ths) / (th0 * np.maximum(wsl0, 0.1) ** 2)
    rho = z['PSFC'][sl].astype(np.float64) / (287.0 * z['T2'][sl].astype(np.float64))
    dth = ths - th0
    with np.errstate(divide='ignore', invalid='ignore'):
        r10 = np.where(wsl0 > 1.0, ws10 / wsl0, np.nan)
        cdn = np.where(wsl0 > 1.0, (z['UST'][sl].astype(np.float64) / wsl0) ** 2, np.nan)
        chx = np.where((wsl0 > 1.0) & (np.abs(dth) > 0.5), z['HFX'][sl] / (rho * 1004.5 * wsl0 * dth), np.nan)
    return dict(r10=r10, cdn=cdn, chx=chx, rib=rib, zl0=zl0)


def class_masks(z):
    lm = z['s_LANDMASK'] > 0.5
    k = np.ones((5, 5), bool)
    edge = binary_dilation(lm, k) & binary_dilation(~lm, k)
    iv = z['s_IVGTYP']
    m = {}
    for nm, cls in CLASSES.items():
        if nm == 'water':
            m[nm] = ~lm & ~edge
        elif nm == 'coast_water':
            m[nm] = ~lm & edge
        else:
            m[nm] = lm & np.isin(iv, cls)
            m['coast_' + nm] = m[nm] & edge
    return {k: v for k, v in m.items() if v.sum() >= 5}


def binned(dg, dc, mask, var, rng, nb=500):
    out = []
    cells = np.argwhere(mask)
    for lo, hi in zip(RIB_BINS[:-1], RIB_BINS[1:]):
        # stability bin from each model's OWN Rib (relation test: conditional on own state)
        vals = {}
        per_cell = {}
        for tag, d in (('gpu', dg), ('cpu', dc)):
            x = d[var][:, mask]; r = d['rib'][:, mask]
            sel = (r >= lo) & (r < hi) & np.isfinite(x)
            vals[tag] = x[sel]
            s = np.where(sel, x, 0.0).sum(0); c = sel.sum(0)
            per_cell[tag] = (s, c)
        if min(vals['gpu'].size, vals['cpu'].size) < 50:
            continue
        mg, mc = float(np.median(vals['gpu'])), float(np.median(vals['cpu']))
        ag, ac = float(vals['gpu'].mean()), float(vals['cpu'].mean())
        # cell-block bootstrap of the mean difference
        sg, cg = per_cell['gpu']; sc, cc = per_cell['cpu']
        diffs = []
        n = len(cells)
        for _ in range(nb):
            i = rng.integers(0, n, n)
            if cg[i].sum() == 0 or cc[i].sum() == 0:
                continue
            diffs.append(sg[i].sum() / cg[i].sum() - sc[i].sum() / cc[i].sum())
        out.append(dict(rib_bin=[float(lo), float(hi)], n_gpu=int(vals['gpu'].size), n_cpu=int(vals['cpu'].size),
                        median_gpu=mg, median_cpu=mc, mean_gpu=ag, mean_cpu=ac, mean_diff=ag - ac,
                        rel_mean_diff=(ag - ac) / ac if ac else None,
                        mean_diff_ci95=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]))
    return out


def main():
    g, c = np.load(sys.argv[1]), np.load(sys.argv[2])
    tmax = int(sys.argv[4]) if len(sys.argv) > 4 else min(g['U10'].shape[0], c['U10'].shape[0]) - 1
    for k in ('s_LANDMASK', 's_IVGTYP', 's_HGT'):
        assert np.array_equal(g[k], c[k]), k
    dg, dc = derive(g, 1, tmax), derive(c, 1, tmax)
    assert np.allclose(dg['zl0'], dc['zl0'], rtol=0.05), 'lowest level height differs'
    rng = np.random.default_rng(7)
    res = {'note': __doc__, 'tau': [1, tmax], 'classes': {}}
    for nm, m in class_masks(g).items():
        res['classes'][nm] = {'ncells': int(m.sum()), 'z_L0_median': float(np.median(dc['zl0'][:, m]))}
        for var in ('r10', 'cdn', 'chx'):
            res['classes'][nm][var] = binned(dg, dc, m, var, rng)
    json.dump(res, open(sys.argv[3], 'w'), indent=1)
    for nm, r in res['classes'].items():
        print(f"== {nm} n={r['ncells']} zL0={r['z_L0_median']:.1f} m")
        for var in ('r10', 'cdn', 'chx'):
            for b in r[var]:
                print(f"   {var} Rib[{b['rib_bin'][0]:+.2f},{b['rib_bin'][1]:+.2f}) n {b['n_cpu']:6d}/{b['n_gpu']:6d} "
                      f"mean C {b['mean_cpu']:.4g} G {b['mean_gpu']:.4g} rel {100*b['rel_mean_diff']:+.2f}% "
                      f"CI [{b['mean_diff_ci95'][0]:+.3g},{b['mean_diff_ci95'][1]:+.3g}]")


if __name__ == '__main__':
    main()
