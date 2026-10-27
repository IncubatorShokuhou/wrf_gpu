"""A2: per-lead X (GPU v0.4.2 − CPU ref) vs C (CPU twin − CPU ref) for 10 m wind, from the w43-teacher frozen-scorer
receipts (aggregate statistics only; no field data read). Cases with an own twin: wn3_20260227 (C_0227), era5_20240302
(C_era5_20240302).

usage: python a2_xc_leads.py OUT.json
Per domain and field: per-lead rmse/bias of X and C, X/C, block means (rms over leads) and spatial-split pooled rows
(land/ocean, elevation) for X and C. No JAX."""
import json
import sys

import numpy as np

R = '<USER_HOME>/wrf_gpu2_lanes/w43-teacher/score'
PAIRS = {'wn3_20260227': 'C_0227', 'era5_20240302': 'C_era5_20240302'}
FIELDS = ('U10', 'V10', 'T2', 'PSFC', 'U', 'V', 'W')
BLOCKS = ((1, 24), (25, 40), (41, 72), (1, 72))


def load(case, dom):
    return json.load(open(f'{R}/{case}/x/{dom}.json'))['field_summaries']


def main():
    out = {}
    for case, twin in PAIRS.items():
        for dom in ('d02', 'd03'):
            X, C = load(case, dom), load(twin, dom)
            for f in FIELDS:
                if f not in X or f not in C:
                    continue
                xl = {r['lead_h']: r for r in X[f]['by_lead']}
                cl = {r['lead_h']: r for r in C[f]['by_lead']}
                leads = sorted(set(xl) & set(cl))
                xr = np.array([xl[t]['rmse'] for t in leads]); cr = np.array([cl[t]['rmse'] for t in leads])
                xb = np.array([xl[t]['bias'] for t in leads]); cb = np.array([cl[t]['bias'] for t in leads])
                row = {'leads': leads, 'x_rmse': xr.tolist(), 'c_rmse': cr.tolist(), 'x_bias': xb.tolist(),
                       'c_bias': cb.tolist(), 'blocks': {}}
                for a, b in BLOCKS:
                    s = [i for i, t in enumerate(leads) if a <= t <= b]
                    xq = float(np.sqrt((xr[s] ** 2).mean())); cq = float(np.sqrt((cr[s] ** 2).mean()))
                    row['blocks'][f'{a}-{b}'] = dict(x_rms=xq, c_rms=cq, ratio=xq / cq if cq else None,
                                                    x_bias_mean=float(xb[s].mean()), c_bias_mean=float(cb[s].mean()),
                                                    n_leads_x_gt_c=int((xr[s] > cr[s]).sum()), n=len(s),
                                                    max_ratio=float(np.max(xr[s] / np.maximum(cr[s], 1e-9))),
                                                    lead_of_max_ratio=int(leads[s[int(np.argmax(xr[s] / np.maximum(cr[s], 1e-9)))]]))
                row['splits'] = {}
                for grp in ('land_ocean', 'elevation'):
                    for k in X[f]['spatial_splits'].get(grp, {}):
                        xs = X[f]['spatial_splits'][grp][k]; cs = C[f]['spatial_splits'][grp][k]
                        row['splits'][k] = dict(x_rmse=xs['rmse'], c_rmse=cs['rmse'], x_bias=xs['bias'], c_bias=cs['bias'],
                                                ratio=xs['rmse'] / cs['rmse'] if cs['rmse'] else None)
                out[f'{case}/{dom}/{f}'] = row
    json.dump(out, open(sys.argv[1], 'w'), indent=1)
    for k, r in out.items():
        if k.split('/')[-1] not in ('U10', 'V10', 'T2'):
            continue
        bl = ' | '.join(f"{b}: X {v['x_rms']:.3f} C {v['c_rms']:.3f} X/C {v['ratio']:.2f} (X>C {v['n_leads_x_gt_c']}/{v['n']}, max {v['max_ratio']:.2f}@{v['lead_of_max_ratio']})"
                        for b, v in r['blocks'].items())
        sp = ' '.join(f"{s}:{v['ratio']:.2f}" for s, v in r['splits'].items())
        print(f'{k:28s} {bl}\n{"":28s} pooled X/C by split {sp}')


if __name__ == '__main__':
    main()
