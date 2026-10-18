"""GPU vs original CPU-WRF TKE_PBL / EL_PBL per frame for a Shin-Hong case (lane w41-shinhongtke).

No D6 bound exists for these fields (REPORT_ONLY); this records per lead: mass-level bias/RMSE/
max abs, Pearson r, field magnitude, the unused Z top (must be 0 in both), lead-0 byte equality
(both = WRF's pre-solve values) and nonfinite counts. No JAX.
usage: python tke_compare.py <case_dir> <out.json>
"""
import glob, json, sys
from pathlib import Path
import numpy as np
import netCDF4

case, out = Path(sys.argv[1]), Path(sys.argv[2])
rows = []
for cpu in sorted((case / 'cpu').glob('wrfout_d01_*')):
    gpu = sorted(glob.glob(str(case / 'gpu' / '**' / cpu.name), recursive=True))
    if not gpu:
        continue
    with netCDF4.Dataset(cpu) as c, netCDF4.Dataset(gpu[0]) as g:
        row = dict(frame=cpu.name[11:])
        for name in ('TKE_PBL', 'EL_PBL'):
            if name not in c.variables:  # e.g. CAM-UW: EL_PBL is not in its Registry package
                row[name] = dict(cpu='absent', gpu_present=name in g.variables)
                continue
            a = np.asarray(c[name][0], np.float64); b = np.asarray(g[name][0], np.float64)
            assert a.shape == b.shape, (name, a.shape, b.shape)
            m, n = a[:-1], b[:-1]
            d = n - m
            row[name] = dict(cpu_mean=float(m.mean()), cpu_max=float(m.max()), gpu_mean=float(n.mean()), gpu_max=float(n.max()),
                             bias=float(d.mean()), rmse=float(np.sqrt((d * d).mean())), max_abs=float(np.abs(d).max()),
                             pearson_r=(float(np.corrcoef(m.ravel(), n.ravel())[0, 1]) if m.std() > 0 and n.std() > 0 else None),
                             top_zero_cpu=bool(not a[-1].any()), top_zero_gpu=bool(not b[-1].any()),
                             byte_equal=bool(np.asarray(c[name][0], np.float32).tobytes() == np.asarray(g[name][0], np.float32).tobytes()),
                             nonfinite_gpu=int(np.count_nonzero(~np.isfinite(b))))
        rows.append(row)
summary = dict(case=str(case), frames=len(rows), rows=rows,
               lead0_byte={k: rows[0][k].get('byte_equal', 'absent') for k in ('TKE_PBL', 'EL_PBL')} if rows else None,
               tops_zero=all(r[k]['top_zero_cpu'] and r[k]['top_zero_gpu'] for r in rows for k in ('TKE_PBL', 'EL_PBL') if 'cpu_mean' in r[k]),
               nonfinite_gpu=sum(r[k]['nonfinite_gpu'] for r in rows for k in ('TKE_PBL', 'EL_PBL') if 'cpu_mean' in r[k]))
out.write_text(json.dumps(summary, indent=1) + '\n')
print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}))
for r in rows:
    print(r['frame'], *(f"{k} bias {r[k]['bias']:+.4g} rmse {r[k]['rmse']:.4g} r {r[k]['pearson_r']}" if 'bias' in r[k] else f"{k} absent in CPU-WRF" for k in ('TKE_PBL', 'EL_PBL')))
