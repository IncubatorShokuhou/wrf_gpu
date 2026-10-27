"""Pallas (native Triton) kernel roofline: unique-byte floor per launch from the kernel's own Triton IR (arguments actually
loaded/stored, pointer provenance; ttir_args.py) x node-trace launches/time (w44-perf N43, 3 ordinary roots).
Unique bytes assume every loaded/stored argument is touched completely (masked boundary-only kernels overcount).
TB/s above peak would indicate cache reuse or partial reads; below ~50 % suggests latency/occupancy/compute limits."""
import json, re, sys
from collections import defaultdict
from pathlib import Path
L = Path('<USER_HOME>/wrf_gpu2_lanes/w44-perf'); P = Path('<USER_HOME>/wrf_gpu2_lanes/o5-perf/pallas')
PEAK = 1.792e12
ARM = sys.argv[2] if len(sys.argv) > 2 else 'N43'
k = json.loads((L / f'TRACE_SUMMARY/{ARM}_kernels.json').read_text())
joins = json.loads((L / f'FAMILY_TIMES_FINAL/{ARM}_joins.json').read_text())
by = {d: json.loads((P / f'PALLAS_BYTES_{d}.json').read_text()) for d in ('d01', 'd02')}
agg = defaultdict(lambda: [0, 0.0])
for r in k:
    d = r['domain']
    if d not in by: continue
    base = re.sub(r'__\d+$', '', r['name'])
    if base not in by[d] or 'error' in by[d][base]: continue
    a = agg[(d, base)]; a[0] += 1; a[1] += (r['end'] - r['start']) * 1e-9
rows = []
for (d, name), (n, t) in agg.items():
    b = by[d][name]['unique_bytes_MB'] * 1e6
    fam = joins.get(f'{d}:{name}', {}).get('family', '?')
    rows.append({'domain': d, 'kernel': name, 'family': fam, 'launches_per_root': n / 3, 'ms_per_root': t * 1e3 / 3,
                 'unique_MB_per_launch': b / 1e6, 'TBps': b * n / t / 1e12, 'pct_peak': 100 * b * n / t / PEAK,
                 'floor_ms_per_root': b * n / PEAK * 1e3 / 3, 'excess_ms_per_root': max(0, t - b * n / PEAK) * 1e3 / 3})
rows.sort(key=lambda x: -x['ms_per_root'])
tot_t = sum(x['ms_per_root'] for x in rows); tot_f = sum(x['floor_ms_per_root'] for x in rows)
fam = defaultdict(lambda: [0.0, 0.0])
for x in rows:
    f = fam[x['family'].split(':')[0]]; f[0] += x['ms_per_root']; f[1] += x['floor_ms_per_root']
res = {'scope': __doc__, 'pallas_ms_per_root': tot_t, 'pallas_floor_ms_per_root': tot_f,
       'aggregate_pct_peak': 100 * tot_f / tot_t if tot_t else None,
       'by_family': {k2: {'ms_per_root': v[0], 'floor_ms_per_root': v[1], 'pct_peak': 100 * v[1] / v[0] if v[0] else None}
                     for k2, v in sorted(fam.items(), key=lambda kv: -kv[1][0])},
       'kernels': rows}
Path(sys.argv[1]).write_text(json.dumps(res, indent=1) + '\n')
print(f"Pallas {tot_t:.2f} ms/root, unique-byte floor {tot_f:.2f} ms/root -> {100*tot_f/tot_t:.0f}% of peak aggregate")
for k2, v in list(res['by_family'].items())[:10]:
    print(f"  {k2:28s} {v['ms_per_root']:6.2f} ms  floor {v['floor_ms_per_root']:5.2f}  {v['pct_peak'] or 0:5.1f}%")
for x in rows[:18]:
    print(f"  {x['domain']} {x['kernel'][:34]:34s} n={x['launches_per_root']:5.1f} {x['ms_per_root']:6.3f} ms {x['unique_MB_per_launch']:7.1f} MB/launch {x['pct_peak']:5.1f}% peak")
