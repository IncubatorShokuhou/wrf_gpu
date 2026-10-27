"""H-CB reconciliation: per-root stepping wall minus node-capture device union.

Pure arithmetic on existing receipts (no model import, no GPU). Mixed scopes:
stepping includes radiation/KF roots; device union covers ordinary roots 68-70.
"""
import hashlib, json
from pathlib import Path
R = Path(__file__).parent / 'receipts'
bench = json.loads((R / 'GPU_BENCH_OCT23.json').read_text())
union_ms = {'v032': 97.073 / 3, 'v043': 107.354 / 3}  # PERF.md same-scope node table
out = {'scope': __doc__.strip(), 'root_dt_s': 54, 'inputs': {
    'GPU_BENCH_OCT23.json': hashlib.sha256((R / 'GPU_BENCH_OCT23.json').read_bytes()).hexdigest()},
    'labels': 'stepping [M]; residual [I, mixed scopes; v032 A/A FAIL]', 'versions': {}}
for v, u in union_ms.items():
    vals = bench['groups'][v]['metrics']['official_stepping_s_per_fch']['values']
    wall = [x * 54 / 3600 * 1e3 for x in vals]
    out['versions'][v] = {'stepping_s_per_fch': vals, 'wall_ms_per_root': [round(w, 3) for w in wall],
        'device_union_ms_per_ordinary_root': round(u, 3),
        'residual_ms_per_root': [round(w - u, 3) for w in wall],
        'mean_residual_ms': round(sum(wall) / len(wall) - u, 3)}
o, n = out['versions']['v032'], out['versions']['v043']
out['delta_mean_wall_ms_per_root'] = round(sum(n['wall_ms_per_root']) / 3 - sum(o['wall_ms_per_root']) / 2, 3)
out['delta_device_union_ms_per_root'] = round(union_ms['v043'] - union_ms['v032'], 3)
out['residual_ceiling_s_per_fch_if_fully_removed'] = round(n['mean_residual_ms'] * 3600 / 54 / 1e3, 3)
print(json.dumps(out, indent=2))
