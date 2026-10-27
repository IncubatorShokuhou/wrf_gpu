"""R2 memory scaling from deviceless compiled memory stats (exact == GPU plans, MEM01). Pairs programs of the same type.
footprint = arguments + outputs + temporaries - aliased (the C-auto per-program footprint). alpha = dln(footprint)/dln(cells)."""
import json, math, sys
from pathlib import Path
S = Path('<DATA_ROOT>/wrf_gpu2/scratch/o5-perf')
pts = {  # measured: PROD d01/d02 == bundle C-auto plan (GPU) == deviceless (d02 MEM01)
    'PROD d01 (root, 120x70)': dict(cells=8400 * 44, arg=126913232, out=114544828, alias=114543364, temp=525257176),
    'PROD d02 (leaf, 267x117)': dict(cells=31239 * 44, arg=346875860, out=304192092, alias=304190732, temp=642901760),
}
geo = {'swiss_d01': ('Swiss d01 (root, 42x42)', 42 * 42 * 44), 'wn3_d03': ('WN3 d03 (leaf, 111x93)', 111 * 93 * 44),
       'wn3_fused': ('WN3 fused d02+d03', (31239 + 10323) * 44)}
for tag, (name, cells) in geo.items():
    r = S / f'r2_{tag}' / 'receipt.json'
    if not r.exists():
        continue
    m = json.loads(r.read_text()).get('memory_analysis') or {}
    if m.get('argument_size_in_bytes') is None:
        continue
    pts[name] = dict(cells=cells, arg=m['argument_size_in_bytes'], out=m['output_size_in_bytes'],
                     alias=m['alias_size_in_bytes'], temp=m['temp_size_in_bytes'])
rows = {}
for k, v in pts.items():
    v['footprint'] = v['arg'] + v['out'] + v['temp'] - v['alias']
    v['B_per_cell_args'] = v['arg'] / v['cells']
    rows[k] = v
def alpha(a, b, key='footprint'):
    A, B = rows.get(a), rows.get(b)
    if not A or not B: return None
    return math.log(B[key] / A[key]) / math.log(B['cells'] / A['cells'])
res = {'points': rows,
       'alpha_leaf (WN3 d03 -> PROD d02)': alpha('WN3 d03 (leaf, 111x93)', 'PROD d02 (leaf, 267x117)'),
       'alpha_leaf_args': alpha('WN3 d03 (leaf, 111x93)', 'PROD d02 (leaf, 267x117)', 'arg'),
       'alpha_leaf_temp': alpha('WN3 d03 (leaf, 111x93)', 'PROD d02 (leaf, 267x117)', 'temp'),
       'alpha_root (Swiss -> PROD d01, different physics)': alpha('Swiss d01 (root, 42x42)', 'PROD d01 (root, 120x70)'),
       'note': 'PROD d01/d02 values are the GPU C-auto plan (== deviceless for d02, MEM01); others deviceless CompileOnly sm_120'}
out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
txt = json.dumps(res, indent=1)
if out: out.write_text(txt + '\n')
print(txt)
