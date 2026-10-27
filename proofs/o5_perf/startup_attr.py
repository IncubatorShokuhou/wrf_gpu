"""Attribute pre-first-segment JAX compile-path time (trace + lower + backend load) to gpuwrf source callers.

Input: an S0 harness receipt with the w44-perf init observer (`w44_init_lowering`, caller stacks per compile event), e.g.
w44-perf R43 (v0.4.3, PROD 3 h proxy). Stdlib only; no JAX. Durations are per-event sums on the main thread (events of one
compile are sequential, but other init work interleaves): an upper bound of avoidable compile-path time, not a wall partition.
"""
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('receipt', type=Path); ap.add_argument('--out', type=Path, required=True)
a = ap.parse_args()
d = json.loads(a.receipt.read_text())
first = d['segments'][0]['t_start']
ev = [e for e in d['w44_init_lowering'] if e.get('start', 0) <= first]
kinds = ('jaxpr_trace_duration', 'jaxpr_to_mlir_module_duration', 'backend_compile_duration')
inner = defaultdict(Counter)
for e in ev:
    c = e.get('gpuwrf_callers') or [{'module': '?', 'function': '?'}]
    k = e['event'].split('/')[-1]
    key = f"{c[0]['module']}:{c[0]['function']}"
    inner[key][k] += e.get('s', 0) or 0
    inner[key]['n_' + k] += 1
rows = []
for key, v in inner.items():
    tot = sum(v[k] for k in kinds)
    rows.append({'caller': key, 'total_s': tot, 'lowerings': v['n_jaxpr_to_mlir_module_duration'],
                 **{k: v[k] for k in kinds}})
rows.sort(key=lambda r: -r['total_s'])
big = [r for r in rows if r['total_s'] / max(1, r['lowerings']) >= 0.03]
small = [r for r in rows if r not in big]
summary = {'receipt': str(a.receipt), 'first_segment_s': first, 'events': len(ev),
           'total_compile_path_s': sum(r['total_s'] for r in rows), 'lowerings': sum(r['lowerings'] for r in rows),
           'large_programs(>=30ms/lowering)': {'s': sum(r['total_s'] for r in big), 'lowerings': sum(r['lowerings'] for r in big)},
           'small_eager_ops': {'s': sum(r['total_s'] for r in small), 'lowerings': sum(r['lowerings'] for r in small)},
           'by_caller': rows}
a.out.write_text(json.dumps(summary, indent=1) + '\n')
print(json.dumps({k: v for k, v in summary.items() if k != 'by_caller'}))
for r in rows[:12]:
    print(f"  {r['total_s']:6.3f}s n={r['lowerings']:4d} {r['caller']}")
