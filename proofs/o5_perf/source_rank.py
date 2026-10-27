"""Rank node-trace kernel time per root by the dominant gpuwrf source function of each kernel's HLO instruction(s).

Inputs: w44-perf N43/R43 kernel rows + FAMILY_TIMES_FINAL joins (source frame counts per exact HLO instruction, E82/E182).
A fusion with mixed sources is assigned WHOLE to its most frequent source function (never split by instruction counts):
ranking/diagnostic only, not a per-line causal cost. Stdlib only.
"""
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path
L = Path('<USER_HOME>/wrf_gpu2_lanes/w44-perf')
ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--arm', default='N43'); ap.add_argument('--out', type=Path, required=True)
a = ap.parse_args()
rows = json.loads((L / f'TRACE_SUMMARY/{a.arm}_kernels.json').read_text())
joins = json.loads((L / f'FAMILY_TIMES_FINAL/{a.arm}_joins.json').read_text())
roots = 3
t = defaultdict(float); n = Counter(); kind = defaultdict(Counter); dom = defaultdict(Counter)
for r in rows:
    cands = joins.get(f"{r['domain']}:{r['name']}", {}).get('candidates', [])
    srcs = Counter()
    for c in cands:
        for s, cnt in c['sources'].items():
            f, _line, fn = s.split(':', 2)
            srcs[f.replace('/gpuwrf/', '') + ':' + fn] += cnt
    src = srcs.most_common(1)[0][0] if srcs else ('UNBOUND' if not cands else 'no-frames')
    k = cands[0]['kind'] if cands else 'unbound'
    ms = (r['end'] - r['start']) / 1e6 / roots
    t[src] += ms; n[src] += 1 / roots; kind[src][k] += ms; dom[src][r['domain']] += ms
total = sum(t.values())
ranked = [{'source': s, 'ms_per_root': v, 'launches_per_root': n[s], 'by_kind_ms': dict(kind[s]), 'by_domain_ms': dict(dom[s])}
          for s, v in sorted(t.items(), key=lambda kv: -kv[1])]
a.out.write_text(json.dumps({'scope': __doc__, 'arm': a.arm, 'roots': roots, 'total_ms_per_root': total, 'ranked': ranked}, indent=1) + '\n')
cum = 0
for x in ranked[:25]:
    cum += x['ms_per_root']
    print(f"{x['ms_per_root']:6.3f} ms {100 * cum / total:5.1f}% n={x['launches_per_root']:6.1f} {x['source']}")
