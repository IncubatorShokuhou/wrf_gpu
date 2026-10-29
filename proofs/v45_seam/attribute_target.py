"""Attribute both roots of the oversized fusion from its original protobuf."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(root / 'src'), str(root / 'scripts'), str(root / 'proofs/w44_e84fix')]
from e41.hlo_scan import INSTR, scan_text
from lowering_audit import debug_from_proto, paths

arm = Path(sys.argv[1])
out = Path(sys.argv[2])
text = (arm / 'optimized.hlo').read_text()
lines = text.splitlines()
row = next(r for r in scan_text(text)['transpose_hero_fusions'] if r['kind'] == 'kInput' and r['n_instr'] == 574)
start, end = row['comp_lines']
block = lines[start-1:end]
nodes = {}
for line in block:
    match = INSTR.match(line)
    if match:
        nodes[match[1]] = match[2]
        if line.lstrip().startswith('ROOT '):
            root_name = match[1]
proto = (arm / 'optimized.hlo.pb').read_bytes()
debug = debug_from_proto(proto)

def closure(name):
    visited = set()
    def visit(n):
        if n in visited:
            return
        visited.add(n)
        for arg in re.findall(r'%([\w.\-]+)', nodes[n]):
            visit(arg)
    visit(name)
    return visited

roots = re.findall(r'%([\w.\-]+)', nodes[root_name])
report = dict(fusion=row, original_proto_sha256=hashlib.sha256(proto).hexdigest(), roots=[])
for name in roots:
    reachable = closure(name)
    sites = Counter()
    for node in reachable:
        frames = paths(debug, debug['metadata'].get((row['comp'], node), {}))
        if frames:
            f = frames[0]
            sites[(f['file'], f['function'], f['line'])] += 1
    report['roots'].append(dict(root=name, expression=nodes[name], reachable_instructions=len(reachable),
        sites=[dict(file=f, function=fn, line=line, instructions=n) for (f,fn,line),n in sites.most_common()]))
report['shared_instructions'] = len(set.intersection(*(closure(n) for n in roots)))
out.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(fusion=row, roots=[dict(root=r['root'], n=r['reachable_instructions'], sites=r['sites'][:4]) for r in report['roots']], shared=report['shared_instructions']), indent=2))
