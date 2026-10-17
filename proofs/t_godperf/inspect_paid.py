"""Read an immutable paid CUDA executable on CPU; never deserialize onto GPU."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('pin', type=Path)
p.add_argument('out', type=Path)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--hlo', type=Path, help='Reuse previously extracted optimized HLO')
a = p.parse_args()
assert os.environ['JAX_PLATFORMS'] == 'cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
sys.path[:0] = [str(a.source/'src'), str(a.source/'scripts')]
import zstandard
import wave_cubin_gate as gate
sys.path.insert(0, str(a.source/'proofs/t_e41opt'))
from map_flags import source_stacks
from gpuwrf.runtime import aot_executable as ax, aot_slim_module as asm
from e41 import hlo_scan

started = time.monotonic()
a.out.mkdir(parents=True, exist_ok=False)
packed = a.pin.read_bytes()
if a.hlo:
    with gzip.open(a.hlo, 'rt') as f:
        text = f.read()
else:
    blob = zstandard.ZstdDecompressor().decompress(packed, max_output_size=1 << 33)[4:]
    text = gate._optimized_hlo(blob, ax, asm)
    del blob
with gzip.open(a.out/'optimized.hlo.gz', 'wt', compresslevel=1) as f:
    f.write(text)
comps = {}
name = None
for line in text.splitlines():
    if name is None:
        if not line.startswith('HloModule') and (m := hlo_scan.HEADER.match(line)):
            name = m[1]
            comps[name] = []
    elif line.strip() == '}':
        name = None
    elif m := hlo_scan.INSTR.match(line):
        op = hlo_scan.OP.search(m[2])
        assert op
        comps[name].append((m[1], op[1], line))
assert comps and name is None
chain = source_stacks(text)
def sources(line):
    frame = re.search(r'stack_frame_id=(\d+)', line)
    return chain(frame[1]) if frame else []
def matches(family, line):
    return any(Path(f).name == family for f, _, _ in sources(line))
families = ('ra_goddard_lw.py', 'ra_goddard_sw.py', 'ra_goddard_common.py',
            'goddard_radiation.py')
result = {}
for family in ('ALL',) + families:
    instructions = [(c, n, op, line) for c, body in comps.items() for n, op, line in body
                    if family == 'ALL' or matches(family, line)]
    compute = [(c, n, op, line) for c, n, op, line in instructions
               if op not in ('parameter', 'constant', 'tuple', 'get-tuple-element')]
    result[family] = dict(instructions=len(instructions), compute_instructions=len(compute),
        result_f64_compute=sum(bool(re.search(r'=\s*f64\[', line)) for _, _, _, line in compute),
        result_f32_compute=sum(bool(re.search(r'=\s*f32\[', line)) for _, _, _, line in compute),
        opcodes=dict(Counter(op for _, _, op, _ in instructions)),
        computations=dict(Counter(c for c, _, _, _ in instructions).most_common(20)))
gods = [(c, n, op, line) for c, body in comps.items() for n, op, line in body
        if any(matches(f, line) for f in families)]
loops = []
for c, n, op, line in gods:
    if op != 'while':
        continue
    b = re.search(r'body=%?([\w.\-]+)', line)
    cond = re.search(r'condition=%?([\w.\-]+)', line)
    body = comps.get(b[1], []) if b else []
    loops.append(dict(parent=c, name=n, body=b[1] if b else None,
        condition=cond[1] if cond else None, body_opcodes=dict(Counter(x[1] for x in body)),
        body_launch_sites=[x[0] for x in body if x[1] in ('fusion', 'custom-call', 'copy')],
        sources=sources(line), instruction=line))
fusions = []
for c, n, op, line in gods:
    if op == 'fusion' and (m := hlo_scan.CALLS.search(line)):
        fusions.append(dict(parent=c, name=n, body=m[1], instructions=len(comps[m[1]]),
            kind=hlo_scan.KIND.search(line)[1]))
rec = dict(scope='static optimized HLO from actual paid c45 JIT; not runtime counts/roofline',
    pin=str(a.pin), pin_sha256=hashlib.sha256(packed).hexdigest(),
    optimized_hlo_sha256=hashlib.sha256(text.encode()).hexdigest(), optimized_hlo_bytes=len(text.encode()),
    computations=len(comps), by_source=result, goddard_loops=loops,
    largest_goddard_fusions=sorted(fusions, key=lambda r:r['instructions'], reverse=True)[:30],
    wall_s=time.monotonic()-started, peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
(a.out/'summary.json').write_text(json.dumps(rec, indent=2)+'\n')
(a.out/'goddard_instructions.txt').write_text('\n'.join(c+' '+line for c, _, _, line in gods)+'\n')
print(json.dumps({k:rec[k] for k in ('computations','optimized_hlo_bytes','wall_s','peak_rss_KiB')}))
print(json.dumps(result, indent=2))
