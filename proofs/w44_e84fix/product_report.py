"""Bind each optimized RN site to emitted PTX in its actual fusion kernel."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re

from lowering_audit import audit, PATTERNS, HOST

p = argparse.ArgumentParser()
p.add_argument('--source', required=True, type=Path)
p.add_argument('--out', required=True, type=Path)
p.add_argument('--child', action='store_true')
a = p.parse_args()
proto = (a.out / 'optimized.hlo.pb').read_bytes()
text = (a.out / 'optimized.hlo').read_text()
required = set(PATTERNS) - HOST
if a.child:
    required = {name for name in required if name.startswith('solar_')}
result = audit(proto, text, a.source, required)
native_bodies = []
for line in text.splitlines():
    if '__gpu$xla.gpu.triton' in line:
        payload = re.search(r'ir = "((?:\\.|[^"\\])*)"', line)
        assert payload, 'Native Pallas body missing from rich optimized HLO'
        native_bodies.append(hashlib.sha256(payload[1].encode()).hexdigest())
result['optimized_native_Pallas_calls'] = len(native_bodies)
result['optimized_unique_Pallas_bodies'] = len(set(native_bodies))
assert native_bodies, 'E222: actual native physics must be present'
callers = {}
for line in text.splitlines():
    m = re.match(r'\s+(?:ROOT )?%([^ =]+) = .*\bfusion\(.*\bcalls=%([^ ,]+)', line)
    if m:
        callers.setdefault(m[2], []).append(m[1])
ptx_files = sorted((a.out / 'dump').glob('*.ptx')) + sorted((a.out / 'dump').glob('*.ptx.gz'))
assert ptx_files, 'No PTX: module filter missed this product program'
entries = {}
for file in ptx_files:
    s = gzip.decompress(file.read_bytes()).decode() if file.suffix == '.gz' else file.read_text()
    positions = list(re.finditer(r'\.visible \.entry (\w+)\(', s))
    for i, pos in enumerate(positions):
        end = positions[i+1].start() if i+1 < len(positions) else len(s)
        entries[pos[1]] = s[pos.start():end]
thunk_bindings = {}
thunk_files = list((a.out / 'dump').glob('*.thunk_sequence.txt')) + list((a.out / 'dump').glob('*.thunk_sequence.txt.gz'))
for file in thunk_files:
    contents = gzip.decompress(file.read_bytes()).decode() if file.suffix == '.gz' else file.read_text()
    for line in contents.splitlines():
        match = re.search(r'kernel=([^,]+), profile_annotation=([^,]+),', line)
        if match:
            thunk_bindings.setdefault(match[2], []).append((match[1], line.strip()))
proofs = []
for record in result['records']:
    op = ('div.rn.f64' if record['dtype']=='f64' and record['opcode']=='divide' else
          'cvt.rn.f32.f64' if record['dtype']=='f32' and record['opcode']=='convert' else None)
    if op is None:
        continue
    kernels = callers.get(record['computation'], [])
    matches = []
    for kernel in kernels:
        bindings = thunk_bindings.get(kernel, [])
        for normalized, thunk_line in bindings:
            assert normalized in entries, ('thunk kernel missing PTX entry', normalized)
            code = entries[normalized]
            assert op in code, ('RN PTX missing in exact fusion kernel', record, kernel, op)
            matches.append(dict(kernel=normalized, instruction=op,
                thunk_binding=thunk_line,
                code_sha256=hashlib.sha256(code.encode()).hexdigest(),
                matching_lines=[line.strip() for line in code.splitlines() if op in line]))
    assert matches, ('No exact HLO fusion -> PTX entry binding', record, kernels)
    proofs.append(dict(families=record['families'], computation=record['computation'],
                       hlo_instruction=record['instruction'], shape=record['shape'], PTX=matches))
result['PTX_site_proofs'] = proofs
fusion_inventory = []
for record in result['records']:
    bindings = []
    for fusion in callers.get(record['computation'], []):
        bindings.extend(dict(fusion=fusion, kernel=kernel, thunk_binding=line)
                        for kernel, line in thunk_bindings.get(fusion, []))
    fusion_inventory.append(dict(families=record['families'], dtype=record['dtype'],
        opcode=record['opcode'], shape=record['shape'], computation=record['computation'],
        instruction=record['instruction'], fusion_bindings=bindings))
result['RN_fusion_inventory'] = fusion_inventory
result['static_scheduled_kernel_references'] = sum(len(v) for v in thunk_bindings.values())
result['PTX_files'] = [{ 'name': f.name.removesuffix('.gz'),
    'sha256': hashlib.sha256(gzip.decompress(f.read_bytes()) if f.suffix == '.gz' else f.read_bytes()).hexdigest() } for f in ptx_files]
result['GPU_executed'] = False
result['scope'] = 'Actual product CompileOnly optimized HLO and PTX, original-protobuf caller frames; final RC GPU remains authoritative'
result['required_families'] = sorted(required)
result['host_constants'] = dict(pi180_bits='3c8efa35', dpd360365_bits='3f7c7e3f',
    proof='Exact host REAL4 quotients; deletion regression rejects device division at these source spans. Full-constant raw+optimized proto hashes bind folded values.')
(a.out / 'RN_AUDIT.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('records', 'PTX_site_proofs', 'RN_fusion_inventory')}, indent=2))
