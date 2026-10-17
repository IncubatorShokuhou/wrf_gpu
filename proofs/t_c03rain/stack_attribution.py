"""Map actual c03 E41 stack kernels to optimized HLO source metadata."""
import collections
import importlib.util
import json
from pathlib import Path
import re
import sys

import zstandard
from gpuwrf.runtime import aot_executable as ax, aot_slim_module as asm

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('e41gate', ROOT/'scripts/wave_cubin_gate.py')
gate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gate
spec.loader.exec_module(gate)

case = Path(sys.argv[1])
out = Path(sys.argv[2])
pins = json.loads((case/'e41/PIN_MAP.json').read_text())['pins']
pin = next(p for p in pins if '-cache' in p['pin'])
blob = zstandard.ZstdDecompressor().decompress(Path(pin['pin']).read_bytes())[4:]
hlo = gate._optimized_hlo(blob, ax, asm)
# HloModuleProto field17 is its StackFrameIndexProto. IDs are 1-based.
gpu = gate._gpu_payload(blob, ax, asm)
hwc = next(v for f,w,v in ax._proto_fields(gpu) if f == 1)
proto = next(v for f,w,v in ax._proto_fields(hwc) if f == 1)
index = next(v for f,w,v in ax._proto_fields(proto) if f == 17)
parts = list(ax._proto_fields(index))
files = [''] + [v.decode() for f,w,v in parts if f == 1]
functions = [''] + [v.decode() for f,w,v in parts if f == 2]
locations = [{}] + [dict((f,v) for f,w,v in ax._proto_fields(v)) for f,w,v in parts if f == 3]
frames = [{}] + [dict((f,v) for f,w,v in ax._proto_fields(v)) for f,w,v in parts if f == 4]
def source(frame_id):
    frame = frames[frame_id]
    loc = locations[frame.get(1,0)]
    return (files[loc.get(1,0)], functions[loc.get(2,0)], loc.get(3,0))
original = json.loads((case/'e41/wave_gate.json').read_text())
flags = original['entries'][0]['elfs'][0]['stack_over256']
computations = list(re.finditer(r'^([^\n ]+)\s*\([^\n]*\)\s*->[^\n]*\{\n(.*?)^\}',hlo,re.M|re.S))
rows = []
for kernel, stack in flags:
    target = kernel.replace('_fusion','_fusion')
    matched = []
    for line in hlo.splitlines():
        name = re.match(r'\s*(?:ROOT )?%?([\w.]+)\s*=', line)
        if name and name[1].replace('.','_') == target:
            callee = re.search(r'calls=%?([\w.]+)',line)
            if callee:
                body = next((m[2] for m in computations if m[1].lstrip('%') == callee[1]),None)
                matched.append({'call':line[:500],'computation':callee[1],
                    'sources': collections.Counter(source(int(i)) for i in re.findall(r'stack_frame_id=(\d+)',body or '')).most_common(12),
                    'ops':collections.Counter(re.findall(r'op_name="([^"]+)"',body or '')).most_common(8),
                    'body_found':body is not None})
    rows.append({'kernel':kernel,'stack_B':stack,'matched':matched})
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps({'source_pin':pin,'rows':rows},indent=2)+'\n')
for row in rows: print(row['kernel'],row['stack_B'],str(row['matched'])[:900])
