"""Read only the already bound Thompson PTX; no device or new code generation."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from datetime import datetime, timezone
import zstandard

ROOT = Path(__file__).resolve().parents[2]
L = Path('<USER_HOME>/wrf_gpu2_lanes/w44-perf')
assert os.environ['JAX_PLATFORMS'] == 'cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
sys.path.insert(0, str(ROOT/'src'))
from gpuwrf.runtime import aot_executable as ax, aot_slim_module as asm
spec = importlib.util.spec_from_file_location('q3_gate', ROOT/'scripts/wave_cubin_gate.py')
gate = importlib.util.module_from_spec(spec); sys.modules[spec.name] = gate; spec.loader.exec_module(gate)

result = {'schema':'w44-perf.q3-existing-ptx.v1', 'gpu_executed':False,
          'model_lower_or_compile':False, 'cache':'NONE', 'records':[],
          'begin_utc':datetime.now(timezone.utc).isoformat()}
for rec in json.loads((L/'ACTUAL_HLO/INDEX.json').read_text())['records']:
    if rec['program'] != 'd02': continue
    data = zstandard.ZstdDecompressor().decompress(Path(rec['cache_entry']).read_bytes(), max_output_size=1<<33)[4:]
    native = gate._record(data, 2, ax, asm)
    assert hashlib.sha256(native).hexdigest() == rec['native_record2_sha256']
    ptx = gate._record(data, 1, ax, asm).decode('utf-8')
    m = re.search(r'(?m)^.*\.(?:entry|func)[^\n]*\bthompson_full_column\b[^\n]*', ptx)
    row = {'version':rec['version'], 'native_record2_sha256':rec['native_record2_sha256'],
           'ptx_sha256':hashlib.sha256(ptx.encode()).hexdigest(), 'ptx_bytes':len(ptx),
           'thompson_entry_found':bool(m)}
    if m:
        start=ptx.find('{',m.end()); depth=1; end=start+1
        while depth:
            depth+=(ptx[end]=='{')-(ptx[end]=='}'); end+=1
        body=ptx[start:end];lines=body.splitlines(); loc=None; interesting=[]
        for i,line in enumerate(lines):
            if '.loc ' in line: loc=line.strip()
            if re.search(r'\b(?:bar\.sync|barrier\.|redux\.|red\.|shfl\.)',line):
                interesting.append({'line':i,'instruction':line.strip(),'location':loc,
                                    'context':[v.strip() for v in lines[max(0,i-2):i+3]]})
        row.update(thompson_body_sha256=hashlib.sha256(body.encode()).hexdigest(),
                   body_lines=len(lines),sync_and_reduce_sites=interesting,
                   file_directives=[line.strip() for line in ptx.splitlines() if line.strip().startswith('.file ')])
        redux=[s['line'] for s in interesting if s['instruction'].startswith('redux.')]
        row['adjacent_integer_max_pairs']=[{'lines':[a,b],
            'context':[{ 'line':i, 'instruction':lines[i].strip()}
                       for i in range(max(0,a-32),min(len(lines),b+18))]}
            for a,b in zip(redux,redux[1:]) if b-a<20]
        for pair in row['adjacent_integer_max_pairs']:
            end=pair['lines'][1];definitions={}
            for i,line in enumerate(lines[:end]):
                d=re.match(r'\s*\S+\s+(%p\d+),',line)
                if d:definitions[d[1]]={'line':i,'instruction':line.strip()}
            roots=set(re.findall(r'%p\d+', ' '.join(v['instruction'] for v in pair['context'])))
            pending=list(roots);seen={}
            while pending:
                pred=pending.pop()
                if pred in seen or pred not in definitions:continue
                definition=definitions[pred];seen[pred]=definition
                pending+=re.findall(r'%p\d+',definition['instruction'].split(',',1)[1])
            pair['predicate_dependencies']=seen
    result['records'].append(row)
    print(rec['version'],'ptx_bytes',len(ptx),'thompson_entry',bool(m),
          'sync/reduce_sites',len(row.get('sync_and_reduce_sites',[])),flush=True)
result['end_utc']=datetime.now(timezone.utc).isoformat()
result['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
partial=L/'Q3_EXISTING_PTX.json.partial'
partial.write_text(json.dumps(result,indent=2)+'\n')
partial.replace(L/'Q3_EXISTING_PTX.json')
