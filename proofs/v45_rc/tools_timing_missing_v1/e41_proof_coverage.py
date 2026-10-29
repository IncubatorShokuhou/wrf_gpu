"""Reuse one full native scan; check each producer inventory with product coverage rules."""
import hashlib,importlib.util,json,sys
from pathlib import Path
source,cache,full,proof,out=map(Path,sys.argv[1:])
spec=importlib.util.spec_from_file_location('product_e41',source/'scripts/wave_cubin_gate.py')
gate=importlib.util.module_from_spec(spec);sys.modules[spec.name]=gate;spec.loader.exec_module(gate)
r=json.loads(full.read_text());required=gate.proof_programs(proof)
gate.assert_coverage(gate.discover(cache),r['entries'],required)
r['coverage']=dict(r['coverage'],required_run_programs=sorted(required),run_proofs=[str(proof)])
r['scan_reused_from']=dict(path=str(full),sha256=hashlib.sha256(full.read_bytes()).hexdigest(),reason='same immutable bundle cache, exact product assert_coverage; no duplicate native scan')
out.mkdir(exist_ok=True);(out/'wave_gate.json').write_text(json.dumps(r,indent=1)+'\n')
print('E41 producer coverage PASS',proof,'required',sorted(required))
