"""Approved CPU admission scope; never manufacture model/GPU evidence."""
from pathlib import Path
import hashlib,json,sys
arm,case=map(Path,sys.argv[1:3]);D=Path('<DATA_ROOT>/wrf_gpu2/scratch/w44-rc')
g=json.loads((D/'COMPILEONLY_GATE_TABLE.json').read_text())
assert len(g['rows'])==5 and all(r['prescreen'] for r in g['rows'])
identity=D/'ITEM9_NATIVE_CARRY.json';i=json.loads(identity.read_text());assert i['actual_default_program_evidence']['verdict']=='IDENTICAL'
r=dict(scope='PLAN_ONLY',GPU_native_model='NOT_APPLICABLE',actual_large_case_model_executed=False,
       exact_CLI_dry_run=True,explicit_alignment_required=True,input_dir=str(case),
       authority='MAIN manager2026-10-26T09:54:33Z',source=(arm/'source_head.txt').read_text().strip(),
       five_program_CompileOnly=dict(path=str(D/'COMPILEONLY_GATE_TABLE.json'),sha256=hashlib.sha256((D/'COMPILEONLY_GATE_TABLE.json').read_bytes()).hexdigest()),
       host_only_default_identity=dict(path=str(identity),sha256=hashlib.sha256(identity.read_bytes()).hexdigest()))
(arm/'CPU_PLAN_ONLY.json').write_text(json.dumps(r,indent=2)+'\n')
