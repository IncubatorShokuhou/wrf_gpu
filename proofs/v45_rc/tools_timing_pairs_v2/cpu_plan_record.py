"""Carried CPU admission: exact CLI plan plus the source-specific native prescreens; no model/GPU PASS."""
from pathlib import Path
import hashlib,json,subprocess,sys
arm,case=map(Path,sys.argv[1:3]);D=Path(__file__).resolve().parent
source=(arm/'source_head.txt').read_text().strip()
head='254c4823745275e3e9d397ae4d1db0fd35ec361d'
if source==head:
 proof=D/'COMPILEONLY_IDENTITIES.json';g=json.loads(proof.read_text());assert g['head']==head
 assert len(g['rows'])==5 and all(r['E41']=='PASS' and r['interpreted']==0 for r in g['rows'])
else:
 # The read-only v0.4.4 consumer stand-in has its own already-qualified programs.
 proof=D/'V044_COMPILEONLY_GATE_TABLE.json';g=json.loads(proof.read_text())
 assert len(g['rows'])==5 and all(r['prescreen'] for r in g['rows'])
 actual_tree=subprocess.check_output(['git','-C',str(Path('<DATA_ROOT>/wrf_gpu2/scratch/w44-rc/rc_bundle_3e5219bc8088/src')),'rev-parse',source+':src'],text=True).strip()
 assert actual_tree=='a22dea711202a48d00c86eafa34391ff2cb60fe2'
r=dict(scope='PLAN_ONLY',GPU_native_model='NOT_APPLICABLE',actual_large_case_model_executed=False,exact_CLI_dry_run=True,explicit_alignment_required=True,input_dir=str(case),authority='CARRIED v0.4.4 CPU admission unchanged in kind; manager2026-10-28T12:47:13Z',source=source,five_program_CompileOnly=dict(path=str(proof),sha256=hashlib.sha256(proof.read_bytes()).hexdigest()),program_identity='changed by included graupel; no identity to previous release claimed' if source==head else 'v0.4.4 read-only consumer stand-in, not candidate model')
(arm/'CPU_PLAN_ONLY.json').write_text(json.dumps(r,indent=2)+'\n')
