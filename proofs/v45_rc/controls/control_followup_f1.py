"""F1 exact documented three-argument shell invocation; inert scorer, zero model/reference reads."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys
ROOT=Path('<USER_HOME>/wrf_gpu2_lanes/v45-rc');SOURCE=Path('<DATA_ROOT>/wrf_gpu2/worktrees/w44-rc/proofs/v45_rc/tools_v6');O=ROOT/'F1_FOLLOWUP_CONTROL';O.mkdir(exist_ok=True)
old=subprocess.check_output(['git','show','580876b88a:proofs/v45_rc/tools_v6/followup/followup_rw3.sh'],text=True)
new=(SOURCE/'followup/followup_rw3.sh').read_text();rows=[]
def run(label,code,canonical=True,wrong_env=False,bad=None):
 base=O/label;tools=base/'tools';(tools/'followup').mkdir(parents=True);lane=base/'lane';srun=base/'short';rrun=base/'rw3';entryrec=base/'entry_receipts';entryrec.mkdir();pin='F1_HOST_ONLY_'+label
 man=base/'PIN_MANIFEST.json';man.write_text(json.dumps(dict(pin=pin,expected_receipts=str(entryrec))));(base/'PREREG.json').write_text('{}')
 idx=lane/'receipts'/('cpu72_'+pin[:12])/'STAGED_INDEX.json';idx.parent.mkdir(parents=True);idx.write_text('{}')
 for d in [srun/'runs/A3_wn3_0227_3h',rrun/'runs/N72']:
  d.mkdir(parents=True);(d/'wrfout_sha.txt').write_text('a'*64+'  wrfout_d01_inert\n')
 for d in [entryrec/'rw3t',rrun/'rw3t']if canonical else [entryrec/'rw3t']:
  d.mkdir(parents=True)
  for name,obj in [('RW3T_VERDICT.json',dict(verdict='INCONCLUSIVE (INERT HOST CONTROL)',primary_metric='NO MODEL READ')),('RW3T_DETAIL24.json',dict(arms={})),('RW3T_DETAIL72.json',dict(arms={}))]:(d/name).write_text(json.dumps(obj))
 if canonical:
  binding=dict(schema='v45-rc.rw3-receipts.v1',run_dir=str(rrun.resolve()),receipts_dir=str(entryrec.resolve()),manifest=str(man.resolve()))
  if bad=='run':binding['run_dir']=str(base/'ANOTHER_RUN')
  if bad=='manifest':binding['manifest']=str(base/'ANOTHER_MANIFEST')
  (rrun/'rw3t/ENTRY_RECEIPTS.json').write_text(json.dumps(binding))
  if bad=='missing_post':(rrun/'rw3t/RW3T_VERDICT.json').unlink()
  if bad=='missing_binding':(rrun/'rw3t/ENTRY_RECEIPTS.json').unlink()
 lib=(SOURCE/'lib.sh').read_text().replace('W45R_TOOLS=<USER_HOME>/wrf_gpu2_lanes/v45-rc/tools_v6','W45R_TOOLS='+str(tools)).replace('W45R_LANE=<USER_HOME>/wrf_gpu2_lanes/v45-rc','W45R_LANE='+str(lane));(tools/'lib.sh').write_text(lib)
 for name in ['bytes_check.py','rw_verdict_v044.py']:shutil.copy2(SOURCE/'followup'/name,tools/'followup'/name)
 # Only scientific transport is inert. Real shell, re-exec, byte checker and final report readers execute.
 (tools/'reference_io.py').write_text("""import json,os,sys
from pathlib import Path
out=Path(sys.argv[sys.argv.index('--out')+1]);out.mkdir(parents=True,exist_ok=True)
(out/'SCORE_CALL.json').write_text(json.dumps(dict(scope='INERT HOST WIRING ONLY',original_payload_reads=0,receipt_binding=os.environ.get('W45R_RECEIPTS'))))
(out/'FLOOR72.json').write_text(json.dumps(dict(binding72_A3='INERT_NOT_A_SCIENCE_RESULT',raw72='INERT')))
(out/'summary.json').write_text(json.dumps(dict(scope='INERT HOST WIRING ONLY',output_integrity_pass=False,domains={})))
""")
 script=tools/'followup/followup_rw3.sh';script.write_text(code.replace('source <USER_HOME>/wrf_gpu2_lanes/v45-rc/tools_v6/lib.sh','source '+str(tools/'lib.sh')))
 env=dict(os.environ,JAX_PLATFORMS='cpu',PYTHONDONTWRITEBYTECODE='1');env.pop('W45R_RECEIPTS',None);env.pop('W45R_FU_INNER',None)
 if wrong_env:env['W45R_RECEIPTS']='/WRONG_INHERITED_VALUE'
 cmd=['bash',str(script),str(man),str(srun),str(rrun)]
 p=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=30);(base/'wrapper.log').write_text(p.stdout+p.stderr)
 out=lane/'receipts'/('followup_rw3_'+pin[:12]);call=out/'d6_N72/SCORE_CALL.json'
 result=dict(case=label,command=cmd,rc=p.returncode,outer_reexec=True,receipt_env_initial='wrong'if wrong_env else'unset',scorer_invoked=call.is_file(),no_model_or_original_reference_reads=True)
 if call.exists():result['bound_receipts']=json.loads(call.read_text())['receipt_binding']
 if label=='old_counterexample':assert p.returncode!=0 and 'unbound variable'in p.stderr and call.exists()
 elif bad or label=='binding_removed_mutant':assert p.returncode!=0 and not call.exists(),result
 else:
  assert p.returncode==0 and call.exists() and result['bound_receipts']==str(entryrec),result
  verdict=json.loads((out/'RW_VERDICT_v045.json').read_text());assert verdict['RW3_speed']['verdict']=='INCONCLUSIVE (INERT HOST CONTROL)' and verdict['overall']!='PASS'
 rows.append(result)
run('old_counterexample',old,canonical=False)
run('documented_call',new)
run('wrong_inherited_receipts',new,wrong_env=True)
for case in ['run','manifest','missing_post','missing_binding']:run(case,new,bad=case)
# Deletion-sensitive binding control: remove resolution, leave its actual consumer intact.
a=new.index('W45R_RECEIPTS=$("$W45R_PY"');b=new.index('[[ -d $W45R_RECEIPTS ]]',a);mutant=new[:a]+new[b:];run('binding_removed_mutant',mutant)
(O/'VERDICT.json').write_text(json.dumps(dict(PASS=True,checks=len(rows),rows=rows,source_sha256=hashlib.sha256(new.encode()).hexdigest(),source_changes='only private tool/lib/output paths; inert scorer replaces scientific transport; shell argv/re-exec and receipt readers unchanged',scientific_result=False),indent=2)+'\n');print('F1 follow-up controls PASS',len(rows),'documented call unset env rc0; wrong inherited value corrected; guards before scorer; binding mutant killed')
