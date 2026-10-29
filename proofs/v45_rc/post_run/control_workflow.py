"""Actual post scientific-completeness checks and paid-Swiss resume catalog; inert files only."""
from pathlib import Path
import importlib.util,json,sys,hashlib
R=Path('<USER_HOME>/wrf_gpu2_lanes/v45-rc');O=R/'POST_WORKFLOW_CONTROL';O.mkdir(exist_ok=True);s=importlib.util.spec_from_file_location('pc',R/'post_run/controller.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
F=O/'short';Q=O/'rw3';(F/'d6_swiss').mkdir(parents=True,exist_ok=True);(Q/'d6_N72').mkdir(parents=True,exist_ok=True)
def write(p,j):p.write_text(json.dumps(j))
from fixtures_complete import fixture
short,measured_rw=fixture();write(F/'SHORT_VERDICT.json',short);write(Q/'RW_VERDICT_v045.json',measured_rw)
d6=dict(domains={k:dict(pairing=dict(paired_file_count=73))for k in ['d01','d02','d03']},all_common_variables_scored=True,all_cpu_variables_required=True,pass_=False,output_integrity_pass=False);d6['pass']=False
floor=dict(coverage_complete=True,binding72_A3='INERT_NOT_PASS');sw=dict(domains=dict(d01=dict(pairing=dict(paired_file_count=25))),all_common_variables_scored=True)
write(Q/'d6_N72/summary.json',d6);write(Q/'d6_N72/FLOOR72.json',floor);write(F/'d6_swiss/summary.json',sw)
result=m.check_scores_paths(F,Q);assert result['required_score_complete'] and result['scientific_verdict']=='STOP'
rows=[dict(case='complete_negative_scientific_result',complete=True,scientific_PASS=False)]
for label,obj,path in [('missing_domain',dict(d6,domains={'d01':d6['domains']['d01']}),Q/'d6_N72/summary.json'),('missing_frame',dict(d6,domains={**d6['domains'],'d03':dict(pairing=dict(paired_file_count=72))}),Q/'d6_N72/summary.json'),('incomplete_floor',dict(floor,coverage_complete=False),Q/'d6_N72/FLOOR72.json'),('incomplete_swiss',dict(sw,domains=dict(d01=dict(pairing=dict(paired_file_count=24)))),F/'d6_swiss/summary.json')]:
 old=path.read_bytes();write(path,obj);failed=False
 try:m.check_scores_paths(F,Q)
 except AssertionError:failed=True
 finally:path.write_bytes(old)
 assert failed;rows.append(dict(case=label,refused=True))
# Controller's declaration is an explicit managed post-window handoff, not an orphaned GPU child.
launch=(R/'post_run/launch.sh').read_text();controller=(R/'post_run/controller.py').read_text()
assert 'timeout -k 15 2100'in launch and '--scope'in launch and 'JAX_PLATFORMS=cpu'in launch
assert controller.index("ready=O/'POST_GPU_READY.json'")<controller.index("cmds=[")<controller.index('proof=check_scores_paths')<controller.index('recovery=remove_literal')
rows.append(dict(case='handoff_order',model_reaped_gpu_unheld_then_score_then_delete=True))
(O/'VERDICT.json').write_text(json.dumps(dict(PASS=True,rows=rows,no_GPU_or_original_CPU_payload_reads=True,no_deletes=True,controller_sha256=hashlib.sha256((R/'post_run/controller.py').read_bytes()).hexdigest()),indent=2)+'\n');print('POST workflow controls PASS',len(rows),'complete evidence checked; incomplete frame/domain/floor/Swiss refused')
