from pathlib import Path
import hashlib,importlib.util,json
R=Path('<USER_HOME>/wrf_gpu2_lanes/v45-rc');O=R/'PLAN_PIN_CONTROL';O.mkdir(exist_ok=True);s=importlib.util.spec_from_file_location('pinpolicy',R/'post_run/dependency_policy.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
plan=O/'reviewed_plan.json';plan.write_text('{"scope":"INERT REVIEWED PLAN"}');digest=hashlib.sha256(plan.read_bytes()).hexdigest();policy=O/'POLICY.json';policy.write_text(json.dumps(dict(current_plan_sha256=[digest],closed_artifacts={})))
assert m.expected(plan,policy)[0];copy=O/'copy.json';copy.write_bytes(plan.read_bytes());assert m.expected(copy,policy)[0]
plan.write_text('{"command":"NEW_QUEUED_READER","scope":"INERT"}');assert not m.expected(plan,policy)[0]
# Adding the changed path as a supposed authority must have no effect.
j=json.loads(policy.read_text());j['current_plan_paths']=[str(plan)];policy.write_text(json.dumps(j));assert not m.expected(plan,policy)[0]
actual=json.loads((R/'post_run/DEPENDENCY_POLICY.json').read_text());assert actual['current_plan_sha256']==['746ded545e34259fb5b712f1d9d46567a9345fd3bb20d465fa632ec618c6146a'] and'current_plan_paths'not in actual
(O/'VERDICT.json').write_text(json.dumps(dict(PASS=True,reviewed_digest_allowed=True,identical_bytes_copy_allowed=True,replaced_same_path_refused=True,path_authority_mutant_refused=True,production_reviewed_GPU_digest_pinned=True,actual_plan_mutations=0,raw_reads=0,deletes=0),indent=2)+'\n');print('F3 plan pin controls PASS5: same-pathreplacement REFUSED/no pathauthority')
