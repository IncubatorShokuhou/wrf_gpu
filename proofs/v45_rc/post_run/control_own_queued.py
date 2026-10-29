"""Own-lane queued readers must block raw cleanup, regardless of directory ownership."""
from pathlib import Path
import hashlib,importlib.util,json,sys
R=Path('<USER_HOME>/wrf_gpu2_lanes/v45-rc');O=R/'OWN_QUEUED_CONTROL';O.mkdir(exist_ok=True)
s=importlib.util.spec_from_file_location('dep_policy',R/'post_run/dependency_policy.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
policy=O/'POLICY.json';current=O/'CURRENT_PLAN.json';current.write_text('{"scope":"INERT CURRENT PLAN"}');policy.write_text(json.dumps(dict(current_plan_sha256=[hashlib.sha256(current.read_bytes()).hexdigest()],closed_artifacts={})))
reader=O/'queued_reader.sh';reader.write_text('# INERT queued reader\ncat '+str(O/'RAW_PRIVATE/wrfout')+'\n')
ok,_=m.expected(reader,policy);assert not ok
queue=O/'queued_reader.json';queue.write_text(json.dumps(dict(command='read',input=str(O/'RAW_PRIVATE'))));assert not m.expected(queue,policy)[0]
# End-to-end actual scanner also refuses a queued reader inside this lane.
ops_spec=importlib.util.spec_from_file_location('dep_ops',R/'post_run/data_ops.py');ops=importlib.util.module_from_spec(ops_spec);ops_spec.loader.exec_module(ops)
blocked=False
try:ops.dependencies([O/'RAW_PRIVATE'],O/'SCAN.json',scan_roots=[O])
except AssertionError:blocked=True
assert blocked

assert m.expected(current,policy)[0]
receipt=O/'CLOSED.json';receipt.write_text('{"scope":"INERT closed evidence"}');j=json.loads(policy.read_text());j['closed_artifacts'][str(receipt)]=dict(sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),reason='completed inert receipt');policy.write_text(json.dumps(j));assert m.expected(receipt,policy)[0];receipt.write_text('{"command":"queued raw reader"}');assert not m.expected(receipt,policy)[0]
# Terminal arm receipt classification must not allow unrelated files beside it.
arm=O/'arm';arm.mkdir(exist_ok=True);(arm/'REGISTRY_START.json').write_text('{"run_id":"inert"}');(arm/'REGISTRY_END.json').write_text('{"run_id":"inert","status":"DONE_HELD"}');(arm/'timer.json').write_text('{"teardown":{"all_reaped":true}}');(arm/'command.txt').write_text('INERT closed command');assert m.expected(arm/'command.txt',policy)[0];q=arm/'queued_reader.sh';q.write_text('INERT future reader');assert not m.expected(q,policy)[0]
(O/'VERDICT.json').write_text(json.dumps(dict(PASS=True,checks=8,actual_own_lane_scanner_refused=True,own_lane_queued_shell_refused=True,own_lane_queued_json_refused=True,current_exact_plan_allowed=True,closed_artifact_hash_bound=True,modified_closed_artifact_refused=True,terminal_receipt_allowed=True,queued_file_beside_receipt_refused=True,no_GPU_or_reference_reads_or_deletes=True),indent=2)+'\n');print('OWN QUEUED controls PASS7; no own-path blanket exemptions')
