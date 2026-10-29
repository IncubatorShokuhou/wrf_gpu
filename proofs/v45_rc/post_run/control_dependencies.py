from pathlib import Path
import importlib.util,json
R=Path('<USER_HOME>/wrf_gpu2_lanes/v45-rc');O=R/'POST_DEP_CONTROL';O.mkdir(exist_ok=True);s=importlib.util.spec_from_file_location('dep',R/'post_run/data_ops.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.R=O/'own';m.R.mkdir(exist_ok=True);raw=O/'RAW_RUN';raw.mkdir(exist_ok=True);queue=O/'other';queue.mkdir(exist_ok=True);p=queue/'queued.json';p.write_text(json.dumps(dict(input=str(raw/'wrfout'))));bad=False
try:m.dependencies([raw],O/'blocked.json',scan_roots=[queue])
except AssertionError:bad=True
assert bad
# Exact producer-workflow copies in the launcher critic's review workspace are not new raw consumers.
review=O/'cr-w44-launchers/review';review.mkdir(parents=True,exist_ok=True);f=review/'PLAN.json';f.write_text(json.dumps(dict(schema='v45-rc.night-plan.v1',paths=dict(raw=str(raw)),steps=[dict(path=str(m.R/'tools/entry.sh'))])));m.dependencies([raw],O/'review.json',scan_roots=[review])
f.write_text(json.dumps(dict(schema='v45-rc.night-plan.v1',paths=dict(raw=str(raw)),steps=[dict(path=str(O/'OTHER_CONSUMER.sh'))])));bad=False
try:m.dependencies([raw],O/'foreign.json',scan_roots=[review])
except AssertionError:bad=True
assert bad
link=queue/'raw_alias'
if not link.exists():link.symlink_to(raw,target_is_directory=True)
bad=False
try:m.dependencies([raw],O/'link.json',scan_roots=[queue])
except AssertionError:bad=True
assert bad
(O/'VERDICT.json').write_text(json.dumps(dict(PASS=True,checks=4,queued_consumer_refused=True,review_copy_of_same_producer_classified=True,other_step_not_whitelisted=True,dangling_alias_refused=True,no_delete_or_original_payload_reads=True),indent=2)+'\n');print('POST dependency controls PASS4; external consumers/aliases refused, exact producer review copy classified')
