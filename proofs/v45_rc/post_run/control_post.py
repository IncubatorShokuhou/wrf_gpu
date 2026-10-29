"""Deletion-order integration controls. All apply operations intercepted: NO DATA DELETES."""
from pathlib import Path
import importlib.util,json,os,sys,time
R=Path('<USER_HOME>/wrf_gpu2_lanes/v45-rc');O=R/'POST_CONTROLS';O.mkdir(exist_ok=True);spec=importlib.util.spec_from_file_location('postdata',R/'post_run/data_ops.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
D=O/'private_data';D.mkdir(exist_ok=True);m.D=D;roots=[D/'session_254c48237452_gpu_v6',*[D/'runs'/(n+'_CONTROL')for n in ['prime_swiss','prime_prod','prime_wn3','rw_warm','rw3']]]
for p in roots:
 (p/'runs/N72/wrfout').mkdir(parents=True,exist_ok=True);(p/'runs/N72/wrfout/INERT_RAW').write_bytes(b'PRIVATE INERT PAYLOAD ONLY');(p/'ENTRY.json').write_text('{"scope":"INERT"}')
rows=[];events=[];original_rmtree=m.shutil.rmtree;m.shutil.rmtree=lambda p:events.append(('DELETE_REQUEST',str(p)))
# Deterministic inert-fixture filesystem budget; the production guard is unchanged.
original_statvfs=m.os.statvfs
class Budget:
 f_bavail=20*2**30+2**20
 f_frsize=1
m.os.statvfs=lambda p: Budget() if Path(p).is_relative_to(O) else original_statvfs(p)
# Test actual ordering and literal root set without removing a path.
m.dependencies=lambda roots,out: (m.save(out,dict(inert_fixture_scan=True)) or 1)
try:
 try:m.remove_literal(roots,O/'incomplete',dict(required_score_complete=False),apply=True);raise AssertionError('missing score allowed')
 except AssertionError:assert not events;rows.append(dict(case='incomplete_score',refused=True))
 m.readers=lambda roots:[dict(pid=123,path='INERT_READER')]
 try:m.remove_literal(roots,O/'reader',dict(required_score_complete=True),apply=True);raise RuntimeError('reader allowed')
 except AssertionError:assert not events;rows.append(dict(case='active_reader',refused=True))
 m.readers=lambda roots:[]
 def blocked(roots,out):raise AssertionError('INERT queued dependency')
 m.dependencies=blocked
 try:m.remove_literal(roots,O/'dependency',dict(required_score_complete=True),apply=True);raise RuntimeError('dependency allowed')
 except AssertionError:assert not events;rows.append(dict(case='queued_dependency',refused=True))
 m.dependencies=lambda roots,out:(m.save(out,dict(inert_fixture_scan=True))or 1)
 result=m.remove_literal(roots,O/'dry',dict(required_score_complete=True),apply=False);assert result['state']=='DRY_RUN'and not events;rows.append(dict(case='dry_run',no_deletes=True))
 # Intercept final deletion, then simulate a clean final filesystem census without touching disk.
 real_rglob=m.Path.rglob
 # Directly validate receipt creation precedes any deletion request by observing the callback.
 def request(p):
  assert (O/'complete/SCI_COMPLETE.json').exists() and (O/'complete/DELETE_SHA_SIZE.json').exists()
  j=json.loads((O/'complete/DELETE_SHA_SIZE.json').read_text());assert len(j['roots'])==6 and len(j['payload'])==6
  events.append(('DELETE_REQUEST',str(p)))
 (O/'complete').mkdir(exist_ok=True);m.save(O/'complete/SCI_COMPLETE.json',dict(required_score_complete=True));m.shutil.rmtree=request
 m.remove_literal(roots,O/'complete',dict(required_score_complete=True),apply=True)
 assert [v for _,v in events]==[str(p)for p in roots] and all(p.exists()for p in roots);rows.append(dict(case='complete_evidence',six_literal_requests=True,real_deletes=0))
 # Forbidden symlink root must be refused before any deletion.
 keep=O/'protected_reference';keep.mkdir(exist_ok=True);link=D/'runs/rw3_LINK'
 if not link.exists():link.symlink_to(keep,target_is_directory=True)
 try:m.validate_roots(roots[:-1]+[link]);raise RuntimeError('symlink allowed')
 except AssertionError:rows.append(dict(case='reference_symlink',refused=True))
 # The same real preserve code must refuse just below its floor, before any delete request.
 before=len(events)
 class LowBudget:
  f_bavail=20*2**30-1
  f_frsize=1
 m.os.statvfs=lambda p: LowBudget() if Path(p).is_relative_to(O) else original_statvfs(p)
 try:m.remove_literal(roots,O/'root_floor',dict(required_score_complete=True),apply=True);raise RuntimeError('ROOT floor ignored')
 except AssertionError:assert len(events)==before;rows.append(dict(case='ROOT_floor_20GiB_minus_one',refused=True))
finally:m.shutil.rmtree=original_rmtree;m.os.statvfs=original_statvfs
(O/'VERDICT.json').write_text(json.dumps(dict(PASS=True,rows=rows,actual_deletes=0,scope='inert host controls; SCI output NOT_APPLICABLE',post_code_sha=__import__('hashlib').sha256((R/'post_run/controller.py').read_bytes()).hexdigest()),indent=2)+'\n');print('POST controls PASS',len(rows),'zero actual deletes')
