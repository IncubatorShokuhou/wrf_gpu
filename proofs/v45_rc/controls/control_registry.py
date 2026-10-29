import importlib.util,json,os,subprocess,sys,time,hashlib
from pathlib import Path
R=Path('<USER_HOME>/wrf_gpu2_lanes/v45-rc');T=Path('<DATA_ROOT>/wrf_gpu2/worktrees/w44-rc/proofs/v45_rc/tools_v6');O=R/'v6_controls_r3';O.mkdir(exist_ok=True)
os.environ.update(JAX_PLATFORMS='cpu',WRF_GPU2_SEQ_SMOKE='1',W45R_PINNED_SOURCE=str(R/'archive'),W45R_PIN_MANIFEST=str(R/'tools_v6/PIN_MANIFEST.json'),W45R_PRIVATE_ATLAS=str(O/'private_atlas'),PYTHONDONTWRITEBYTECODE='1')
sys.path.insert(0,str(T));import registry_hook
m=json.load(open(os.environ['W45R_PIN_MANIFEST']));case=m['inputs']['WN3_0227'];rows=[]
fixture=O/'arm.sh';fixture.write_text('#!/bin/bash\nexit 0\n')
for label,body,cap,expected in [('done','exit 0',3,'DONE_HELD'),('failed','exit 7',3,'FAILED'),('timeout','sleep 20',.3,'ABORTED')]:
 fixture.write_text('#!/bin/bash\n'+body+'\n');out=O/label
 cmd=[sys.executable,'-I',str(T/'arm_timer.py'),'--seconds',str(cap),'--term-grace','.2','--kill-grace','.2','--receipt',str(out/'timer.json'),'--','bash',str(fixture),'unused',label,case,str(out),'--','--hours','72']
 result=subprocess.run(cmd,capture_output=True,text=True);(O/(label+".log")).write_text(result.stdout+result.stderr);rc=result.returncode
 assert(out/'REGISTRY_START.json').is_file(),label
 end=json.loads((out/'REGISTRY_END.json').read_text());assert end['status']==expected,(label,rc,end)
 assert json.loads((out/'timer.json').read_text())['teardown']['all_reaped']
 rows.append(dict(case=label,rc=rc,status=end['status'],one_start=True,one_end=True))
# Signal path is actual supervisor TERM, distinct from per-arm cap.
fixture.write_text('#!/bin/bash\nsleep 20\n');out=O/'signal';cmd=[sys.executable,'-I',str(T/'arm_timer.py'),'--seconds','30','--term-grace','.2','--kill-grace','.2','--receipt',str(out/'timer.json'),'--','bash',str(fixture),'unused','signal',case,str(out),'--','--hours','24'];p=subprocess.Popen(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
for _ in range(100):
 if(out/'REGISTRY_START.json').exists():break
 time.sleep(.03)
assert(out/'REGISTRY_START.json').exists();p.terminate();rc=p.wait(timeout=10);assert json.loads((out/'REGISTRY_END.json').read_text())['status']=='ABORTED';rows.append(dict(case='signal',rc=rc,status='ABORTED'))
# Update after receipted deletion runs the actual registry schema and append.
out=O/'done';(out/'wrfout').mkdir();(out/'wrfout/wrfout_d01_control').write_text('OWN FIXTURE ONLY\n');subprocess.run([sys.executable,'-I',str(T/'arm_receipt.py'),str(O),str(out),str(out/'timer.json'),str(O/'receipts'),'--delete-wrfout'],check=True,capture_output=True,text=True);assert not (out/'wrfout').exists();assert json.loads((out/'REGISTRY_END.json').read_text())['raw_deleted'] is True
# Reconcile private append journal, not the lane's self-reported counts.
files=list((O/'private_atlas').glob('*.jsonl'));print('private files',[p.name for p in files]);events=[json.loads(l)for p in files for l in p.read_text().splitlines()if l.strip()];(O/'REGISTRY_EVENTS.json').write_text(json.dumps(events,indent=2)+'\n')
(O/'REGISTRY_CONTROLS.json').write_text(json.dumps(dict(rows=rows,raw_deleted_update=True,private_atlas_only=True,actual_register_schema_and_disk_append=True,event_count=len(events)),indent=2)+'\n');print(rows,'update passed')
