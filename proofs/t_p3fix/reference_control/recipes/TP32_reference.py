from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,signal,subprocess,time,sys
import numpy as np
from netCDF4 import Dataset
sys.path.insert(0,'<USER_HOME>/src/wrf_gpu2_wt/t-p3fix/proofs/t_p3fix')
from stage_case import link_assets,write_namelist
p=argparse.ArgumentParser();p.add_argument('--case',choices=['c21','c22','c23','c24'],required=True);p.add_argument('--arm',choices=['a','b'],required=True);a=p.parse_args()
base=Path('<USER_HOME>/wrf_gpu2_lanes/t-p3fix');mp={'c21':50,'c22':51,'c23':52,'c24':53}[a.case]
assert not Path('/tmp/wrf_gpu2_quiet').exists()
admission=json.loads((base/'TP27z/receipt.json').read_text())
if str(mp) in admission['schemes']:
 assert admission['schemes'][str(mp)]['all_BYTE'] and admission['schemes'][str(mp)]['all_finite']
source=base/'TP26_zero/WRF/main/wrf.exe';original=source.read_bytes();source_sha=hashlib.sha256(original).hexdigest();assert source_sha==admission['source_executable_sha256']
root=base/'TP32'/f'{mp}{a.arm}';root.mkdir(parents=True,exist_ok=False)
seed=Path('<USER_HOME>/wrf_gpu2_lanes/v-prep')/a.case/'cpu'
link_assets(seed,root,lane_root=base)
for name in ['qr_acr_qg_V4.dat','qr_acr_qsV2.dat','freezeH2O.dat']:
 if (root/name).is_symlink():(root/name).unlink()
for name in ['wrfinput_d01','wrfbdy_d01']:
 if not (root/name).exists():(root/name).symlink_to((seed/name).resolve())
nml=(base/'NML_RECOVERY'/f'{a.case}.namelist.input').read_bytes()
sealed=json.loads((seed.parent/'cpu_run.json').read_text())['input_sha256']['namelist.input']
assert hashlib.sha256(nml).hexdigest()==sealed
write_namelist(root,nml,lane_root=base)
data=bytearray(original);patches=[]
for old,leaf in [('<DATA_ROOT>/wrf_gpu2/wrf_truth/em_quarter_ss_center_dump.txt','d0'),('<DATA_ROOT>/wrf_gpu2/wrf_truth/em_grav2d_x_touchdown_dump.txt','d1')]:
 old=old.encode();new=str(root/leaf).encode();assert len(new)<=len(old) and data.count(old)==1
 i=data.find(old);data[i:i+len(old)]=new.ljust(len(old),b' ');patches.append({'offset':i,'before':old.decode(),'after':new.decode()})
exe=root/'wrf.exe';exe.write_bytes(data);exe.chmod(0o700)
record={'scope':'originalactiveZERO CPU-WRF24h Swiss; no portexecution; sourcebodiesunchanged/inactiveMP40exceptionTP26b','case':a.case,'mp':mp,'arm':a.arm,'source_executable_sha256':source_sha,'private_executable_sha256':hashlib.sha256(data).hexdigest(),'filename_patches':patches,'input_sha256':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ['wrfinput_d01','wrfbdy_d01','namelist.input']},'started_utc':datetime.now(timezone.utc).isoformat(),'cache':'NONE','status':'RUNNING','GPU_execution':False}
assert record['input_sha256']==json.loads((seed.parent/'cpu_run.json').read_text())['input_sha256']
record['compiler_initialization']='active physics -finit-real=zero -finit-integer=0 -finit-logical=false; FP traps off; inactiveMP40 original flags'
(root/'receipt.json').write_text(json.dumps(record,indent=2)+'\n');start=time.monotonic()
with (root/'stdout.log').open('w') as log:
 child=subprocess.Popen(['timeout','--kill-after=10','1800','/usr/bin/time','-v',str(exe)],cwd=root,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1'},start_new_session=True)
 try:rc=child.wait(timeout=1820)
 except BaseException:
  os.killpg(child.pid,signal.SIGKILL);child.wait();raise
paths=sorted(root.glob('wrfout_d01_*'));manifest=[];nonfinite=[]
for q in paths:
 h=hashlib.sha256()
 with q.open('rb') as stream:
  for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
 with Dataset(str(q)) as d:
  fields={}
  for name in d.variables:
   x=np.asarray(d[name][:]);fields[name]={'dtype':str(x.dtype),'shape':list(x.shape),'sha256':hashlib.sha256(x.tobytes()).hexdigest()}
   if x.dtype.kind=='f':
    n=int(np.count_nonzero(~np.isfinite(x)))
    if n:nonfinite.append({'file':q.name,'field':name,'count':n})
  times=[b''.join(row).decode() for row in np.asarray(d['Times'][:])]
 manifest.append({'file':q.name,'sha256':h.hexdigest(),'times':times,'fields':fields})
(root/'outputs.json').write_text(json.dumps(manifest,indent=1)+'\n')
success='SUCCESS COMPLETE WRF' in (root/'stdout.log').read_text()
record.update(rc=rc,wall_s=time.monotonic()-start,finished_utc=datetime.now(timezone.utc).isoformat(),WRF_SUCCESS=success,frames=len(paths),nonfinite=nonfinite,output_manifest_sha256=hashlib.sha256((root/'outputs.json').read_bytes()).hexdigest(),status='COMPLETE' if rc==0 and success and len(paths)==25 and not nonfinite else 'FAIL')
(root/'receipt.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');print(record,flush=True)
assert record['status']=='COMPLETE'
