from pathlib import Path
import hashlib,json,os,re,shutil,signal,subprocess,sys
import numpy as np
from netCDF4 import Dataset
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from stage_case import link_assets,write_namelist
base=Path('<USER_HOME>/wrf_gpu2_lanes/t-p3fix');root=base/'TP25b_mp8';root.mkdir(exist_ok=False)
assert not Path('/tmp/wrf_gpu2_quiet').exists()
case=root/'c';case.mkdir();seed=Path('<USER_HOME>/wrf_gpu2_lanes/v-prep/c00/cpu')
link_assets(seed,case,lane_root=base,exclude=('qr_acr_qg_V4.dat','qr_acr_qsV2.dat','freezeH2O.dat'))
for name in ['wrfinput_d01','wrfbdy_d01']:
 if not (case/name).exists():(case/name).symlink_to((seed/name).resolve())
tables=root/'tables';tables.mkdir();table_sha={}
for name in ['qr_acr_qg_V4.dat','qr_acr_qsV2.dat','freezeH2O.dat']:
 shutil.copyfile(seed/name,tables/name);table_sha[name]=hashlib.sha256((tables/name).read_bytes()).hexdigest();(case/name).symlink_to(tables/name)
nml=(seed/'namelist.input').read_text()
for key,value in [('run_days',0),('run_hours',0),('run_minutes',0),('run_seconds',36),('end_second',36),('history_interval',0)]:
 nml,n=re.subn(rf'(?im)^\s*{key}\s*=.*$',f' {key} = {value},',nml);assert n==1
nml=nml.replace(' history_interval = 0,',' history_interval = 0,\n history_interval_s = 18,')
write_namelist(case,nml.encode(),lane_root=base)
data=bytearray(Path('<USER_HOME>/src/wrf_pristine/WRF/main/wrf.exe').read_bytes());original_sha=hashlib.sha256(data).hexdigest();assert original_sha=='52f914cb38e0899d8a1da54bd52f3e08d81ba0f6a7015d3159e48393f5226bd6'
for old,leaf in [('<DATA_ROOT>/wrf_gpu2/wrf_truth/em_quarter_ss_center_dump.txt','d0'),('<DATA_ROOT>/wrf_gpu2/wrf_truth/em_grav2d_x_touchdown_dump.txt','d1')]:
 old=old.encode();new=str(case/leaf).encode();assert len(new)<=len(old) and data.count(old)==1
 i=data.find(old);data[i:i+len(old)]=new.ljust(len(old),b' ')
exe=case/'wrf.exe';exe.write_bytes(data);exe.chmod(0o700)
report={'scope':'release-default mp8 Swiss fromc00; sameoriginal52f executable/cwd/inputs/nml twice; retainedoriginalWRFtables privatecopies','original_sha256':original_sha,'private_sha256':hashlib.sha256(data).hexdigest(),'namelist_sha256':hashlib.sha256(nml.encode()).hexdigest(),'tables':table_sha,'cache':str(tables),'runs':[]}
for arm in ['a','b']:
 dest=root/arm;dest.mkdir()
 with (dest/'stdout.log').open('w') as log:
  p=subprocess.Popen(['timeout','--kill-after=5','110','/usr/bin/time','-v',str(exe)],cwd=case,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1'},start_new_session=True)
  try:rc=p.wait(timeout=120)
  except BaseException:
   os.killpg(p.pid,signal.SIGKILL);p.wait();raise
 assert rc==0,rc
 for path in list(case.glob('wrfout*'))+[case/'d0',case/'d1',case/'namelist.output']:
  if path.exists():path.rename(dest/path.name)
 report['runs'].append({'arm':arm,'rc':rc})
rows=[]
for pa in sorted((root/'a').glob('wrfout*')):
 with Dataset(str(pa)) as a,Dataset(str(root/'b'/pa.name)) as b:
  diff=[]
  for name in a.variables:
   x,y=np.asarray(a[name][:]),np.asarray(b[name][:])
   if x.tobytes()!=y.tobytes():
    item={'field':name}
    if x.dtype.kind in 'fiu':item.update(max_abs=float(np.max(np.abs(x.astype(np.float64)-y))),different=int(np.count_nonzero(x!=y)))
    diff.append(item)
  rows.append({'file':pa.name,'BYTE':len(a.variables)-len(diff),'total':len(a.variables),'different':diff})
report['comparisons']=rows
assert table_sha=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tables.iterdir()}
(root/'receipt.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print([(r['file'],r['BYTE'],r['total'],[(v['field'],v.get('max_abs')) for v in r['different'] if v['field'] in ['T','QKE','U']]) for r in rows],flush=True)
