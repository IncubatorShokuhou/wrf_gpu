from pathlib import Path
import hashlib,json,os,re,subprocess
import numpy as np
from netCDF4 import Dataset
base=Path('<USER_HOME>/wrf_gpu2_lanes/t-p3fix');out=base/'TP25_same';out.mkdir(exist_ok=False)
assert not Path('/tmp/wrf_gpu2_quiet').exists()
case=out/'c';case.mkdir();seed=base/'TP17_caller/control'
for p in seed.iterdir():
 if p.is_symlink():(case/p.name).symlink_to(p.resolve())
data=bytearray(Path('<USER_HOME>/src/wrf_pristine/WRF/main/wrf.exe').read_bytes())
original_sha=hashlib.sha256(data).hexdigest();assert original_sha=='52f914cb38e0899d8a1da54bd52f3e08d81ba0f6a7015d3159e48393f5226bd6'
for old,leaf in [('<DATA_ROOT>/wrf_gpu2/wrf_truth/em_quarter_ss_center_dump.txt','d0'),('<DATA_ROOT>/wrf_gpu2/wrf_truth/em_grav2d_x_touchdown_dump.txt','d1')]:
 old=old.encode();new=str(case/leaf).encode();assert len(new)<=len(old) and data.count(old)==1
 i=data.find(old);data[i:i+len(old)]=new.ljust(len(old),b' ')
exe=case/'wrf.exe';exe.write_bytes(data);exe.chmod(0o700)
report={'scope':'original SAME executable/cwd/input/environment twice for mp50 and release-default mp8','original_sha256':original_sha,'private_sha256':hashlib.sha256(data).hexdigest(),'input_sha256':{n:hashlib.sha256((case/n).read_bytes()).hexdigest() for n in ['wrfinput_d01','wrfbdy_d01']},'schemes':{},'cache':'NONE'}
for mp in [50,8]:
 nml,n=re.subn(r'(?im)^\s*mp_physics\s*=.*$',f' mp_physics = {mp},',(seed/'namelist.input').read_text());assert n==1
 (case/'namelist.input').write_text(nml);record={'namelist_sha256':hashlib.sha256(nml.encode()).hexdigest(),'runs':[]}
 for arm in ['a','b']:
  dest=out/f'mp{mp}_{arm}';dest.mkdir()
  with (dest/'stdout.log').open('w') as log:p=subprocess.run(['/usr/bin/time','-v',str(exe)],cwd=case,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1'},timeout=30)
  assert p.returncode==0
  for path in list(case.glob('wrfout*'))+[case/'d0',case/'d1',case/'namelist.output']:
   if path.exists():path.rename(dest/path.name)
  record['runs'].append({'arm':arm,'rc':p.returncode})
 rows=[]
 for pa in sorted((out/f'mp{mp}_a').glob('wrfout*')):
  with Dataset(str(pa)) as a,Dataset(str(out/f'mp{mp}_b'/pa.name)) as b:
   diff=[]
   for name in a.variables:
    x,y=np.asarray(a[name][:]),np.asarray(b[name][:])
    if x.tobytes()!=y.tobytes():
     item={'field':name}
     if x.dtype.kind in 'fiu':item.update(max_abs=float(np.max(np.abs(x.astype(np.float64)-y))),different=int(np.count_nonzero(x!=y)))
     diff.append(item)
   rows.append({'file':pa.name,'BYTE':len(a.variables)-len(diff),'total':len(a.variables),'different':diff})
 record['comparisons']=rows;report['schemes'][str(mp)]=record
 (out/'receipt.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(mp,[(r['file'],r['BYTE'],r['total'],[(v['field'],v.get('max_abs')) for v in r['different'] if v['field'] in ['T','QKE','U']]) for r in rows],flush=True)
