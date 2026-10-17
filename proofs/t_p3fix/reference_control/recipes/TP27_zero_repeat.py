from pathlib import Path
import hashlib,json,os,re,signal,subprocess,sys
import numpy as np
from netCDF4 import Dataset
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from stage_case import link_assets,write_namelist
base=Path('<USER_HOME>/wrf_gpu2_lanes/t-p3fix');root=base/'TP27z';root.mkdir(exist_ok=False)
assert not Path('/tmp/wrf_gpu2_quiet').exists()
source=base/'TP26_zero/WRF/main/wrf.exe';original=source.read_bytes();source_sha=hashlib.sha256(original).hexdigest()
report={'scope':'activecompilerZERO originalWRF repeatability; inactiveMP40 compilerexception disclosedTP26b; sameprivateexe/cwd/inputs pervariant, FPtrapsoff','source_executable_sha256':source_sha,'build_receipt':str(base/'TP26b/receipt.json'),'schemes':{},'cache':'NONE'}
for mp,cid in [(50,'c21'),(51,'c22'),(8,'c00')]:
 seed=Path('<USER_HOME>/wrf_gpu2_lanes/v-prep')/cid/'cpu';case=root/f'c{mp}';case.mkdir()
 link_assets(seed,case,lane_root=base,exclude=('qr_acr_qg_V4.dat','qr_acr_qsV2.dat','freezeH2O.dat'))
 for name in ['wrfinput_d01','wrfbdy_d01']:
  if not (case/name).exists():(case/name).symlink_to((seed/name).resolve())
 if mp==8:
  for name in ['qr_acr_qg_V4.dat','qr_acr_qsV2.dat','freezeH2O.dat']:(case/name).symlink_to(base/'TP25b_mp8/tables'/name)
 nml=(seed/'namelist.input').read_text()
 for key,value in [('run_days',0),('run_hours',0),('run_minutes',0),('run_seconds',36),('end_second',36),('history_interval',0)]:
  nml,n=re.subn(rf'(?im)^\s*{key}\s*=.*$',f' {key} = {value},',nml);assert n==1,(key,n)
 nml=nml.replace(' history_interval = 0,',' history_interval = 0,\n history_interval_s = 18,')
 write_namelist(case,nml.encode(),lane_root=base)
 data=bytearray(original);patches=[]
 for old,leaf in [('<DATA_ROOT>/wrf_gpu2/wrf_truth/em_quarter_ss_center_dump.txt','d0'),('<DATA_ROOT>/wrf_gpu2/wrf_truth/em_grav2d_x_touchdown_dump.txt','d1')]:
  old=old.encode();new=str(case/leaf).encode();assert len(new)<=len(old) and data.count(old)==1
  i=data.find(old);data[i:i+len(old)]=new.ljust(len(old),b' ');patches.append({'offset':i,'before':old.decode(),'after':new.decode()})
 exe=case/'wrf.exe';exe.write_bytes(data);exe.chmod(0o700)
 record={'namelist_sha256':hashlib.sha256(nml.encode()).hexdigest(),'private_executable_sha256':hashlib.sha256(data).hexdigest(),'filename_patches':patches,'input_sha256':{n:hashlib.sha256((case/n).read_bytes()).hexdigest() for n in ['wrfinput_d01','wrfbdy_d01']},'runs':[]}
 for arm in ['a','b']:
  dest=root/f'mp{mp}_{arm}';dest.mkdir()
  with (dest/'stdout.log').open('w') as log:
   p=subprocess.Popen(['timeout','--kill-after=5','110','/usr/bin/time','-v',str(exe)],cwd=case,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1'},start_new_session=True)
   try:rc=p.wait(timeout=120)
   except BaseException:
    os.killpg(p.pid,signal.SIGKILL);p.wait();raise
  assert rc==0,rc
  assert 'SUCCESS COMPLETE WRF' in (dest/'stdout.log').read_text(),(mp,arm,'missingWRFsuccess')
  assert len(list(case.glob('wrfout_d01_*')))==3,(mp,arm,'missingframes')
  for path in list(case.glob('wrfout*'))+[case/'d0',case/'d1',case/'namelist.output']:
   if path.exists():path.rename(dest/path.name)
  record['runs'].append({'arm':arm,'rc':rc})
 rows=[]
 for pa in sorted((root/f'mp{mp}_a').glob('wrfout*')):
  with Dataset(str(pa)) as a,Dataset(str(root/f'mp{mp}_b'/pa.name)) as b:
   diff=[]
   for name in a.variables:
    x,y=np.asarray(a[name][:]),np.asarray(b[name][:])
    if x.dtype.kind=='f':assert np.isfinite(x).all() and np.isfinite(y).all(),(mp,pa.name,name,'nonfinite')
    if x.tobytes()!=y.tobytes():
     item={'field':name}
     if x.dtype.kind in 'fiu':item.update(max_abs=float(np.max(np.abs(x.astype(np.float64)-y))),different=int(np.count_nonzero(x!=y)))
     diff.append(item)
   rows.append({'file':pa.name,'BYTE':len(a.variables)-len(diff),'total':len(a.variables),'different':diff})
 assert len(rows)==3,(mp,'missingcomparisonframes')
 record['comparisons']=rows;record['all_BYTE']=all(not r['different'] for r in rows);record['all_finite']=True;record['WRF_SUCCESS_both']=True;report['schemes'][str(mp)]=record
 (root/'receipt.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(mp,[(r['file'],r['BYTE'],r['total'],[(v['field'],v.get('max_abs')) for v in r['different'] if v['field'] in ['T','QKE','U']]) for r in rows],flush=True)
report['all_BYTE']=all(v['all_BYTE'] for v in report['schemes'].values())
(root/'receipt.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print('ALL_BYTE',report['all_BYTE'],flush=True)
