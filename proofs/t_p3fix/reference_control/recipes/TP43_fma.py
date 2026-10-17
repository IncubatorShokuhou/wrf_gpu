from pathlib import Path
import hashlib,json,os,subprocess,time,sys
import numpy as np
base=Path('<USER_HOME>/wrf_gpu2_lanes/t-p3fix');old=base/'TP15_oracle44';root=base/'TP43_fma';root.mkdir(exist_ok=False)
assert os.environ['JAX_PLATFORMS']=='cpu' and not Path('/tmp/wrf_gpu2_quiet').exists()
r=json.loads((old/'receipt.json').read_text());hashf=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name,key in [('module_mp_p3.F','source_sha256'),('driver.f90','driver_sha256'),('input.bin','input_sha256')]:
 p=old/name;assert hashf(p)==r[key];(root/name).write_bytes(p.read_bytes())
assert hashf(root/'module_mp_p3.F')==hashf(Path('<USER_HOME>/src/wrf_pristine/WRF/phys/module_mp_p3.F'))
fc='<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/x86_64-conda-linux-gnu-gfortran';flags=[x for x in r['flags'] if not x.startswith('-ffp-contract=')];flags+=['-march=native','-ffp-contract=fast']
record={'scope':'independent original same-input44-level FMA control under existing P348 policy; no port execution','parent_literal':r,'compiler':subprocess.check_output([fc,'--version'],text=True).splitlines()[0],'flags':flags,'cache':'NONE','GPU_execution':False,'status':'BUILDING'};(root/'receipt.json').write_text(json.dumps(record,indent=2)+'\n');start=time.monotonic()
with (root/'compile.log').open('w') as log:subprocess.run([fc,*flags,'module_mp_p3.F','driver.f90','-o','oracle'],cwd=root,check=True,timeout=180,stdout=log,stderr=subprocess.STDOUT)
record['compile_s']=time.monotonic()-start
with (root/'stdout.log').open('w') as log:subprocess.run(['timeout','--kill-after=5','120',str(root/'oracle'),str(old/'tables'),'50'],cwd=root,check=True,timeout=130,stdout=log,stderr=subprocess.STDOUT)
sys.path.insert(0,'<USER_HOME>/src/wrf_gpu2_wt/t-p3fix/proofs/s2_p3');from generate_columns import FIELDS
x=np.fromfile(root/'frames.bin',np.float32).reshape((8,44,1,len(FIELDS),4),order='F');p=np.fromfile(root/'precip.bin',np.float32).reshape((8,1,5,4),order='F');assert np.isfinite(x).all() and np.isfinite(p).all()
fma={name:x[:,:,0,i,:] for i,name in enumerate(FIELDS)};fma.update({name:p[:,0,i,:] for i,name in enumerate(('rainnc','rainncv','sr','snownc','snowncv'))});np.savez_compressed(root/'fma_mp50.npz',**fma)
with np.load(old/'columns44_mp50.npz') as f:literal=dict(f)
for k in literal:assert literal[k].dtype==fma[k].dtype and literal[k].shape==fma[k].shape and np.isfinite(literal[k]).all() and literal[k][...,0].tobytes()==fma[k][...,0].tobytes()
record.update(status='COMPLETE',wall_s=time.monotonic()-start,all_finite=True,all_frame0_BYTE=True,fma_npz_sha256=hashf(root/'fma_mp50.npz'),spread={k:float(np.max(np.abs(fma[k][...,1:].astype(np.float64)-literal[k][...,1:].astype(np.float64)))) for k in literal});(root/'receipt.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record['spread']),flush=True)
# Closed build products: keep source/driver/input, sealed output NPZ and receipt.
for name in ['oracle','microphy_p3.mod','frames.bin','precip.bin']:
 p=root/name
 if p.exists():p.unlink()
