"""Generate fresh-init truth by linking ORIGINAL REAL4 WRF; no port import.

Uses the already-qualified fresh-call wrapper with independent work arrays,
source Exner, SQV, mixlength1, scaleaware1, INITIALIZE_QKE=true, spp0, rmol0.
The last convention is the retained explicit qualification of WRF's undefined
first-call local RMOL. Input sets are the frozen real day/night/ice/snow witnesses.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import numpy as np

root = Path(__file__).resolve().parents[2]
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
wrf = Path('<USER_HOME>/src/wrf_pristine/WRF')
witness = root/'tests/data/w44_warmstart/f3a_wrf_witnesses.npz'
data = np.load(witness)
arrays=[]; scalars=[]; labels=[]
for tag in ('ws11', 'swiss_ice_snow'):
    # One fresh call per physical column; the witness duplicated each as
    # cycling-low/cycling-high, which fresh initialization ignores.
    cold = data[f'mynn__{tag}__cold']
    arrays.extend(data[f'mynn__{tag}__arrays'][cold])
    scalars.extend(data[f'mynn__{tag}__scalars'][cold])
    labels.extend([tag]*int(cold.sum()))
a,s = np.asarray(arrays,np.float32),np.asarray(scalars,np.float32)
wrapper=(root/'tests/v025/b_phys/mynn_cold_oracle.f90').read_text()
wrapper=wrapper.replace('a(nz,13)','a(nz,14)')
wrapper=wrapper.replace('read(10,*) ust,flt,flq,fltv,xland,ts,rhosfc','read(10,*) ust,flt,flq,fltv,xland,ts,rhosfc,dx')
wrapper=wrapper.replace('((a(k,7)/p1000mb)**rcp)','a(k,14)')
wrapper=wrapper.replace('write(20,*) qke(k)','write(20,*) qke(k),work(k,3),work(k,5),work(k,4),work(k,6)')
(out/'oracle.f90').write_text(wrapper)
lines=[f'{a.shape[1]} {len(a)} 18 3000']
for column, scalar in zip(a,s):
    lines.append(' '.join(f'{x:.9e}' for x in scalar))
    lines.extend(' '.join(f'{x:.9e}' for x in layer) for layer in column)
(out/'input.txt').write_text('\n'.join(lines)+'\n')
objects=[wrf/p for p in ('phys/module_bl_mynnedmf.o','phys/module_bl_mynnedmf_common.o','share/module_model_constants.o','phys/ccpp_kind_types.o')]
cmd=['<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran','-I'+str(wrf/'phys'),'-ffree-line-length-none',str(out/'oracle.f90'),*map(str,objects),'-o',str(out/'oracle')]
r=subprocess.run(cmd,capture_output=True,text=True,timeout=60);(out/'compile.log').write_text(r.stdout+r.stderr);r.check_returncode()
r=subprocess.run([str(out/'oracle'),str(out/'input.txt'),str(out/'output.txt')],capture_output=True,text=True,timeout=60);r.check_returncode()
lines=(out/'output.txt').read_text().splitlines(); pos=1; truth=[]
for _ in a:
    pos+=1
    truth.append([list(map(float,line.split())) for line in lines[pos:pos+a.shape[1]]]);pos+=a.shape[1]
truth=np.asarray(truth,np.float32);assert np.isfinite(truth).all()
np.savez_compressed(out/'original.npz',arrays=a,scalars=s,truth=truth,labels=np.asarray(labels))
source=wrf/'phys/module_bl_mynnedmf.F'
meta={'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'objects':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in objects},'command':cmd,'wrapper_sha256':hashlib.sha256(wrapper.encode()).hexdigest(),'witness_sha256':hashlib.sha256(witness.read_bytes()).hexdigest(),'columns':len(a),'nz':a.shape[1],'fields':['QKE','EL','QSQ','TSQ','COV'],'original_sha256':hashlib.sha256((out/'original.npz').read_bytes()).hexdigest()}
(out/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
(out/'oracle').unlink()
print(json.dumps(meta,indent=2))
