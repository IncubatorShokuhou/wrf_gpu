"""Read-only CPU-WRF adapter smoke; formula truth is separate numeric oracles."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np
from gpuwrf.post import read_wrfout,run_products
from gpuwrf.post.registry import required_inputs

names=["app_wind","app_solar","app_icing","app_fire","app_frost","app_visibility"]
root=Path('<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu')
paths=[root/f'wrfout_d01_2023-01-15_{h}:00:00' for h in ['12','13']]
t0=time.perf_counter()
ds=read_wrfout(paths,required_inputs(names))
read_s=time.perf_counter()-t0
results={}
for name in names:
    start=time.perf_counter()
    try:
        out=run_products(ds,[name])
        results[name]={"status":"PASS","seconds":time.perf_counter()-start,
                       "variables":{key:{"min":float(da.min()),"max":float(da.max()),"units":da.attrs.get('units')}
                                    for key,da in out.data_vars.items() if np.issubdtype(da.dtype,np.number)},
                       "metadata":json.loads(out.attrs['product_metadata'])}
    except (ValueError,TypeError,ImportError) as exc:
        results[name]={"status":"REFUSED","reason":str(exc),"seconds":time.perf_counter()-start}
# Preserve the full-grid icing refusal; independently check a real unchanged
# active column whose liquid inputs satisfy the screen's documented domain.
raw=lambda name:np.asarray(ds[name].transpose('Time','bottom_top','south_north','west_east').values,dtype=np.float64)
qc,qr=raw('QCLOUD'),raw('QRAIN')
p=raw('P')+raw('PB')
t=(raw('T')+300)*(p/100000)**(287/1004.5)
valid=np.all((qc>=0)&(qr>=0),axis=(0,1))
peak=np.max(np.where((t<273.15)&(t>233.15),qc+qr,0),axis=(0,1))
y,x=np.unravel_index(np.argmax(np.where(valid,peak,-np.inf)),valid.shape)
column=ds.isel(south_north=slice(int(y),int(y)+1),west_east=slice(int(x),int(x)+1))
icing=run_products(column,['app_icing']).supercooled_liquid_water.values
qv=np.asarray(column.QVAPOR.transpose('Time','bottom_top','south_north','west_east').values,dtype=np.float64)
tc=t[:,:,y:y+1,x:x+1];pc=p[:,:,y:y+1,x:x+1]
literal=np.where((tc>233.15)&(tc<273.15),1000*pc/(287.05*tc*(1+qv/.622))*(qc[:,:,y:y+1,x:x+1]+qr[:,:,y:y+1,x:x+1]),0)
np.testing.assert_allclose(icing,literal,rtol=1e-14,atol=0)
assert icing.max()>.01, 'real icing witness must exceed0.01g/m3, not merely be subnormal/nonzero'
results['app_icing_active_column']={'status':'PASS','source_yx':[int(y),int(x)],
                                    'supercooled_liquid_max_gm3':float(icing.max()),
                                    'scope':'Unchanged valid active CPU-WRF column, literal ideal-gas/LWC check; full-grid strict refusal retained'}
receipt={"source_commit":subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
         "created_utc":datetime.now(timezone.utc).isoformat(),"case":"Swiss pristine CPU-WRF RD11",
         "input_files":[str(p) for p in paths],"input_sha256":[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],
         "read_s":read_s,"results":results,"maxrss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         "scope":"WRF-field adapter compatibility and finiteness. Not a forecast fidelity or GPU speed claim. FWI references use independent sequential table; real24h case lacks preceding-noon coverage."}
Path(sys.argv[1]).write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,sort_keys=True))
assert all(r['status']=='PASS' for k,r in results.items() if k!='app_icing'),results
assert results['app_icing']['status']=='PASS' or 'must be >= 0' in results['app_icing']['reason']
