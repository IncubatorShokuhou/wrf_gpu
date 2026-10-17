import glob, json, subprocess, numpy as np, netCDF4, jax
jax.config.update('jax_enable_x64', True)
from gpuwrf.diagnostics.morrison_radar import morrison_reflectivity
import sys
L='<USER_HOME>/wrf_gpu2_lanes/t-classd'
OUT=sys.argv[1]
res=json.load(open(f'{L}/probe/p10_sensitivity.json'))
cols=[]; cpus=[]
for r in res:
    with netCDF4.Dataset(f'{L}/c08r/run/wrfout_d01_{r["frame"]}') as ds:
        g=lambda n: np.asarray(ds[n][0], np.float32)[:, r['j'], r['i']]
        p=g('P')+g('PB'); t=((g('T')+np.float32(300.))*(p/np.float32(1e5))**np.float32(287./1004.5)).astype(np.float32)
        cols.append(np.stack([g('QVAPOR'),g('QRAIN'),g('QNRAIN'),g('QSNOW'),g('QNSNOW'),g('QGRAUP'),g('QNGRAUPEL'),t,p])); cpus.append(g('REFL_10CM'))
arr=np.stack(cols).astype(np.float32); ncol,_,nz=arr.shape
with open(f'{L}/probe/hist_in.bin','wb') as fh: fh.write(np.asarray([nz,ncol],np.int32).tobytes()); fh.write(arr.tobytes())
out=subprocess.run([f'{L}/morr_oracle/build/fp32_morr/oracle', f'{L}/probe/hist_in.bin'],capture_output=True,text=True,check=True).stdout
orc=np.array([float(l.split()[4]) for l in out.splitlines() if l.startswith('DBZ')],np.float32).reshape(ncol,nz)
port=np.asarray(jax.jit(lambda *a: morrison_reflectivity(*a))(*[arr[:,i,:] for i in range(9)]))
cpu=np.maximum(np.stack(cpus),-35.)
print('port==oracle(history inputs) BYTE', bool((port==orc).all()), 'levels', port.size)
rec=dict(columns=ncol, levels=int(port.size), port_equals_pristine_oracle_on_same_inputs=bool((port==orc).all()), max_abs_oracle_minus_cpu_refl=float(np.abs(np.maximum(orc,-35.)-cpu).max()), frames=sorted({r['frame'] for r in res}), note='Residual vs CPU REFL_10CM is history-state reconstruction (MP-exit state not in history); operator identical to pristine WRF on identical inputs.')
json.dump(rec,open(OUT,'w'),indent=1); print(rec)
