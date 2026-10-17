"""Supplemental every-frame writer metadata audit; no numerical gate changes."""
import json, os, sys
from pathlib import Path
import numpy as np
from gpuwrf.io.netcdf_lock import Dataset
assert os.environ['JAX_PLATFORMS']=='cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
case=Path(sys.argv[1]); report=json.loads((case/'compare.json').read_text()); frames=[]
for pair in report['pairing']['pairs']:
    with Dataset(pair['cpu_file']) as cpu, Dataset(pair['gpu_file']) as gpu:
        mismatches=[]
        for name in sorted(set(cpu.variables)&set(gpu.variables)):
            a,b=cpu[name],gpu[name]
            if a.dtype!=b.dtype:
                mismatches.append({'field':name,'kind':'dtype','cpu':str(a.dtype),'gpu':str(b.dtype)})
            for attr in a.ncattrs():
                if attr not in b.ncattrs():
                    mismatches.append({'field':name,'kind':'missing_attribute','attribute':attr})
                elif not np.array_equal(a.getncattr(attr),b.getncattr(attr)):
                    mismatches.append({'field':name,'kind':'attribute_value','attribute':attr,'cpu':str(a.getncattr(attr)),'gpu':str(b.getncattr(attr))})
        globals=[{'attribute':k,'cpu':str(cpu.getncattr(k)),'gpu':str(gpu.getncattr(k)) if k in gpu.ncattrs() else None} for k in cpu.ncattrs() if k not in gpu.ncattrs() or not np.array_equal(cpu.getncattr(k),gpu.getncattr(k))]
        controls={k:{'cpu':str(cpu.getncattr(k)) if k in cpu.ncattrs() else None,'gpu':str(gpu.getncattr(k)) if k in gpu.ncattrs() else None} for k in ('MP_PHYSICS','CU_PHYSICS','BL_PBL_PHYSICS','SF_SFCLAY_PHYSICS','SF_SURFACE_PHYSICS','RA_LW_PHYSICS','RA_SW_PHYSICS')}
        frames.append({'lead_h':pair['lead_h'],'variable_metadata_mismatches':mismatches,'global_mismatches':globals,'physics_controls':controls})
result={'schema':'vrun-writer-audit-v1','paired_frames':len(frames),'by_frame':frames,'variable_metadata_mismatch_count':sum(len(f['variable_metadata_mismatches']) for f in frames),'global_mismatch_count':sum(len(f['global_mismatches']) for f in frames),'scope':'Original CPU-WRF every paired frame. Audit only; numerical D6 and coverage verdict preserved separately.'}
(case/'WRITER_AUDIT.json').write_text(json.dumps(result,indent=1,allow_nan=False)+'\n')
print(json.dumps({k:result[k] for k in ('paired_frames','variable_metadata_mismatch_count','global_mismatch_count')}))
