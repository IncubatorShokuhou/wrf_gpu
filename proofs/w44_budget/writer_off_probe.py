"""Default writer schema/data regression against immutable main, CPU only.

Use the existing in-repo writer schema case, not a scientific fidelity case.
All metadata, variable order, dtypes, dimensions and data bytes are recorded.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    assert os.environ.get('JAX_PLATFORMS')=='cpu' and not Path('/tmp/wrf_gpu2_quiet').exists()
    sys.path[:0]=[str(a.source/'src'),str(a.source/'tests')]
    import numpy as np
    from netCDF4 import Dataset
    from gpuwrf.io.wrfout_writer import write_wrfout_netcdf
    spec=importlib.util.spec_from_file_location('writer_schema_case',a.source/'tests/test_m7_netcdf_writer.py')
    M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
    a.out.mkdir(exist_ok=False);(a.out/'frames').mkdir()
    def plain(value):
        if isinstance(value,np.ndarray):return value.tolist()
        if isinstance(value,np.generic):return value.item()
        return value
    record=dict(scope=__doc__,source=str(a.source),source_commit=subprocess.check_output(
        ['git','-C',str(a.source),'rev-parse','HEAD'],text=True).strip(),platform='cpu',streams={})
    for label,full in (('minimum',False),('full',True)):
        state,grid,namelist=M.synthetic_case()
        path=a.out/'frames'/f'{label}.nc'
        write_wrfout_netcdf(state,grid,namelist,path,domain='d02',domain_authority=M.writer_authority(grid),
            valid_time='2026-05-25_21:00:00',run_start='2026-05-25_18:00:00',lead_hours=3.,full_variable_set=full)
        with Dataset(path) as ds:
            ds.set_auto_mask(False)
            metadata=dict(dimensions={k:dict(size=len(v),unlimited=v.isunlimited()) for k,v in ds.dimensions.items()},
                globals={k:plain(ds.getncattr(k)) for k in ds.ncattrs()},variable_order=list(ds.variables),variables={})
            for name,variable in ds.variables.items():
                values=np.asarray(variable[:])
                metadata['variables'][name]=dict(dimensions=list(variable.dimensions),dtype=str(variable.dtype),
                    attrs={k:plain(variable.getncattr(k)) for k in variable.ncattrs()},
                    data_sha256=hashlib.sha256(values.tobytes()).hexdigest())
        record['streams'][label]=metadata
    record['schema_and_data_sha256']=hashlib.sha256(json.dumps(record['streams'],sort_keys=True).encode()).hexdigest()
    (a.out/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(sha=record['schema_and_data_sha256'],variables={k:len(v['variables']) for k,v in record['streams'].items()})))


if __name__=='__main__':main()
