"""Derive a small, real-input launch case; never modify reference inputs.

Horizontal subsets only: d01 16x16 around the original nest SW corner,
d02 15x15, all 44 vertical levels and original physics/dt/cadences. Boundary
strips retain original WRF boundary data cropped along their tangent. This is
launcher coverage, not a geographically complete forecast or fidelity oracle.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from netCDF4 import Dataset

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--out',type=Path,required=True)
args=ap.parse_args()
assert os.environ['JAX_PLATFORMS']=='cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
source=Path('<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725')
args.out.mkdir(parents=True,exist_ok=False)
receipt={'scope':__doc__,'files':{},'grid':{'d01':[16,16,44],'d02':[15,15,44]},
         'hours':3,'dt_s':[54,18],'parent_start':[4,4]}
for name in ['wrfinput_d01','wrfinput_d02','wrfbdy_d01']:
    parent=name!='wrfinput_d02';offset=12 if parent else 0;side=16 if parent else 15
    path=source/name;target=args.out/name
    with Dataset(path) as src,Dataset(target,'w',format='NETCDF4') as dst:
        dst.setncatts({key:src.getncattr(key) for key in src.ncattrs()})
        dst.setncattr('WEST-EAST_GRID_DIMENSION',side+1)
        dst.setncattr('SOUTH-NORTH_GRID_DIMENSION',side+1)
        if not parent:
            dst.setncattr('I_PARENT_START',4);dst.setncattr('J_PARENT_START',4)
        for key,dim in src.dimensions.items():
            n=side+(key.endswith('_stag')) if key.startswith(('west_east','south_north')) else len(dim)
            if key=='Time' and name.startswith('wrfbdy'):n=2
            dst.createDimension(key,n)
        for key,var in src.variables.items():
            kwargs={'fill_value':var.getncattr('_FillValue')} if '_FillValue' in var.ncattrs() else {}
            out=dst.createVariable(key,var.dtype,var.dimensions,**kwargs)
            out.setncatts({k:var.getncattr(k) for k in var.ncattrs() if k!='_FillValue'})
            slices=[]
            for dim in var.dimensions:
                n=len(dst.dimensions[dim])
                slices.append(slice(offset,offset+n) if dim.startswith(('west_east','south_north')) else slice(0,n))
            out[:]=var[tuple(slices)] if slices else var[...]
    receipt['files'][name]={'source':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                            'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'bytes':target.stat().st_size}
text=(source/'namelist.input').read_text()
for key,value in {'e_we':'17, 16','e_sn':'17, 16','i_parent_start':'1, 4','j_parent_start':'1, 4'}.items():
    text,n=re.subn(r'(?m)^(\s*'+key+r'\s*=)[^\n]*',lambda m:m[1]+' '+value+',',text)
    assert n==1,key
(args.out/'namelist.input').write_text(text)
receipt['files']['namelist.input']={'source':str(source/'namelist.input'),
    'source_sha256':hashlib.sha256((source/'namelist.input').read_bytes()).hexdigest(),
    'sha256':hashlib.sha256(text.encode()).hexdigest()}
(args.out/'MINI_INPUTS.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'case':str(args.out),'grid':receipt['grid'],'physics_and_vertical_unchanged':True}))
