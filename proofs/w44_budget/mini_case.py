"""Crop admitted WN3 inputs for a launch test, preserving all physics options.

The three 15 x 15 mass grids retain all 44 vertical levels. Only horizontal
grid dimensions/registration, output cadence and forecast endpoint change.
This cropped tower is a launcher test, never a WRF fidelity reference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source, destination, receipt):
    import netCDF4
    assert not destination.exists()
    destination.mkdir(parents=True)
    n=15
    dims={'west_east':n,'west_east_stag':n+1,
          'south_north':n,'south_north_stag':n+1}
    original=json.loads(receipt.read_text())['input_files']
    for name,entry in original.items():
        path=source/name
        assert digest(path)==entry['sha256'],name
        if name=='namelist.input':
            text=path.read_text()
            for key,value in [('e_we','16, 16, 16'),('e_sn','16, 16, 16'),
                              ('i_parent_start','1, 5, 5'),('j_parent_start','1, 5, 5'),
                              ('history_interval','0.9, 0.9, 0.9')]:
                text,count=re.subn(r'(?m)^(\s*'+key+r'\s*=)[^\n]+',r'\g<1> '+value+',',text)
                assert count==1,key
            (destination/name).write_text(text)
            continue
        with netCDF4.Dataset(path) as src, netCDF4.Dataset(destination/name,'w',format='NETCDF4') as dst:
            src.set_auto_mask(False)
            for key,dim in src.dimensions.items():
                dst.createDimension(key,dims.get(key,len(dim)))
            attrs={key:src.getncattr(key) for key in src.ncattrs()}
            for key in ('WEST-EAST_GRID_DIMENSION','SOUTH-NORTH_GRID_DIMENSION'):
                if key in attrs:attrs[key]=n+1
            for key in ('I_PARENT_START','J_PARENT_START','i_parent_start','j_parent_start'):
                if key in attrs:attrs[key]=1 if name.endswith('d01') else 5
            dst.setncatts(attrs)
            for key,var in src.variables.items():
                kw={}
                if '_FillValue' in var.ncattrs():kw['fill_value']=var.getncattr('_FillValue')
                output=dst.createVariable(key,var.datatype,var.dimensions,**kw)
                output.setncatts({a:var.getncattr(a) for a in var.ncattrs() if a!='_FillValue'})
                output[:]=var[tuple(slice(0,dims[d]) if d in dims else slice(None) for d in var.dimensions)]
    result=dict(schema='wrf_gpu2.w44_budget.mini_inputs.v1',
        scope='cropped launch test, not a scientific reference',mass_grid=[44,n,n],max_dom=3,
        duration_args=['--minutes','1.8'],root_steps=2,
        original_input_receipt_sha256=digest(receipt),
        input_files={name:dict(path=str(destination/name),bytes=(destination/name).stat().st_size,
                              sha256=digest(destination/name)) for name in original})
    (destination/'input_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('out',type=Path)
    parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(build(args.source,args.out,args.receipt)))
