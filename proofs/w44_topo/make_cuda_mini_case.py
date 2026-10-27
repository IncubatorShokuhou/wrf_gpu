"""Grid-only 24x24x44 WN3 0227 crop, preserving clocks/schemes/dt and nesting.

New specified outer boundaries hold the crop's actual initial fields, coupled
with the same hybrid-mass/map-factor formula as d02_replay. This is a launcher
fixture, not a scientifically qualified replacement forecast/reference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from netCDF4 import Dataset

CROPS = ((44, 24), (93, 33), (12, 0))
N = 24


def crop(src, dst, x0, y0):
    with Dataset(src) as a, Dataset(dst, 'w', format=a.data_model) as b:
        for name, dim in a.dimensions.items():
            count = {'west_east': N, 'west_east_stag': N+1, 'south_north': N, 'south_north_stag': N+1}.get(name, len(dim))
            b.createDimension(name, None if dim.isunlimited() else count)
        for name, var in a.variables.items():
            attrs = {k: var.getncattr(k) for k in var.ncattrs()}
            fill = attrs.pop('_FillValue', None)
            out = b.createVariable(name, var.datatype, var.dimensions, **({'fill_value': fill} if fill is not None else {}))
            out.setncatts(attrs)
            slices = [slice(x0, x0+N+(d.endswith('_stag'))) if d.startswith('west_east') else
                      slice(y0, y0+N+(d.endswith('_stag'))) if d.startswith('south_north') else slice(None) for d in var.dimensions]
            out[:] = var[tuple(slices)]
        attrs = {k: a.getncattr(k) for k in a.ncattrs()}
        for name in ('WEST-EAST_GRID_DIMENSION','SOUTH-NORTH_GRID_DIMENSION','WEST-EAST_PATCH_END_STAG','SOUTH-NORTH_PATCH_END_STAG'):
            if name in attrs:attrs[name] = np.int32(N+1)
        for name in ('WEST-EAST_PATCH_END_UNSTAG','SOUTH-NORTH_PATCH_END_UNSTAG'):
            if name in attrs:attrs[name] = np.int32(N)
        for name in ('I_PARENT_START','J_PARENT_START'):
            if name in attrs and int(attrs.get('GRID_ID',1)) > 1:attrs[name] = np.int32(3)
        if 'XLAT' in b.variables:
            attrs['CEN_LAT'] = np.float32(b['XLAT'][0,N//2,N//2]);attrs['CEN_LON'] = np.float32(b['XLONG'][0,N//2,N//2])
        b.setncatts(attrs)


def boundaries(src, initial, dst):
    with Dataset(initial) as i, Dataset(src) as a, Dataset(dst, 'w', format=a.data_model) as b:
        total = np.asarray(i['MU'][0], np.float64) + np.asarray(i['MUB'][0], np.float64)
        mu = np.pad((total[:,1:]+total[:,:-1])*.5, ((0,0),(1,1)), mode='edge');mu[:,0]=total[:,0];mu[:,-1]=total[:,-1]
        mv = np.pad((total[1:]+total[:-1])*.5, ((1,1),(0,0)), mode='edge');mv[0]=total[0];mv[-1]=total[-1]
        c = lambda name:np.asarray(i[name][0],np.float64)[:,None,None]
        mh = c('C1H')*total + c('C2H');mf = c('C1F')*total + c('C2F')
        mass_u=(c('C1H')*mu+c('C2H'))/np.asarray(i['MAPFAC_UY'][0],np.float64)
        mass_v=(c('C1H')*mv+c('C2H'))/np.asarray(i['MAPFAC_VX'][0],np.float64)
        for name, dim in a.dimensions.items():
            count={'west_east':N,'west_east_stag':N+1,'south_north':N,'south_north_stag':N+1}.get(name,len(dim))
            b.createDimension(name,None if dim.isunlimited() else count)
        b.setncatts({k:a.getncattr(k) for k in a.ncattrs()})
        for name,var in a.variables.items():
            out=b.createVariable(name,var.datatype,var.dimensions);out.setncatts({k:var.getncattr(k) for k in var.ncattrs() if k!='_FillValue'})
            match=re.fullmatch(r'(.+)_B(T?)([XY])([SE])',name)
            if not match:
                out[:]=var[:];continue
            field,tendency,axis,edge=match.groups()
            if tendency:
                out[:]=np.zeros(tuple(len(b.dimensions[d]) for d in var.dimensions),var.dtype);continue
            if field == 'HT_SHAD':
                # WRF cold allocator field is not written in wrfinput.
                assert np.count_nonzero(np.asarray(var[:])) == 0
                out[:]=np.zeros(tuple(len(b.dimensions[d]) for d in var.dimensions),var.dtype)
                continue
            if field not in i.variables:raise ValueError('no crop IC field for boundary '+field)
            source_field='THM' if field=='T' and int(getattr(i,'USE_THETA_M',1))==1 and 'THM' in i.variables else field
            value=np.asarray(i[source_field][0],np.float64)
            if value.ndim == 3 and field != 'W':
                value=value*({'U':mass_u,'V':mass_v,'PH':mf}.get(field,mh))
            width=len(a.dimensions['bdy_width'])
            strips=[]
            for k in range(width):
                index=k if edge=='S' else -1-k
                strips.append(value[..., :, index] if axis=='X' else value[..., index, :])
            value=np.stack(strips)
            out[:]=np.broadcast_to(value,(len(a.dimensions['Time']),*value.shape)).astype(var.dtype)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(exist_ok=False,parents=True)
    for domain,(x,y) in enumerate(CROPS,1):
        for prefix in ('wrfinput','wrflowinp'):
            crop(a.source/f'{prefix}_d0{domain}',a.out/f'{prefix}_d0{domain}',x,y)
    boundaries(a.source/'wrfbdy_d01',a.out/'wrfinput_d01',a.out/'wrfbdy_d01')
    text=(a.source/'namelist.input').read_text()
    for name,value in dict(e_we='25,25,25',e_sn='25,25,25',i_parent_start='1,3,3',j_parent_start='1,3,3').items():
        text,count=re.subn(r'(?im)^\s*'+name+r'\s*=.*$',f' {name} = {value},',text);assert count==1,name
    (a.out/'namelist.input').write_text(text)
    files={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in a.out.iterdir()}
    report=dict(source=str(a.source),shape=[3,44,N,N],crops=CROPS,clock='original2026-02-28T00Z unchanged',schemes_dt_radt='unchanged',
        boundary='crop IC held with zero tendencies; real hybrid mass and map factors; grid-conversion launcher fixture only',files=files)
    (a.out/'MINI_MANIFEST.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
