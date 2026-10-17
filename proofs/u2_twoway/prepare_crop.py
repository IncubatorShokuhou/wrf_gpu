"""Stage a real WN3 child crop; retain the complete parent and real root BCs.

The child is the 36x30 original 3-km cells covering the Tenerife inner-nest
region. Native input operands are sliced, never interpolated or invented.
Original CPU-WRF supplies the independent coupled reference on this geometry.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,re,shutil,sys
import numpy as np
from gpuwrf.io.netcdf_lock import Dataset

source,target=map(lambda value:Path(value).resolve(),sys.argv[1:])
target.mkdir(exist_ok=False,parents=True)
plan=json.loads((source/'PLAN.json').read_text())
x0,y0,nx,ny=90,30,36,30
assert x0%3==0 and y0%3==0
with Dataset(source/'wrfinput_d02') as original, Dataset(target/'wrfinput_d02','w',format=original.file_format) as cropped:
    original.set_auto_mask(False)
    cuts={'west_east':(x0,nx),'west_east_stag':(x0,nx+1),
          'south_north':(y0,ny),'south_north_stag':(y0,ny+1)}
    for name,dimension in original.dimensions.items():
        cropped.createDimension(name,None if dimension.isunlimited() else cuts.get(name,(0,len(dimension)))[1])
    attrs={name:original.getncattr(name) for name in original.ncattrs()}
    for axis,size in (('WEST-EAST',nx),('SOUTH-NORTH',ny)):
        for name in tuple(attrs):
            if name==axis+'_GRID_DIMENSION' or (axis in name and 'PATCH_END_STAG' in name):attrs[name]=np.int32(size+1)
            elif axis in name and 'PATCH_END_UNSTAG' in name:attrs[name]=np.int32(size)
    for name,value in (('I_PARENT_START',np.int32(46)),('J_PARENT_START',np.int32(26)),
                       ('CEN_LAT',original['XLAT'][0,y0+ny//2,x0+nx//2]),
                       ('CEN_LON',original['XLONG'][0,y0+ny//2,x0+nx//2])):
        if name in attrs:attrs[name]=value
    cropped.setncatts(attrs)
    for name,var in original.variables.items():
        attrs={key:var.getncattr(key) for key in var.ncattrs()}
        fill=attrs.pop('_FillValue',None)
        dest=cropped.createVariable(name,var.datatype,var.dimensions,**({'fill_value':fill} if fill is not None else {}))
        dest.setncatts(attrs)
        index=tuple(slice(* (lambda start,count:(start,start+count))(*cuts[dim])) if dim in cuts else slice(None)
                    for dim in var.dimensions)
        dest[:]=var[index]
    terrain=np.asarray(cropped['HGT'][0])
    assert np.isfinite(terrain).all() and terrain.max()>1000.,'crop must exercise real Tenerife terrain'
for name in ('wrfinput_d01','wrfbdy_d01','wrf.exe'):
    (target/name).symlink_to(source/name)
for file in source.iterdir():
    if file.is_symlink() and not file.name.startswith(('wrf','rsl')):
        (target/file.name).symlink_to(file.resolve())
tables={}
for name in ('qr_acr_qg_V4.dat','qr_acr_qsV2.dat','freezeH2O.dat'):
    path=Path(plan['source_case'])/name
    shutil.copy2(path,target/name)
    tables[name]=dict(source=str(path),sha256=hashlib.sha256((target/name).read_bytes()).hexdigest())
text=(source/'namelist.input').read_text()
for key,value in (('e_we','121,37'),('e_sn','71,31'),('i_parent_start','1,46'),('j_parent_start','1,26')):
    text,count=re.subn(rf'^\s*{key}\s*=.*$',f' {key} = {value},',text,flags=re.M|re.I)
    assert count==1,key
(target/'namelist.input').write_text(text)
plan.update(created_utc=datetime.now(timezone.utc).isoformat(),case=str(target),
    geometry='original120x70 parent +real36x30 Tenerife childcrop,44levels,ratio3,start46/26',
    crop=dict(child_original='wrfinput_d02',x0=x0,y0=y0,nx=nx,ny=ny,max_hgt_m=float(terrain.max())),
    input_files={name:dict(source=str(source/name),sha256=hashlib.sha256((target/name).read_bytes()).hexdigest())
                 for name in ('wrfinput_d01','wrfinput_d02','wrfbdy_d01')},
    namelist_sha256=hashlib.sha256(text.encode()).hexdigest(),thompson_setup_tables=tables,
    execution='CPU12,13,28,29nice19 MPI1 OMP1 timeout600; noGPU',
    scope='real-input child crop for intermediate CPU qualification, not full-case/release validation')
(target/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
print(json.dumps(plan,indent=2))
