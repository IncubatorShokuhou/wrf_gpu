"""Create a labelled LCZ patch on original Swiss inputs; originals are read-only."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import sys
# Pure input preparation: use the package's offline bootstrap, without JAX.
sys.argv.insert(1,'doctor')
from gpuwrf.io.netcdf_lock import Dataset

root=Path(__file__).resolve().parents[3]
src=Path('<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu')
out=Path('<USER_HOME>/wrf_gpu2_lanes/u2-lcz/case_swiss_lcz')
out.mkdir(exist_ok=True)
manifest={'source':str(src),'source_wrfinput_sha256':hashlib.sha256((src/'wrfinput_d01').read_bytes()).hexdigest(),
    'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'scope':'synthetic LCZ patch and morphology on real Swiss meteorology; coupled smoke, not trajectory fidelity'}
for p in src.iterdir():
    if p.name in ('wrfinput_d01','namelist.input'):
        shutil.copy2(p,out/p.name)
    elif not (out/p.name).exists():
        (out/p.name).symlink_to(p)
# Keep the LCZ synthetic input structurally valid for original WRF too.
# real.exe supplies LANDUSEF on the declared category axis; pad it to61.
expanded=out/'wrfinput_d01.expanded'
with Dataset(out/'wrfinput_d01') as original:
    category_dim=original['LANDUSEF'].dimensions[1] if 'LANDUSEF' in original.variables else None
    if category_dim is not None and len(original.dimensions[category_dim])!=61:
        with Dataset(expanded,'w',format='NETCDF4') as target:
            target.setncatts(original.__dict__)
            for name,dim in original.dimensions.items():
                target.createDimension(name,61 if name==category_dim else None if dim.isunlimited() else len(dim))
            for name,var in original.variables.items():
                kw={'fill_value':var.getncattr('_FillValue')} if '_FillValue' in var.ncattrs() else {}
                new=target.createVariable(name,var.dtype,var.dimensions,**kw)
                new.setncatts({k:var.getncattr(k) for k in var.ncattrs() if k!='_FillValue'})
                data=np.asarray(var[:])
                if category_dim in var.dimensions:
                    axis=var.dimensions.index(category_dim);shape=list(data.shape);shape[axis]=61
                    padded=np.zeros(shape,data.dtype);sl=[slice(None)]*data.ndim;sl[axis]=slice(0,data.shape[axis])
                    padded[tuple(sl)]=data;data=padded
                new[:]=data
if expanded.exists(): expanded.replace(out/'wrfinput_d01')
with Dataset(out/'wrfinput_d01','a') as ds:
    y=ds.dimensions['south_north'].size;x=ds.dimensions['west_east'].size
    patch=(slice(15,18),slice(15,19))
    for key in ('IVGTYP','LU_INDEX'):
        data=np.asarray(ds[key][:]);data[(0,)+patch]=np.tile([[53,53,56,56]],(3,1));ds[key][:]=data
    if 'LANDUSEF' in ds.variables:
        lu=np.asarray(ds['LANDUSEF'][:]);lu[(0,slice(None))+patch]=0.
        lu[0,52,15:18,15:17]=1.;lu[0,55,15:18,17:19]=1.;ds['LANDUSEF'][:]=lu
    ds.NUM_LAND_CAT=61
    for name,value in [('FRC_URB2D',.7),('BUILD_HEIGHT',8.),('MH_URB2D',8.),
                        ('STDH_URB2D',2.),('BUILD_AREA_FRACTION',.3),('BUILD_SURF_RATIO',1.)]:
        var=ds.variables.get(name)
        if var is None: var=ds.createVariable(name,'f4',('Time','south_north','west_east'))
        data=np.zeros((1,y,x),np.float32);data[(0,)+patch]=value;var[:]=data
    if 'urban_directions' not in ds.dimensions: ds.createDimension('urban_directions',4)
    lf=ds.createVariable('LF_URB2D','f4',('Time','urban_directions','south_north','west_east'))
    data=np.zeros((1,4,y,x),np.float32)
    for k,value in enumerate((.1,.3,.5,.7)): data[(0,k)+patch]=value
    lf[:]=data
    if 'month_hour' not in ds.dimensions: ds.createDimension('month_hour',288)
    ahe=ds.createVariable('AHE','f4',('Time','month_hour','south_north','west_east'))
    ahe.units='W m-2'
    data=np.zeros((1,288,y,x),np.float32);data[(0,slice(None))+patch]=25
    ahe[:]=data
    times=np.asarray(ds['Times'][:])
with Dataset(out.parent/'hourly_ahe.nc','w') as ds:
    ds.createDimension('Time',1);ds.createDimension('DateStrLen',19)
    ds.createDimension('south_north',y);ds.createDimension('west_east',x)
    ds.createVariable('Times','S1',('Time','DateStrLen'))[:]=times[:1]
    ahe=ds.createVariable('AHE','f4',('Time','south_north','west_east'));ahe.units='W m-2'
    data=np.zeros((1,y,x),np.float32);data[(0,)+patch]=40;ahe[:]=data
heatroot=out.parent/'wrf_heat_root'
(heatroot/'run').mkdir(parents=True,exist_ok=True)
pristine=Path('<USER_HOME>/src/wrf_pristine/WRF')
for name in ('phys','frame','share'):
    if not (heatroot/name).exists(): (heatroot/name).symlink_to(pristine/name)
for p in (pristine/'run').iterdir():
    target=heatroot/'run'/p.name
    if p.name!='URBPARM_LCZ.TBL' and not target.is_symlink() and not target.exists(): target.symlink_to(p)
shutil.copy2(root/'proofs/v040/urban_lcz/savepoints/URBPARM.TBL.heat',heatroot/'run/URBPARM_LCZ.TBL')
nl=(out/'namelist.input').read_text()
nl=nl.replace('&physics','&physics\n use_wudapt_lcz=1,\n distributed_ahe_opt=2,',1)
import re
nl=re.sub(r'^\s*sf_urban_physics\s*=.*$',' sf_urban_physics=1,',nl,flags=re.M|re.I)
nl=re.sub(r'num_land_cat\s*=\s*\d+','num_land_cat=61',nl,flags=re.I)
(out/'namelist.input').write_text(nl)
manifest['edited_wrfinput_sha256']=hashlib.sha256((out/'wrfinput_d01').read_bytes()).hexdigest()
manifest['heated_lcz_table_sha256']=hashlib.sha256((heatroot/'run/URBPARM_LCZ.TBL').read_bytes()).hexdigest()
manifest['hourly_ahe_sha256']=hashlib.sha256((out.parent/'hourly_ahe.nc').read_bytes()).hexdigest()
(out/'patch.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(out)
