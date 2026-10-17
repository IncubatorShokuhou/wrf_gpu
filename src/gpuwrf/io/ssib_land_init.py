"""SSiB initialization from WRF real.exe three-layer inputs (D31 locked IO)."""
from dataclasses import replace
from pathlib import Path
import numpy as np
from gpuwrf.io.netcdf_lock import Dataset


def load_ssib_land(run_dir,domain):
    from gpuwrf.physics.lsm_ssib import initial_ssib_land
    from gpuwrf.coupling.ssib_surface_hook import SsibStatic
    path=Path(run_dir)/f'wrfinput_{domain}'
    required=('TSLB','SMOIS','ZS','XLAND','IVGTYP','XLAT','SNOW','CANWAT')
    with Dataset(path,'r') as ds:
        missing=[name for name in required if name not in ds.variables]
        if missing: raise ValueError(f'SSiB wrfinput missing {missing}; prepare real.exe sf=8 inputs')
        values={}
        for name in required+('SEAICE','SFROFF','SFCEVP'):
            if name not in ds.variables: continue
            var=ds.variables[name]; data=np.asarray(var[0] if var.dimensions[0]=='Time' else var[:])
            if not np.all(np.isfinite(data)): raise ValueError(f'SSiB nonfinite input {name}')
            values[name]=data
        scheme=getattr(ds,'SF_SURFACE_PHYSICS',8)
        if int(scheme)!=8: raise ValueError('SSiB requires real.exe-prepared sf_surface_physics=8 inputs')
        lut=str(getattr(ds,'MMINLU','')).strip()
        from gpuwrf.io.ssib_landuse import ssib_landuse_from_input
        startup=ssib_landuse_from_input(ds,run_dir)
        values.update(EMISS=startup['emiss'],ALBEDO=startup['albedo'])
    enums={'USGS':0,'SSIB':1,'MODIS':2,'MODIFIED_IGBP_MODIS_NOAH':2}
    if lut not in enums: raise ValueError(f'SSiB unsupported vegetation map {lut!r}')
    shape=values['XLAND'].shape
    if values['TSLB'].shape!=(3,)+shape or values['SMOIS'].shape!=(3,)+shape:
        raise ValueError('SSiB requires three soil levels; Noah four-layer inputs need real.exe preprocessing')
    if not np.allclose(np.asarray(values['ZS']).reshape(-1),[0.,.05,1.05],rtol=0,atol=1e-7):
        raise ValueError('SSiB ZS must equal WRF init_soil_depth_8:0,.05,1.05m')
    for name in required:
        if name in ('TSLB','SMOIS','ZS'): continue
        if values[name].shape!=shape: raise ValueError(f'SSiB {name} shape mismatch')
    iv=values['IVGTYP']; maximum={0:24,1:13,2:20}[enums[lut]]
    active=values['XLAND']<1.5
    if np.any(active & ((iv!=np.floor(iv)) | (iv<1) | (iv>maximum))):
        raise ValueError('SSiB invalid vegetation category; no category clipping')
    # Original module_sf_ssib.F:1004/1007 maps water to ITYPE=0, which
    # is fatal on XLAND<1.5. Reject inconsistent prepared input before JIT.
    water_category={0:16,1:None,2:17}[enums[lut]]
    if water_category is not None and np.any(active & (iv==water_category)):
        raise ValueError('SSiB vegetation maps to water at a land point (WRF ITYPE=0 fatal)')
    ice=values.get('SEAICE',np.zeros(shape,np.float32))
    if np.any(ice>0): raise ValueError('SSiB fractional sea-ice tiles are not coupled; land/land-ice only')
    static=SsibStatic(iv.astype(np.int32),values['XLAND'].astype(np.float32),ice.astype(np.float32),
        values['XLAT'].astype(np.float32),enums[lut],values['EMISS'].astype(np.float32),bool(np.any(ice>=.02)),.02,startup['mavail'],startup['znt'])
    land=initial_ssib_land(tslb=values['TSLB'],smois=values['SMOIS'],snow=values['SNOW'],
        canwat=values['CANWAT'],albedo=values['ALBEDO'],sfcrunoff=values.get('SFROFF'),sfcevp=values.get('SFCEVP'))
    return static,land


def bind_ssib_land(namelist,run_dir,domain,*,num_soil_layers,state):
    if int(num_soil_layers)!=3: raise ValueError('SSiB requires num_soil_layers=3')
    if (int(namelist.sf_sfclay_physics),int(namelist.bl_pbl_physics))!=(1,1):
        raise ValueError('SSiB requires revised-MM5 sf_sfclay_physics=1 and YSU bl_pbl_physics=1 (WRF physics_init)')
    if int(namelist.ra_sw_physics)!=1 or int(namelist.ra_lw_physics)!=4:
        raise ValueError('SSiB CLI currently requires Dudhia SW=1 and RRTMG LW=4; spectral SW=3/4 lacks four-component coupling')
    if int(namelist.mp_physics)!=8 or int(namelist.sf_lake_physics)!=0 or int(namelist.sf_urban_physics)!=0:
        raise ValueError('SSiB requires Thompson precipitation (mp=8), no lake/urban model')
    static,land=load_ssib_land(run_dir,domain)
    if land.tc.shape!=state.t_skin.shape: raise ValueError('SSiB input and atmospheric surface grids differ')
    return replace(namelist,ssib_static=static,ssib_land=land,use_noahmp=False,sf_surface_physics=8)
