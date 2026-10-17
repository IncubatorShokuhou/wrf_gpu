"""Bind Liu crop's original cold initialization, geography and crop tables."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from gpuwrf.io.netcdf_lock import Dataset


def crop_initialization(inputs,static,table,field):
    """NOAHMP_INIT:2198-2236; strict corn/soy maxima and generic tie fallback."""
    f32=np.float32;veg=np.asarray(static.ivgtyp,np.int32)
    no_veg=np.isin(veg,(13,15,16,17))|((veg>=51)&(veg<=61))
    lai,sai=(np.asarray(inputs[k],np.float32) for k in ('LAI_INIT','XSAI_INIT'))
    fraction=field
    ag=(fraction[4]>=f32(.5))&~no_veg
    corn=ag&(fraction[0]>fraction[1])&(fraction[0]>fraction[2])&(fraction[0]>fraction[3])
    soy=ag&(fraction[1]>fraction[0])&(fraction[1]>fraction[2])&(fraction[1]>fraction[3])
    default=int(table['DEFAULT_CROP'][0])
    cat=np.where(no_veg,0,np.where(corn,1,np.where(soy,2,default))).astype(np.int32)
    pools=[np.asarray(inputs[k],np.float32) for k in ('LFMASS','RTMASS','STMASS','WOOD','STBLCP','FASTCP')]
    pools[0]=np.where(ag,lai/np.where(corn,f32(.015),np.where(soy,f32(.030),f32(.035))),pools[0])
    pools[1]=np.where(ag,f32(0.),pools[1]);pools[3]=np.where(ag,f32(0.),pools[3])
    pools[2]=np.where(ag,sai/f32(.003),pools[2])
    return cat,pools


def bind_crop(namelist,run_dir,domain,*,state):
    from gpuwrf.config.paths import wrf_run_dir
    from gpuwrf.io.gen2_accessor import Gen2Run
    from gpuwrf.io.land_history import load_land_history_inputs
    from gpuwrf.physics.noahmp.tables import _parse_mptable_block
    from gpuwrf.physics.noahmp.carbon import CarbonState,CarbonParameters,CarbonStatic
    from gpuwrf.physics.noahmp.crop import CropCarbonState,CropParameters,CropStaticParameters
    from gpuwrf.io.noahmp_land_init import build_noahmp_params
    from gpuwrf.io.noahmp_carbon import validate_restart
    table_dir=wrf_run_dir();text=(table_dir/'MPTABLE.TBL').read_text()
    table=_parse_mptable_block(text,'noahmp_crop_parameters')
    vegetation=_parse_mptable_block(text,'noahmp_modis_parameters')
    static=namelist.noahmp_static;shape=tuple(state.t_skin.shape)
    if np.any((np.asarray(static.ivgtyp)<1)|(np.asarray(static.ivgtyp)>20)):
        raise ValueError('Liu crop currently requires the MODIS 20-category table')
    path=Gen2Run(run_dir).wrfinput_file(domain)
    inputs=load_land_history_inputs(path,table_dir=table_dir,parameters=static.parameters)
    with Dataset(path) as ds:
        if str(ds.getncattr('MMINLU')).strip()!='MODIFIED_IGBP_MODIS_NOAH':
            raise ValueError('Liu crop currently requires MODIFIED_IGBP_MODIS_NOAH')
        def read(name,layered=False):
            key=next((k for k in (name,name.lower(),name.upper()) if k in ds.variables),None)
            if key is None:return np.zeros((5,*shape) if layered else shape,np.float32)
            var=ds[key];dims=list(var.dimensions);raw=np.ma.asarray(var[:])
            if dims and dims[0]=='Time':raw=raw[0];dims=dims[1:]
            if layered:
                axis=next((i for i,d in enumerate(dims) if 'crop' in d.lower()),None)
                if axis is None:raise ValueError('CROPTYPE requires its original five-category crop dimension')
                raw=np.moveaxis(raw,axis,0)
            expected=(5,*shape) if layered else shape
            if raw.shape!=expected or np.ma.getmaskarray(raw).any() or not np.isfinite(raw).all():
                raise ValueError(f'Liu crop {name} must be finite and have shape {expected}')
            value=np.asarray(raw,np.float32)
            if not np.isfinite(value).all():raise ValueError(f'Liu crop {name} overflows REAL4')
            return value
        fractions=read('CROPTYPE',True)
        if np.any((fractions<0.)|(fractions>1.)):raise ValueError('Liu crop fractions must be within [0,1]')
        cat,pools=crop_initialization(inputs,static,table,fractions)
        planting,harvest,season=(read(k) for k in ('PLANTING','HARVEST','SEASON_GDD'))
        active=cat>0
        if np.any(active&((planting<1.)|(planting>366.)|(harvest<1.)|(harvest>366.)|(season<=0.))):
            raise ValueError('Active Liu crop requires original PLANTING/HARVEST dates and positive SEASON_GDD maps')
    # Driver:768-772 overrides VEGTYP/FVGMAX when initialized CROPCAT > 0.
    veg=np.where(active,int(static.parameters.iscrop),np.asarray(static.ivgtyp,np.int32))
    static=static.replace(ivgtyp=jnp.asarray(veg,jnp.int32),shdmax=jnp.asarray(np.where(active,np.float32(.95),static.shdmax),jnp.float32))
    ep,rp,nroot=build_noahmp_params(static)
    index=np.maximum(cat-1,0)
    scalar_keys=('PLTDAY','HSDAY','GDDTBASE','GDDTCUT','GDDS1','GDDS2','GDDS3','GDDS4','GDDS5','Q10MR','FOLN_MX','LEFREEZ','FRA_GR','LFMR25','STMR25','RTMR25','GRAINMR25','BIO2LAI','MRP')
    array_keys=('DILE_FC','DILE_FW','LF_OVRC','ST_OVRC','RT_OVRC','LFPT','STPT','RTPT','GRAINPT','LFCT','STCT','RTCT')
    scalars={key:np.asarray(vegetation['MRP'],np.float32)[veg-1] if key=='MRP' else np.asarray(table[key],np.float32)[index] for key in scalar_keys}
    # INTEGER parameters assigned the REAL map: truncate before comparing Julian.
    scalars['PLTDAY']=np.where(active,planting,scalars['PLTDAY']).astype(np.int32).astype(np.float32)
    scalars['HSDAY']=np.where(active,harvest,scalars['HSDAY']).astype(np.int32).astype(np.float32)
    for key in ('GDDS1','GDDS2','GDDS3','GDDS4','GDDS5'):
        scalars[key]=np.where(active,season/np.float32(1770.)*scalars[key],scalars[key])
    arrays={key:np.stack([np.asarray(table[key+'_S'+str(s)],np.float32)[index] for s in range(1,9)]) for key in array_keys}
    cp=CropParameters(*(jnp.asarray(scalars[k],jnp.float32) for k in scalar_keys),*(jnp.asarray(arrays[k],jnp.float32) for k in array_keys))
    natural=CarbonParameters(*(jnp.asarray(np.asarray(vegetation[key.upper()],np.float32)[veg-1],jnp.float32) for key in CarbonParameters._fields))
    initial=CropCarbonState(*(jnp.asarray(v,jnp.float32) for v in pools),jnp.full(shape,1.e-10,jnp.float32),jnp.zeros(shape,jnp.float32),
        # PGSXY is untouched by NOAHMP_INIT; original WRF allocated integer zero.
        jnp.zeros(shape,jnp.int32))
    overrides={key:jnp.asarray(np.where(active,np.asarray(table[key.upper()],np.float32)[index],np.asarray(getattr(ep,key))),jnp.float32)
        for key in ('c3psn','kc25','akc','ko25','ako','avcmx','vcmx25','bp','mp','folnmx','qe25')}
    ep=ep._replace(**overrides)
    land=namelist.noahmp_land.replace(lai=jnp.asarray(inputs['LAI_INIT'],jnp.float32),sai=jnp.asarray(inputs['XSAI_INIT'],jnp.float32))
    validate_restart(initial,namelist,1,shape)
    return replace(namelist,noahmp_static=static,noahmp_energy_params=ep,noahmp_rad_params=rp,noahmp_nroot=nroot,
        noahmp_land=land,noahmp_carbon_static=CarbonStatic(CropStaticParameters(natural,cp,jnp.asarray(cat,jnp.int32)),initial))
