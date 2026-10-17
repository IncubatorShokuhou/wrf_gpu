"""Resident WRF auxinput9 surface analysis pairs and alarm cursor.

Only fields actually read by SFDDAGD are transferred. T2/RH/PSL/PS/TOB are
unused by that operator; TH2 is absolute dry theta, Q2 mixing ratio, ODIS km.
The stored U10/V10 fields have X/Y staggering in WRF's Registry; SFDDAGD
indexes them directly at the interior faces (no spatial interpolation).
"""
from datetime import datetime
from pathlib import Path
import numpy as np
from gpuwrf.io.fdda_namelist import _value, _index, _raw


def surface_code(namelist, domain):
    return int(_value(namelist,'grid_sfdda',_index(domain)))


def surface_interval(namelist, index):
    return int(_value(namelist,'sgfdda_interval_m',index) or _value(namelist,'sgfdda_interval',index))


def effective_surface_radius(namelist, index):
    """WRF check_a_mundo:1679–1692 normalizes every child if any domain uses mode 1.

    All domains in the port's selected domain list are grid_allowed. An omitted
    max_dom in component callers is inferred from the written surface menu.
    """
    raw=(namelist.get('fdda') or {}).get('grid_sfdda',0)
    written=len(raw) if isinstance(raw,(list,tuple)) else 1
    declared=(namelist.get('domains') or {}).get('max_dom',written)
    if isinstance(declared,(list,tuple)):
        if len(declared)!=1:raise ValueError('domains.max_dom must contain one scalar value')
        declared=declared[0]
    max_dom=int(declared)
    radius=float(_value(namelist,'rinblw',index))
    if index>0 and radius==-1 and any(int(_value(namelist,'grid_sfdda',j))==1 for j in range(max_dom)):
        return float(_value(namelist,'rinblw',0))
    return radius


def surface_errors(namelist,domain):
    i=_index(domain);v=lambda k:_value(namelist,k,i)
    code=surface_code(namelist,domain)
    if code==0:return []
    errors=[]
    if code not in (1,2):errors.append(f'&fdda grid_sfdda={code} ({domain}): only 0, 1 or 2 are supported')
    if v('grid_fdda')!=1:errors.append(f'&fdda grid_sfdda ({domain}): WRF requires grid_fdda=1')
    if surface_interval(namelist,i)<=0:errors.append(f'&fdda sgfdda_interval_m ({domain}): must be > 0')
    if int(v('io_form_sgfdda'))!=2:errors.append(f'&fdda io_form_sgfdda ({domain}): only NetCDF (2) is supported')
    for part in ('begin_y','begin_d','begin_h','begin_m','begin_s','end_y','end_d','end_m','end_s',
                 'interval_y','interval_d','interval_h','interval_s'):
        key='sgfdda_'+part
        if int(v(key))!=0:errors.append(f'&fdda {key} ({domain}): WRF fddagd_driver is fatal; use sgfdda_interval_m/sgfdda_end_h')
    for k in ('guv_sfc','gt_sfc','gq_sfc'):
        if not np.isfinite(float(v(k))) or float(v(k))<0:errors.append(f'&fdda {k} ({domain}): must be positive')
    radius=effective_surface_radius(namelist,i)
    if not np.isfinite(radius) or radius<.001:errors.append(f'&fdda rinblw ({domain}): must be >= .001 km')
    from gpuwrf.io.fdda_namelist import _physics_int
    pbl=_physics_int(namelist,'bl_pbl_physics',i,5)
    sfclay=_physics_int(namelist,'sf_sfclay_physics',i,5)
    lsm=_physics_int(namelist,'sf_surface_physics',i,4)
    if pbl!=5:errors.append(f'&fdda grid_sfdda ({domain}): current-step PBLH qualification requires MYNN (5)')
    if sfclay!=5:errors.append(f'&fdda grid_sfdda ({domain}): current SFCLAY REGIME qualification requires sf_sfclay_physics=5')
    if lsm!=4:
        reason='classic Noah indirect FASDAS soil updates are not ported' if code==2 and lsm==2 else 'surface FDDA is qualified with Noah-MP only'
        errors.append(f'&fdda grid_sfdda={code} ({domain}): {reason}; use sf_surface_physics=4')
    return ['surface FDDA: '+e for e in errors]


def attach_surface_fdda(base,run_dir,namelist,domain,*,run_start,dt_s,ny,nx,dx_m,run_seconds):
    if surface_code(namelist,domain)==0:return base
    from gpuwrf.io.netcdf_lock import Dataset
    from netCDF4 import chartostring
    from gpuwrf.assimilation.fdda_surface import SurfaceAnalysis,SurfaceGridFddaInputs,SurfaceFddaOptions
    import jax.numpy as jnp
    i=_index(domain);v=lambda k:_value(namelist,k,i)
    template=str(_value(namelist,'sgfdda_inname',0)).strip().strip("'\"")
    filename=template.replace('<domain>',f'{i+1:02d}')
    if '<' in filename:raise ValueError(f'{domain}: unresolved sgfdda_inname {filename!r}')
    path=Path(run_dir)/filename
    if not path.is_file():raise ValueError(f'{domain}: grid_sfdda requires {path} (auxinput9)')
    interval=surface_interval(namelist,i)*60
    end=int(v('sgfdda_end_h'))*3600
    def read(t):return (end==0 or t<=end) and (run_seconds is None or np.ceil(t/dt_s)*dt_s<run_seconds)
    with Dataset(path) as ds:
        times=[datetime.strptime(str(t),'%Y-%m-%d_%H:%M:%S') for t in chartostring(ds['Times'][:])]
        offsets=np.array([(t-run_start.replace(tzinfo=None)).total_seconds() for t in times])
        if offsets.size==0 or not np.array_equal(offsets,np.arange(offsets.size)*interval):
            raise ValueError(f'{path}: auxinput9 records must start at run start and follow sgfdda_interval_m')
        if read(float(offsets[-1]+interval)) and run_seconds is not None:
            raise ValueError(f'{path}: insufficient auxinput9 records for the run')
        count=sum(read(float(t)) for t in offsets)
        if count<1:raise ValueError(f'{path}: run reads no auxinput9 records')
        arrays={}
        for n,var in (('u10','U10_NDG'),('v10','V10_NDG'),('th2','TH2_NDG'),('q2','Q2_NDG'),('odis','ODIS_NDG')):
            for suffix in ('OLD','NEW'):
                key=var+'_'+suffix
                if key not in ds.variables:raise ValueError(f'{path}: missing {key}')
                a=np.ma.asarray(ds[key][:count])
                shape=(count,ny,nx+1) if n=='u10' else (count,ny+1,nx) if n=='v10' else (count,ny,nx)
                if np.ma.getmaskarray(a).any() or a.shape!=shape or not np.isfinite(a).all():
                    raise ValueError(f'{path}: {key} must be finite {shape} with no missing values')
                with np.errstate(over='ignore',invalid='ignore'):
                    a=np.asarray(a,np.float32)
                if not np.isfinite(a).all():
                    raise ValueError(f'{path}: {key} must remain finite when converted to REAL32')
                arrays[n+'_'+suffix.lower()]=jnp.asarray(a)
    radius=effective_surface_radius(namelist,i)
    options=SurfaceFddaOptions(**dict(vars(base.options),end_h_sfc=int(v('sgfdda_end_h'))),
        surface_code=surface_code(namelist,domain),interval_m_sfc=surface_interval(namelist,i),
        guv_sfc=float(v('guv_sfc')),gt_sfc=float(v('gt_sfc')),gq_sfc=float(v('gq_sfc')),rinblw=radius,dx_m=dx_m)
    return SurfaceGridFddaInputs(**dict(vars(base),options=options),surface=SurfaceAnalysis(**arrays),
        surface_steps=tuple(int(s) for s in np.ceil(offsets[:count]/dt_s)))
