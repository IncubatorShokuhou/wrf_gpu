"""Original REAL phy_prep -> CAMMGMP -> moist_physics_finish seam."""
import jax
import jax.numpy as jnp

from .registry import MASS_FIELDS, STATE_FIELDS


def operands(state, grid, *, cu=0, convection=None, land_state=None):
    from gpuwrf.coupling.physics_couplers import (
        _to_columns, _dry_theta_view, _wrf_hydrostatic_pressure_profiles_from_state,
        _wrf_phy_prep_rho_from_state,
    )
    if grid is None or grid.metrics is None:
        raise ValueError('CAM-MG requires original WRF hybrid metrics')
    missing = [name for name in STATE_FIELDS if getattr(state, name, None) is None]
    if missing:
        raise ValueError(f'CAM-MG is missing held driver state (E78): {missing}')
    r = jnp.float32
    real = state.replace(theta=jnp.asarray(state.theta,r), qv=jnp.asarray(state.qv,r), _cast=False)
    th = jnp.asarray(_dry_theta_view(real),r)
    pi = (jnp.asarray(state.p,r)/r(1.e5))**(r(287.)/r(1004.5))
    p, pf, _ = _wrf_hydrostatic_pressure_profiles_from_state(real,grid.metrics,output_dtype=r)
    alt = _wrf_phy_prep_rho_from_state(real,grid.metrics,output_dtype=r,return_alt=True)
    zw = jnp.asarray(state.ph,r)/r(9.81)
    z = r(.5)*(zw[:-1]+zw[1:])
    nz,ny,nx=state.theta.shape;ncol=ny*nx
    # CAMMGMP's memory profile has a held top element above the physical mass
    # levels. It is not an extra model level and is never transported.
    def mass(a):
        c=_to_columns(jnp.asarray(a,r)).reshape(ncol,nz)
        return jnp.concatenate((c,jnp.zeros((ncol,1),r)),axis=1)
    face=lambda a:_to_columns(jnp.asarray(a,r)).reshape(ncol,nz+1)
    flat=lambda a:jnp.asarray(a,r).reshape(ncol)
    zero=jnp.zeros_like(th);zf=jnp.zeros_like(zw);zs=jnp.zeros_like(state.rain_acc,dtype=r)
    fields={name:mass(getattr(state,source)) for name,source in {
        'qv_curr':'qv','qc_curr':'qc','qi_curr':'qi','qr_curr':'qr','qs_curr':'qs',
        'nc3d':'Nc','ni3d':'Ni','nr3d':'Nr','ns3d':'Ns',**{n:n for n in MASS_FIELDS}}.items()}
    fields.update(th=mass(th),t_phy=mass(th*pi),pi_phy=mass(pi),p_hyd=mass(p),
                  p8w=face(pf),z_at_w=face(zw),z_sea_level=mass(z),alt=mass(alt),
                  qndrop=mass(zero),f_ice_phy=mass(zero),f_rain_phy=mass(zero),
                  qfx=flat(state.qfx),ht=flat(grid.terrain_height),
                  xland=flat(state.xland),snowh=flat(state.cam_snowh if land_state is None else land_state.snowh),rliq2d=flat(zs),
                  rainnc=flat(state.cam_rainnc),rainncv=flat(zs),snownc=flat(state.cam_snownc),
                  snowncv=flat(zs),sr=flat(state.cam_sr))
    for name,source in (('kvh3d','cam_kvh3d'),('tke_pbl','cam_tke_pbl'),
                        ('turbtype3d','cam_turbtype3d'),('smaw3d','cam_smaw3d')):
        fields[name]=face(getattr(state,source))
    for name in ('icwmrdp3d','icwmrsh3d','shfrc3d','dlf3d','dlf2_3d'):
        fields[name]=mass(zero)
    fields.update(cmfmc3d=face(zf),cmfmc2_3d=face(zf))
    if int(cu)==7:
        if convection is None:
            raise ValueError('CAM-MG cu=7 requires actual held CAM-ZM outputs')
        held=convection.held
        for name,key in (('icwmrdp3d','ICWMRDP'),('dlf3d','DLF_OUT')):
            fields[name]=jnp.concatenate((jnp.asarray(held[key],r),jnp.zeros((ncol,1),r)),axis=1)
        fields['cmfmc3d']=jnp.asarray(held['CMFMC'],r)
        fields['rliq2d']=jnp.asarray(held['RLIQ_OUT'],r)
    return fields,th


def cammg_adapter(state,dt,grid=None,*,timestep=1,cu=0,shcu=0,
                  accum_mode=1.e9,aitken_mode=3.e8,coarse_mode=2.e5,
                  convection=None,land_state=None,return_precipitation=False):
    from gpuwrf.coupling.physics_couplers import _from_columns,_theta_m_from_microphysics
    from gpuwrf.kernels.phys_cammg import full_column
    from gpuwrf.backends import pallas as backend
    fields,old_th=operands(state,grid,cu=cu,convection=convection,land_state=land_state)
    result,fatal=full_column(fields,dt,timestep,cu=cu,shcu=shcu,accum_mode=accum_mode,
        aitken_mode=aitken_mode,coarse_mode=coarse_mode,interpret=not backend.cuda_native())
    nz,ny,nx=state.theta.shape
    back=lambda a:_from_columns(a[:,:-1].reshape(ny,nx,nz)).astype(jnp.float32)
    updates={target:back(result[source]) for source,target in {
        'qv_curr':'qv','qc_curr':'qc','qi_curr':'qi','qr_curr':'qr','qs_curr':'qs',
        'nc3d':'Nc','ni3d':'Ni','nr3d':'Nr','ns3d':'Ns',**{n:n for n in MASS_FIELDS}}.items()}
    updates['theta']=_theta_m_from_microphysics(old_th,jnp.asarray(state.qv,jnp.float32),
        back(result['th']),updates['qv'],jnp.float32)
    shape=state.rain_acc.shape
    total=result['rainncv'].reshape(shape);solid=result['snowncv'].reshape(shape);liquid=total-solid
    updates.update(rain_acc=state.rain_acc+liquid,snow_acc=state.snow_acc+solid,
        cam_rainnc=result['rainnc'].reshape(shape),cam_snownc=result['snownc'].reshape(shape),
        cam_sr=result['sr'].reshape(shape),cam_fatal=state.cam_fatal+fatal.reshape(shape))
    # Original endrun is fatal. The cumulative resident status cannot be erased
    # by a later successful call or a dynamics guard. The existing output/chunk
    # boundary checks it even when optional finite checking is disabled.
    updated=state.replace(**updates)
    precip=dict(rain=liquid,snow=solid,graupel=jnp.zeros_like(solid),ice=jnp.zeros_like(solid),
                sr=result['sr'].reshape(shape))
    return (updated,precip) if return_precipitation else updated
