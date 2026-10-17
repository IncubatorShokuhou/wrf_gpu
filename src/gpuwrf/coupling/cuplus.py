"""Original cu=7/10/11 caller operands and resident uncoupled output fields."""
from __future__ import annotations
from typing import NamedTuple
import jax
import jax.numpy as jnp


class CuPlusCarry(NamedTuple):
    w0avg: object                 # (column, mass level), WRF REAL
    held: dict                   # all scheme-owned REAL/INTEGER driver fields


def initial_carry(state,scheme):
    from gpuwrf.kernels.phys_cuplus_driver import initial_column
    nz,ny,nx=state.theta.shape
    one=initial_column(scheme,nz)
    held={key:jnp.broadcast_to(value,(ny*nx,)+value.shape) for key,value in one.items()}
    return CuPlusCarry(jnp.zeros((ny*nx,nz),jnp.float32),held)


def mynn_kpbl(zw,pblh,*,mass_levels=None):
    """Original GET_PBLH:5568–5574, with its initialized fallback index 2.

    ``zw`` is the PBL's AGL interface profile (nz+1, y, x), not mass heights.
    No extra PBL solve or bulk-Richardson substitute is performed.
    """
    k=jnp.arange(1,zw.shape[0]+1,dtype=jnp.int32)[:,None,None]
    nz=zw.shape[0]-1 if mass_levels is None else int(mass_levels)
    match=(k>=2)&(k<nz)&(zw>=jnp.asarray(pblh,jnp.float32)[None])
    first=jnp.min(jnp.where(match,k,zw.shape[0]+1),axis=0)
    return jnp.where(first<=zw.shape[0],first-1,2).astype(jnp.int32)


def driver(state,surface_state,grid,carry,namelist,step,*,zol=None,pbl_inputs=None,camuw=None,cldfra=None):
    """Read time-n phy_prep profiles and the actual surface/PBL diagnostics.

    Returns resident uncoupled WRF rates. The shared RK caller owns mass
    coupling, A2A/A2C insertion, moist-theta conversion and final decoupling.
    """
    from gpuwrf.coupling.physics_couplers import (
        _wrf_phy_rv_over_rd,_wrf_hydrostatic_pressure_profiles_from_state,
        _wrf_phy_prep_rho_from_state)
    from gpuwrf.kernels.phys_cuplus_driver import column_step
    scheme=int(namelist.cu_physics);pbl=int(namelist.bl_pbl_physics)
    if scheme not in (7,10,11):raise ValueError('cuplus requires cu=7/10/11')
    if grid is None or grid.metrics is None:raise ValueError('cuplus requires WRF hybrid metrics')
    if carry is None:raise ValueError('cuplus restart is missing resident cloud/rate fields (E78)')
    nz,ny,nx=state.theta.shape;ncol=ny*nx;r=jnp.float32
    cols=lambda a:jnp.moveaxis(jnp.asarray(a,r),0,-1).reshape(ncol,-1)
    flat=lambda a:jnp.asarray(a,r).reshape(ncol)
    phyd,faces,psfc=_wrf_hydrostatic_pressure_profiles_from_state(state,grid.metrics,output_dtype=r)
    pi=(jnp.asarray(state.p,r)/r(1e5))**(r(287.)/r(1004.5))
    th=jnp.asarray(state.theta,r)/(r(1.)+_wrf_phy_rv_over_rd(r)*jnp.asarray(state.qv,r));t=th*pi
    u_face=jnp.asarray(state.u,r);v_face=jnp.asarray(state.v,r)
    u=r(.5)*(u_face[:,:,:-1]+u_face[:,:,1:]);v=r(.5)*(v_face[:,:-1,:]+v_face[:,1:,:])
    zw=jnp.asarray(state.ph,r)/r(9.81);z=r(.5)*(zw[:-1]+zw[1:]);dz=zw[1:]-zw[:-1]
    rho=_wrf_phy_prep_rho_from_state(state,grid.metrics,output_dtype=r)
    terrain=jnp.asarray(grid.terrain_height,r)
    pblh=flat(surface_state.pblh);hfx=flat(surface_state.hfx);qfx=flat(surface_state.qfx)
    ust=flat(surface_state.ustar);land=flat(surface_state.xland)
    dt=float(namelist.dt_s);dx=float(grid.projection.dx_m);cudt=float(namelist.cudt_minutes)
    stepcu=int(namelist.cumulus_cadence_steps)
    dtfield=jnp.full((ncol,),dt,r);dxfield=jnp.full((ncol,),dx,r)
    w=cols(state.w)
    if pbl_inputs is None:
        if pbl==5:
            from gpuwrf.coupling.physics_couplers import _mynn_column_from_state
            from gpuwrf.physics.mynn_pbl import _wrf_zw
            # Use the exact geometry supplied to this PBL, as the observation
            # driver does, including the retained adapter's dtype/spacing view.
            column=_mynn_column_from_state(state,grid)
            pbl_zw=jnp.moveaxis(jnp.asarray(_wrf_zw(column.dz),r),-1,0)
            kpbl=mynn_kpbl(pbl_zw,surface_state.pblh,mass_levels=nz)
        elif scheme in (7,10):raise ValueError('cuplus requires actual KPBL from the selected PBL producer')
        else:kpbl=jnp.zeros((ny,nx),jnp.int32)  # MSKF does not consume caller KPBL.
        tpert=jnp.zeros((ny,nx),r)
    else:tpert,kpbl=pbl_inputs
    if scheme==11:
        if zol is None:raise ValueError('MSKF requires actual surface ZOL; MOL is a temperature scale')
        arrays=tuple(cols(a) for a in (t,state.qv,phyd,dz,rho,state.qv,u,v,pi))
        controls=jnp.stack((dtfield,dxfield,pblh,hfx,ust,flat(zol),jnp.full((ncol,),cudt,r),jnp.zeros(ncol,r)),axis=1)
    elif scheme==10:
        arrays=tuple(cols(a) for a in (t,state.qv,phyd,dz,rho,state.qv,u,v,th,z,zw[:-1],pi))
        controls=jnp.stack((dtfield,dxfield,pblh,flat(kpbl),hfx,qfx,land,flat(terrain)),axis=1)
    else:
        if pbl==9:
            if camuw is None:raise ValueError('CAM-ZM requires the actual CAM-UW TKE producer carry')
            tke=camuw.tke_pbl[:nz]
        else:tke=r(.5)*jnp.asarray(surface_state.qke,r)
        cf=(state.cldfra_mp_all if int(namelist.mp_physics)==11 else
            jnp.zeros_like(state.qv,dtype=r) if cldfra is None else cldfra)
        arrays=tuple(cols(a) for a in (t,state.qv,phyd,faces,pi,u,v,z,zw,tke,th,state.qc,state.qi,cf,dz))
        controls=jnp.stack((dtfield,dxfield,jnp.full((ncol,),cudt,r),jnp.full((ncol,),step,r),
            flat(psfc),flat(terrain),pblh,flat(kpbl),hfx,qfx,ust,flat(surface_state.mavail),land,
            flat(surface_state.t_skin),flat(tpert)),axis=1)
    def one(a,w0,w,held,c):
        return column_step(a,w,w0,held,c,step,scheme=scheme,stepcu=stepcu,cudt=cudt,pbl=pbl,
            num_bins=int(getattr(namelist,'cup_num_bins',1)),th_bin_size=float(getattr(namelist,'cup_th_bin_size',1.)),
            r_bin_size=float(getattr(namelist,'cup_r_bin_size',1.)),min_deep_freq=float(getattr(namelist,'cup_min_deep_freq',1.)),
            min_shallow_freq=float(getattr(namelist,'cup_min_shallow_freq',1.)),
            small_domain=nx<=3 and ny<=3,interpret=jax.default_backend()!='gpu')
    w0,held=jax.vmap(one)(arrays,carry.w0avg,w,carry.held,controls)
    return CuPlusCarry(w0,held)


def rates(carry,state,scheme,mp_physics):
    """Raw rate dictionary in leading-z layout; number units follow State's MP."""
    nz,ny,nx=state.theta.shape
    back=lambda a:jnp.moveaxis(a.reshape(ny,nx,nz),-1,0)
    keys={'theta':'RTHCUTEN','qv':'RQVCUTEN','qc':'RQCCUTEN','qi':'RQICUTEN'}
    if scheme in (10,11):keys.update(qr='RQRCUTEN',qs='RQSCUTEN')
    output={name:back(carry.held[key]) for name,key in keys.items()}
    if scheme in (7,11):
        output.update(u=back(carry.held['RUCUTEN' if scheme==7 else 'DUDT']),
                      v=back(carry.held['RVCUTEN' if scheme==7 else 'DVDT']))
    if scheme==7:
        # WRF Thompson Ni and the written QNICE field are kg-1 despite the old
        # generic State docstring. WDM Nc is m-3 and needs the dry-density view
        # at the RK seam; admission restricts number-unit combinations there.
        from gpuwrf.contracts.physics_registry import number_species_for_mp
        active=number_species_for_mp(mp_physics)
        if 'Ni' in active:output['Ni']=back(carry.held['RQINCUTEN'])
        if 'Nc' in active:output['Nc']=back(carry.held['RQCNCUTEN'])
    return output


def finish(carry,dt,scheme):
    from gpuwrf.kernels.phys_cuplus_driver import finish_column
    held,rain=jax.vmap(lambda x:finish_column(x,dt,scheme=scheme))(carry.held)
    return CuPlusCarry(carry.w0avg,held),rain


def coupled_keys(scheme,mp_physics):
    from gpuwrf.contracts.physics_registry import number_species_for_mp
    keys=['RTHCUTEN','RQVCUTEN','RQCCUTEN','RQICUTEN']
    if scheme in (10,11):keys+=['RQRCUTEN','RQSCUTEN']
    if scheme==11:keys+=['DUDT','DVDT']
    if scheme==7:
        keys+=['RUCUTEN','RVCUTEN']
        active=number_species_for_mp(mp_physics)
        if 'Ni' in active:keys+=['RQINCUTEN']
        if 'Nc' in active:keys+=['RQCNCUTEN']
    return tuple(keys)


def mass_columns(mu,metrics):
    r=jnp.float32
    mass=jnp.asarray(metrics.c1h,r)[:,None,None]*jnp.asarray(mu,r)[None]+jnp.asarray(metrics.c2h,r)[:,None,None]
    return jnp.moveaxis(mass,0,-1).reshape(-1,mass.shape[0])


def couple(carry,mu,metrics,scheme,mp_physics):
    """calculate_phy_tend couples each original active REAL source separately."""
    mass=mass_columns(mu,metrics);held=dict(carry.held)
    for key in coupled_keys(scheme,mp_physics):held[key]=mass*held[key]
    return CuPlusCarry(carry.w0avg,held)


def finish_coupled(carry,mu,metrics,dt,scheme,mp_physics):
    """advance_ppt then phy_prep_part2, once every RK stage has consumed CU."""
    carry,rain=finish(carry,dt,scheme)
    mass=mass_columns(mu,metrics);held=dict(carry.held)
    for key in coupled_keys(scheme,mp_physics):held[key]=held[key]/mass
    return CuPlusCarry(carry.w0avg,held),rain


def momentum_sources(u_rate,v_rate,*,specified_or_nested,periodic_x=False,periodic_y=False):
    """Original add_a2c mass-to-face insertion of already coupled CU momentum."""
    nz,ny,nx=u_rate.shape
    ru=jnp.zeros((nz,ny,nx+1),u_rate.dtype)
    rv=jnp.zeros((nz,ny+1,nx),v_rate.dtype)
    if specified_or_nested:
        ru=ru.at[:,1:-1,1:nx].set(.5*(u_rate[:,1:-1,:-1]+u_rate[:,1:-1,1:]))
        x=slice(None) if periodic_x else slice(1,-1)
        rv=rv.at[:,1:ny,x].set(.5*(v_rate[:,:-1,x]+v_rate[:,1:,x]))
    else:
        ru=ru.at[:,:,1:nx].set(.5*(u_rate[:,:,:-1]+u_rate[:,:,1:]))
        rv=rv.at[:,1:ny,:].set(.5*(v_rate[:,:-1,:]+v_rate[:,1:,:]))
    if periodic_x:
        edge=.5*(u_rate[:,:,-1]+u_rate[:,:,0])
        ru=ru.at[:,:,0].set(edge).at[:,:,-1].set(edge)
    if periodic_y:
        edge=.5*(v_rate[:,-1,:]+v_rate[:,0,:])
        rv=rv.at[:,0,:].set(edge).at[:,-1,:].set(edge)
    return ru,rv


def radiation_merge(fraction,qc,qi,flags,held,*,before_pbl,shallowcu_forced_ra=False):
    """Original radiation-driver order: MSKF, MYNN-SGS, then CuP.

    Columns use the radiation assembler's trailing-level layout. The caller
    keeps condensate changes in its temporary radiation view and saves only
    CuP's INOUT cloud fraction, as WRF does at radiation_driver:3260–3280.
    """
    r=jnp.float32
    if before_pbl and 'CLDFRA_DP_KF' in held:
        conv=(held['CLDFRA_DP_KF']+held['CLDFRA_SH_KF']).reshape(fraction.shape)
        fraction=jnp.minimum(r(1.),(r(1.)-conv)*fraction+conv)
        qc=qc+held['QC_KF'].reshape(qc.shape)*conv
        qi=qi+held['QI_KF'].reshape(qi.shape)*conv
    cup=None
    if not before_pbl and 'CLDFRA_CUP' in held:
        from gpuwrf.physics.cumulus_kfcup_support import cup_radiation_feedback
        shape=fraction.shape;nz=shape[-1]
        def one(cf,q,cup,flag,shall,bottom,top):
            return cup_radiation_feedback(cf,q,cup,flag,ishall=shall,
                cloud_bottom=bottom-r(1.),cloud_top=top-r(1.),icloud=1,
                shallowcu_forced_ra=shallowcu_forced_ra)
        out=jax.vmap(one)(fraction.reshape(-1,nz),qc.reshape(-1,nz),held['CLDFRA_CUP'],
            flags.reshape(-1,nz),held['ISHALL'].astype(r),held['CUBOT'],held['CUTOP'])
        fraction=out['cldfra'].reshape(shape);qc=out['qc'].reshape(shape);cup=out['cldfra_cup']
    return fraction,qc,qi,cup
