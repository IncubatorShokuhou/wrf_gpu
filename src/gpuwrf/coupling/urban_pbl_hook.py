"""Compact BEP/BEM columns replacing their rural PBL tendencies."""
import jax.numpy as jnp
import jax


def urban_pbl_forcing(state,grid,surface_layer):
    """first_rk_step_part1: p_hyd/p_hyd_w, nonhydrostatic PI_PHY/T_PHY."""
    from gpuwrf.coupling.physics_couplers import (_u_mass,_v_mass,_wrf_phy_rv_over_rd,
        _wrf_phy_prep_rho_from_state,_wrf_hydrostatic_pressure_profiles_from_state)
    F=jnp.float32;nz,ny,nx=state.theta.shape;ncol=ny*nx
    cols=lambda a:jnp.moveaxis(jnp.asarray(a,F),0,-1).reshape(ncol,-1)
    flat=lambda a:jnp.asarray(a,F).reshape(ncol)
    theta=jnp.asarray(state.theta,F)/(1+_wrf_phy_rv_over_rd(F)*jnp.asarray(state.qv,F))
    p,faces,psfc=_wrf_hydrostatic_pressure_profiles_from_state(state,grid.metrics,output_dtype=F)
    pi=(jnp.asarray(state.p,F)/F(1.e5))**F(287./1004.5)
    zw=jnp.asarray(state.ph,F)/F(9.81)
    return dict(ncol=ncol,nz=nz,ny=ny,nx=nx,u_cols=cols(_u_mass(state)),v_cols=cols(_v_mass(state)),
        theta_cols=cols(theta),T_cols=cols(theta*pi),qv_cols=cols(state.qv),qc_cols=cols(state.qc),qi_cols=cols(state.qi),
        tke_cols=cols(state.qke),p_cols=cols(p),p_int_cols=cols(faces),pii_cols=cols(pi),
        rho_cols=cols(_wrf_phy_prep_rho_from_state(state,grid.metrics)),dz_cols=cols(zw[1:]-zw[:-1]),
        psfc=flat(psfc),ust=flat(state.ustar),hfx=flat(state.hfx),qfx=flat(state.qfx),
        znt=flat(state.roughness_m),xland=flat(state.xland),wspd=flat(surface_layer.wspd),
        br=flat(surface_layer.br),psim=flat(surface_layer.psim),psih=flat(surface_layer.psih),
        u10=flat(surface_layer.u10),v10=flat(surface_layer.v10))


def _ysu_cloud_kwargs(qc,qi,rthraten,ysu_kwargs,index=None):
    """module_pbl_driver.F:1208-1251 QC3D/QI3D, RTHRATEN, YSU_TOPDOWN_PBLMIX, CTOPO/CTOPO2."""
    from gpuwrf.coupling.scan_adapters import _topo_wind_column_kwargs
    ysu_kwargs=ysu_kwargs or {}
    return dict(qc=qc,qi=qi,rthraten=rthraten,ysu_topdown_pblmix=int(ysu_kwargs.get('ysu_topdown_pblmix',1)),
                **_topo_wind_column_kwargs(ysu_kwargs.get('topo_wind_ctopo'),index))


def urban_pbl_full_columns(f,dt,scheme,ysu_kwargs=None):
    """Ordinary rural columns in WRF REAL; the compact urban replacement follows."""
    from gpuwrf.physics.pbl_ysu import ysu_columns
    from gpuwrf.physics.pbl_boulac import boulac_columns
    with jax.enable_x64(False):
        if scheme==1:
            return ysu_columns(f['u_cols'],f['v_cols'],f['T_cols'],f['qv_cols'],f['p_cols'],f['p_int_cols'],f['pii_cols'],f['dz_cols'],
                psfc=f['psfc'],znt=f['znt'],ust=f['ust'],hfx=f['hfx'],qfx=f['qfx'],wspd=f['wspd'],br=f['br'],
                psim=f['psim'],psih=f['psih'],dt=dt,xland=f['xland'],u10=f['u10'],v10=f['v10'],
                **_ysu_cloud_kwargs(f['qc_cols'],f['qi_cols'],
                    None if (ysu_kwargs or {}).get('rthraten') is None else
                    jnp.moveaxis(jnp.asarray(ysu_kwargs['rthraten'],jnp.float32),0,-1).reshape(f['ncol'],f['nz']),
                    ysu_kwargs))
        return boulac_columns(f['u_cols'],f['v_cols'],f['theta_cols'],f['qv_cols'],f['qc_cols'],f['rho_cols'],f['dz_cols'],
            jnp.maximum(f['tke_cols'],1.e-4),hfx=f['hfx'],qfx=f['qfx'],ust=f['ust'],dt=dt)


def replace_urban_tendencies(state,forcing,out,*,dt,grid,bundle,carry,scheme,ysu_kwargs=None):
    from gpuwrf.physics.pbl_ysu import ysu_columns
    from gpuwrf.physics.pbl_boulac import boulac_columns
    idx=jnp.asarray(bundle.indices,jnp.int32);nz=forcing['nz'];dtype=jnp.float32
    cols=lambda value:jnp.moveaxis(jnp.asarray(value,dtype),0,-1).reshape(-1,nz)[idx]
    take=lambda key:jnp.asarray(forcing[key],dtype)[idx]
    pp=take('p_cols');th=take('theta_cols')
    if scheme==1:
        t=take('T_cols');exner=take('pii_cols')
        urb=ysu_columns(take('u_cols'),take('v_cols'),t,cols(state.qv),pp,take('p_int_cols'),exner,take('dz_cols'),
            psfc=take('psfc'),znt=take('znt'),ust=take('ust'),hfx=jnp.asarray(state.hfx,dtype).reshape(-1)[idx],
            qfx=jnp.asarray(state.qfx,dtype).reshape(-1)[idx],wspd=take('wspd'),br=take('br'),psim=take('psim'),psih=take('psih'),
            dt=dt,xland=take('xland'),u10=take('u10'),v10=take('v10'),urban=carry.sources,sf_urban_physics=bundle.option,
            **_ysu_cloud_kwargs(cols(state.qc),cols(state.qi),
                None if (ysu_kwargs or {}).get('rthraten') is None else cols(ysu_kwargs['rthraten']),ysu_kwargs,idx))
    else:
        urb=boulac_columns(take('u_cols'),take('v_cols'),th,cols(state.qv),cols(state.qc),
            take('rho_cols'),take('dz_cols'),jnp.maximum(cols(state.qke),1.e-4),
            hfx=jnp.asarray(state.hfx,dtype).reshape(-1)[idx],qfx=jnp.asarray(state.qfx,dtype).reshape(-1)[idx],
            ust=take('ust'),dt=dt,urban=carry.sources,sf_urban_physics=bundle.option)
    result=dict(out)
    for key,value in urb.items():
        if key in out:result[key]=out[key].at[idx].set(jnp.asarray(value,out[key].dtype))
    return result
