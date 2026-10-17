"""WRF Milbrandt driver mediation, six number scalars, hail and all-phase precipitation."""
import jax
import jax.numpy as jnp
from .kernel import run_fields

_SPECIES=('qv','qc','qr','qi','qs','qg','qh','Nc','Nr','Ni','Ns','Ng','Nh')


def _round_real(value):
    return jax.lax.optimization_barrier(jnp.nextafter(value,value))


def milbrandt_adapter(state,dt,grid=None,*,return_precipitation=False,return_radar=False):
    from gpuwrf.coupling.physics_couplers import (
        P0_PA,R_D_OVER_CP,_dry_theta_view,_to_columns,
        _theta_m_from_microphysics,_output_dtype,_wrf_phy_prep_p8w,
        _interface_pressure_from_state)
    state=state.ensure_conditional_leaves(mp_physics=9)
    R=jnp.float32
    # Keep WRF REAL prep's multiply, add, divide and temperature stores
    # distinct: whole-adapter CPU fusion otherwise contracts/reassociates
    # them and changes cold number sources beyond the same-input oracle band.
    ratio=R(461.6)/R(287.)
    old_theta=jnp.asarray(state.theta,R)
    old_qv=jnp.asarray(state.qv,R)
    scaled=_round_real(ratio*old_qv)
    denominator=_round_real(R(1.)+scaled)
    theta=_round_real(old_theta/denominator)
    p=jnp.asarray(state.p,R)
    pii=_round_real((p/R(P0_PA))**R(R_D_OVER_CP))
    temperature=_round_real(theta*pii)
    if grid is not None and getattr(grid,'metrics',None) is not None:
        m=grid.metrics
        p8w=_wrf_phy_prep_p8w(p,jnp.asarray(state.ph,R)/R(9.81),m.fnm,m.fnp)
    else:
        # Grid-less unit callers only. Operational callers always supply eta weights.
        p8w=jnp.moveaxis(_interface_pressure_from_state(state).astype(R),-1,0)
    fields=(temperature,*(jnp.asarray(getattr(state,n),R) for n in _SPECIES))
    out,rates,zet=run_fields(fields,p,p8w[0],jnp.asarray(state.w,R),R(dt))
    qv=out[1].astype(_output_dtype(state,'qv'))
    dry=out[0] / pii
    changes={n:out[i+1].astype(_output_dtype(state,n)) for i,n in enumerate(_SPECIES)}
    changes['theta']=_theta_m_from_microphysics(theta,state.qv,dry,qv,_output_dtype(state,'theta'))
    # The nine type rates are disjoint; hail and ice pellets share HAILNCV.
    factor=R(1.e3)*R(dt)
    rainncv=rates[0]
    for i in range(1,9): rainncv=rainncv+rates[i]
    rainncv=rainncv*factor
    snow=(rates[4]+rates[5])*factor
    graupel=rates[6]*factor
    hail=(rates[7]+rates[8])*factor
    liquid=(rates[0]+rates[1]+rates[2]+rates[3])*factor
    shape=state.t_skin.shape
    amounts={'rain':(liquid+hail).reshape(shape),'snow':snow.reshape(shape),
             'graupel':graupel.reshape(shape),'ice':jnp.zeros_like(state.t_skin),
             'rainncv':rainncv.reshape(shape),'hailncv':hail.reshape(shape),
             'sr':((snow+hail+graupel)/(rainncv+R(1.e-12))).reshape(shape)}
    for acc,x in (('rain_acc',liquid),('snow_acc',snow),('graupel_acc',graupel),('hail_acc',hail)):
        old=getattr(state,acc)
        changes[acc]=(old+x.reshape(shape)).astype(_output_dtype(state,acc))
    updated=state.replace(**changes)
    if return_radar: return updated,amounts,zet.astype(R)
    return (updated,amounts) if return_precipitation else updated
