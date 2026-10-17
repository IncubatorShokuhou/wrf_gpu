"""WRF caller wiring: three ice habits, dry-theta handoff and held radii."""
import jax.numpy as jnp
from .registry import input_variables,output_mapping
from gpuwrf.contracts.ice_diagnostics import ice_diagnostic_mapping


def jensen_adapter(state,dt,grid=None,*,timestep=1,return_precipitation=False,return_radar=False):
    """``State[, precipitation][, REFL_10CM]``. WRF fills DIAG_DBZ3D=refl_10cm on EVERY
    mp_jensen_ishmael call (module_mp_jensen_ishmael.F:411, max(dbz1d,-35)); its Registry
    package always allocates refl_10cm, independently of do_radar_ref."""
    required=input_variables()
    missing=[n for n in required if getattr(state,n,None) is None]
    if missing:raise ValueError(f'Jensen requires initialized habit moments: {missing}')
    from gpuwrf.coupling.physics_couplers import (
        _to_columns,_from_columns,_dry_theta_view,_column_dz_from_state,_theta_m_from_microphysics)
    f=lambda a:jnp.asarray(a,jnp.float32)
    real=state.replace(theta=f(state.theta),qv=f(state.qv),_cast=False)
    old_theta,old_qv=_dry_theta_view(real),f(state.qv)
    mapping=output_mapping()
    fields={n:_to_columns(f(getattr(state,target))) for n,target in mapping.items() if n not in ('effc','effi')}
    fields.update(th=_to_columns(old_theta),p=_to_columns(f(state.p)),
                  dz=_column_dz_from_state(real,grid,dtype=jnp.float32))
    # Diagnostics are outputs only; never feed held history back into ME.
    mapping.update(ice_diagnostic_mapping(55))
    from gpuwrf.backends import pallas as backend
    if backend.cuda_native():
        from gpuwrf.kernels.phys_jensen import full_column
        output,precip=full_column(fields,f(dt),timestep,output_names=(*mapping,'th',*(('dbz',) if return_radar else ())))
    else:
        from .column import step
        output,precip=step(fields,f(dt),timestep)
    updates={target:f(_from_columns(output[n])) for n,target in mapping.items()}
    updates['theta']=_theta_m_from_microphysics(old_theta,old_qv,_from_columns(output['th']),updates['qv'],jnp.float32)
    # The shared State separates liquid and solid, unlike WRF RAINNC total.
    liquid=precip['rainstep']-precip['snowstep'];solid=precip['snowstep']
    updates.update(rain_acc=state.rain_acc+liquid.reshape(state.rain_acc.shape),
                   snow_acc=state.snow_acc+solid.reshape(state.snow_acc.shape))
    updated=state.replace(**updates)
    # JENSEN_ISHMAEL passes no SR (microphysics_driver.F:1718): Registry 0 reaches the LSM.
    water=dict(rain=liquid,snow=solid,graupel=jnp.zeros_like(solid),ice=jnp.zeros_like(solid),sr=jnp.zeros_like(solid))
    side=((water,) if return_precipitation else ())+((f(_from_columns(output['dbz'])),) if return_radar else ())
    return (updated,*side) if side else updated
