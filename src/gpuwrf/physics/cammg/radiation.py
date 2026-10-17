"""RRTMG_SW:11156 / RRTMG_LW:12607 CAM-MG caller override."""
import jax.numpy as jnp
from gpuwrf.physics.rrtmg_mp_re import RadiationRadii


def optical_view(state,qc,qi,qs):
    from gpuwrf.coupling.physics_couplers import _to_columns
    r=jnp.float32
    liquid=jnp.maximum(r(2.5),jnp.minimum(r(60),jnp.where(qc>r(1.e-20),_to_columns(state.lradius),r(10))))
    ice=jnp.maximum(r(5),jnp.minimum(r(140),jnp.where((qi>r(1.e-20))|(qs>r(1.e-20)),_to_columns(state.iradius),r(25))))
    # WRF has_reqc/i/s are zero for mp11; CAM-MG overrides radii after CAM
    # combines qi+qs into the ice water path. These radius operands are microns.
    return qi+qs,jnp.zeros_like(qs),dict(re_cloud=liquid,re_ice=ice,
        re_snow=jnp.full_like(ice,r(10)),xland=state.xland,cammg_radii=True)


def radius_view(liquid,ice):
    return RadiationRadii(liquid,ice,jnp.full_like(ice,10),jnp.zeros_like(ice))
