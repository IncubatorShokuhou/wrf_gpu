"""WRF PBL-driver Fitch seam (after MYNN; before physical RK coupling)."""
from gpuwrf.physics.wind_fitch import dragforce
import jax.numpy as jnp


def windfarm_sources(entry, after, du, dv, namelist):
    from gpuwrf.coupling.physics_couplers import _u_mass, _v_mass
    z = jnp.asarray(entry.ph,jnp.float32)/jnp.float32(9.81)
    dz = z[1:]-z[:-1]
    du,dv,qke,power = dragforce(_u_mass(entry,dtype=jnp.float32),_v_mass(entry,dtype=jnp.float32),
                              z,dz,after.qke,du,dv,dx=namelist.grid.projection.dx_m,dt=namelist.dt_s,farm=namelist.windfarm)
    return after.replace(qke=qke),du,dv,power
