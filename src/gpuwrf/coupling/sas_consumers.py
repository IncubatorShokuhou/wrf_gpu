"""New SAS-family consumers with original WRF add_a2a/add_a2c bounds.

module_physics_addtendc.F:2435-2481,2531-2636. Static two-dimensional
ownership broadcasts fuse with the updates. Holds and rain retain their ranges.
The retained cu=4 API and the PBL interpolation helpers are not changed.
"""
import jax.numpy as jnp

from gpuwrf.coupling.physics_couplers import (
    _output_dtype, _theta_m_tendency_from_dry,
    _add_a2c_u_increment, _add_a2c_v_increment,
)


def apply_sas_scalar_rates(state, rates, dt, *, entry_state,
                           specified_or_nested=False, periodic_x=False):
    rth, rqv, rqc, _, rqi, _, pratec = rates[:7]
    if specified_or_nested:
        ny, nx = state.qv.shape[-2:]
        y, x = jnp.arange(ny), jnp.arange(nx)
        owned = ((y>0)&(y<ny-1))[:,None]
        if not periodic_x:
            owned = owned & ((x>0)&(x<nx-1))[None,:]
    def add(name, increment):
        value = getattr(state, name)
        if specified_or_nested:
            result = jnp.where(owned, value+increment, value)
        else:
            result = value + increment
        return result.astype(_output_dtype(state, name))
    thermal = _theta_m_tendency_from_dry(rth, rqv, entry_state.theta,
                                       entry_state.qv, state.theta.dtype)
    return state.replace(theta=add('theta', float(dt)*thermal),
        qv=add('qv', float(dt)*rqv), qc=add('qc', float(dt)*rqc),
        qi=add('qi', float(dt)*rqi),
        rainc_acc=(state.rainc_acc + float(dt)*jnp.asarray(pratec, state.rainc_acc.dtype))
                  .astype(_output_dtype(state, 'rainc_acc')))


def add_sas_momentum(state, du, dv, *, specified_or_nested=False,
                     periodic_x=False, periodic_y=False):
    u, v = state.u, state.v
    u = _add_a2c_u_increment(u, du)
    v = _add_a2c_v_increment(v, dv)
    if periodic_x:
        edge = (.5*(du[:, :, -1]+du[:, :, 0])).astype(u.dtype)
        u = u.at[:, :, 0].add(edge).at[:, :, -1].add(edge)
    if specified_or_nested:
        ny, nx = du.shape[-2:]
        y, x = jnp.arange(ny), jnp.arange(nx)
        u = jnp.where(((y>0)&(y<ny-1))[None,:,None], u, state.u)
        if not periodic_x:
            v = jnp.where(((x>0)&(x<nx-1))[None,None,:], v, state.v)
    # periodic_x never restores the specified/nested j bounds in add_a2c_v.
    if periodic_y and not specified_or_nested:
        edge = (.5*(dv[:, -1, :]+dv[:, 0, :])).astype(v.dtype)
        v = v.at[:, 0, :].add(edge).at[:, -1, :].add(edge)
    return state.replace(u=u, v=v)
