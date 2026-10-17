"""Station-only WRF diagnostics and a bounded, device-resident step buffer.

Nothing here calls device_get or converts a traced value to a host scalar.
All large fields are gathered BEFORE arithmetic/casts: temporary storage is
O(stations * levels), never O(domain cells * levels).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class TimeSeriesBuffer:
    hours: object
    surface: object
    profiles: object
    count: object
    # Station-only constants are traced inputs (also included in donation).
    y: object
    x: object
    land: object
    unstagger: bool = False
    sst_update: bool = False

    def tree_flatten(self):
        return ((self.hours, self.surface, self.profiles, self.count, self.y, self.x, self.land),
                (self.unstagger, self.sst_update))

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children, *aux)


def seed_buffer(locations, levels, capacity, *, land, unstagger=False, sst_update=False):
    if capacity < 1 or levels < 1:
        raise ValueError("positive time series capacity and level count required")
    shape = (int(capacity), len(locations))
    return TimeSeriesBuffer(
        jnp.zeros((capacity,), jnp.float32), jnp.zeros((*shape, 14), jnp.float32),
        jnp.zeros((*shape, 7, levels), jnp.float32), jnp.asarray(0, jnp.int32),
        jnp.asarray([loc.j-1 for loc in locations], jnp.int32),
        jnp.asarray([loc.i-1 for loc in locations], jnp.int32), jnp.asarray(land, bool),
        bool(unstagger), bool(sst_update),
    )


def sample_fields(state, history, radiation, soil_top, metrics, x, y, levels, *, unstagger=False):
    """Gather ARW calc_ts quantities; State.theta carries full moist theta_m.

    WRF CLW includes QC/QR/QI/QS/QG on ALL mass levels, independent of the
    selected profile height. Pressure-face interpolation is calc_p8w, not a
    hydrostatic layer-mass approximation. Its species/level sum follows WRF's
    literal REAL order (unrolled only over these small station columns).
    """
    f = jnp.float32

    def column(value):
        return jnp.asarray(value[:, y, x], f)

    def surface(value):
        return jnp.asarray(value[y, x], f)

    cosa, sina = surface(metrics.cosa), surface(metrics.sina)
    u, v = column(state.u)[:levels], column(state.v)[:levels]
    uu = u*cosa-v*sina
    vv = v*cosa+u*sina
    if unstagger:
        up, vp = jnp.asarray(state.u[:levels, y, x+1], f), jnp.asarray(state.v[:levels, y, x+1], f)
        cp, sp = jnp.asarray(metrics.cosa[y, x+1], f), jnp.asarray(metrics.sina[y, x+1], f)
        uu = (uu + (up*cp-vp*sp))/f(2)
        up, vp = jnp.asarray(state.u[:levels, y+1, x], f), jnp.asarray(state.v[:levels, y+1, x], f)
        cp, sp = jnp.asarray(metrics.cosa[y+1, x], f), jnp.asarray(metrics.sina[y+1, x], f)
        vv = (vv + (vp*cp+up*sp))/f(2)
    w, ph, pressure = column(state.w), column(state.ph_total), column(state.p_total)
    ww = (w[:levels]+w[1:levels+1])/f(2)
    height = f(.5)*(ph[:levels]+ph[1:levels+1])/f(9.81)
    theta_dry = column(state.theta)[:levels]/(f(1)+(f(461.6)/f(287))*column(state.qv)[:levels])
    profiles = jnp.stack((uu, vv, ww, height, theta_dry,
                          column(state.qv)[:levels], pressure[:levels]), axis=0)
    # Compute full-column face pressure, including the source's two extrapolations.
    zw = ph/f(9.81)
    z = f(.5)*(zw[:-1]+zw[1:])
    weight0 = (zw[0]-z[1])/(z[0]-z[1])
    weightn = (zw[-1]-z[-2])/(z[-1]-z[-2])
    bottom = weight0*pressure[0]+(f(1)-weight0)*pressure[1]
    top = jnp.exp(weightn*jnp.log(pressure[-1])+(f(1)-weightn)*jnp.log(pressure[-2]))
    interior = (jnp.asarray(metrics.fnm[1:], f)[:, None]*pressure[1:]
                + jnp.asarray(metrics.fnp[1:], f)[:, None]*pressure[:-1])
    p8w = jnp.concatenate((bottom[None], interior, top[None]), axis=0)
    dp = p8w[:-1]-p8w[1:]
    clw = jnp.zeros_like(bottom)
    for species, name in enumerate(("qc", "qr", "qi", "qs", "qg")):
        q = column(getattr(state, name))
        for k in range(q.shape[0]):
            clw = clw + q[k]*dp[k]
            # The literal 220-term WRF sum otherwise joins the station scatter
            # and producer transposes into a measured 754-instruction kInput
            # fusion. Bound the add chain without changing its REAL order.
            # Each boundary materializes only one scalar per station.
            if (species*q.shape[0]+k+1) % 96 == 0:
                clw = jax.lax.optimization_barrier(clw)
    clw /= f(9.81)
    u10, v10 = surface(history.u10), surface(history.v10)
    # Surface rotation uses the unaveraged station rotation, even when UU/VV
    # select the WRF rotated-then-averaged profile option.
    fields = (surface(history.t2), surface(history.q2), u10*cosa-v10*sina,
              v10*cosa+u10*sina, surface(history.psfc), surface(radiation.glw),
              surface(radiation.swdown)-surface(radiation.swup), surface(state.hfx),
              surface(history.lh), surface(state.t_skin), jnp.asarray(soil_top, f),
              surface(state.rainc_acc), surface(state.rain_acc), clw)
    return jnp.stack(fields, axis=-1), jnp.transpose(profiles, (2, 0, 1))


def record_step(carry, namelist, step_index):
    """Called after solve and before nest feedback, matching med_after_solve_io."""
    buf = carry.timeseries
    if buf is None:
        return carry
    soil = jnp.asarray(carry.noahmp_land.tslb[0, buf.y, buf.x], jnp.float32)
    # WRF Noah-MP initializes ocean TSLB to 273.16 on the first call. Aux4
    # overwrites its top layer only on later calls (same rule as land_history).
    ocean = jnp.where((step_index > 1) & buf.sst_update,
                      carry.state.t_skin[buf.y, buf.x], jnp.float32(273.16))
    soil = jnp.where(buf.land, soil, ocean)
    fields, profiles = sample_fields(
        carry.state, carry.history_diagnostics, carry.radiation_diagnostics, soil,
        namelist.metrics, buf.x, buf.y, buf.profiles.shape[-1], unstagger=buf.unstagger,
    )
    hour = (jnp.asarray(step_index, jnp.float32)*jnp.float32(namelist.dt_s)/jnp.float32(60)) / jnp.float32(60)
    n = buf.count
    # Host segmentation guarantees capacity; never wrap and silently lose rows.
    buf = replace(buf, hours=buf.hours.at[n].set(hour), surface=buf.surface.at[n].set(fields),
                  profiles=buf.profiles.at[n].set(profiles), count=n+jnp.int32(1))
    return carry.replace(timeseries=buf)
