"""WRF MYNN CLOUDMIX number scalars beyond QNI: QNC, QNWFA, QNIFA.

module_bl_mynnedmf.F mynn_tendencies (WRF 4.7.1):
* QNC  :4549-4605 -- pre ``max(qnc,0)``, ``max(.,nc_min)`` where sqc>1e-12, the
  QNI matrix with s_awqnc, post ``max(.,nc_min)`` where the mixed sqc2>1e-12.
* QNWFA :4610-4670 / QNIFA :4675-4735 -- pre ``max(x,0)``, the same matrix with
  s_awqnwfa / s_awqnifa, post ``max(x,0)`` then the height bounds
  ``min(max(x_min*exp(-zw/ht), x), x_max*exp(-zw/ht))`` on the bottom interface zw.
Constants :3937-3953.  sd_* (EDMF downdrafts) are zero for the admitted
bl_mynn_edmf_dd=0.  QNBCA has no State leaf (zero with wif_input_opt=1).
"""
import jax.numpy as jnp

NC_MIN = 100.0
AEROSOL_BOUNDS = {"nwfa": (5e6, 800e6, 2000.0), "nifa": (1e4, 270e6, 10000.0)}  # (min, max, ht)


def _bottom_interfaces(dz):
    """zw(k) of mynn_tendencies: zw(kts)=0, zw(k+1)=zw(k)+dz(k) summed sequentially
    in REAL like the caller (an associative cumsum changes the rounding, which the
    exp(-zw/ht) aerosol bounds amplify ~|zw/ht| times)."""
    import jax
    levels = jnp.moveaxis(dz, -1, 0)

    def step(z, d):
        return z + d, z

    _, zw = jax.lax.scan(step, jnp.zeros_like(levels[0]), levels)
    return jnp.moveaxis(zw, 0, -1)


def mf_flux_key(name: str) -> str:
    """DMP_mf passive flux key of one State number leaf (s_awqnc1 etc.)."""
    return "s_awq" + name


def mix_passive_numbers(passive, sqc, sqc2, state, turb, dt, mf=None):
    """Mixed number scalars (dict, State leaf names) at the column's REAL precision."""
    from gpuwrf.kernels.phys_mynn_cloudmix import _scalar
    from gpuwrf.physics import mynn_pbl as P

    out = {}
    for name, x in passive.items():
        c = lambda value: jnp.asarray(value, x.dtype)  # noqa: E731
        kdz = P._rho_interfaces(state, turb["dfh"])
        empty = jnp.zeros(x.shape[:-1] + (x.shape[-1] + 1,), x.dtype)
        if mf is None:
            aw, awx = empty, empty
        else:
            aw, awx = mf["s_aw"], mf[mf_flux_key(name)]
            kdz = P._apply_s_aw_stability_floor(kdz, aw)
        before = jnp.maximum(x, c(0.0))
        if name == "Nc":
            before = jnp.where(sqc > c(1e-12), jnp.maximum(before, c(NC_MIN)), before)
        mixed = _scalar(before, state, dt, kdz, aw, awx, jnp.zeros_like(x[..., 0]))
        if name == "Nc":
            out[name] = jnp.where(sqc2 > c(1e-12), jnp.maximum(mixed, c(NC_MIN)), mixed)
            continue
        low, high, height = AEROSOL_BOUNDS[name]
        zw = _bottom_interfaces(state.dz)
        decay = jnp.exp(-zw / c(height))
        out[name] = jnp.minimum(jnp.maximum(c(low) * decay, jnp.maximum(mixed, c(0.0))), c(high) * decay)
    return out
