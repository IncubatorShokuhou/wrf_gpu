"""Original WRF ``diff4d`` passive-tracer column mixing (REAL arithmetic).

Reference: pristine WRF phys/module_pbl_driver.F:2690-2931. This is the
uncoupled tendency; the dynamics caller supplies hybrid dry-mass coupling.
"""

import jax
import jax.numpy as jnp
from jax import lax


def tracer_pbl_tendency(tracers, rho, dz, exch_h, exch_m, elapsed_dt):
    """Return WRF's raw mixing tendency in tracer units per second.

    ``tracers`` is [N,z,y,x]; ``rho`` (kg/m3) and ``dz`` (m) are [z,y,x];
    exchange coefficients (m2/s) are [z+1,y,x], on lower level faces.
    ``elapsed_dt`` is the PBL solve interval (WRF DTBL), not necessarily the
    dynamics timestep. The caller applies the returned tendency on its usual
    physics cadence. All inputs must be float32; dt, rho and dz must be positive.

    WRF zeros the bottom heat exchange, holds the top mass level fixed, and
    ignores momentum exchange for passive scalars. No clipping is applied.
    """
    raw_arrays = (tracers, rho, dz, exch_h, exch_m)
    if any(getattr(a, "dtype", jnp.float32) != jnp.float32 for a in raw_arrays):
        raise TypeError("WRF tracer PBL inputs must be float32")
    arrays = tuple(jnp.asarray(a) for a in raw_arrays)
    tracers, rho, dz, exch_h, exch_m = arrays
    if any(a.dtype != jnp.float32 for a in arrays):
        raise TypeError("WRF tracer PBL inputs must be float32")
    if tracers.ndim != 4 or rho.ndim != 3 or tracers.shape[1:] != rho.shape:
        raise ValueError("expected tracers [N,z,y,x] and rho [z,y,x]")
    if dz.shape != rho.shape or rho.shape[0] < 1:
        raise ValueError("dz must match the nonempty rho column")
    face_shape = (rho.shape[0] + 1, *rho.shape[1:])
    if exch_h.shape != face_shape or exch_m.shape != face_shape:
        raise ValueError("exchange fields must have shape [z+1,y,x]")
    dt = jnp.asarray(elapsed_dt, dtype=jnp.float32)
    if dt.ndim:
        raise ValueError("elapsed_dt must be scalar")
    if rho.shape[0] == 1:
        # WRF fixes the only row: co_out=co_in. Keep subtraction/division so
        # invalid q or zero dt propagate as in the source; avoid empty scans.
        return (tracers - tracers) / dt
    # Materialize the literal source intermediates. Native FMA contraction is
    # qualified separately against the original WRF FMA/no-FMA spread.
    b = lax.optimization_barrier
    rhof = b(b(b(rho[1:] * dz[:-1]) + b(rho[:-1] * dz[1:])) /
             b(dz[:-1] + dz[1:]))
    interior = b(b(b(jnp.float32(2) * rhof) * exch_h[1:-1]) /
                 b(dz[1:] + dz[:-1]))
    # diff4d initializes the local exchange arrays to zero; bottom and unused
    # top faces therefore vanish even when the caller passes nonzero values.
    cddz = jnp.concatenate((jnp.zeros_like(rho[:1]), interior,
                           jnp.zeros_like(rho[:1])), axis=0)
    lower = b(b(b(-cddz[:-1] * dt) / dz) / rho)
    upper = b(b(b(-cddz[1:] * dt) / dz) / rho)
    diag = b(jnp.float32(1) + b(b(b(dt * b(cddz[:-1] + cddz[1:])) / dz) / rho))
    lower = lower.at[-1].set(jnp.float32(0))
    upper = upper.at[-1].set(jnp.float32(0))
    diag = diag.at[-1].set(jnp.float32(1))
    rhs = jnp.moveaxis(tracers, 1, 0)

    def backward(carry, row):
        next_d, next_c, next_l = carry
        d, c, l, u = row
        c = b(c - b(b(u * next_c) / next_d))
        d = b(d - b(b(u * next_l) / next_d))
        return (d, c, l), (d, c)

    _, (reduced_d, reduced_c) = lax.scan(
        backward, (diag[-1], rhs[-1], lower[-1]),
        (diag[:-1], rhs[:-1], lower[:-1], upper[:-1]), reverse=True,
    )
    reduced_d = jnp.concatenate((reduced_d, diag[-1:]), axis=0)
    reduced_c = jnp.concatenate((reduced_c, rhs[-1:]), axis=0)

    def forward(carry, row):
        previous_c, previous_d = carry
        c, l, d = row
        c = b(c - b(b(l * previous_c) / previous_d))
        return (c, d), c

    _, solved_c = lax.scan(
        forward, (reduced_c[0], reduced_d[0]),
        (reduced_c[1:], lower[1:], reduced_d[1:]),
    )
    solved_c = jnp.concatenate((reduced_c[:1], solved_c), axis=0)
    mixed = b(solved_c / reduced_d[:, None])
    return jnp.moveaxis(b(b(mixed - rhs) / dt), 0, 1)


__all__ = ["tracer_pbl_tendency"]
