"""Nt_c-dependent mp8 closures from the original Thompson REAL/table rules.

All-nu gamma families and the full cloud-number freezing axis are independent
of Nt_c in thompson_init. The traced value selects those fixed WRF entries and
forms the dependent scalar closures. Do that scalar work before the column
kernel, so the DOUBLE logarithm is not repeated in every column.
"""
from __future__ import annotations

from functools import lru_cache
import numpy as np

ASSET = "data/fixtures/thompson-ntc-v1.npz"
PACK_NAMES = ("nu_c", "idx_n", "lamc_prefix", "dc_g_prefix", "pnr_wau_den", "ccg5", "ocg2")


@lru_cache(maxsize=1)
def host_tables():
    """Cache host arrays only (E206); JAX arrays are made at the use site."""
    from gpuwrf.assets import asset_path, ensure_asset
    with np.load(ensure_asset(asset_path(ASSET)), allow_pickle=False) as table:
        arrays = {key: np.asarray(table[key]) for key in table.files}
    for name in ("tpi_qcfz", "tni_qcfz"):
        assert arrays[name].shape == (37, 100, 45) and arrays[name].dtype == np.float64
    assert arrays["ccg"].shape == (5, 15) and arrays["ccg"].dtype == np.float32
    assert int(arrays["nic1"]) == 7 and int(arrays["fixed_in_index"]) == 27
    for value in arrays.values():
        assert np.isfinite(value).all()
        value.setflags(write=False)
    return arrays


def cloud_parameters(nt_c):
    """Seven REAL scalar operands; source :2172–2191, :2278, :3658–3661.

    nu_c uses REAL division/NINT, whereas idx_n uses WRF's DOUBLE DLOG and
    INTEGER nic1. The gamma and derived nu-only coefficients are WRF's own
    REAL results, not gamma() from another math library.
    """
    import jax
    import jax.numpy as jnp
    from gpuwrf.physics.thompson_constants import WRF_REAL_CONSTANTS
    tables = host_tables()
    nt_c = jnp.asarray(nt_c, jnp.float32)
    nu = jnp.minimum(jnp.int32(15), jnp.floor(jnp.float32(1000.e6) / nt_c + jnp.float32(.5)).astype(jnp.int32) + jnp.int32(2))
    row = nu - jnp.int32(1)
    ccg2 = jnp.asarray(tables["ccg"][1], jnp.float32)[row]
    ocg1 = jnp.asarray(tables["ocg1"], jnp.float32)[row]
    # Original REAL left association: nc*am_r*ccg(2,nu_c)*ocg1(nu_c).
    lamc_prefix = nt_c * jnp.float32(WRF_REAL_CONSTANTS["AM_R"]) * ccg2 * ocg1
    with jax.enable_x64(True):
        index = jnp.float64(1) + jnp.float64(100) * jnp.log(nt_c.astype(jnp.float64) / jnp.float64(tables["t_Nc"][0])) / jnp.float64(int(tables["nic1"]))
        nearest = jnp.sign(index) * jnp.floor(jnp.abs(index) + jnp.float64(.5))
        index = jnp.clip(nearest.astype(jnp.int32), 1, 100) - 1
    return jnp.stack((nu.astype(jnp.float32), index.astype(jnp.float32), lamc_prefix,
                      jnp.asarray(tables["dc_g_prefix"], jnp.float32)[row],
                      jnp.asarray(tables["pnr_wau_den"], jnp.float32)[row],
                      jnp.asarray(tables["ccg"][4], jnp.float32)[row],
                      jnp.asarray(tables["ocg2"], jnp.float32)[row]))


def freezing_tables(cold):
    """Replace only the two default-Nc planes with the independent full Nc axis."""
    import jax.numpy as jnp
    arrays = list(cold)
    host = host_tables()
    for name in ("tpi_qcfz", "tni_qcfz"):
        arrays[cold._fields.index(name)] = jnp.asarray(host[name], jnp.float64)
    return arrays


def effective_radii(out, nt_c):
    """Original calc_effectRad with its private Nt_c binding changed (:5627).

    Keep the equation code and WRF DOUBLE islands intact; no module mutation
    and no cached tracer. The caller skips the old fixed-Nt producer.
    """
    from types import FunctionType
    from gpuwrf.physics.rrtmg_mp_re import calc_thompson_effective_radii
    original = calc_thompson_effective_radii
    fn = FunctionType(original.__code__, {**original.__globals__, "NT_C": nt_c},
                      original.__name__, original.__defaults__, original.__closure__)
    fn.__kwdefaults__ = original.__kwdefaults__
    return fn(out.T, out.p, out.qv, out.qc, out.qi, out.Ni, out.qs)
