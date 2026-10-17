"""Column-local Pallas execution of the shared Ferrier equations.

One tile handles adjacent columns directly in (z,y,x) storage. All level
iteration, incoming precipitation, flags and saturation iteration remain
inside the kernel. No explicit full-domain intermediate fields or column transposes.
"""
from types import FunctionType

import jax
from jax import lax
import jax.numpy as jnp
from gpuwrf.backends import pallas as pl
from jax.experimental.pallas import triton as plt

from . import column as col
from . import saturation as sat
from .saturation import F, host_tables


def _copy(fn, env):
    copy = FunctionType(fn.__code__, env, fn.__name__, fn.__defaults__, fn.__closure__)
    copy.__kwdefaults__ = fn.__kwdefaults__
    return copy


def _namespace(table_refs, valid, *, interpret):
    def load(name, *indices):
        indices = tuple(jnp.broadcast_to(jnp.asarray(i, jnp.int32), valid.shape) for i in indices)
        return plt.load(table_refs[name].at[indices], mask=valid, other=F(0))

    def fpvs(t, *, water):
        t = jnp.broadcast_to(jnp.asarray(t, jnp.float32), valid.shape)
        name = "tbpvs0" if water else "tbpvs"
        xinc = F(F(150) / F(7500))
        c1, c2 = F(F(1) - F(F(180) / xinc)), F(F(1) / xinc)
        # GPVS interpolation has a large subtraction. Preserve its literal
        # REAL product even when the surrounding Triton kernel permits FMA.
        product = c2 * t if interpret else plt.elementwise_inline_asm(
            "mul.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[t, jnp.full_like(t, c2)],
            result_shape_dtypes=[jax.ShapeDtypeStruct(t.shape, jnp.float32)], pack=1)[0]
        xj = jnp.clip(c1 + product, F(1), F(7501))
        j = jnp.minimum(xj, F(7500)).astype(jnp.int32)
        lo, hi = load(name, j - 1), load(name, j)
        value = lo + (xj - j.astype(jnp.float32)) * (hi - lo)
        liquid = F(.61078) * jnp.exp(F(17.2694) * (t - F(273.16)) / (t - F(35.86)))
        ice = F(.61078) * jnp.exp(F(21.8746) * (t - F(273.16)) / (t - F(7.66)))
        outside = liquid if water else jnp.where(t > F(330), liquid, ice)
        return jnp.where((t >= F(180)) & (t <= F(330)), value, outside)

    # Rebind only table access. All physics code objects remain shared with
    # the oracle-tested column path, without global mutation/JIT cache races.
    saturation_env = dict(vars(sat))
    saturation_env["fpvs"] = fpvs
    for name in ("_ws", "_adjust", "condense", "deposit"):
        saturation_env[name] = _copy(getattr(sat, name), saturation_env)
    env = dict(vars(col))
    env.update(fpvs=fpvs, condense=saturation_env["condense"], deposit=saturation_env["deposit"])
    env["_table"] = lambda name, index: load(name, jnp.asarray(index, jnp.int32) - 50)
    env["_vel_rime"] = lambda ixs, ixrf: load("vel_rf", ixs - 2, ixrf)
    env["_growth"] = lambda index, dt: (dt / F(600)) ** F(1.5) * load("my600", index - 1) * F(1e-3)
    env["_rain_dsd"] = _copy(col._rain_dsd, env)
    return _copy(col.level, env)


def native_column(theta, vapor, cloud, rain, ice, p, pii, rho, dz, dt, *, option=5,
                  rimef=None, qrimef=None, gid=1, interpret=False, block=32):
    """Return actual public mp5/mp15 fields; inputs have shape (z,y,x)."""
    if option not in (5, 15):
        raise ValueError("Ferrier option must be 5 or 15")
    inputs = tuple(jnp.asarray(x, jnp.float32) for x in (theta, vapor, cloud, rain, ice, p, pii, rho, dz))
    nz, ny, nx = inputs[0].shape
    if any(x.shape != (nz, ny, nx) for x in inputs):
        raise ValueError("Ferrier inputs must share (z,y,x) shape")
    extra = qrimef if option == 15 else rimef
    if extra is None:
        extra = jnp.zeros_like(ice) if option == 15 else jnp.ones_like(ice)
    inputs += (jnp.asarray(extra, jnp.float32), jnp.asarray(dt, jnp.float32))
    tables = {**host_tables(), "my600": col.MY600}
    table_names = tuple(sorted(tables))
    table_args = tuple(jnp.asarray(tables[n], jnp.float32) for n in table_names)
    if option == 5:
        slots = (0, 1, 2, 3, 4, 5, 6, 7, 8)
        names = ("TH_OUT", "QV_OUT", "QT_OUT", "QC_OUT", "QI_OUT", "QR_OUT", "F_ICE_PHY_OUT", "F_RAIN_PHY_OUT", "F_RIMEF_PHY_OUT")
    else:
        slots = (0, 1, 3, 4, 5, 9)
        names = ("TH_OUT", "QV_OUT", "QC_OUT", "QI_OUT", "QR_OUT", "QRIMEF_OUT")
    shapes = tuple(jax.ShapeDtypeStruct((nz, ny, nx), jnp.float32) for _ in slots)
    shapes += (jax.ShapeDtypeStruct((ny, nx), jnp.float32),) * 2
    def kernel(*refs):
        src = refs[:11]
        tbl = dict(zip(table_names, refs[11:11 + len(table_names)]))
        dst = refs[11 + len(table_names):]
        y = pl.program_id(0)
        x = pl.program_id(1) * block + jnp.arange(block, dtype=jnp.int32)
        valid = x < nx
        compute = _namespace(tbl, valid, interpret=interpret)
        step_dt = plt.load(src[10].at[()])

        def body(k, flux):
            iz = nz - 1 - k
            # All loads precede stores. Input and output refs do not alias.
            values = [plt.load(ref.at[iz, y, x], mask=valid,
                               other=F(1) if i in (0, 5, 6, 7, 8) else F(0))
                      for i, ref in enumerate(src[:10])]
            extra_kw = {"qrimef": values[-1]} if option == 15 else {"rimef": values[-1]}
            row = col.prepare_level(*values[:9], top=k == 0, gid=gid, **extra_kw)
            result, flux, _ = compute(*row, step_dt, flux)
            output = col.finish_level(result, values[6])
            for ref, slot in zip(dst, slots):
                plt.store(ref.at[iz, y, x], output[slot], mask=valid)
            return flux

        flux = lax.fori_loop(0, nz, body, col.initial_flux(jnp.zeros((block,), jnp.float32)))
        amounts = (flux[0] + flux[1]) * F(F(1) / F(1000))
        sr = jnp.where(amounts < F(1e-8), F(0), F(F(1) / F(1000)) * flux[1] / amounts)
        plt.store(dst[-2].at[y, x], amounts * F(1000), mask=valid)
        plt.store(dst[-1].at[y, x], sr, mask=valid)

    outputs = pl.pallas_call(kernel, out_shape=shapes, grid=(ny, pl.cdiv(nx, block)),
                             interpret=interpret, compiler_params=plt.CompilerParams(num_warps=4),
                             name=f"ferrier_mp{option}_columns")(*inputs, *table_args)
    return dict(zip((*names, "RAINNCV", "SR"), outputs))
