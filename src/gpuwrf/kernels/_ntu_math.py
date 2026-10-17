"""Original REAL/DOUBLE ERF through libdevice in NTU's resident kernel.

The pinned Pallas Triton backend lacks an erf lowering. This local primitive
uses the same typed libdevice dispatch mechanism as its existing exp/log/sqrt
rules, without changing any global JAX primitive or another physics scheme.
"""
import jax
import jax.numpy as jnp
from jax.extend import core
from jax.interpreters import mlir
from jax._src.pallas.triton import lowering

erf_p=core.Primitive('ntu_source_erf')
erf_p.def_impl(jax.lax.erf)
erf_p.def_abstract_eval(lambda x:jax.core.ShapedArray(x.shape,x.dtype))
mlir.register_lowering(erf_p,mlir.lower_fun(jax.lax.erf,multiple_results=False))
lowering.register_lowering(erf_p)(lowering._make_dispatch_table(
    'ntu_source_erf',cuda=(lowering._Extern((jnp.float32,),'__nv_erff',jnp.float32),
                           lowering._Extern((jnp.float64,),'__nv_erf',jnp.float64))))


def erf(value):return erf_p.bind(value)
