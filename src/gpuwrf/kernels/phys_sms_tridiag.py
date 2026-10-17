"""Literal REAL module_diffusion_em.tridiag, one level-major column kernel."""
from functools import partial
from functools import lru_cache
import math
import jax
import jax.numpy as jnp
from gpuwrf.backends import pallas as pl
from jax.experimental.pallas import triton as pt
from gpuwrf.kernels.dyn_ring_relax_fp32 import _rn

BLOCK=128


def _kernel(a,b,c,d,out,*,nz,nc,interpret):
    col=pl.program_id(0)*BLOCK+jnp.arange(BLOCK,dtype=jnp.int32);valid=col<nc
    ld=lambda r,k:pt.load(r.at[k,col],mask=valid,other=1)
    rn=partial(_rn,interpret=interpret)
    neg=lambda x:rn('sub',jnp.zeros_like(x),x)
    q=[rn('div',neg(ld(c,0)),ld(b,0))]
    dd=[rn('div',ld(d,0),ld(b,0))]
    for k in range(1,nz):
        p=rn('div',jnp.ones_like(dd[-1]),rn('add',ld(b,k),rn('mul',ld(a,k),q[-1])))
        # WRF overwrites c(n)=0 before the forward sweep.
        ck=ld(c,k) if k<nz-1 else jnp.zeros_like(p)
        q.append(rn('mul',neg(ck),p))
        dd.append(rn('mul',rn('sub',ld(d,k),rn('mul',ld(a,k),dd[-1])),p))
    val=dd[-1];pt.store(out.at[nz-1,col],val,mask=valid)
    for k in range(nz-2,-1,-1):
        val=rn('add',dd[k],rn('mul',q[k],val));pt.store(out.at[k,col],val,mask=valid)


@lru_cache(maxsize=32)
def _solver(nz,nc,interpret):
    # Cache only the shape-specialized function, never arrays/tracers (E206).
    return pl.pallas_call(partial(_kernel,nz=nz,nc=nc,interpret=interpret),grid=((nc+BLOCK-1)//BLOCK,),
        out_shape=jax.ShapeDtypeStruct((nz,nc),jnp.float32),interpret=interpret,
        compiler_params=pt.CompilerParams(num_warps=4),name='sms_tridiag_real')


def solve(a,b,c,d):
    shape=d.shape;nz=shape[0];nc=math.prod(shape[1:]);interpret=jax.default_backend()=='cpu'
    if any(x.dtype!=jnp.float32 for x in (a,b,c,d)):raise TypeError('SMS tridiag requires WRF REAL')
    out=_solver(nz,nc,interpret)(*(x.reshape(nz,nc) for x in (a,b,c,d)))
    return out.reshape(shape)
