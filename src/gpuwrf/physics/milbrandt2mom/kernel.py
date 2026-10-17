"""One GPU Pallas program per column, adaptive sedimentation local to that column."""
from functools import partial
import jax
import jax.numpy as jnp
from gpuwrf.backends import pallas as pl
from jax.experimental.pallas import triton as pt
from .column import column, columns


def _field_kernel(*refs,nz,width,nx):
    col=pl.program_id(0); y,x=col//nx,col%nx
    k=jnp.arange(width,dtype=jnp.int32); valid=k<nz
    f=tuple(pt.load(refs[i].at[k,y,x],mask=valid,other=220. if i==0 else 0.) for i in range(14))
    pressure=pt.load(refs[14].at[k,y,x],mask=valid,other=1.)
    surface=pt.load(refs[15].at[y,x])
    wind=pt.load(refs[16].at[k,y,x],mask=valid,other=0.)
    step=pt.load(refs[17].at[()])
    fo,r,_unused_dm,z,_unused_zec=column(f,pressure,surface,wind,step,nz=nz)
    for ref,value in zip(refs[18],fo):pt.store(ref.at[k,y,x],value,mask=valid)
    for ref,value in zip(refs[19],r):pt.store(ref.at[y,x],value)
    pt.store(refs[20].at[k,y,x],z,mask=valid)


def run_fields(fields,pressure,ps,w,dt):
    """Layout-neutral production ABI: native level-major fields, no packed input copy.

    WRF's discarded six Dm profiles/ZEC stay local and are not global outputs.
    They remain available on the independent column qualification boundary.
    """
    nz,ny,nx=pressure.shape;dtype=pressure.dtype
    if not pl.cuda_native():
        stack=jnp.stack([jnp.moveaxis(v,0,-1).reshape(-1,nz) for v in fields],axis=1)
        out,rates,_dm,zet,_zec=columns(stack,jnp.moveaxis(pressure,0,-1).reshape(-1,nz),
            ps.reshape(-1),jnp.moveaxis(w[:nz],0,-1).reshape(-1,nz),dt)
        restore=lambda v:jnp.moveaxis(v.reshape(ny,nx,nz),-1,0)
        return tuple(restore(out[:,i]) for i in range(14)),tuple(rates[:,i].reshape(ny,nx) for i in range(11)),restore(zet)
    if dtype!=jnp.float32 or any(v.dtype!=jnp.float32 for v in fields):
        raise TypeError('Milbrandt native fields require WRF REAL32')
    width=1 << (nz-1).bit_length()
    shape3=jax.ShapeDtypeStruct((nz,ny,nx),dtype)
    shape2=jax.ShapeDtypeStruct((ny,nx),dtype)
    return pl.pallas_call(partial(_field_kernel,nz=nz,width=width,nx=nx),grid=(ny*nx,),
        out_shape=((shape3,)*14,(shape2,)*11,shape3),name='milbrandt2mom_fields',
        compiler_params=pt.CompilerParams(num_warps=2))(*fields,pressure,ps,w,jnp.asarray(dt,dtype))
