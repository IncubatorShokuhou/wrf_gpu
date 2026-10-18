"""NTU mp56: one Pallas program per column, original DOUBLE islands retained.

All microsteps, aerosol inversion and sedimentation are kernel-local. CPU
interpretation and device-less lowering are qualification tools; actual GPU
performance and E41 checks are scheduled separately by the manager.
"""
import math
import numpy as np
import jax
import jax.numpy as jnp
from gpuwrf.backends import pallas as pl
from jax.experimental.pallas import triton as plt
from gpuwrf.physics.ntu import _native
from gpuwrf.physics.ntu.column import micro,PROGNOSTICS,SURFACE,PRIVATE_PRECIP
from gpuwrf.physics.ntu.tables import selected_tables
from gpuwrf.kernels._column_memory import load as _load, store as _store


class TableRef:
    _ntu_table_ref=True

    def __init__(self,ref,*,column=None):
        self.ref,self.fixed_column=ref,column
        self.shape=ref.shape if column is None else ref.shape[:-1]
        self.ndim,self.dtype=len(self.shape),ref.dtype

    def __getitem__(self,index):
        if not isinstance(index,tuple):index=(index,)
        index=tuple(x for x in index if x is not Ellipsis)
        if self.fixed_column is not None:index=index+(self.fixed_column,)
        return _load(self.ref.at[index])

    def column(self,index):
        assert self.fixed_column is None and self.ndim==2
        return TableRef(self.ref,column=index)


def full_column(fields,dt,*,ccnty=2,interpret=False,output_names=None):
    # TBLRC/TBLXF are original DOUBLE arrays. The scope must include their
    # conversion, not just pallas_call (ambient x64=False would truncate them).
    with jax.enable_x64(True):
        return _full_column(fields,dt,ccnty=ccnty,interpret=interpret,output_names=output_names)


def _full_column(fields,dt,*,ccnty=2,interpret=False,output_names=None):
    shape=fields['th'].shape;nz=shape[-1];ncol=math.prod(shape[:-1]);width=1<<(nz-1).bit_length()
    volume=('th','p','pii','dz')+tuple(PROGNOSTICS)
    arrays=tuple(jnp.asarray(fields[n],jnp.float32).reshape(ncol,nz) for n in volume)
    surface_in=('rainnc','snownc','grapnc','hailnc')
    surface_arrays=tuple(jnp.broadcast_to(jnp.asarray(fields[n],jnp.float32),shape[:-1]).reshape(ncol) for n in surface_in)
    tables=selected_tables(np.float32,ccnty)
    array_names=tuple(n for n,a in tables.items() if isinstance(a,np.ndarray) and a.ndim>0 and a.dtype.kind=='f')
    table_arrays=tuple(jnp.asarray(tables[n]) for n in array_names)
    assert all(a.dtype==tables[n].dtype for n,a in zip(array_names,table_arrays))
    outputs=('th',)+tuple(PROGNOSTICS)+tuple(SURFACE) if output_names is None else tuple(output_names)
    allowed=set(('th',)+tuple(PROGNOSTICS)+tuple(SURFACE)+tuple(PRIVATE_PRECIP))
    if len(set(outputs))!=len(outputs) or not set(outputs)<=allowed:raise ValueError('Unknown/duplicate NTU output')
    shapes=tuple(jax.ShapeDtypeStruct((ncol,) if n in SURFACE+PRIVATE_PRECIP else (ncol,nz),jnp.float32) for n in outputs)

    def kernel(*refs):
        col=pl.program_id(0);k=jnp.arange(width,dtype=jnp.int32);valid=k<nz;index=_native.model_index(k,nz)
        pads={'th':300.,'p':100000.,'pii':1.,'dz':250.}
        local={n:_load(ref.at[col,index],mask=valid,other=np.float32(pads.get(n,0.))) for n,ref in zip(volume,refs[:len(volume)])}
        offset=len(volume)
        w0=_load(refs[offset].at[col,index],mask=valid,other=np.float32(0))
        w1=_load(refs[offset].at[col,index+1],mask=valid,other=np.float32(0));offset+=1
        original={n:plt.load(ref.at[col]) for n,ref in zip(surface_in,refs[offset:offset+len(surface_in)])};offset+=len(surface_in)
        c=dict(tables)
        c.update({n:TableRef(ref) for n,ref in zip(array_names,refs[offset:offset+len(array_names)])});offset+=len(array_names)
        local_dt=plt.load(refs[offset].at[()]);offset+=1
        zero=jnp.zeros_like(local['th'])
        v=dict(tk3d=local['th']*local['pii'],p3d=local['p'],dz3d=local['dz'],w3d=np.float32(.5)*(w0+w1),s3d=zero,
               dt=local_dt,dtmn=local_dt/np.float32(60),sdtl=jnp.asarray(0.,jnp.float32),dtl=jnp.asarray(0.,jnp.float32))
        for n in PROGNOSTICS:
            value=local[n]
            if n in ('qdcn','qtcn','qccn','qrcn','qnin'):value=jnp.maximum(c['rlimit'],value)
            v[n+'3d']=value
        for n in SURFACE+PRIVATE_PRECIP:v[n]=original[n] if n in original else jnp.asarray(0.,jnp.float32)
        with _native.column(nz):
            v=micro(v,c,xp=jnp)
        result={'th':v['tk3d']/local['pii']}
        result.update({n:jnp.fmax(np.float32(0),v[n+'3d']) for n in PROGNOSTICS})
        result.update({n:v[n] for n in SURFACE+PRIVATE_PRECIP})
        for n,ref in zip(outputs,refs[offset:]):
            if n in SURFACE+PRIVATE_PRECIP:_store(ref.at[col],result[n])
            else:_store(ref.at[col,index],result[n],mask=valid)

    with jax.enable_x64(True):
        values=pl.pallas_call(kernel,grid=(ncol,),out_shape=shapes,interpret=interpret,
                             compiler_params=plt.CompilerParams(num_warps=2),name='ntu_mp56_column')(
            *arrays,jnp.asarray(fields['w'],jnp.float32).reshape(ncol,nz+1),*surface_arrays,*table_arrays,jnp.asarray(dt,jnp.float32))
    return {n:a.reshape(shape[:-1] if n in SURFACE+PRIVATE_PRECIP else shape) for n,a in zip(outputs,values)}
