"""One Pallas program per Jensen column; all adaptive/vertical loops are local.

CPU interpretation and device-less Triton lowering qualify this route without
opening a GPU. GPU execution, E41 and performance qualification remain pending.
"""
import math
import jax
import jax.numpy as jnp
from gpuwrf.backends import pallas as pl
from jax.experimental.pallas import triton as plt
from gpuwrf.physics.jensen_ishmael import _native
from gpuwrf.physics.jensen_ishmael.column import _me,public_outputs
from gpuwrf.physics.jensen_ishmael.tables import get_tables
from gpuwrf.kernels._column_memory import load as _load, store as _store


class TableRef:
    _jensen_table_ref=True
    def __init__(self,ref):
        self.ref,self.shape,self.ndim,self.dtype=ref,ref.shape,len(ref.shape),ref.dtype
    def __getitem__(self,index):return _load(self.ref.at[index])


def full_column(fields,dt,it,*,rain=0.,snow=0.,interpret=False,output_names=None):
    shape=fields['qv'].shape;nz=shape[-1];ncol=math.prod(shape[:-1]);width=1<<(nz-1).bit_length()
    names=('p','dz','th','qv','qc','qr','nr')+tuple(n+str(i) for i in (1,2,3) for n in ('qi','ni','ai','ci'))
    arrays=tuple(jnp.asarray(fields[n],jnp.float32).reshape((ncol,nz)) for n in names)
    tables={n:jnp.asarray(a,jnp.float32) for n,a in get_tables(jnp.float32).items()}
    table_names=tuple(tables)
    outputs=names+('effc','effi','dbz')+tuple(n+str(i) for i in (1,2,3) for n in ('vm','di','rho','phi'))+('type1','type2','type3')
    if output_names is not None:
        selected=tuple(output_names)
        if len(set(selected))!=len(selected) or any(n not in outputs for n in selected):raise ValueError('Unknown/duplicate Jensen column output')
        outputs=selected
    precip=('rain','rainstep','snow','snowstep')
    output_shapes=tuple(jax.ShapeDtypeStruct((ncol,nz),jnp.float32) for _ in outputs)+tuple(jax.ShapeDtypeStruct((ncol,),jnp.float32) for _ in precip)

    def kernel(*refs):
        col=pl.program_id(0);level=jnp.arange(width,dtype=jnp.int32);valid=level<nz
        pads={'p':100000.,'dz':250.,'th':300.}
        local={n:_load(ref.at[col,level],mask=valid,other=pads.get(n,0.)) for n,ref in zip(names,refs[:len(names)])}
        offset=len(names)
        local_tables={n:TableRef(ref) for n,ref in zip(table_names,refs[offset:offset+len(table_names)])}
        offset+=len(table_names)
        local_dt=plt.load(refs[offset].at[()]);local_it=plt.load(refs[offset+1].at[()])
        old_rain=plt.load(refs[offset+2].at[col]);old_snow=plt.load(refs[offset+3].at[col])
        zero=jnp.zeros_like(local['qv'])
        f=dict(theta=local['th'],qv=local['qv'],qc=local['qc'],qr=local['qr'],nr=local['nr'],
               pres_e=local['p'],dzmic=local['dz'],dt=local_dt,it=local_it)
        for n in ('qi','ni','ai','ci'):f[n]=_native.stack4(tuple(local[n+str(i)] for i in (1,2,3))+(zero,))
        cat_zero=jnp.zeros_like(f['qi'])
        f.update(effc1d=zero,effi1d=zero,dbz1d=jnp.full_like(zero,-35.),qrpre1d=zero,qipre1d=cat_zero)
        for n in ('icetype1d','vmi1d','di1d','rhopo1d','phii1d'):f[n]=cat_zero
        with _native.column(nz):
            state=_me(f,local_tables,jnp)
            result,accum=public_outputs(local,state,old_rain,old_snow)
        offset+=4
        for n,ref in zip(outputs,refs[offset:offset+len(outputs)]):_store(ref.at[col,level],result[n],mask=valid)
        for n,ref in zip(precip,refs[offset+len(outputs):]):_store(ref.at[col],accum[n])

    with jax.enable_x64(True):
        values=pl.pallas_call(kernel,grid=(ncol,),out_shape=output_shapes,interpret=interpret,
                             compiler_params=plt.CompilerParams(num_warps=4),name='jensen_mp55_column')(
            *arrays,*tables.values(),jnp.asarray(dt,jnp.float32),jnp.asarray(it,jnp.int32),
            jnp.broadcast_to(jnp.asarray(rain,jnp.float32),shape[:-1]).reshape(ncol),
            jnp.broadcast_to(jnp.asarray(snow,jnp.float32),shape[:-1]).reshape(ncol))
    result={n:a.reshape(shape) for n,a in zip(outputs,values[:len(outputs)])}
    accum={n:a.reshape(shape[:-1]) for n,a in zip(precip,values[len(outputs):])}
    return result,accum
