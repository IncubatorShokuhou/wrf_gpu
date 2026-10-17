"""One Pallas program per CAM-MG column; all vertical/adaptive loops local.

CPU interpretation / deviceless CUDA lowering are separate from GPU execution,
which remains pending manager scheduling. No host callback runs inside a column.
On CUDA the column is compiled by STOCK Triton 3.6 and launched by jaxlib
triton_kernel_call (KernelCallCompilerParams, kernels/_triton_kernel_call.py):
XLA's bundled Triton (in-graph and PJRT-plugin compiler) miscompiles this kernel
on sm_120 to zero precipitation (t-rainnc/t-cammgws).
"""
import math
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

from gpuwrf.physics.cammg import _native
from gpuwrf.physics.cammg.column import step
from gpuwrf.physics.cammg.tables import get_globals
from gpuwrf.physics.bl_camuw import ESTBL
from gpuwrf.kernels._triton_kernel_call import KernelCallCompilerParams

SURFACE_FIELDS=('ht','qfx','rliq2d','xland','snowh','rainnc','rainncv','snownc','snowncv','sr')
MODIFIED_PROFILES=('th','cldfra_old_mp','rh_old_mp','lcd_old_mp','qv_curr','qc_curr','qi_curr','qs_curr','qr_curr',
                   'ni3d','nc3d','ns3d','nr3d','qndrop','wsedl3d','cldfra_mp','cldfra_conv','cldfra_mp_all',
                   'lradius','iradius','cldfrai','cldfral')
MODIFIED_SURFACES=('rainnc','rainncv','snownc','snowncv','sr')


class TableRef:
    _cammg_table_ref=True
    def __init__(self,ref):self.ref,self.shape,self.ndim,self.dtype=ref,ref.shape,len(ref.shape),ref.dtype
    def __getitem__(self,index):return plt.load(self.ref.at[index])


def full_column(fields,dt,it,*,cu=0,shcu=0,accum_mode=1.e9,aitken_mode=3.e8,coarse_mode=2.e5,
                interpret=False,output_names=None):
    shape=fields['t_phy'].shape;nz=shape[-1]-1;ncol=math.prod(shape[:-1]);width=1<<nz.bit_length()
    names=tuple(fields)
    profile_names=tuple(n for n in names if n not in SURFACE_FIELDS)
    arrays=tuple(jnp.asarray(fields[n],jnp.float32).reshape((ncol,nz+1) if n in profile_names else (ncol,)) for n in names)
    globals=get_globals(nz)
    flat={}
    def visit(value,path=()):
        if isinstance(value,dict):
            for name,child in value.items():visit(child,(*path,name))
        elif isinstance(value,np.ndarray) and value.ndim:flat[path]=value
    visit(globals)
    table_paths=tuple(flat)
    with jax.enable_x64(True):
        table_values=tuple(jnp.asarray(flat[p]) for p in table_paths)
    outputs=(*MODIFIED_PROFILES,*MODIFIED_SURFACES) if output_names is None else tuple(output_names)
    if len(set(outputs))!=len(outputs) or any(n not in (*MODIFIED_PROFILES,*MODIFIED_SURFACES) for n in outputs):
        raise ValueError('Unknown/duplicate CAM-MG native column output')
    out_shapes=tuple(jax.ShapeDtypeStruct((ncol,nz+1) if n in profile_names else (ncol,),jnp.float32) for n in outputs)
    out_shapes+= (jax.ShapeDtypeStruct((ncol,),jnp.int32),)

    def kernel(*refs):
        col=pl.program_id(0);level=jnp.arange(width,dtype=jnp.int32);valid=level<nz+1
        neutral={'t_phy':300.,'th':300.,'pi_phy':1.,'p8w':100000.,'p_hyd':100000.,'alt':1.}
        local={}
        for n,ref in zip(names,refs[:len(names)]):
            local[n]=plt.load(ref.at[col,level],mask=valid,other=neutral.get(n,0.)) if n in profile_names else plt.load(ref.at[col])
        def copy(value):return {k:copy(v) for k,v in value.items()} if isinstance(value,dict) else value
        bound=copy(globals)
        offset=len(names)
        for path,ref in zip(table_paths,refs[offset:offset+len(table_paths)]):
            destination=bound
            for key in path[:-1]:destination=destination[key]
            destination[path[-1]]=TableRef(ref)
        offset+=len(table_paths)
        bound['__estbl_ref']=TableRef(refs[offset]);offset+=1
        local_dt=plt.load(refs[offset].at[()]);local_it=plt.load(refs[offset+1].at[()]);offset+=2
        modes=tuple(plt.load(refs[offset+i].at[()]) for i in range(3));offset+=3
        with _native.column(nz,local_it):
            result,fatal=step(local,local_dt,itimestep=local_it,cu=cu,shcu=shcu,accum_mode=modes[0],
                              aitken_mode=modes[1],coarse_mode=modes[2],globals=bound,xp=jnp)
        for name,ref in zip(outputs,refs[offset:offset+len(outputs)]):
            if name in profile_names:plt.store(ref.at[col,level],result[name],mask=valid)
            else:plt.store(ref.at[col],jnp.reshape(result[name],()))
        plt.store(refs[-1].at[col],jnp.reshape(fatal,()))

    with jax.enable_x64(True):
        values=pl.pallas_call(kernel,grid=(ncol,),out_shape=out_shapes,interpret=interpret,
                             compiler_params=KernelCallCompilerParams(num_warps=4),name='cammg_mp11_column')(
            *arrays,*table_values,jnp.asarray(ESTBL,jnp.float64),jnp.asarray(dt,jnp.float32),jnp.asarray(it,jnp.int32),
            jnp.asarray(accum_mode,jnp.float32),jnp.asarray(aitken_mode,jnp.float32),jnp.asarray(coarse_mode,jnp.float32))
    result={n:a.reshape(shape if n in profile_names else shape[:-1]) for n,a in zip(outputs,values[:-1])}
    return result,values[-1].reshape(shape[:-1])
