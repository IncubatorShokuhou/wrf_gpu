"""One column kernel per QNSE driver; all vertical work stays inside it.

Profiles have (level, flattened horizontal column) layout, preserving WRF
coalesced horizontal traffic. Surface operands have (column,) layout. CPU
interpretation is for qualification only; GPU performance remains unmeasured.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from gpuwrf.backends import pallas as pl
from jax.experimental.pallas import triton as pt
from gpuwrf.backends import pallas as backend

from gpuwrf.physics.pbl_qnse import qnsepbl_column
from gpuwrf.physics.sfclay_qnse import qnsesfc_column, _tables
from gpuwrf.physics.qnse_edmf import mfshconvpbl_column
from gpuwrf.physics._qnse_levels import ColumnLevels
from gpuwrf.kernels._qnse_memory import load, store

_PBL_PROFILES=('RUBLTEN','RVBLTEN','RTHBLTEN','RQVBLTEN','RQCBLTEN','tke','EXCH_H','EXCH_M','EL_MYJ')
_PBL_SCALARS=('PBLH','KPBL','thz0','qz0','qsfc','ct')
_SFC_SCALARS=('ustar','znt','akhs','akms','rmol','rib','chs','chs2','cqs2','hfx','qfx','flx_lh',
              'flhc','flqc','qgh','cpm','ct','qsfc','thz0','qz0','uz0','vz0','u10','v10','th02',
              'th10','tshltr','q02','qshltr','q10','pshltr','u10e','v10e','akhs02','akhs10','akms10','t02','pblh')
_MF_PROFILES=('RTHBLTEN','RQVBLTEN','RQCBLTEN','RUBLTEN','RVBLTEN','massflux_EDKF','entr_EDKF',
              'detr_EDKF','thl_up','thv_up','rt_up','rv_up','rc_up','u_up','v_up','frac_up','rc_mf','WTHV','PLM_BL89')


class _Table:
    def __init__(self,ref):self.ref=ref
    def __getitem__(self,index):return load(self.ref.at[index])


def qnse_columns(scheme, inputs, *, interpret=None, block=64, requested=None):
    """Evaluate one complete WRF driver on a batch of REAL4 columns."""
    if scheme not in ('pbl','sfc','mf'):raise ValueError('Unknown QNSE driver')
    primary=jnp.asarray(inputs['theta'],jnp.float32)
    if primary.ndim!=2 or primary.shape[0]<3:raise ValueError('QNSE batch profiles require (level,column)')
    nz,ncol=primary.shape
    names=tuple(inputs)
    values=[]
    for name in names:
        value=jnp.asarray(inputs[name],jnp.float32)
        if value.ndim==0:value=jnp.broadcast_to(value,(ncol,))
        if value.shape not in ((ncol,),(nz,ncol),(nz+1,ncol)):
            raise ValueError(f'QNSE {name} has invalid batch shape {value.shape}')
        values.append(value)
    profiles,scalars,fn={
        'pbl':(_PBL_PROFILES,_PBL_SCALARS,qnsepbl_column),
        'sfc':((),_SFC_SCALARS,qnsesfc_column),
        'mf':(_MF_PROFILES,(),mfshconvpbl_column)}[scheme]
    output_names=profiles+scalars
    if requested is not None:
        if not set(requested)<=set(output_names):raise ValueError('Unknown requested QNSE output')
        output_names=tuple(requested)
    if interpret is None and not backend.cuda_native():
        # The portable CPU path uses the SAME register-level equations and
        # saturation loop. Avoid expanding Pallas Ref emulation over every
        # grid block; explicit interpretation remains available for the oracle.
        arguments={name:ColumnLevels(list(value)) if value.ndim==2 else value
                   for name,value in zip(names,values)}
        if scheme=='sfc':
            pm,ph,_,_=_tables('float32')
            arguments['_tables_input']=(jnp.asarray(pm),jnp.asarray(ph))
        if scheme=='mf':arguments['_diagnostics']='PLM_BL89' in output_names
        result=fn(**arguments,_column_kernel=True)
        return {name:jnp.stack(result[name]) if name in profiles else
                     jnp.broadcast_to(result[name],(ncol,)) for name in output_names}
    specs=tuple(jax.ShapeDtypeStruct((nz,ncol) if name in profiles else (ncol,),
                                   jnp.int32 if name=='KPBL' else jnp.float32) for name in output_names)
    shapes=tuple(value.shape for value in values)
    if scheme=='sfc':
        pm,ph,_,_=_tables('float32')
        values.extend((jnp.asarray(pm),jnp.asarray(ph)))
    nin=len(values)

    def kernel(*refs):
        cols=pl.program_id(0)*block+jnp.arange(block)
        valid=cols<ncol
        arguments={}
        for name,shape,ref in zip(names,shapes,refs):
            if len(shape)==1:
                arguments[name]=load(ref.at[cols],mask=valid,other=jnp.float32(1.))
            else:
                arguments[name]=ColumnLevels([load(ref.at[k,cols],mask=valid,other=jnp.float32(1.))
                                           for k in range(shape[0])])
        if scheme=='sfc':arguments['_tables_input']=(_Table(refs[nin-2]),_Table(refs[nin-1]))
        if scheme=='mf':arguments['_diagnostics']='PLM_BL89' in output_names
        result=fn(**arguments,_column_kernel=True)
        for name,ref in zip(output_names,refs[nin:]):
            if name in profiles:
                for k in range(nz):store(ref.at[k,cols],result[name][k],mask=valid)
            else:
                store(ref.at[cols],jnp.broadcast_to(result[name],(block,)),mask=valid)

    if interpret is None:interpret=False
    outputs=pl.pallas_call(kernel,out_shape=specs,grid=((ncol+block-1)//block,),
                           interpret=interpret,name='qnse_'+scheme+'_columns',
                           compiler_params=pt.CompilerParams(num_warps=4))(*values)
    return dict(zip(output_names,outputs))
