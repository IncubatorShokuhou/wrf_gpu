"""One program per column: all five adaptive sedimentation passes on device.

The Pallas path shares every equation code object with the oracle-qualified
column transcription. Only table/neighbor/point access changes. Existing qg/qh
output buffers provide the two neighbor readbacks; no extra domain arrays.
"""
from types import FunctionType
import jax
import jax.numpy as jnp
from gpuwrf.backends import pallas as pl
from jax.experimental.pallas import triton as plt
from . import column as col, equations as eq
from .support import C,F

CORE=('theta','qv','qc','qr','qi','qs','qg','qh')
RADII=tuple('re_'+name+'_gsfc' for name in ('cloud','rain','ice','snow','graupel','hail'))
AC=tuple('acphys'+name for name in 'cedsmf')
INSTANT=tuple('phys'+name for name in 'cedsmf')
FLUX=('preci3d','precs3d','precg3d','prech3d','precr3d')
SURFACE=('rainnc','snownc','graupelnc','hailnc','sr','rain_other')
TABLES=('aa1','aa2','rn12a','rn12b','rn13','rn25a','bergcon1','bergcon2','bergcon3','bergcon4')


def _copy(fn,env):
    result=FunctionType(fn.__code__,env,fn.__name__,fn.__defaults__,fn.__closure__)
    result.__kwdefaults__=fn.__kwdefaults__
    return result


def native_column(theta,qv,qc,qr,qi,qs,qg,qh,rho,exner,pressure,dz,z,w,
                  dt,dx=3000.,xland=1.,topo=0.,itimestep=1,diagnostics=None,
                  *,interpret=False,full=False):
    """Inputs (z,y,x), surface xland/topo (y,x); return retained WRF carries.

    ``full=True`` additionally returns the Registry '-' diagnostics for source
    qualification. They are elided from normal forecasts.
    """
    inputs=tuple(jnp.asarray(v,jnp.float32) for v in
                 (theta,qv,qc,qr,qi,qs,qg,qh,rho,exner,pressure,dz,z,w))
    nz,ny,nx=inputs[0].shape
    if any(v.shape!=(nz,ny,nx) for v in inputs):
        raise ValueError('Goddard4ice inputs must share (z,y,x) shape')
    xland=jnp.broadcast_to(jnp.asarray(xland,jnp.float32),(ny,nx))
    topo=jnp.broadcast_to(jnp.asarray(topo,jnp.float32),(ny,nx))
    diagnostics={} if diagnostics is None else diagnostics
    inputs+=tuple(jnp.asarray(diagnostics.get(key,jnp.zeros_like(theta)),jnp.float32) for key in AC)
    inputs+=(jnp.asarray(dt,jnp.float32),jnp.asarray(dx,jnp.float32),xland,topo,jnp.asarray(itimestep,jnp.int32))
    table_args=tuple(jnp.asarray(C[key],jnp.float32) for key in TABLES)
    columns=(*CORE,*RADII,*AC,*INSTANT,*FLUX) if full else (*CORE,*RADII,*AC)
    columns=(*columns,"phys_tot")
    output_names=(*columns,*SURFACE)
    shapes=tuple(jax.ShapeDtypeStruct((nz,ny,nx),jnp.float32) for key in columns)
    shapes+=tuple(jax.ShapeDtypeStruct((ny,nx),jnp.float32) for key in SURFACE)
    padded=pl.next_power_of_2(nz)

    def kernel(*refs):
        src=refs[:25]
        tables=dict(zip(TABLES,refs[25:25+len(TABLES)]))
        dst=dict(zip(output_names,refs[25+len(TABLES):]))
        y,x=pl.program_id(0),pl.program_id(1)
        k=jnp.arange(padded,dtype=jnp.int32)
        valid=k<nz
        values=[plt.load(ref.at[k,y,x],mask=valid,other=F(1) if i in (0,8,9,10,11) else F(0))
                for i,ref in enumerate(src[:14])]
        ac={key:plt.load(ref.at[k,y,x],mask=valid,other=F(0)) for key,ref in zip(AC,src[14:20])}
        step_dt=plt.load(src[20].at[()]);step_dx=plt.load(src[21].at[()])
        land=plt.load(src[22].at[y,x]);terrain=plt.load(src[23].at[y,x])
        step=plt.load(src[24].at[()])

        def lookup(name,index):
            index=jnp.broadcast_to(jnp.asarray(index,jnp.int32),k.shape)
            return plt.load(tables[name].at[index-1],mask=valid&(index>=1)&(index<=31),other=F(0))

        def shift(value,name):
            if name in ('qh_original','qg_original'):
                ref=src[7 if name=='qh_original' else 6]
            elif name=='qg_current':
                # sgmap option2 never reads its qcg2 argument; no access needed.
                return value
            else:
                ref=dst['qh' if name=='qhwrf' else 'qg']
            return plt.load(ref.at[k+1,y,x],mask=(k+1)<nz,other=F(0))

        def point(value,index):
            return jnp.sum(jnp.where(k==index,value,F(0)))

        def distance(value,surface):
            previous=plt.load(src[12].at[k-1,y,x],mask=valid&(k>0),other=surface)
            return value-previous

        def prepare(graupel,hail):
            plt.store(dst['qg'].at[k,y,x],graupel,mask=valid)
            plt.store(dst['qh'].at[k,y,x],hail,mask=valid)
            if not interpret:
                plt.debug_barrier()

        def saturation_ratio(numerator, denominator):
            if interpret:
                return numerator / denominator
            # An approximate positive SSI at exact saturation selects WRF's
            # 0/0 deposition normalization (:4063-4065). Preserve REAL IEEE
            # division here; do not change the physical branch or clamp rates.
            return plt.elementwise_inline_asm(
                "div.rn.f32 $0, $1, $2;", args=[numerator, denominator],
                constraints="=f,f,f", pack=1,
                result_shape_dtypes=[jax.ShapeDtypeStruct(numerator.shape,jnp.float32)])[0]

        eqenv=dict(vars(eq),table_lookup=lookup,shift_up=shift,
                   _saturation_ratio=saturation_ratio)
        # Avoid a gather for the fixed k=1 reference density.
        eqenv['first_density']=lambda value:plt.load(src[8].at[0,y,x])
        for name in ('sgmap','vqrqi','saticel_s'):
            eqenv[name]=_copy(getattr(eq,name),eqenv)
        env=dict(vars(col),shift_up=shift,_point=point,_distance=distance,_prepare_saticel=prepare)
        env.update({name:eqenv[name] for name in ('sgmap','vqrqi','saticel_s')})
        env['_fall_species']=_copy(col._fall_species,env)
        calculate=_copy(col.column,env)
        result=calculate(*values,step_dt,step_dx,land,terrain,step,ac,_physical_nz=nz)
        for key in columns:
            plt.store(dst[key].at[k,y,x],result[key],mask=valid)
        for key in SURFACE:
            plt.store(dst[key].at[y,x],result[key])

    result=pl.pallas_call(kernel,out_shape=shapes,grid=(ny,nx),interpret=interpret,
        compiler_params=plt.CompilerParams(num_warps=4),name='goddard4ice_columns')(*inputs,*table_args)
    return dict(zip(output_names,result))
