"""Resident original-WRF cadence and cloud lifetime for cu=7/10/11.

One Pallas program owns each column, including the decision to reuse an active
cloud. Held columns skip the parcel/ensemble/DOUBLE solves inside the program.
The RK consumer finishes countdowns only after all stage consumers have run.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from gpuwrf.backends import pallas as pl
from jax.experimental.pallas import triton as plt


MSKF_PROFILES=('RTHCUTEN','RQVCUTEN','RQCCUTEN','RQRCUTEN','RQICUTEN','RQSCUTEN',
               'CLDFRA_DP_KF','CLDFRA_SH_KF','QC_KF','QI_KF','DUDT','DVDT')
MSKF_SCALARS=('RAINCV','PRATEC','NCA','CUTOP','CUBOT','ISHALL','TIMEC')
INTEGER_FIELDS=frozenset(('ISHALL','CUPFLAG','ENTROPY_FAILED','DLF_CLIPPED','VAPOR_FLOORED'))


def fields(scheme):
    if scheme==11:return MSKF_PROFILES,MSKF_SCALARS
    if scheme==10:
        from gpuwrf.kernels.phys_kfcup_cps import _OUTPUT_PROFILE,_OUTPUT_SCALAR
        return _OUTPUT_PROFILE,_OUTPUT_SCALAR
    if scheme==7:
        from gpuwrf.kernels.phys_camzm_column import _CAM_PROFILE,_CAM_SCALAR
        return _CAM_PROFILE,_CAM_SCALAR
    raise ValueError(f'unsupported cuplus selector {scheme}')


def initial_column(scheme,nz):
    profiles,scalars=fields(scheme)
    out={key:jnp.zeros((nz+1 if key=='CMFMC' else nz,),jnp.float32) for key in profiles}
    out.update({key:jnp.asarray(0,jnp.int32 if key in INTEGER_FIELDS else jnp.float32) for key in scalars})
    if scheme in (10,11):out['NCA']=jnp.float32(-100.)
    return out


def column_step(arrays,w,w0avg,held,controls,step,*,scheme,stepcu,cudt,pbl=9,
                num_bins=1,th_bin_size=1.,r_bin_size=1.,min_deep_freq=1.,
                min_shallow_freq=1.,warm_rain=False,f_qi=True,f_qs=True,small_domain=False,
                interpret=False):
    """Run the original two caller gates with all held REAL fields resident.

    ``arrays``/``controls`` use the corresponding qualified fresh-column ABI;
    MSKF and CuP include W0AVG at position five. W contains nz+1 interfaces.
    ``cudt`` is minutes and ``step`` the original one-based model index.
    Adaptive timesteps are refused by operational admission.
    """
    from gpuwrf.physics import cumulus_kfcup_tables as tables
    from gpuwrf.kernels.phys_mskf_cmt import _ALFA1,_ALFA2,_BSJ0,_BSJ1,_TAU
    from gpuwrf.physics.bl_camuw import ESTBL
    profiles,scalars=fields(scheme);keys=profiles+scalars
    kx=arrays[0].shape[0];size=1<<(kx+2).bit_length();nargs=len(arrays)
    def pad(value):
        value=jnp.asarray(value,jnp.float32)
        return jnp.pad(value,(1,size-value.shape[0]-1))
    inputs=tuple(pad(a) for a in arrays)
    # Triton cannot load a 15-element CAM-ZM control vector. Keep its live
    # entries in place and append an unused lane outside the resident kernel.
    controls=jnp.asarray(controls,jnp.float32)
    ncontrols=controls.shape[0]
    controls=jnp.pad(controls,(0,(1<<(ncontrols-1).bit_length())-ncontrols))
    old=tuple(pad(held[key]) if key in profiles else jnp.asarray(held[key]) for key in keys)
    with jax.enable_x64(True):
        constants=(jnp.asarray(tables.THE0K_R4),jnp.asarray(tables.TTAB_R4),
                   jnp.asarray(tables.QSTAB_R4),jnp.asarray(tables.ALU),
                   *(jnp.asarray(a+(0.,)*6,jnp.float64) for a in (_ALFA1,_ALFA2,_BSJ0,_BSJ1,_TAU)),
                   jnp.asarray(ESTBL+(-99999.,)*6,jnp.float64))
    shapes=(jax.ShapeDtypeStruct((size,),jnp.float32),)+tuple(
        jax.ShapeDtypeStruct((size,) if key in profiles else (),jnp.int32 if key in INTEGER_FIELDS else jnp.float32)
        for key in keys)

    def kernel(*refs):
        a=tuple(ref[...] for ref in refs[:nargs])
        wr,w0r,ctlr,stepr=refs[nargs:nargs+4]
        oldrefs=refs[nargs+4:nargs+4+len(keys)]
        oldout={key:ref[...] for key,ref in zip(keys,oldrefs,strict=True)}
        const=refs[nargs+4+len(keys):nargs+14+len(keys)]
        output_refs=refs[nargs+14+len(keys):]
        step_i=stepr[...]
        outer=(step_i==1)|(cudt==0.)|(stepcu==1)|(step_i%stepcu==0)

        def call(_):
            with jax.enable_x64(False):
                # These are scalars in the WRF caller. Loading the vector then
                # indexing it creates unsupported lax.slice operations.
                ctl=tuple(plt.load(ctlr.at[i]) for i in range(ncontrols))
                dt=ctl[0]
                w0=w0r[...]
                if scheme in (10,11):
                    # Load the next padded interface directly. jnp.roll emits
                    # slice/concatenate primitives unsupported by Triton. The
                    # wrap preserves the old value even on the unused last lane.
                    next_w=plt.load(wr.at[(jnp.arange(size)+1)%size])
                    mass_w=.5*(wr[...]+next_w)
                    denominator=jnp.float32(stepcu*2)
                    w0=(w0*(denominator-1.)+mass_w)/denominator
                    a_use=a[:5]+(w0,)+a[6:]
                    fresh=~(oldout['NCA']>=.5*dt)  # original IF/ELSE, including IEEE nonfinite comparisons
                    if scheme==10:
                        inner=(step_i%stepcu==0)&(step_i>1) if small_domain else ((step_i==1)|(step_i%stepcu==0))
                        fresh=fresh&inner
                else:
                    a_use=a;fresh=jnp.array(True)

                def solve(_):
                    if scheme==11:
                        from gpuwrf.kernels.phys_mskf_column import _mskf_column
                        with jax.enable_x64(True):harmonics=tuple(ref[...] for ref in const[4:9])
                        out=_mskf_column(*a_use[:8],ctl[0],ctl[1],kx,*const[:4],warm_rain,f_qi,f_qs,
                            jnp.float32(0.),ctl[2],ctl[3],ctl[4],ctl[5],jnp.float32(cudt),True,cmt_tables=harmonics)
                        out['RTHCUTEN']=out['DTDT']/a_use[8]
                        # MSKF_CPS leaves NCA unchanged when PARA returns before
                        # creating a cloud (original:3843, 3886, 6617).
                        out['NCA']=jnp.where(out['CUBOT']>out['CUTOP'],oldout['NCA'],out['NCA'])
                    elif scheme==10:
                        from gpuwrf.kernels.phys_kfcup_cps import _cps_column
                        out=_cps_column(a_use,ctl,kx,*const[:4],num_bins=num_bins,
                            th_bin_size=jnp.float32(th_bin_size),r_bin_size=jnp.float32(r_bin_size),
                            min_deep_freq=jnp.float32(min_deep_freq),min_shallow_freq=jnp.float32(min_shallow_freq),
                            warm_rain=warm_rain,f_qi=f_qi,f_qs=f_qs)
                        empty=(out['CUPFLAG']==0)&(out['CUBOT']>out['CUTOP'])
                        out['NCA']=jnp.where(empty,oldout['NCA'],out['NCA'])
                        # Fallback PARA intentionally produces no CuP cloud
                        # diagnostics (original:1178–1186).
                        out['CLDFRA_CUP']=jnp.where(out['CUPFLAG']==0,oldout['CLDFRA_CUP'],out['CLDFRA_CUP'])
                    else:
                        from gpuwrf.kernels.phys_camzm_column import _camzm_cps
                        with jax.enable_x64(True):out=_camzm_cps(a_use,ctl,const[9],kx=kx,pbl=pbl)
                        # Retain original-fatal status and pre-adjustment event
                        # counts until the host output guard/census consumes them.
                        # A later successful call must not hide an earlier fatal.
                        out['ENTROPY_FAILED']=jnp.maximum(oldout['ENTROPY_FAILED'],out['ENTROPY_FAILED'])
                        for key in ('DLF_CLIPPED','VAPOR_FLOORED'):out[key]=oldout[key]+out[key]
                    return tuple(jnp.asarray(out[key],jnp.int32 if key in INTEGER_FIELDS else jnp.float32) for key in keys)
                # CAM-ZM always solves on an admitted outer call. A literal
                # true cond trips JAX 0.10's Triton constant-index lowering.
                out=solve(None) if scheme==7 else jax.lax.cond(
                    fresh,solve,lambda _:tuple(oldout[key] for key in keys),None)
            return (w0,)+out
        result=jax.lax.cond(outer,call,lambda _:(w0r[...],)+tuple(oldout[key] for key in keys),None)
        for ref,value in zip(output_refs,result,strict=True):ref[...]=value

    with jax.enable_x64(True):
        result=pl.pallas_call(kernel,out_shape=shapes,grid=(),interpret=interpret,
            compiler_params=plt.CompilerParams(num_warps=4),name=f'cuplus_{scheme}_caller')(
            *inputs,pad(w),pad(w0avg),jnp.asarray(controls,jnp.float32),jnp.asarray(step,jnp.int32),*old,*constants)
    out={key:value[1:kx+2 if key=='CMFMC' else kx+1] if key in profiles else value
         for key,value in zip(keys,result[1:],strict=True)}
    return result[0][1:kx+1],out


def finish_column(held,dt,*,scheme):
    """advance_ppt after RK consumption, before next-step decoupling."""
    if scheme==7:return held,jnp.float32(dt)*held['PRATEC']
    out=dict(held);nca=held['NCA'];dt=jnp.float32(dt)
    clear=(nca>0)&(jnp.floor(nca/dt+jnp.float32(.5))<=1)
    for key in MSKF_PROFILES[:6]:out[key]=jnp.where(clear,jnp.float32(0.),held[key])
    nca=jnp.where(nca>0,nca-dt,nca);out['NCA']=nca
    if scheme==10:
        # Original:2391–2399 observes NCA AFTER decrement.
        clear_cloud=(nca>0)&(jnp.floor(nca/dt+jnp.float32(.5))<=1)
        out['CLDFRA_CUP']=jnp.where(clear_cloud,jnp.float32(0.),held['CLDFRA_CUP'])
    return out,dt*held['PRATEC']
