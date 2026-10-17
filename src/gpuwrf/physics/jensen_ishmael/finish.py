"""Original final k traversal, preserving its cross-level aggregate density.

Only RHOBAR is read before a current-level assignment on an active path. Axes
and delta are assigned at original2851..2876 before use. Carrying those unused
old values through the scan would add four category arrays without changing
any prognostic or public diagnostic.
"""
import jax
import jax.numpy as jnp
from .math import _real
from .final_checks import final_checks
from . import _native


def finish(fields,tables,held_rhobar,*,xp=jnp):
    _,r=_real(fields['qv'],xp)
    scalar=('theta','qv','qr','nr','pres_e')
    category=('qi','ni','ai','ci')
    state=scalar[:-1]+category
    constants=dict(gammnu=r(6.),i_gammnu=r(1.)/r(6.),i_cp=r(1.)/r(1004.),
                   fourthirdspi=r(4.)/r(3.)*r(3.14159265))
    if _native.active.get():
        level=jnp.arange(fields['qv'].shape[-1],dtype=jnp.int32)
        output={n:fields[n] for n in state}
        def body(k,carry):
            rho,output=carry;selected=level==k
            values={n:jnp.sum(jnp.where(selected,fields[n],jnp.zeros_like(fields[n])),axis=-1) for n in scalar}
            values.update({n:jnp.sum(jnp.where(selected[...,None],fields[n],jnp.zeros_like(fields[n])),axis=-2) for n in category})
            result=final_checks(dict(values,**constants,rhobar=rho),tables,xp=xp)
            output={n:jnp.where(selected if n in scalar else selected[...,None],result[n] if n in scalar else result[n][None,:],output[n]) for n in state}
            return result['rhobar'],output
        rho,output=jax.lax.fori_loop(0,_native.levels.get(),body,(held_rhobar,output))
        return dict(fields,**output,held_rhobar=rho)
    sequence={n:xp.moveaxis(fields[n],-1 if n in scalar else -2,0) for n in scalar+category}

    def point(rho,values):
        incoming=dict(values,**constants,rhobar=rho)
        result=final_checks(incoming,tables,xp=xp)
        return result['rhobar'],{n:result[n] for n in state}

    if xp is jnp:
        rho,output=jax.lax.scan(point,held_rhobar,sequence)
    else:
        rho=held_rhobar;collected={n:[] for n in state}
        for k in range(fields['qv'].shape[-1]):
            rho,values=point(rho,{n:a[k] for n,a in sequence.items()})
            for n,value in values.items():collected[n].append(value)
        output={n:xp.stack(values,axis=0) for n,values in collected.items()}
    result=dict(fields)
    result.update({n:xp.moveaxis(a,0,-1 if n in scalar else -2) for n,a in output.items()})
    result['held_rhobar']=rho
    return result


def last_active_density(processed,*,xp=jnp):
    """Original last active main-loop level leaves its category scratch held."""
    mask=processed['domicro'];nz=mask.shape[-1]
    last=xp.max(xp.where(mask,xp.arange(nz,dtype='int32'),-1),axis=-1)
    index=xp.maximum(last,0)[...,None,None]
    if _native.active.get():
        value=xp.sum(xp.where((xp.arange(nz,dtype='int32')==last)[...,None],processed['rhobar'],xp.zeros_like(processed['rhobar'])),axis=-2)
    else:value=xp.take_along_axis(processed['rhobar'],index,axis=-2)[...,0,:]
    return xp.where((last>=0)[...,None],value,xp.zeros_like(value))
