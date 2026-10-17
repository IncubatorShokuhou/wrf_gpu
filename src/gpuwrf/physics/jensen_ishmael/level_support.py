"""Trace-time support for literal Jensen level processes, with explicit operands."""
from functools import reduce
import jax.numpy as jnp
import numpy as np
from .math import _real,gamma,gammln,polysvp,get_igr,capacitance_gamma,wet_growth_check
from .shape import var_check
from .lookup import access_lookup_table
from .vaporgrow import vaporgrow
from .aggregation import aggregation
from . import _native


def _trailing(value,n,xp):
    return xp.asarray(value)[(...,)+(None,)*n]


def _maximum(values,xp):return reduce(xp.maximum,values)
def _minimum(values,xp):return reduce(xp.minimum,values)


def _fortran_nint(a,xp):
    r=lambda v:xp.asarray(v,dtype=xp.asarray(a).dtype)
    return xp.where(a>=r(0.),xp.floor(a+r(.5)),-xp.floor(-a+r(.5))).astype('int32')


def _initial(like,kind,trailing,xp):
    dtype={'real':like.dtype,'integer':np.int32,'logical':np.bool_}[kind]
    if _native.active.get():trailing=tuple(4 if n==3 else 8 if n==6 else n for n in trailing)
    return xp.zeros(like.shape+trailing,dtype=dtype)


def _get(value,index):
    if _native.active.get():
        for i in reversed(index):
            select=jnp.arange(value.shape[-1],dtype=jnp.int32)==jnp.asarray(i,jnp.int32)[...,None]
            if value.dtype==jnp.bool_:
                value=jnp.max(jnp.where(select,value.astype(jnp.int32),jnp.int32(0)),axis=-1)!=0
            else:value=jnp.sum(jnp.where(select,value,jnp.zeros_like(value)),axis=-1,dtype=value.dtype)
        return value
    return value[(Ellipsis,*index)]


def _put(values,name,index,value,mask,xp):
    original=values[name] if index is None else _get(values[name],index)
    extra=max(0,xp.ndim(original)-xp.ndim(mask))
    result=xp.where(_trailing(mask,extra,xp),value,original)
    if index is None:values[name]=result
    elif _native.active.get():
        old=values[name];select=jnp.ones(old.shape[-len(index):],dtype=bool)
        for axis,i in enumerate(index):
            shape=[1]*len(index);shape[axis]=old.shape[old.ndim-len(index)+axis]
            select=select&(jnp.arange(shape[axis],dtype=jnp.int32).reshape(shape)==i)
        values[name]=jnp.where(select,_trailing(result,len(index),jnp),old)
    elif xp is np or not hasattr(values[name],'at'):
        changed=values[name].copy();changed[(Ellipsis,*index)]=result;values[name]=changed
    else:values[name]=values[name].at[(Ellipsis,*index)].set(result)


_VAPOR_NAMES=('dt ani cni rni igr nidum temp rimesum presdum nu alphstr sui sup qvs qvi mu iwci rhodum qidum '
              'dv kt ao nsch npr gammnu i_gammnu fourthirdspi svpi xxls xxlv xxlf capgam vtbarb vtbarbm vtbarbz '
              'anf cnf rnf iwcf fvdum fhdum rbdum dsdum rdout dsdumout').split()
_AGG_NAMES=('dtdum rhoairdum tempdum qdum1 ndum1 ddum1 qdum2 ndum2 ddum2 qdum3 ndum3 ddum3 '
            'rhodum1 rhodum2 phidum1 phidum2').split()


def _call(name,args,tables,xp):
    if name=='var_check':
        # Original incoming RNI is an output; the other original args map
        # exactly to the qualified helper's input names.
        fields=dict(zip(('nu','ao','fourthirdspi','gammnu','qidum','dsdum','ani','cni'),args[:8]))
        fields.update(zip(('rbdum','nidum','aidum','cidum'),args[9:13]))
        out=var_check(**fields,gamma_tab=tables['gamma_tab'],xp=xp)
        return tuple(out[n] for n in ('dsdum','ani','cni','rni','rbdum','nidum','aidum','cidum','alphstr','alphv','betam'))
    if name=='access_lookup_table':return (access_lookup_table(*args[:10],xp=xp),)
    if name=='wet_growth_check':return (wet_growth_check(*args,xp=xp),)
    if name=='vaporgrow':
        out=vaporgrow(dict(zip(_VAPOR_NAMES,args)),tables['gamma_tab'],xp=xp)
        return tuple(out[n] for n in ('vtbarb','vtbarbm','vtbarbz','anf','cnf','rnf','iwcf','fvdum','fhdum','rbdum','rdout','dsdumout'))
    if name=='aggregation':
        out=aggregation(dict(zip(_AGG_NAMES,args[:16])),args[16],args[17],xp=xp)
        return tuple(out[n] for n in ('ddum3','qagg1','qagg2','qagg3','nagg1','nagg2','nagg3'))
    raise ValueError(name)


__all__=['jnp','_real','gamma','gammln','polysvp','get_igr','capacitance_gamma',
         '_initial','_put','_get','_call','_trailing','_maximum','_minimum','_fortran_nint']
