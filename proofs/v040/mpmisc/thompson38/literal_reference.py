"""Literal REAL executor of the production mp38 equation code objects.

Only array/lax primitives and state storage are replaced. Physics functions,
including shared Thompson/Aero helpers and HailContext methods, retain their
exact code objects. The independent reference remains pristine Fortran.
"""
from contextlib import contextmanager
import ctypes
import importlib.util
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace

import numpy as np

_spec = importlib.util.spec_from_file_location('shared_ferrier_literal', Path(__file__).resolve().parents[1]/'literal_reference.py')
_shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared)
_RealArray, _LiteralNumpy, _LiteralLax, _wrap, _LIBM = (_shared._RealArray, _shared._LiteralNumpy, _shared._LiteralLax, _shared._wrap, _shared._LIBM)
from gpuwrf.physics import thompson_hail as hail
from gpuwrf.physics.thompson_hail import column, context, native
from gpuwrf.physics import thompson_saturation


for name in ('log10', 'cos'):
    for suffix, dtype in (('f', ctypes.c_float), ('', ctypes.c_double)):
        fn = getattr(_LIBM, name+suffix)
        fn.argtypes, fn.restype = [dtype], dtype


class LiteralNumpy(_LiteralNumpy):
    @staticmethod
    def log10(value):
        array = np.asarray(value)
        fn = _LIBM.log10f if array.dtype == np.float32 else _LIBM.log10
        return _wrap(np.asarray([fn(float(x)) for x in array.ravel()],dtype=array.dtype).reshape(array.shape))

    @staticmethod
    def cos(value):
        array = np.asarray(value)
        fn = _LIBM.cosf if array.dtype == np.float32 else _LIBM.cos
        return _wrap(np.asarray([fn(float(x)) for x in array.ravel()],dtype=array.dtype).reshape(array.shape))


class LiteralLax(_LiteralLax):
    @staticmethod
    def scan(body, carry, fields, length=None):
        outputs=[]
        size=length if fields is None else (fields[0].shape[0] if isinstance(fields,tuple) else fields.shape[0])
        for i in range(size):
            item=None if fields is None else tuple(a[i] for a in fields) if isinstance(fields,tuple) else fields[i]
            carry,result=body(carry,item)
            outputs.append(result)
        def stack(values):
            if values[0] is None: return None
            if isinstance(values[0],tuple): return tuple(stack([v[i] for v in values]) for i in range(len(values[0])))
            return _wrap(np.stack(values))
        return carry,stack(outputs)


class LiteralState:
    __slots__=column.ThompsonHailColumnState.__slots__

    def __init__(self, **values):
        for name in self.__slots__:
            setattr(self,name,_wrap(np.asarray(values.get(name,np.zeros_like(values['qv'])),np.float32)))

    def replace(self, **updates):
        return type(self)(**{name:updates.get(name,getattr(self,name)) for name in self.__slots__})


def _copy(fn, env):
    result=FunctionType(fn.__code__,env,fn.__name__,fn.__defaults__,fn.__closure__)
    result.__kwdefaults__=fn.__kwdefaults__
    return result


def _host(value):
    if isinstance(value,tuple) and hasattr(value,'_fields'):
        return type(value)(*(_host(a) for a in value))
    if isinstance(value,np.ndarray) or hasattr(value,'__jax_array__') or (hasattr(value,'dtype') and isinstance(getattr(value,'shape',None),tuple)):
        return _wrap(np.asarray(value))
    return value


@contextmanager
def executor():
    numpy=LiteralNumpy()
    fakejax=SimpleNamespace(lax=LiteralLax,dtypes=SimpleNamespace(canonicalize_dtype=lambda dtype:dtype))
    sat_env=dict(vars(thompson_saturation),jnp=numpy)
    for name,fn in tuple(sat_env.items()):
        if isinstance(fn,FunctionType) and fn.__module__==thompson_saturation.__name__:
            sat_env[name]=_copy(fn,sat_env)

    def copy_namespace(namespace):
        env={name:_host(value) for name,value in vars(namespace).items()}
        env.update(jnp=numpy,jax=fakejax)
        for name in ('latent_heat_vaporization','cp_inverse','saturation_mixing_ratio_liquid','saturation_mixing_ratio_ice'):
            if name in env: env[name]=sat_env[name]
        for name,fn in tuple(env.items()):
            if isinstance(fn,FunctionType) and fn.__module__ in ('gpuwrf.physics.thompson_column','gpuwrf.physics.thompson_aero_column'):
                env[name]=_copy(fn,env)
        return env

    tc_env=copy_namespace(native.TC)
    tc=SimpleNamespace(**tc_env)
    aero_env=copy_namespace(native.AERO)
    # Imported shared TC helpers must use the native/literal TC environment.
    for name,fn in vars(native.AERO).items():
        if isinstance(fn,FunctionType) and fn.__module__=='gpuwrf.physics.thompson_column' and name in tc_env:
            aero_env[name]=tc_env[name]
    aero=SimpleNamespace(**aero_env)
    hail_env=dict(vars(hail),jnp=numpy,jax=fakejax,tc=tc)
    for name,fn in tuple(hail_env.items()):
        if isinstance(fn,FunctionType) and fn.__module__==hail.__name__:
            hail_env[name]=_copy(fn,hail_env)
    tables=_host(native.THOMPSON_AERO_TABLES)
    real_tables=_copy(native.real_tables,dict(vars(native),jnp=numpy,THOMPSON_AERO_TABLES=tables))
    ctx_env=dict(vars(context),jnp=numpy,tc=tc,aero=aero,real_tables=real_tables)
    ctx_env.update({name:hail_env[name] for name in ('entry_distribution','volume_tendency')})
    methods={name:_copy(fn,ctx_env) if isinstance(fn,FunctionType) else fn for name,fn in vars(context.HailContext).items() if name not in ('__dict__','__weakref__')}
    literal_context=type('LiteralHailContext',(),methods)
    env=dict(vars(column),jnp=numpy,jax=fakejax,tc=tc,aero=aero,HailContext=literal_context)
    env.update({name:hail_env[name] for name in ('entry_distribution','fall_speeds','sediment_hail','final_balance')})
    for name,fn in tuple(env.items()):
        if isinstance(fn,FunctionType) and fn.__module__==column.__name__:
            env[name]=_copy(fn,env)
    # One import inside the shared TC body resolves this identical code object
    # through the package. Temporarily bind its literal array environment.
    saved=hail.volume_tendency
    hail.volume_tendency=hail_env['volume_tendency']
    try:
        with np.errstate(invalid='ignore',divide='ignore',over='ignore',under='ignore'):
            yield env['_body']
    finally:
        hail.volume_tendency=saved
