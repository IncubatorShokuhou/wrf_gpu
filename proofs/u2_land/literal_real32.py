"""Diagnostic host execution of unchanged port bodies with scalar REAL4/libm.

Adapted from the wave's proofs/s2_keps/literal_real32.py. This is a source-order
audit, not a runtime backend. Compiled JAX has its own independent ULP gate.
"""
from contextlib import contextmanager
import ctypes

import numpy as np

_libm=ctypes.CDLL("libm.so.6")
for name in ("expf","logf","tanhf","powf","cosf","tanf","acosf","atanf"):
    fn=getattr(_libm,name)
    fn.restype=ctypes.c_float
    fn.argtypes=[ctypes.c_float]*(2 if name=="powf" else 1)
_libm.pow.restype=ctypes.c_double
_libm.pow.argtypes=[ctypes.c_double,ctypes.c_double]


def wrap(value):
    return np.asarray(value).view(Real) if isinstance(value,(np.ndarray,np.generic)) else value


def libm(name,*args):
    dtype=np.float64 if name=="pow" else np.float32
    arrays=np.broadcast_arrays(*(np.asarray(value,dtype) for value in args))
    out=np.empty(arrays[0].shape,dtype)
    for index in np.ndindex(out.shape):
        out[index]=getattr(_libm,name)(*(float(value[index]) for value in arrays))
    return wrap(out)


class Real(np.ndarray):
    __array_priority__=1000
    def __getitem__(self,key):return wrap(super().__getitem__(key))
    def __array_ufunc__(self,ufunc,method,*args,**kwargs):
        integer_power=ufunc is np.power and np.asarray(args[1]).dtype.kind in "iu"
        args=tuple(np.asarray(value) if isinstance(value,Real) else value for value in args)
        if "out" in kwargs:
            kwargs["out"]=tuple(np.asarray(value) if isinstance(value,Real) else value for value in kwargs["out"])
        strong_double=any(isinstance(value,np.ndarray) and value.dtype==np.float64 for value in args)
        if not strong_double and any(isinstance(value,np.ndarray) and value.dtype==np.float32 for value in args):
            args=tuple(np.asarray(value,np.float32) if np.asarray(value).dtype.kind in "ifu" else value for value in args)
        if ufunc is np.power and method=="__call__":
            exponent=np.asarray(args[1])
            if integer_power and np.all(exponent==2):return wrap(args[0]*args[0])
            if integer_power and np.all(exponent==3):return wrap(args[0]*args[0]*args[0])
            if integer_power and np.all(exponent==4):
                square=args[0]*args[0]
                return wrap(square*square)
            return libm("pow" if strong_double else "powf",*args)
        result=getattr(ufunc,method)(*args,**kwargs)
        return tuple(wrap(value) for value in result) if isinstance(result,tuple) else wrap(result)
    @property
    def at(self):return At(self)


class At:
    def __init__(self,value,key=None):self.value,self.key=value,key
    def __getitem__(self,key):return At(self.value,key)
    def set(self,value):
        result=self.value.copy();result[self.key]=value
        return result
    def add(self,value):
        result=self.value.copy();result[self.key]=result[self.key]+value
        return result


class Float32(np.float32):
    """NumPy dtype-compatible constructor whose scalar operations are audited."""
    def __new__(cls,value):return np.asarray(value,np.float32).view(Real)


class NumpyReal:
    def __getattr__(self,name):
        if name=="newaxis":return None
        if name=="float32":return Float32
        if name in ("float64","int32","int64","bool_","integer","floating"):
            return getattr(np,name)
        if name in ("exp","log","tanh","cos","tan","arccos","arctan"):
            if name=="arccos":name="acos"
            if name=="arctan":name="atan"
            return lambda *args:libm(name+"f",*args)
        fn=getattr(np,name)
        def call(*args,**kwargs):
            if name=="where":args=tuple(np.float32(value) if isinstance(value,float) else value for value in args)
            result=fn(*args,**kwargs)
            explicit_dtype=len(args)>1 or kwargs.get("dtype") is not None
            if name=="asarray" and result.dtype==np.float64 and not explicit_dtype:result=result.astype(np.float32)
            return wrap(result)
        return call


@contextmanager
def source_backend():
    from gpuwrf.physics.noahmp import precip_heat,energy_radiation,precision,water_hydro,irrigation
    from gpuwrf.physics.noahmp import surface_options,phenology,soil_thermo,radiation_options,canopy_options,carbon,groundwater_options,crop,soil_options,runoff_options,tile_drain,land_columns
    from gpuwrf.coupling import sst_skin_hook
    modules=(precip_heat,energy_radiation,precision,water_hydro,irrigation,sst_skin_hook,surface_options,phenology,soil_thermo,radiation_options,canopy_options,carbon,groundwater_options,crop,soil_options,runoff_options,tile_drain,land_columns)
    saved=[module.jnp for module in modules]
    saved_lax=groundwater_options.lax
    saved_runoff_lax=runoff_options.lax
    saved_tile_lax=tile_drain.lax
    class LiteralLax:
        @staticmethod
        def fori_loop(start,stop,body,state):
            for k in range(start,stop):state=body(k,state)
            return state
    try:
        for module in modules:module.jnp=NumpyReal()
        groundwater_options.lax=LiteralLax
        runoff_options.lax=LiteralLax
        tile_drain.lax=LiteralLax
        with np.errstate(all="ignore"):yield
    finally:
        for module,original in zip(modules,saved):module.jnp=original
        groundwater_options.lax=saved_lax
        runoff_options.lax=saved_runoff_lax
        tile_drain.lax=saved_tile_lax
