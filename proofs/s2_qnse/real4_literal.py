"""Unfused IEEE REAL4 evaluator for the SAME Python column source.

NumPy/XLA elementary-function approximations need not return the same last bit
as gfortran's libm expf/logf/powf. This proof-only backend uses that scalar libm
ABI and rounds each arithmetic operation to REAL4; no production imports it.
"""
from __future__ import annotations

import contextlib
import ctypes
import ctypes.util

import numpy as np

_libm=ctypes.CDLL(ctypes.util.find_library('m'))
_functions={}
for _name,_argc in (('expf',1),('logf',1),('sqrtf',1),('powf',2)):
    _fn=getattr(_libm,_name)
    _fn.argtypes=[ctypes.c_float]*_argc
    _fn.restype=ctypes.c_float
    _functions[_name]=_fn


def _elementary(name,*args):
    args=np.broadcast_arrays(*(np.asarray(a,np.float32) for a in args))
    out=np.empty(args[0].shape,np.float32)
    for index in np.ndindex(out.shape):
        out[index]=_functions[name](*(float(a[index]) for a in args))
    return out


class Real4(np.ndarray):
    __array_priority__=10000

    def __new__(cls,value,dtype=None):
        return np.asarray(value,dtype=dtype).view(cls)

    def __getitem__(self,index):
        return Real4(np.ndarray.__getitem__(self,index))

    def __iter__(self):
        for index in range(len(self)):yield self[index]

    def __pow__(self,exponent):
        # Fortran INTEGER powers round each multiply; powf(x,3) rounds the
        # exact cubic only once and is therefore a different REAL operation.
        if isinstance(exponent,(int,np.integer)) and exponent>=0:
            if exponent==0:return Real4(np.ones(self.shape,self.dtype))
            result=Real4(np.ones(self.shape,self.dtype));factor=self
            power=int(exponent)
            while power:
                if power & 1:result=result*factor
                power//=2
                if power:factor=factor*factor
            return result
        return np.power(self,exponent)

    def __array_ufunc__(self,ufunc,method,*inputs,**kwargs):
        if method!='__call__':return NotImplemented
        # Python literals are WRF REAL operands of the surrounding expression.
        floating=any(isinstance(a,Real4) and np.issubdtype(a.dtype,np.floating) for a in inputs)
        dtype=np.float32 if floating else self.dtype
        arrays=[np.asarray(a,dtype=dtype) if isinstance(a,(float,int,np.floating)) else np.asarray(a)
                for a in inputs]
        if ufunc==np.power and floating:
            out=_elementary('powf',*arrays)
        elif ufunc in (np.exp,np.log,np.sqrt) and floating:
            out=_elementary({np.exp:'expf',np.log:'logf',np.sqrt:'sqrtf'}[ufunc],*arrays)
        else:
            out=ufunc(*arrays,**kwargs)
        return Real4(out)


class Real4Numpy:
    float32=np.float32;float64=np.float64;int32=np.int32;dtype=np.dtype
    @staticmethod
    def asarray(a,dtype=None):return Real4(a,dtype)
    @staticmethod
    def zeros_like(a):return Real4(np.zeros(a.shape,a.dtype))
    @staticmethod
    def ones_like(a):return Real4(np.ones(a.shape,a.dtype))
    @staticmethod
    def zeros(shape,dtype=np.float32):return Real4(np.zeros(shape,dtype))
    @staticmethod
    def stack(values,axis=0):return Real4(np.stack([np.asarray(a) for a in values],axis=axis))
    @staticmethod
    def where(condition,a,b):
        anchor=a if isinstance(a,Real4) else b
        dtype=getattr(anchor,'dtype',None)
        return Real4(np.where(np.asarray(condition),np.asarray(a,dtype=dtype),np.asarray(b,dtype=dtype)))
    @staticmethod
    def maximum(a,b):
        anchor=a if isinstance(a,Real4) else b
        dtype=getattr(anchor,'dtype',None)
        return Real4(np.maximum(np.asarray(a,dtype=dtype),np.asarray(b,dtype=dtype)))
    @staticmethod
    def minimum(a,b):
        anchor=a if isinstance(a,Real4) else b
        dtype=getattr(anchor,'dtype',None)
        return Real4(np.minimum(np.asarray(a,dtype=dtype),np.asarray(b,dtype=dtype)))
    @classmethod
    def clip(cls,a,lo,hi):return cls.minimum(cls.maximum(a,lo),hi)
    @staticmethod
    def sqrt(a):return Real4(_elementary('sqrtf',a))
    @staticmethod
    def exp(a):return Real4(_elementary('expf',a))
    @staticmethod
    def log(a):return Real4(_elementary('logf',a))
    @staticmethod
    def abs(a):return Real4(np.abs(np.asarray(a)))
    @staticmethod
    def signbit(a):return Real4(np.signbit(np.asarray(a)))


@contextlib.contextmanager
def real4_backend(*modules):
    previous=[m.jnp for m in modules]
    try:
        for module in modules:module.jnp=Real4Numpy
        yield
    finally:
        for module,old in zip(modules,previous):module.jnp=old
