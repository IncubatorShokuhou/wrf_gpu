"""Validation-only Fortran libm and no-FMA REAL arithmetic.

Adapted from the independently qualified P3 literal executor. This executes
the port expressions, never calls original NTU, and never changes references.
"""
import ctypes
import numpy as np

LIBM=ctypes.CDLL("libm.so.6")


def intrinsic(name,*values):
    dtype=np.result_type(*(np.asarray(v).dtype for v in values))
    real=ctypes.c_double if dtype==np.float64 else ctypes.c_float
    fn=getattr(LIBM,name+("" if real==ctypes.c_double else "f"))
    fn.argtypes=[real]*len(values);fn.restype=real
    arrays=np.broadcast_arrays(*map(np.asarray,values))
    result=np.fromiter((fn(*x) for x in zip(*(a.flat for a in arrays))),dtype=dtype,count=arrays[0].size)
    return result.reshape(arrays[0].shape).view(Real)


class Real(np.ndarray):
    __array_priority__=1000

    def __rpow__(self,base):
        return intrinsic('pow',np.asarray(base,dtype=self.dtype),self)

    def __pow__(self,exponent):
        e=np.asarray(exponent)
        if e.ndim==0 and np.issubdtype(e.dtype,np.integer) and int(e)>=0:
            n=int(e)
            if n==0:return np.ones_like(self)
            if n==1:return self
            if n==2:return self*self
            if n==3:return (self*self)*self
            if n in (4,6):
                v=self**(n//2);return v*v
            if n==5:return (self**4)*self
        return intrinsic('pow',self,np.asarray(exponent,dtype=self.dtype))


class LiteralNumpy:
    literal=True

    def power(self,base,exponent):
        # NumPy scalars bypass ndarray.__rpow__; dispatch explicitly so both
        # field-base and constant-base REAL powers use the same host powf.
        return self.asarray(base).__pow__(exponent)

    def __getattr__(self,name):
        if name in ('exp','log','log10','sqrt','sin','cos','tanh','arccos','arctan','erf'):
            return lambda x:intrinsic({'arccos':'acos','arctan':'atan'}.get(name,name),x)
        fn=getattr(np,name)
        if not callable(fn) or isinstance(fn,type):return fn
        def wrap(value):
            if isinstance(value,tuple):return tuple(map(wrap,value))
            if isinstance(value,np.ndarray) and np.issubdtype(value.dtype,np.floating):return value.view(Real)
            return value
        return lambda *a,**kw:wrap(fn(*a,**kw))

    def asarray(self,value,dtype=None):
        array=np.asarray(value,dtype=dtype)
        return array.view(Real) if np.issubdtype(array.dtype,np.floating) else array
