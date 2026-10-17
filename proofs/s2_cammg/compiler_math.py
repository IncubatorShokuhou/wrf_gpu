"""Proof-only scalar compiler libm backend; product uses ordinary JAX math."""
import math
import numpy as np


class CompilerMath:
    _literal_numpy=True

    def __getattr__(self,name):
        if name in ('log','log10','exp','sin','cos','sqrt','tanh','trunc'):
            func=getattr(math,name)
            return lambda x: np.float64(func(float(x))) if np.ndim(x)==0 else np.vectorize(func,otypes=[np.float64])(x)
        if name=='arccos':return lambda x:np.float64(math.acos(float(x)))
        if name=='arctan':return lambda x:np.float64(math.atan(float(x)))
        return getattr(np,name)

    @staticmethod
    def power(x,y):
        if np.ndim(y)==0 and float(y)==2.:return x*x
        if np.ndim(y)==0 and float(y)==3.:return (x*x)*x
        if np.ndim(x)==0 and np.ndim(y)==0:return np.float64(math.pow(float(x),float(y)))
        return np.vectorize(math.pow,otypes=[np.float64])(x,y)


compiler_math=CompilerMath()
