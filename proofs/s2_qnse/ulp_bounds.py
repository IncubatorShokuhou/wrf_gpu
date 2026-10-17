"""Independent forward input-ULP/error enclosure for the QNSE proof packet.

This backend evaluates the literal source with interval arithmetic in binary64,
without reading compiled results or oracle outputs. Nonzero REAL4 input values
start in +/- one input ULP; exact zero and discrete option values stay exact.
Each arithmetic operation widens by its IEEE rounding bound (0.5 REAL4 ULP).
Division allows 2 ULP vs rounded reference (E84), pow 4, exp 2, log 1,
sqrt 0 (correctly rounded); each includes the reference's 0.5 ULP rounding.
Function budgets follow NVIDIA CUDA13.1 mathematical-functions Tables47/48:
https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/05-appendices/mathematical-functions.html
These contracts must be verified for the executed backend before GPU acceptance;
this packet's measurements are CPU-only and do not qualify fast intrinsics.
Selection unions both branches if an input interval crosses its predicate.
An infinite enclosure is REFUSED, never accepted as a numerical tolerance.

The resulting per-field envelope is the independently derived propagation
bound. It has no fitted absolute floors and costs nothing in model stepping.
"""
from __future__ import annotations

import contextlib

import numpy as np


def _spacing(value,dtype):
    with np.errstate(all='ignore'):
        x=np.asarray(np.abs(value),dtype=dtype)
        return np.asarray(np.nextafter(x,np.asarray(np.inf,dtype=dtype))-x,np.float64)


class Interval:
    __array_priority__=10000

    def __init__(self,lo,hi=None,dtype=np.float32):
        self.dtype=np.dtype(dtype)
        self.lo=np.asarray(lo,np.float64)
        self.hi=np.asarray(lo if hi is None else hi,np.float64)

    @classmethod
    def exact(cls,value,dtype=None):
        a=np.asarray(value,dtype=dtype)
        return cls(a,a,a.dtype)

    @classmethod
    def input(cls,value,dtype=np.float32,*,fixed=False):
        a=np.asarray(value,dtype=dtype)
        error=np.where(a==0.,0.,_spacing(a,dtype)) if not fixed else np.zeros(a.shape)
        return cls(a.astype(np.float64)-error,a.astype(np.float64)+error,a.dtype)

    @property
    def shape(self):return self.lo.shape

    @property
    def ndim(self):return self.lo.ndim

    def __len__(self):return len(self.lo)

    def __iter__(self):
        for k in range(len(self)):yield self[k]

    def __array__(self,*args,**kwargs):
        raise TypeError('Interval cannot silently materialize a midpoint')

    def __getitem__(self,index):
        if isinstance(index,Interval):
            low=int(np.min(index.lo));high=int(np.max(index.hi))
            return Interval(np.min(self.lo[low:high+1],axis=0),np.max(self.hi[low:high+1],axis=0),self.dtype)
        return Interval(self.lo[index],self.hi[index],self.dtype)

    def astype(self,dtype):
        dtype=np.dtype(dtype)
        if np.issubdtype(dtype,np.integer):
            return Interval(self.lo.astype(dtype),self.hi.astype(dtype),dtype)
        return Interval(self.lo,self.hi,dtype)

    def reshape(self,*shape):
        return Interval(self.lo.reshape(*shape),self.hi.reshape(*shape),self.dtype)

    def _coerce(self,other):
        return other if isinstance(other,Interval) else Interval.exact(other,self.dtype)

    def _rounded(self,lo,hi,ulps=.5):
        if not np.issubdtype(self.dtype,np.floating):return Interval(lo,hi,self.dtype)
        lo=np.where(np.isnan(lo),-np.inf,lo);hi=np.where(np.isnan(hi),np.inf,hi)
        magnitude=np.maximum(np.abs(lo),np.abs(hi))
        error=np.where(np.isfinite(magnitude),ulps*_spacing(magnitude,self.dtype),0.)
        exact_zero=(lo==0.) & (hi==0.)
        error=np.where(exact_zero,0.,error)
        low=lo-error;high=hi+error
        low=np.where(np.isnan(low),-np.inf,low);high=np.where(np.isnan(high),np.inf,high)
        low=np.where(exact_zero,lo,np.nextafter(low,-np.inf))
        high=np.where(exact_zero,hi,np.nextafter(high,np.inf))
        return Interval(low,high,self.dtype)

    def __add__(self,other):
        b=self._coerce(other)
        if getattr(self,'negated_from',None) is b or getattr(b,'negated_from',None) is self:
            return Interval.exact(np.zeros(np.broadcast_shapes(self.shape,b.shape)),self.dtype)
        if np.all(b.lo==0.) and np.all(b.hi==0.):return self
        if np.all(self.lo==0.) and np.all(self.hi==0.):return b
        return self._rounded(self.lo+b.lo,self.hi+b.hi)
    __radd__=__add__

    def __sub__(self,other):
        b=self._coerce(other)
        if self is b:return Interval.exact(np.zeros(self.shape),self.dtype)
        if np.all(b.lo==0.) and np.all(b.hi==0.):return self
        return self._rounded(self.lo-b.hi,self.hi-b.lo)

    def __rsub__(self,other):return self._coerce(other).__sub__(self)

    def __neg__(self):
        result=Interval(-self.hi,-self.lo,self.dtype)
        result.negated_from=self
        return result

    def __mul__(self,other):
        b=self._coerce(other)
        if np.all(b.lo==1.) and np.all(b.hi==1.):return self
        if np.all(self.lo==1.) and np.all(self.hi==1.):return b
        with np.errstate(all='ignore'):
            corners=np.stack(np.broadcast_arrays(self.lo*b.lo,self.lo*b.hi,self.hi*b.lo,self.hi*b.hi))
        zero=((self.lo==0.) & (self.hi==0.)) | ((b.lo==0.) & (b.hi==0.))
        low=np.min(corners,axis=0);high=np.max(corners,axis=0)
        low=np.where(zero,0.,np.where(np.isnan(low),-np.inf,low))
        high=np.where(zero,0.,np.where(np.isnan(high),np.inf,high))
        return self._rounded(low,high)
    __rmul__=__mul__

    def __truediv__(self,other):
        b=self._coerce(other)
        if np.all(b.lo==1.) and np.all(b.hi==1.):return self
        with np.errstate(all='ignore'):
            corners=np.stack(np.broadcast_arrays(self.lo/b.lo,self.lo/b.hi,self.hi/b.lo,self.hi/b.hi))
            cross=(b.lo<=0.) & (b.hi>=0.)
            low=np.where(cross,-np.inf,np.min(corners,axis=0))
            high=np.where(cross,np.inf,np.max(corners,axis=0))
        return self._rounded(low,high,2.5)

    def __rtruediv__(self,other):return self._coerce(other).__truediv__(self)

    def __pow__(self,exponent):
        if isinstance(exponent,(int,np.integer)) and exponent>=0:
            result=Interval.exact(np.ones(self.shape),self.dtype)
            power=int(exponent);factor=self
            while power:
                if power & 1:result=result*factor
                power//=2
                if power:factor=factor*factor
            if exponent==2:result.squared_from=self
            return result
        b=self._coerce(exponent)
        with np.errstate(all='ignore'):
            values=np.stack(np.broadcast_arrays(self.lo**b.lo,self.lo**b.hi,self.hi**b.lo,self.hi**b.hi))
            low=np.min(values,axis=0);high=np.max(values,axis=0)
        if np.all(b.lo==b.hi) and np.all(b.lo==np.floor(b.lo)) and np.all(b.lo%2==0):
            low=np.where((self.lo<=0.) & (self.hi>=0.) & (b.lo>0.),0.,low)
        low=np.where(np.isnan(low),-np.inf,low);high=np.where(np.isnan(high),np.inf,high)
        return self._rounded(low,high,4.5)

    def __abs__(self):
        if np.all(self.lo>=0.):return self
        if np.all(self.hi<=0.):return -self
        low=np.where((self.lo<=0.) & (self.hi>=0.),0.,np.minimum(np.abs(self.lo),np.abs(self.hi)))
        return Interval(low,np.maximum(np.abs(self.lo),np.abs(self.hi)),self.dtype)

    def _compare(self,other,operation):
        b=self._coerce(other)
        if operation=='gt':low=self.lo>b.hi;high=self.hi>b.lo
        elif operation=='ge':low=self.lo>=b.hi;high=self.hi>=b.lo
        elif operation=='lt':low=self.hi<b.lo;high=self.lo<b.hi
        elif operation=='le':low=self.hi<=b.lo;high=self.lo<=b.hi
        elif operation=='eq':
            low=(self.lo==self.hi) & (b.lo==b.hi) & (self.lo==b.lo)
            high=(self.lo<=b.hi) & (self.hi>=b.lo)
        else:raise ValueError(operation)
        return Interval(low,high,np.bool_)

    def __gt__(self,b):return self._compare(b,'gt')
    def __ge__(self,b):return self._compare(b,'ge')
    def __lt__(self,b):return self._compare(b,'lt')
    def __le__(self,b):return self._compare(b,'le')
    def __eq__(self,b):return self._compare(b,'eq')
    def __and__(self,b):
        b=b if isinstance(b,Interval) else Interval.exact(b,np.bool_)
        return Interval(self.lo.astype(bool)&b.lo.astype(bool),self.hi.astype(bool)&b.hi.astype(bool),np.bool_)
    __rand__=__and__
    def __or__(self,b):
        b=b if isinstance(b,Interval) else Interval.exact(b,np.bool_)
        return Interval(self.lo.astype(bool)|b.lo.astype(bool),self.hi.astype(bool)|b.hi.astype(bool),np.bool_)
    __ror__=__or__
    def __invert__(self):return Interval(~self.hi.astype(bool),~self.lo.astype(bool),np.bool_)

    def __array_ufunc__(self,ufunc,method,*inputs,**kwargs):
        if method!='__call__' or kwargs:return NotImplemented
        a,b=inputs if len(inputs)==2 else (inputs[0],None)
        funcs={np.add:lambda:interval(a,self.dtype)+b,np.subtract:lambda:interval(a,self.dtype)-b,
               np.multiply:lambda:interval(a,self.dtype)*b,np.divide:lambda:interval(a,self.dtype)/b,
               np.true_divide:lambda:interval(a,self.dtype)/b,np.power:lambda:interval(a,self.dtype)**b}
        return funcs[ufunc]() if ufunc in funcs else NotImplemented


def interval(value,dtype=np.float32):
    return value if isinstance(value,Interval) else Interval.exact(value,dtype)


class IntervalNumpy:
    float32=np.float32;float64=np.float64;int32=np.int32;dtype=np.dtype

    @staticmethod
    def asarray(value,dtype=None):
        if isinstance(value,Interval):return value if dtype is None else value.astype(dtype)
        return Interval.exact(value,dtype)

    @staticmethod
    def zeros_like(value):return Interval.exact(np.zeros(value.shape),value.dtype)
    @staticmethod
    def ones_like(value):return Interval.exact(np.ones(value.shape),value.dtype)
    @staticmethod
    def zeros(shape,dtype=np.float32):return Interval.exact(np.zeros(shape,dtype),dtype)
    @staticmethod
    def stack(values,axis=0):
        return Interval(np.stack([v.lo for v in values],axis),np.stack([v.hi for v in values],axis),values[0].dtype)

    @staticmethod
    def where(condition,a,b):
        anchor=a if isinstance(a,Interval) else b
        a=interval(a,anchor.dtype);b=interval(b,anchor.dtype)
        c=condition if isinstance(condition,Interval) else Interval.exact(condition,np.bool_)
        if np.all(c.lo):return a
        if not np.any(c.hi):return b
        low=np.where(c.lo.astype(bool),a.lo,np.where(~c.hi.astype(bool),b.lo,np.minimum(a.lo,b.lo)))
        high=np.where(c.lo.astype(bool),a.hi,np.where(~c.hi.astype(bool),b.hi,np.maximum(a.hi,b.hi)))
        return Interval(low,high,a.dtype)

    @staticmethod
    def maximum(a,b):
        anchor=a if isinstance(a,Interval) else b
        a=interval(a,anchor.dtype);b=interval(b,anchor.dtype)
        return Interval(np.maximum(a.lo,b.lo),np.maximum(a.hi,b.hi),a.dtype)

    @staticmethod
    def minimum(a,b):
        anchor=a if isinstance(a,Interval) else b
        a=interval(a,anchor.dtype);b=interval(b,anchor.dtype)
        return Interval(np.minimum(a.lo,b.lo),np.minimum(a.hi,b.hi),a.dtype)

    @classmethod
    def clip(cls,a,lo,hi):return cls.minimum(cls.maximum(a,lo),hi)

    @staticmethod
    def sqrt(a):
        parent=getattr(a,'squared_from',None)
        # IEEE round-to-nearest sqrt(round(x*x)) recovers |x| when x*x
        # stays normal/finite. This preserves the exact zero-energy BL89
        # cancellation instead of inventing a loose interval at its LUP=0.
        # Requires correctly rounded sqrt; native fast-sqrt is unqualified.
        if parent is not None:
            magnitude=np.maximum(np.abs(parent.lo),np.abs(parent.hi))
            minimum=np.where((parent.lo<=0.)&(parent.hi>=0.),0.,np.minimum(np.abs(parent.lo),np.abs(parent.hi)))
            info=np.finfo(parent.dtype)
            if np.all(minimum>np.sqrt(info.tiny)) and np.all(magnitude<np.sqrt(info.max)):
                return abs(parent)
        with np.errstate(all='ignore'):
            lo=np.sqrt(np.maximum(a.lo,0.));hi=np.sqrt(a.hi)
        lo=np.where(a.hi<0.,-np.inf,lo);hi=np.where(a.hi<0.,np.inf,hi)
        return a._rounded(lo,hi,.5)

    @staticmethod
    def exp(a):return a._rounded(np.exp(a.lo),np.exp(a.hi),2.5)
    @staticmethod
    def log(a):
        with np.errstate(all='ignore'):lo=np.log(a.lo);hi=np.log(a.hi)
        lo=np.where(a.lo<=0.,-np.inf,lo);hi=np.where(a.hi<=0.,np.inf,hi)
        return a._rounded(lo,hi,1.5)
    @staticmethod
    def abs(a):return abs(a)
    @staticmethod
    def signbit(a):
        exact=(a.lo==a.hi)
        return Interval(np.where(exact,np.signbit(a.lo),a.hi<0.),
                        np.where(exact,np.signbit(a.hi),a.lo<0.),np.bool_)


@contextlib.contextmanager
def interval_backend(*modules):
    previous=[m.jnp for m in modules]
    interpolators=[(m,m._interp_psi) for m in modules if hasattr(m,'_interp_psi')]
    try:
        for module in modules:module.jnp=IntervalNumpy
        # Preserve the correlation between the integer table bin and its
        # fractional coordinate. Independent intervals for table[k+1] and
        # table[k] otherwise invent jumps that do not exist in WRF's linear
        # interpolant. Enumerate every reachable bin, never a sampled subset.
        for module,_ in interpolators:module._interp_psi=_interp_psi_enclosure
        yield
    finally:
        for module,old in zip(modules,previous):module.jnp=old
        for module,old in interpolators:module._interp_psi=old


def _interp_psi_enclosure(table,zeta,ztmin,dzeta):
    rz=(zeta-ztmin)/dzeta
    if not np.isfinite(rz.lo).all() or not np.isfinite(rz.hi).all():
        return Interval(-np.inf,np.inf,zeta.dtype)
    lo=int(np.min(rz.lo.astype(np.int32)));hi=int(np.max(rz.hi.astype(np.int32)))
    values=[]
    for raw in range(lo,hi+1):
        # rz is nonnegative at every caller, from WRF's ZTMIN clamp.
        # Include endpoints conservatively at integer transitions.
        restricted=Interval(max(float(rz.lo),raw),min(float(rz.hi),raw+1),zeta.dtype)
        fraction=restricted-Interval.exact(raw,zeta.dtype)
        k=min(max(raw,0),9999)
        value=(table[k+1]-table[k])*fraction+table[k]
        values.append(value)
    return Interval(min(float(v.lo) for v in values),max(float(v.hi) for v in values),zeta.dtype)


def input_intervals(kwargs):
    fixed={'itimestep','stepbl','scm_force_flux','xland'}
    return {key:(None if value is None else Interval.input(value,fixed=key in fixed))
            for key,value in kwargs.items()}
