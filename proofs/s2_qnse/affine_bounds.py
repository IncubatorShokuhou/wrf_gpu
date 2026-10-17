"""Independent MF input-ULP propagation retaining shared input dependence.

An affine form is c + sum(a_i*e_i) + [-r,r], |e_i|<=1. Each original input
ULP has its own symbol. Nonlinear remainders use supremum second derivatives
over the full enclosing interval; arithmetic/library rounding enters r with
the ulp_bounds.py contracts. No reference or candidate output is read.
Uncertain branches use a conservative hull; infinite results are refused.
"""
from contextlib import contextmanager
import math

import numpy as np

from ulp_bounds import Interval, IntervalNumpy, _spacing


class Affine(Interval):
    symbols=0
    next_symbol=0
    symbol_limit=64

    def __init__(self,center,coeff=None,radius=0.,dtype=np.float32):
        self.dtype=np.dtype(dtype)
        self.center=np.asarray(center,np.float64)
        if coeff is None:self.coeff={}
        elif isinstance(coeff,dict):
            self.coeff={k:(float(v) if np.ndim(v)==0 else np.asarray(v,np.float64))
                        for k,v in coeff.items() if (v!=0. if isinstance(v,float) else np.any(v!=0.))}
        else:
            coeff=np.asarray(coeff,np.float64)
            self.coeff={k:coeff[...,k] for k in range(coeff.shape[-1]) if np.any(coeff[...,k]!=0.)}
        self.radius=np.broadcast_to(np.asarray(radius,np.float64),self.center.shape)
        if len(self.coeff)>self.symbol_limit:
            # A representation budget, NOT a tolerance: merging discarded
            # symbols by their full L1 norm can only widen the enclosure.
            order=sorted(self.coeff,key=lambda k:abs(self.coeff[k]) if isinstance(self.coeff[k],float)
                         else float(np.max(np.abs(self.coeff[k]))),reverse=True)
            discarded=order[self.symbol_limit:]
            error=sum((abs(self.coeff[k]) if isinstance(self.coeff[k],float) else np.abs(self.coeff[k])
                       for k in discarded),0.)
            error=np.nextafter(error/(1.-(len(discarded)+1)*np.finfo(np.float64).eps),np.inf)
            self.radius=np.nextafter(self.radius+error,np.inf)
            self.coeff={k:self.coeff[k] for k in order[:self.symbol_limit]}

    @property
    def width(self):
        if not hasattr(self,'_width'):
            total=sum((abs(v) if isinstance(v,float) else np.abs(v) for v in self.coeff.values()),0.)+self.radius
            # Gamma_n for the positive binary64 accumulation, rounded outward.
            n=len(self.coeff)+2;eps=np.finfo(np.float64).eps
            self._width=np.where(total==0.,0.,np.nextafter(total/(1.-n*eps),np.inf))
        return self._width
    @property
    def lo(self):return np.where(self.width==0.,self.center,np.nextafter(self.center-self.width,-np.inf))
    @property
    def hi(self):return np.where(self.width==0.,self.center,np.nextafter(self.center+self.width,np.inf))
    @property
    def shape(self):return self.center.shape
    @property
    def ndim(self):return self.center.ndim

    @classmethod
    def exact(cls,value,dtype=None):
        value=np.asarray(value,dtype=dtype)
        return cls(value,dtype=value.dtype)

    @classmethod
    def hull(cls,lo,hi,dtype):
        if not np.isfinite(lo).all() or not np.isfinite(hi).all():
            return cls(np.zeros(np.broadcast_shapes(np.shape(lo),np.shape(hi))),radius=np.inf,dtype=dtype)
        return cls(.5*(lo+hi),radius=.5*(hi-lo),dtype=dtype)

    def __getitem__(self,index):
        return Affine(self.center[index],{k:np.broadcast_to(v,self.shape)[index] for k,v in self.coeff.items()},self.radius[index],self.dtype)
    def __len__(self):return len(self.center)
    def astype(self,dtype):
        if np.dtype(dtype)==self.dtype:return self
        if np.issubdtype(dtype,np.integer):return Interval(self.lo.astype(dtype),self.hi.astype(dtype),dtype)
        return Affine(self.center,self.coeff,self.radius,dtype)
    def reshape(self,*shape):
        return Affine(self.center.reshape(*shape),{k:np.broadcast_to(v,self.shape).reshape(*shape) for k,v in self.coeff.items()},self.radius.reshape(*shape),self.dtype)

    def _lift(self):
        # A reused expression has the SAME rounding/nonlinear residual each
        # time it is read. Lift its aggregate radius into a shared symbol on
        # first use; otherwise the two nearly cancelling plume-temperature
        # products incorrectly receive independent ~300*delta(entr) errors.
        if not np.isfinite(self.radius).all():return
        for index in np.ndindex(self.shape):
            if self.radius[index]==0.:continue
            if self.shape:
                value=np.zeros(self.shape);value[index]=self.radius[index]
            else:value=float(self.radius)
            self.coeff[Affine.next_symbol]=value;Affine.next_symbol+=1
        self.radius=np.zeros(self.shape)

    @staticmethod
    def _sum_coeff(a,b):
        result=dict(a)
        for key,value in b.items():result[key]=result.get(key,0.)+value
        return {k:v for k,v in result.items() if (v!=0. if isinstance(v,float) else np.any(v!=0.))}

    @staticmethod
    def _scale_coeff(a,scale):
        if np.ndim(scale)==0:scale=float(scale)
        result={k:v*scale for k,v in a.items()}
        return {k:v for k,v in result.items() if (v!=0. if isinstance(v,float) else np.any(v!=0.))}

    def _coerce(self,x):
        if isinstance(x,Affine):return x
        if isinstance(x,Interval):return Affine.hull(x.lo,x.hi,x.dtype)
        return Affine.exact(x,self.dtype)

    def __array_ufunc__(self,ufunc,method,*inputs,**kwargs):
        if method!='__call__' or kwargs:return NotImplemented
        a,b=inputs if len(inputs)==2 else (inputs[0],None)
        functions={np.add:lambda:self._coerce(a)+b,np.subtract:lambda:self._coerce(a)-b,
                   np.multiply:lambda:self._coerce(a)*b,np.divide:lambda:self._coerce(a)/b,
                   np.true_divide:lambda:self._coerce(a)/b,np.power:lambda:self._coerce(a)**b}
        return functions[ufunc]() if ufunc in functions else NotImplemented

    def _finish(self,center,coeff,radius,ulps=.5):
        if not np.isfinite(center).all() or any(not (math.isfinite(v) if isinstance(v,float) else np.isfinite(v).all())
                                               for v in coeff.values()) or not np.isfinite(radius).all():
            return Affine.hull(np.full(np.shape(center),-np.inf),np.full(np.shape(center),np.inf),self.dtype)
        width=sum((abs(v) if isinstance(v,float) else np.abs(v) for v in coeff.values()),0.)+radius
        magnitude=np.abs(center)+width
        noise=ulps*_spacing(magnitude,self.dtype)
        if not np.isfinite(noise).all():
            return Affine.hull(np.full(np.shape(center),-np.inf),np.full(np.shape(center),np.inf),self.dtype)
        # Multiplication by zero and exact cancellation are dealt with before
        # reaching here. Outward binary64 widening is below a REAL4 rounding.
        return Affine(center,coeff,np.nextafter(radius+noise,np.inf),self.dtype)

    def __add__(self,x):
        b=self._coerce(x)
        if getattr(self,'negated_from',None) is b or getattr(b,'negated_from',None) is self:
            return Affine.exact(np.zeros(np.broadcast_shapes(self.shape,b.shape)),self.dtype)
        if np.all(b.center==0.) and np.all(b.width==0.):return self
        if np.all(self.center==0.) and np.all(self.width==0.):return b
        self._lift();b._lift()
        return self._finish(self.center+b.center,self._sum_coeff(self.coeff,b.coeff),self.radius+b.radius)
    __radd__=__add__
    def __neg__(self):
        self._lift()
        result=Affine(-self.center,self._scale_coeff(self.coeff,-1.),self.radius,self.dtype);result.negated_from=self
        return result
    def __sub__(self,x):
        if self is x:return Affine.exact(np.zeros(self.shape),self.dtype)
        return self+(-self._coerce(x))
    def __rsub__(self,x):return self._coerce(x)-self

    def __mul__(self,x):
        b=self._coerce(x)
        if (np.all(self.center==0.) and np.all(self.width==0.)) or (np.all(b.center==0.) and np.all(b.width==0.)):
            return Affine.exact(np.zeros(np.broadcast_shapes(self.shape,b.shape)),self.dtype)
        if np.all(b.center==1.) and np.all(b.width==0.):return self
        if np.all(self.center==1.) and np.all(self.width==0.):return b
        self._lift();b._lift()
        coeff=self._sum_coeff(self._scale_coeff(self.coeff,b.center),self._scale_coeff(b.coeff,self.center))
        rad=np.abs(b.center)*self.radius+np.abs(self.center)*b.radius+self.width*b.width
        return self._finish(self.center*b.center,coeff,rad)
    __rmul__=__mul__

    def unary(self,func,derivative,curvature,ulps):
        self._lift()
        center=func(self.center)
        slope=derivative(self.center)
        radius=np.abs(slope)*self.radius+.5*curvature(self.lo,self.hi)*self.width**2
        return self._finish(center,self._scale_coeff(self.coeff,slope),radius,ulps)

    def reciprocal(self):
        if np.any((self.lo<=0.)&(self.hi>=0.)):return Affine.hull(-np.inf,np.inf,self.dtype)
        return self.unary(lambda x:1./x,lambda x:-1./x**2,
                          lambda lo,hi:2./np.minimum(np.abs(lo),np.abs(hi))**3,2.5)
    def __truediv__(self,x):
        b=self._coerce(x)
        if np.all(b.center==1.) and np.all(b.width==0.):return self
        return self*b.reciprocal()
    def __rtruediv__(self,x):return self._coerce(x)/self

    def __pow__(self,x):
        if isinstance(x,(int,np.integer)) and x>=0:
            result=Affine.exact(np.ones(self.shape),self.dtype);factor=self;power=int(x)
            while power:
                if power&1:result=result*factor
                power//=2
                if power:factor=factor*factor
            if x==2:result.squared_from=self
            return result
        exponent=self._coerce(x)
        if np.any(exponent.width!=0.):raise ValueError('MF power exponent must be exact')
        e=exponent.center
        lo,hi=self.lo,self.hi
        if np.any(lo<=0.):
            with np.errstate(all='ignore'):
                a,b=lo**e,hi**e
            result=Affine.hull(np.minimum(a,b),np.maximum(a,b),self.dtype)
            return result._finish(result.center,result.coeff,result.radius,4.5)
        return self.unary(lambda a:a**e,lambda a:e*a**(e-1.),
            lambda a,b:np.abs(e*(e-1.))*np.maximum(a**(e-2.),b**(e-2.)),4.5)

    def __abs__(self):
        if np.all(self.lo>=0.):return self
        if np.all(self.hi<=0.):return -self
        return Affine.hull(np.zeros(self.shape),np.maximum(np.abs(self.lo),np.abs(self.hi)),self.dtype)


class AffineNumpy(IntervalNumpy):
    @staticmethod
    def asarray(value,dtype=None):
        if isinstance(value,(Affine,Interval)):return value if dtype is None else value.astype(dtype)
        value=np.asarray(value,dtype=dtype)
        return Affine.exact(value) if np.issubdtype(value.dtype,np.floating) else Interval.exact(value)
    @staticmethod
    def zeros_like(a):return Affine.exact(np.zeros(a.shape),a.dtype)
    @staticmethod
    def ones_like(a):return Affine.exact(np.ones(a.shape),a.dtype)
    @staticmethod
    def zeros(shape,dtype=np.float32):return Affine.exact(np.zeros(shape),dtype)
    @staticmethod
    def stack(values,axis=0):
        if not isinstance(values[0],Affine):return IntervalNumpy.stack(values,axis)
        keys=set().union(*(v.coeff for v in values))
        coefficients={k:np.stack([np.broadcast_to(v.coeff.get(k,0.),v.shape) for v in values],axis) for k in keys}
        return Affine(np.stack([v.center for v in values],axis),coefficients,
                      np.stack([v.radius for v in values],axis),values[0].dtype)
    @staticmethod
    def where(condition,a,b):
        c=condition if isinstance(condition,Interval) else Interval.exact(condition,np.bool_)
        anchor=a if isinstance(a,Interval) else b
        if not isinstance(anchor,Affine):return IntervalNumpy.where(c,a,b)
        a=anchor._coerce(a);b=anchor._coerce(b)
        if np.all(c.lo):return a
        if not np.any(c.hi):return b
        lo=np.where(c.lo.astype(bool),a.lo,np.where(~c.hi.astype(bool),b.lo,np.minimum(a.lo,b.lo)))
        hi=np.where(c.lo.astype(bool),a.hi,np.where(~c.hi.astype(bool),b.hi,np.maximum(a.hi,b.hi)))
        return Affine.hull(lo,hi,a.dtype)
    @staticmethod
    def maximum(a,b):
        anchor=a if isinstance(a,Affine) else b
        a=anchor._coerce(a);b=anchor._coerce(b)
        if np.all(a.lo>=b.hi):return a
        if np.all(b.lo>=a.hi):return b
        # Equal exact operands, especially max(0,0), stay exact.
        if np.all(a.width==0.) and np.all(b.width==0.):return Affine.exact(np.maximum(a.center,b.center),a.dtype)
        return Affine.hull(np.maximum(a.lo,b.lo),np.maximum(a.hi,b.hi),a.dtype)
    @staticmethod
    def minimum(a,b):return -AffineNumpy.maximum(-a if isinstance(a,Affine) else -np.asarray(a),-b if isinstance(b,Affine) else -np.asarray(b))
    @staticmethod
    def sqrt(a):
        parent=getattr(a,'squared_from',None)
        if (parent is not None and
                np.all(np.minimum(np.abs(parent.lo),np.abs(parent.hi))>np.sqrt(np.finfo(a.dtype).tiny)) and
                np.all(np.maximum(np.abs(parent.lo),np.abs(parent.hi))<np.sqrt(np.finfo(a.dtype).max))):
            return abs(parent)
        if np.any(a.lo<=0.):
            result=Affine.hull(np.sqrt(np.maximum(a.lo,0.)),np.sqrt(np.maximum(a.hi,0.)),a.dtype)
            return result._finish(result.center,result.coeff,result.radius,.5)
        return a.unary(np.sqrt,lambda x:.5/np.sqrt(x),lambda lo,hi:.25/lo**1.5,.5)
    @staticmethod
    def exp(a):return a.unary(np.exp,np.exp,lambda lo,hi:np.exp(hi),2.5)
    @staticmethod
    def log(a):
        if np.any(a.lo<=0.):return Affine.hull(-np.inf,np.inf,a.dtype)
        return a.unary(np.log,lambda x:1./x,lambda lo,hi:1./lo**2,1.5)


@contextmanager
def affine_backend(*modules):
    previous=[m.jnp for m in modules]
    thermos=[(m,m._thermo_liquid) for m in modules if hasattr(m,'_thermo_liquid')]
    entrainment=[(m,m._entr_detr) for m in modules if hasattr(m,'_entr_detr')]
    try:
        for module in modules:module.jnp=AffineNumpy
        for module,original in thermos:
            module._thermo_liquid=lambda p,thl,rt,c,original=original:_thermo_fixedpoint(p,thl,rt,c,original)
        for module,original in entrainment:
            module._entr_detr=lambda *args,original=original:_inactive_entrainment(original,*args)
        yield
    finally:
        for module,old in zip(modules,previous):module.jnp=old
        for module,old in thermos:module._thermo_liquid=old
        for module,old in entrainment:module._entr_detr=old


def _inactive_entrainment(original,*args):
    # In the ONLY model caller, before=OTEST masks every returned e/d/buoyancy
    # value and preserves active=False. This prunes discarded work in the
    # proof, rather than propagating invalid log arguments from a dead path.
    test,w2=args[0],args[9]
    if not np.any(test.hi):
        zero=Affine.exact(np.zeros(w2.shape),w2.dtype)
        return test,zero,zero,zero
    return original(*args)


def _thermo_fixedpoint(p,thl,rt,c,original):
    """Certify the unchanged state over ALL 20 iterations, or run the source.

    Cold reset: (TH,RV,RC)=(THL,RT,0) each sweep. Warm unsaturated: RLT=0,
    hence RV_NEW=RT, RC_NEW=0 and TH_NEW=THL; this is the same fixed point.
    The branch is taken only if the FULL input/rounding enclosure proves it.
    This avoids creating 20 unrelated noise variables for an identical,
    deterministic calculation. It does not change production iteration count.
    """
    exn=(p/1.e5)**(c.RD/c.CPD)
    temp=thl*exn
    if np.all(temp.hi<=273.15):return thl,rt,Affine.exact(np.zeros(rt.shape),rt.dtype)
    foes=AffineNumpy.exp(c.ALPW-c.BETAW/temp-c.GAMW*AffineNumpy.log(temp))
    qsat=(c.RD/c.RV)*foes/p/(1.+(c.RD/c.RV-1.)*foes/p)
    coef=Affine.exact(.8,rt.dtype)
    rvsat=(1.-coef)*qsat*(1.+rt)+coef*rt
    excess=rt-rvsat
    if np.all(excess.hi<=0.):return thl,rt,Affine.exact(np.zeros(rt.shape),rt.dtype)
    return original(p,thl,rt,c)


def affine_inputs(kwargs):
    fixed={'itimestep','stepbl','scm_force_flux','xland'}
    Affine.symbols=sum(np.size(v) for k,v in kwargs.items() if v is not None and k not in fixed)
    Affine.next_symbol=Affine.symbols
    Affine.symbol_limit=4*max((np.size(v) for v in kwargs.values() if v is not None),default=1)
    result={};offset=0
    for key,value in kwargs.items():
        if value is None:result[key]=None;continue
        a=np.asarray(value,np.float32)
        coeff=np.zeros((*a.shape,Affine.symbols))
        if key not in fixed:
            error=np.where(a==0.,0.,_spacing(a,np.float32))
            for index in np.ndindex(a.shape):
                coeff[(*index,offset)]=error[index];offset+=1
        result[key]=Affine(a,coeff,dtype=np.float32)
    return result
