"""Independent source-input ULP bound for SWR93 flux-divergence heating.

WRF's loops370/1251 compute H=RADCON*(F[k+1]-F[k])/DP[k]. The prior
registered flux budget is16ULP+1e-12, unchanged here. Propagate those two
flux-input budgets through the subtraction/division. PP is a REAL midpoint
(.5*(PRESS[k+1]+PRESS[k])); bound its and DP's rounding independently by
two PP ULPs plus one DP ULP. Include both evaluations' three-operation IEEE
rounding bounds, gamma3, using the source operand norm. No candidate values
enter this bound. Its validity is conditional on the separate flux gate;
it is not a relaxation of that gate or a forecast/D6 qualification.
"""
import numpy as np


def heating_input_bound(flux,pressure,heating,half,radcon):
    values=[np.asarray(v) for v in (flux,pressure,heating,half,radcon)]
    if any(v.dtype!=np.float32 or not np.isfinite(v).all() for v in values):
        raise ValueError('Source bound requires finite native REAL4 operands')
    f,p,h,half,rad=values
    if f.shape!=p.shape or h.shape!=f.shape[:-1]+(f.shape[-1]-1,):
        raise ValueError('Source bound geometry mismatch')
    half=np.broadcast_to(half,f.shape[:-1]);rad=np.broadcast_to(rad,f.shape[:-1])
    if not np.all(half==np.float32(.5)):
        raise ValueError('Source midpoint factor is not native HALF')
    pp=np.zeros_like(p)
    pp[...,-1]=p[...,-1]
    pp[...,1:-1]=half[...,None]*(p[...,1:-1]+p[...,:-2])
    dp=pp[...,1:]-pp[...,:-1]
    ulp=lambda v:np.spacing(np.abs(v)).astype(np.float64)
    bd=2.*(ulp(pp[...,1:])+ulp(pp[...,:-1]))+ulp(dp)
    d=dp.astype(np.float64)
    if not np.all(d>bd):raise ValueError('Unresolved/nonpositive source pressure depth')
    bf=16.*ulp(f)+1.e-12
    scale=np.abs(rad.astype(np.float64))[...,None]/(d-bd)
    u=2.**-24;gamma3=3.*u/(1.-3.*u)
    norm=np.abs(f[...,1:].astype(np.float64))+np.abs(f[...,:-1].astype(np.float64))
    bound=(scale*(bf[...,1:]+bf[...,:-1])+np.abs(h.astype(np.float64))*bd/(d-bd)
           +2.*gamma3*scale*norm+16.*ulp(h)+1.e-12)
    if not np.isfinite(bound).all() or not np.all(bound>0.):
        raise ValueError('Invalid source-input bound')
    return bound
