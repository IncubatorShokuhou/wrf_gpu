"""Initialization-only REAL map transforms used by WRF observation nudging.

Literal set_lc/llij_lc, set_ps/llij_ps and set_merc/llij_merc algebra from
share/module_llxy.F; in4dob adds .5 to the returned mass-grid coordinates.
No JAX, backend query or compiler is needed by this module.
"""
from dataclasses import dataclass
import ctypes
import numpy as np


# gfortran's scalar REAL intrinsics call libm, while NumPy may select SIMD
# approximations that differ by one ulp. Pole subtraction magnifies that into
# 1e-4 grid cells on this WRF map. Keep the original scalar intrinsic boundary
# on the host; this runs only while loading stations, never in the timestep.
_LIBM=ctypes.CDLL(None)
_INTRINSICS={}
for _dtype,_ctype,_suffix in ((np.float32,ctypes.c_float,'f'),(np.float64,ctypes.c_double,'')):
    for _name in ('sin','cos','tan','log','log10','pow'):
        _fn=getattr(_LIBM,_name+_suffix)
        _fn.argtypes=[_ctype]*(2 if _name=='pow' else 1)
        _fn.restype=_ctype
        _INTRINSICS[_dtype,_name]=_fn


@dataclass
class ObservationProjection:
    code: int
    lat1: float
    lon1: float
    truelat1: float
    truelat2: float
    stdlon: float
    dx: float
    dy: float
    dtype: object = np.float32

    def __post_init__(self):
        r=self.dtype
        self.sin=lambda x:r(_INTRINSICS[r,'sin'](float(x)))
        self.cos=lambda x:r(_INTRINSICS[r,'cos'](float(x)))
        self.tan=lambda x:r(_INTRINSICS[r,'tan'](float(x)))
        self.log=lambda x:r(_INTRINSICS[r,'log'](float(x)))
        self.log10=lambda x:r(_INTRINSICS[r,'log10'](float(x)))
        self.power=lambda x,y:r(_INTRINSICS[r,'pow'](float(x),float(y)))
        for key in ('lat1','lon1','truelat1','truelat2','stdlon','dx','dy'):
            setattr(self,key,r(getattr(self,key)))
        self.rad=r(r(3.141592653589793)/r(180))
        self.deg=r(r(180)/r(3.141592653589793))
        self.hemi=r(-1 if self.truelat1<0 else 1)
        self.rebydx=r(6370000)/self.dx
        self.cone=r(0)
        if self.code==1:
            if abs(self.truelat2)>r(90):self.truelat2=self.truelat1
            if abs(self.truelat1-self.truelat2)>r(.1):
                cone=self.log10(self.cos(self.truelat1*self.rad))-self.log10(self.cos(self.truelat2*self.rad))
                self.cone=cone/(self.log10(self.tan((r(45)-abs(self.truelat1)/r(2))*self.rad))
                              -self.log10(self.tan((r(45)-abs(self.truelat2)/r(2))*self.rad)))
            else:self.cone=self.sin(abs(self.truelat1)*self.rad)
            self.rsw=self._lc_radius(self.lat1)
            arg=self.cone*(self._wrap(self.lon1-self.stdlon)*self.rad)
            self.polei=self.hemi*r(1)-self.hemi*self.rsw*self.sin(arg)
            self.polej=self.hemi*r(1)+self.rsw*self.cos(arg)
        elif self.code==2:
            scale=r(1)+self.hemi*self.sin(self.truelat1*self.rad)
            ala=self.lat1*self.rad
            self.rsw=self.rebydx*self.cos(ala)*scale/(r(1)+self.hemi*self.sin(ala))
            alo=(self.lon1-(self.stdlon+r(90)))*self.rad
            self.polei=r(1)-self.rsw*self.cos(alo)
            self.polej=r(1)-self.hemi*self.rsw*self.sin(alo)
        elif self.code==3:
            self.dlon=self.dx/(r(6370000)*self.cos(self.rad*self.truelat1))
            self.rsw=r(0) if self.lat1==r(0) else self.log(self.tan(r(.5)*((self.lat1+r(90))*self.rad)))/self.dlon
        elif self.code==6:
            self.latinc=self.dy*r(360)/(r(2)*r(6370000)*r(3.141592653589793))
            self.loninc=self.dx*r(360)/(r(2)*r(6370000)*r(3.141592653589793))
        else:raise ValueError(f'obs_nudge_opt=1: unsupported ARW map_proj={self.code}')

    def _wrap(self,value):
        r=self.dtype
        value=np.where(value>r(180),value-r(360),value)
        return np.where(value<r(-180),value+r(360),value).astype(self.dtype)

    def _lc_radius(self,lat):
        r=self.dtype
        return (self.rebydx*self.cos(self.truelat1*self.rad)/self.cone
                *self.power(self.tan((r(90)*self.hemi-lat)*self.rad/r(2))
                  /self.tan((r(90)*self.hemi-self.truelat1)*self.rad/r(2)),self.cone))

    def station_coordinates(self,lat,lon,*,mass_offset=True):
        r=self.dtype;lat,lon=r(lat),r(lon)
        if self.code==1:
            rm=self._lc_radius(lat)
            arg=self.cone*(self._wrap(lon-self.stdlon)*self.rad)
            i=(self.polei+self.hemi*rm*self.sin(arg))*self.hemi
            j=(self.polej-rm*self.cos(arg))*self.hemi
        elif self.code==2:
            scale=r(1)+self.hemi*self.sin(self.truelat1*self.rad)
            ala=lat*self.rad
            rm=self.rebydx*self.cos(ala)*scale/(r(1)+self.hemi*self.sin(ala))
            alo=(lon-(self.stdlon+r(90)))*self.rad
            i=self.polei+rm*self.cos(alo);j=self.polej+self.hemi*rm*self.sin(alo)
        elif self.code==3:
            delta=self._wrap(lon-self.lon1)
            i=r(1)+delta/(self.dlon*self.deg)
            j=r(1)+self.log(self.tan(r(.5)*((lat+r(90))*self.rad)))/self.dlon-self.rsw
        else:
            i=r(1)+self._wrap(lon-self.lon1)/self.loninc
            j=r(1)+(lat-self.lat1)/self.latinc
        return (r(i+r(.5)),r(j+r(.5))) if mass_offset else (r(i),r(j))

    def rotate_wind(self,lon,u,v):
        r=self.dtype
        u,v=r(u),r(v)
        if self.code in (1,2):
            diff=self._wrap(self.stdlon-r(lon))
            alpha=diff*self.cone*self.rad*self.hemi if self.code==1 else diff*self.rad*self.hemi
            # WRF assigns REAL expressions to DOUBLE temporaries before writeback;
            # the expression itself is REAL, so no extra precision is introduced.
            udbl=np.float64(v*self.sin(alpha)+u*self.cos(alpha))
            vdbl=np.float64(v*self.cos(alpha)-u*self.sin(alpha))
            u,v=r(udbl),r(vdbl)
        return u,v
