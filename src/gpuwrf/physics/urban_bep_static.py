"""Host-only, REAL-rounded geometry and linear-solve plans for WRF BEP/BEM.

WRF's canyon radiation matrix depends on geometry/material properties only.
Its pivot choices and matrix updates are performed once; the device applies the
same ordered operations to the time-varying RHS, without retaining dense matrices.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gpuwrf.physics.urban_slucm import parse_urbparm, _fortran_reals


@dataclass(frozen=True, eq=False)
class GaussPlan:
    swaps: tuple
    pivots: tuple
    inverse: np.ndarray
    elimination: np.ndarray


def gauss_plan(matrix):
    """Literal matrix half of module_sf_bep.F:gaussj, including last-tie pivot."""
    a=np.array(matrix,copy=True);n=len(a);T=a.dtype.type
    ipiv=np.zeros(n,np.int32);swaps=[];pivots=[];inverse=[];elim=[]
    for _ in range(n):
        big=T(0);irow=icol=0
        for j in range(n):
            if ipiv[j]!=1:
                for k in range(n):
                    if ipiv[k]==0 and abs(a[j,k])>=big:
                        big=abs(a[j,k]);irow=j;icol=k
                    elif ipiv[k]>1:
                        raise ValueError('singular WRF urban radiation matrix')
        ipiv[icol]+=1
        if irow!=icol:
            a[[irow,icol]]=a[[icol,irow]]
        if a[icol,icol]==0:
            raise ValueError('singular WRF urban radiation matrix')
        pivinv=T(1)/a[icol,icol];a[icol,icol]=T(1)
        for l in range(n):a[icol,l]=a[icol,l]*pivinv
        coeff=np.zeros(n,a.dtype)
        for ll in range(n):
            if ll!=icol:
                dum=a[ll,icol];coeff[ll]=dum;a[ll,icol]=T(0)
                for l in range(n):a[ll,l]=a[ll,l]-a[icol,l]*dum
        swaps.append((irow,icol));pivots.append(icol);inverse.append(pivinv);elim.append(coeff)
    return GaussPlan(tuple(swaps),tuple(pivots),np.array(inverse,a.dtype),np.array(elim,a.dtype))


def _fprls(a,b,c,T):
    x=a/c;y=b/c
    if a==0 or b==0:return T(0)
    f=np.log(((T(1)+x**T(2))*(T(1)+y**T(2))/(T(1)+x**T(2)+y**T(2)))**T(.5))
    f=f+y*((T(1)+x**T(2))**T(.5))*np.arctan(y/((T(1)+x**T(2))**T(.5)))
    f=f+x*((T(1)+y**T(2))**T(.5))*np.arctan(x/((T(1)+y**T(2))**T(.5)))
    f=f-y*np.arctan(y)-x*np.arctan(x)
    return f*T(2)/(T(3.141592653)*x*y)


def _fnrms(a,b,c,T):
    x=a/b;y=c/b;z=x**T(2)+y**T(2)
    if x==0 or y==0:return T(0)
    a1=np.log((T(1)+x*x)*(T(1)+y*y)/(T(1)+z))
    a2=y*y*np.log(y*y*(T(1)+z)/z/(T(1)+y*y))
    a3=x*x*np.log(x*x*(T(1)+z)/z/(T(1)+x*x))
    a4=y*np.arctan(T(1)/y);a5=x*np.arctan(T(1)/x)
    a6=np.sqrt(z)*np.arctan(T(1)/np.sqrt(z))
    return (T(.25)*(a1+a2+a3)+a4+a5-a6)/(T(3.141592653)*y)


def view_factors(z,width):
    """module_sf_bep.F:view_factors, index order (emitter, receiver)."""
    n=len(z)-1;T=z.dtype.type;length=T(100000);hut=z[n]
    fww=np.zeros((n,n),z.dtype);fgw=np.zeros(n,z.dtype);fsw=fgw.copy();fwg=fgw.copy();fws=fgw.copy()
    for j in range(n):
        for i in range(n):
            f123=_fprls(length,abs(z[j+1]-z[i]),width,T)
            f23=_fprls(length,abs(z[j+1]-z[i+1]),width,T)
            f12=_fprls(length,abs(z[j]-z[i]),width,T)
            f2=_fprls(length,abs(z[j]-z[i+1]),width,T)
            a123=length*abs(z[j+1]-z[i]);a12=length*abs(z[j]-z[i])
            a23=length*abs(z[j+1]-z[i+1]);a1=length*abs(z[i+1]-z[i])
            a2=length*abs(z[j]-z[i+1]);a3=length*abs(z[j+1]-z[j])
            ftot=T(.5)*(a123*f123-a23*f23-a12*f12+a2*f2)/a1
            fww[i,j]=ftot*a1/a3
        a1=width*length;a12=width*length;a4=(z[j+1]-z[j])*length
        f12=_fnrms(z[j+1],length,width,T);f1=_fnrms(z[j],length,width,T)
        fgw[j]=((a12*f12-a12*f1)/a1)*a1/a4
        f12=_fnrms(hut-z[j],length,width,T);f1=_fnrms(hut-z[j+1],length,width,T)
        fsw[j]=((a12*f12-a12*f1)/a1)*a1/a4
    for i in range(n):
        f12=_fnrms(width,length,hut-z[i],T);f1=_fnrms(width,length,hut-z[i+1],T)
        a1=(z[i+1]-z[i])*length;a2=(hut-z[i+1])*length;a12=(hut-z[i])*length;a4=width*length
        fws[i]=((a12*f12-a2*f1)/a1)*a1/a4
        f12=_fnrms(width,length,z[i+1],T);f1=_fnrms(width,length,z[i],T)
        a2=z[i]*length;a12=z[i+1]*length
        fwg[i]=((a12*f12-a2*f1)/a1)*a1/a4
    return fww,fwg,fgw,fsw,fws,_fprls(length,width,hut,T)


def radiation_matrix(fww,fwg,fgw,pb,wall_reflect,ground_reflect):
    n=len(fwg);T=fgw.dtype.type;a=np.zeros((2*n+1,2*n+1),fgw.dtype)
    for i in range(n):
        a[i,i]=T(1);a[n+i,n+i]=T(1)
        for j in range(n):
            a[i,n+j]=-wall_reflect*fww[j,i]*pb[j+1]
            a[n+i,j]=-wall_reflect*fww[j,i]*pb[j+1]
        a[i,2*n]=-ground_reflect*fgw[i];a[n+i,2*n]=-ground_reflect*fgw[i]
        a[2*n,i]=-wall_reflect*fwg[i]*pb[i+1]
        a[2*n,n+i]=-wall_reflect*fwg[i]*pb[i+1]
    a[2*n,2*n]=T(1)
    return a


@dataclass(frozen=True,eq=False)
class BepClass:
    n: int
    nd: int
    z: np.ndarray
    ss: np.ndarray
    pb: np.ndarray
    width: np.ndarray
    building: np.ndarray
    angle: np.ndarray
    roughness: np.ndarray
    albedo: np.ndarray  # ground,wall,roof
    emissivity: np.ndarray
    capacity: np.ndarray
    diffusivity: np.ndarray
    views: tuple
    long_plans: tuple
    short_plans: tuple
    raw: dict


def read_bep_classes(path,dtype=np.float32):
    """WRF urban_param_init/init_para/icBEP/param for table geometry (no NUDAPT)."""
    table=parse_urbparm(path,dtype=dtype,sf_urban_physics=2);T=np.dtype(dtype).type
    rows=[];heights={};raw={};lines=iter(Path(path).read_text().splitlines())
    for line in lines:
        text=line.strip()
        if not text or text.startswith('#'):continue
        if text.startswith('STREET PARAMETERS:'):
            for row in lines:
                if row.strip()=='END STREET PARAMETERS':break
                if row.strip() and not row.lstrip().startswith('#'):rows.append(_fortran_reals(row))
        elif text.startswith('BUILDING HEIGHTS:'):
            category=int(text.split(':')[1]);values=[]
            for row in lines:
                if row.strip()=='END BUILDING HEIGHTS':break
                if row.strip() and not row.lstrip().startswith('#'):values.append(_fortran_reals(row))
            heights[category]=np.asarray(values,dtype)
        elif ':' in text:
            key,value=text.split(':',1)
            try:raw[key.strip()]=np.asarray(_fortran_reals(value),dtype)
            except ValueError:pass
    classes=[]
    for category in range(table.icate):
        street=np.asarray([row[1:] for row in rows if int(row[0])==category+1],dtype)
        if len(street) not in (1,2):raise ValueError('WRF BEP requires 1 or 2 street directions')
        height=heights[category+1];height=height.copy()
        density=T(0)
        for value in height[:,1]:density=density+value
        height[:,1]=height[:,1]/density
        n=int(np.floor(np.max(height[:,0])/T(5)))+2
        if n>=18:raise ValueError('WRF BEP nz_um=18 too small for URBPARM building heights')
        z=np.arange(n+1,dtype=dtype)*T(5);ss=np.zeros(n+1,dtype);pb=ss.copy();pb[0]=T(1)
        for k in range(n):
            for h,p in height:
                if z[k]<=h and z[k+1]>h:ss[k]=ss[k]+p
            pb[k+1]=max(T(0),pb[k]-ss[k])
        width=street[:,1];building=street[:,2];angle=street[:,0]*T(3.141592653)/T(180)
        rough=np.full((len(street),n),table.z0r[category],dtype);rough[:,0]=table.z0g[category]
        albedo=np.array([table.albg[category],table.albb[category],table.albr[category]],dtype)
        emiss=np.array([table.epsg[category],table.epsb[category],table.epsr[category]],dtype)
        conv=(T(1)/T(4.1868))*T(1.e-6)
        capacity=np.array([table.capg[category]/conv,table.capb[category]/conv,table.capr[category]/conv],dtype)
        conv=(T(1)/T(4.1868))*T(1.e-2)
        diffusivity=np.array([table.aksg[category]/capacity[0]/conv,table.aksb[category]/capacity[1]/conv,table.aksr[category]/capacity[2]/conv],dtype)
        views=tuple(view_factors(z,w) for w in width)
        long_plans=tuple(gauss_plan(radiation_matrix(v[0],v[1],v[2],pb,T(1)-emiss[1],T(1)-emiss[0])) for v in views)
        short_plans=tuple(gauss_plan(radiation_matrix(v[0],v[1],v[2],pb,albedo[1],albedo[0])) for v in views)
        classes.append(BepClass(n,len(street),z,ss,pb,width,building,angle,rough,albedo,emiss,capacity,diffusivity,views,long_plans,short_plans,raw))
    return table,tuple(classes)
