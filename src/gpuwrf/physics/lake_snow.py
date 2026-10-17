"""Literal WRF lake snow water, compaction and layer restructuring (DOUBLE)."""
from jax import numpy as jnp
from .lake_fluxes import TFRZ, CPICE, CPLIQ, real, at, set_at
from .lake_thermodynamics import lake_combine


def geometry(s,enabled):
    z,zi = s.z3d,s.zi3d
    for j in range(4,-1,-1):
        active = enabled&(j > s.snl2d+4)
        z = z.at[...,j].set(jnp.where(active,zi[...,j+1]-.5*s.dz3d[...,j],z[...,j]))
        zi = zi.at[...,j].set(jnp.where(active,zi[...,j+1]-s.dz3d[...,j],zi[...,j]))
    return s._replace(z3d=z,zi3d=zi)


def snow_water(s,dt,rain,sub,evap,dew_snow,dew):
    snow = s.snl2d < 0
    top = s.snl2d.astype(jnp.int32)+5
    wi,wl = s.h2osoi_ice3d,s.h2osoi_liq3d
    original_ice = at(wi,top)
    change = original_ice+(dew_snow-sub)*dt
    new_liq = jnp.maximum(0.,at(wl,top)+jnp.minimum(change,0.)+(rain+dew-evap)*dt)
    wi = set_at(wi,top,jnp.where(snow,jnp.maximum(change,0.),original_ice))
    wl = set_at(wl,top,jnp.where(snow,new_liq,at(wl,top)))
    active = jnp.arange(5) >= top[...,None]
    dz = jnp.where(active,s.dz3d[...,:5],1.)
    vi = jnp.minimum(1.,wi[...,:5]/(dz*917.))
    porosity = 1.-vi
    vl = jnp.minimum(porosity,wl[...,:5]/(dz*1000.))
    qin = jnp.zeros_like(dt)
    for j in range(5):
        enabled = snow&(j >= top)
        current = wl[...,j]+qin
        out = jnp.maximum(0.,(vl[...,j]-real(.033)*porosity[...,j])*s.dz3d[...,j])
        if j < 4:
            out = jnp.minimum(out,(1.-vi[...,j+1]-vl[...,j+1])*s.dz3d[...,j+1])
            out = jnp.where((porosity[...,j] < real(.05))|(porosity[...,j+1] < real(.05)),0.,out)
        out = out*1000.
        wl = wl.at[...,j].set(jnp.where(enabled,current-out,wl[...,j]))
        qin = jnp.where(enabled,out,qin)
    return s._replace(h2osoi_ice3d=wi,h2osoi_liq3d=wl)


def compact(s,dt,imelt,fracold):
    dz = s.dz3d
    burden = jnp.zeros_like(dt)
    for j in range(5):
        active = j > s.snl2d+4
        wi,wl = s.h2osoi_ice3d[...,j],s.h2osoi_liq3d[...,j]
        wx = wi+wl
        denom = jnp.where(active,dz[...,j],1.)
        void = 1.-(wi/917.+wl/1000.)/denom
        bi,fi = wi/denom,wi/jnp.where(active,wx,1.)
        td = TFRZ-s.t_soisno3d[...,j]
        ddz1 = -real(2.777e-6)*jnp.exp(-real(.04)*td)
        ddz1 = jnp.where(bi > 100.,ddz1*jnp.exp(-real(46.e-3)*(bi-100.)),ddz1)
        ddz1 = jnp.where(wl > real(.01)*dz[...,j],ddz1*2.,ddz1)
        ddz2 = -burden*jnp.exp(-real(.08)*td-real(23.e-3)*bi)/9e5
        ddz3 = jnp.where(imelt[...,j] == 1,-1./dt*jnp.maximum(0.,(fracold[...,j]-fi)/fracold[...,j]),0.)
        enabled = active&(void > real(.001))&(wi > real(.1))
        dz = dz.at[...,j].set(jnp.where(enabled,dz[...,j]*(1.+(ddz1+ddz2+ddz3)*dt),dz[...,j]))
        burden = jnp.where(active,burden+wx,burden)
    return s._replace(dz3d=dz)


def combine(s,enabled):
    snl = s.snl2d.astype(jnp.int32)
    oldtop = snl+5
    dz,wi,wl,t = s.dz3d,s.h2osoi_ice3d,s.h2osoi_liq3d,s.t_soisno3d
    for j in range(5):
        remove = enabled&(j >= oldtop)&(wi[...,j] <= real(.1))
        wl = wl.at[...,j+1].set(jnp.where(remove,wl[...,j+1]+wl[...,j],wl[...,j+1]))
        wi = wi.at[...,j+1].set(jnp.where(remove,wi[...,j+1]+wi[...,j],wi[...,j+1]))
        shifting = remove&(j > snl+5)&(snl < -1)
        for k in range(j,0,-1):
            move = shifting&(k >= snl+6)
            t = t.at[...,k].set(jnp.where(move,t[...,k-1],t[...,k]))
            wi = wi.at[...,k].set(jnp.where(move,wi[...,k-1],wi[...,k]))
            wl = wl.at[...,k].set(jnp.where(move,wl[...,k-1],wl[...,k]))
            dz = dz.at[...,k].set(jnp.where(move,dz[...,k-1],dz[...,k]))
        snl = snl+remove.astype(jnp.int32)
    swe,sd,zwice = (jnp.zeros_like(s.h2osno2d) for _ in range(3))
    for j in range(5):
        active = j > snl+4
        swe = jnp.where(active,swe+wi[...,j]+wl[...,j],swe)
        sd = jnp.where(active,sd+dz[...,j],sd)
        zwice = jnp.where(active,zwice+wi[...,j],zwice)
    collapse = enabled&(sd < real(.01))&(sd > 0.)
    snl = jnp.where(collapse,0,snl)
    swe = jnp.where(collapse,zwice,swe)
    sd = jnp.where(collapse&(swe <= 0.),0.,sd)
    oldtop = snl+5
    order = jnp.ones_like(snl)
    minimum = real([.010,.015,.025,.055,.115])
    for i in range(5):
        active = enabled&(snl < -1)&(i >= oldtop)
        small = dz[...,i] < minimum[jnp.minimum(order-1,4)]
        neighbor = jnp.where(i == snl+5,i+1,jnp.where(i == 4,i-1,jnp.where(dz[...,max(i-1,0)]+dz[...,i] < dz[...,min(i+1,4)]+dz[...,i],i-1,i+1)))
        high,low = jnp.maximum(i,neighbor),jnp.minimum(i,neighbor)
        merge = active&small
        nd,nl,ni,nt = lake_combine(at(dz,high),at(wl,high),at(wi,high),at(t,high),at(dz,low),at(wl,low),at(wi,low),at(t,low))
        dz = set_at(dz,high,jnp.where(merge,nd,at(dz,high)))
        wl = set_at(wl,high,jnp.where(merge,nl,at(wl,high)))
        wi = set_at(wi,high,jnp.where(merge,ni,at(wi,high)))
        t = set_at(t,high,jnp.where(merge,nt,at(t,high)))
        for k in range(4,0,-1):
            move = merge&(high-1 > snl+5)&(k <= high-1)&(k >= snl+6)
            t = t.at[...,k].set(jnp.where(move,t[...,k-1],t[...,k]))
            wi = wi.at[...,k].set(jnp.where(move,wi[...,k-1],wi[...,k]))
            wl = wl.at[...,k].set(jnp.where(move,wl[...,k-1],wl[...,k]))
            dz = dz.at[...,k].set(jnp.where(move,dz[...,k-1],dz[...,k]))
        snl = snl+merge.astype(jnp.int32)
        order = order+(active&~small).astype(jnp.int32)
    s = s._replace(snl2d=snl.astype(jnp.float64),dz3d=dz,h2osoi_ice3d=wi,h2osoi_liq3d=wl,t_soisno3d=t,
                   h2osno2d=jnp.where(enabled,swe,s.h2osno2d),snowdp2d=jnp.where(enabled,sd,s.snowdp2d))
    return geometry(s,enabled)


def divide(s,enabled):
    count = -s.snl2d.astype(jnp.int32)
    index = jnp.arange(5)+s.snl2d.astype(jnp.int32)[...,None]+5
    def gather(a):
        return jnp.take_along_axis(a,index,axis=-1)
    dz,wi,wl,t = (gather(a) for a in (s.dz3d,s.h2osoi_ice3d,s.h2osoi_liq3d,s.t_soisno3d))
    split = enabled&(count == 1)&(dz[...,0] > real(.03))
    def halve(a,j,selected):
        half = a[...,j]/2.
        a = a.at[...,j+1].set(jnp.where(selected,half,a[...,j+1]))
        return a.at[...,j].set(jnp.where(selected,half,a[...,j]))
    dz,wi,wl = (halve(a,0,split) for a in (dz,wi,wl))
    t = t.at[...,1].set(jnp.where(split,t[...,0],t[...,1]))
    count = jnp.where(split,2,count)
    limits = (.02,.05,.11,.23)
    split_limits = (.07,.18,.41)
    for j,limit in enumerate(limits):
        transfer = enabled&(count > j+1)&(dz[...,j] > real(limit))
        drr = dz[...,j]-real(limit)
        proportion = drr/dz[...,j]
        zwice,zwliq = proportion*wi[...,j],proportion*wl[...,j]
        proportion = real(limit)/dz[...,j]
        ni,nl = proportion*wi[...,j],proportion*wl[...,j]
        nd,nwl,nwi,nt = lake_combine(dz[...,j+1],wl[...,j+1],wi[...,j+1],t[...,j+1],drr,zwliq,zwice,t[...,j])
        dz = dz.at[...,j+1].set(jnp.where(transfer,nd,dz[...,j+1]))
        wl = wl.at[...,j+1].set(jnp.where(transfer,nwl,wl[...,j+1]))
        wi = wi.at[...,j+1].set(jnp.where(transfer,nwi,wi[...,j+1]))
        t = t.at[...,j+1].set(jnp.where(transfer,nt,t[...,j+1]))
        dz = dz.at[...,j].set(jnp.where(transfer,real(limit),dz[...,j]))
        wi = wi.at[...,j].set(jnp.where(transfer,ni,wi[...,j]))
        wl = wl.at[...,j].set(jnp.where(transfer,nl,wl[...,j]))
        if j < 3:
            split = transfer&(count <= j+2)&(dz[...,j+1] > real(split_limits[j]))
            dz,wi,wl = (halve(a,j+1,split) for a in (dz,wi,wl))
            t = t.at[...,j+2].set(jnp.where(split,t[...,j+1],t[...,j+2]))
            count = jnp.where(split,j+3,count)
    columns = {}
    for name,values in (("dz3d",dz),("h2osoi_ice3d",wi),("h2osoi_liq3d",wl),("t_soisno3d",t)):
        original = getattr(s,name)
        for j in range(5):
            selected = enabled&(j >= 5-count)
            original = original.at[...,j].set(jnp.where(selected,at(values,j+count-5),original[...,j]))
        columns[name] = original
    s = s._replace(snl2d=jnp.where(enabled,-count,s.snl2d).astype(jnp.float64),**columns)
    return geometry(s,enabled)
