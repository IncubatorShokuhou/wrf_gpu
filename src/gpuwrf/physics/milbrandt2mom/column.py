"""WRF MY2 column solve; local recurrences are reused inside the GPU Pallas kernel."""
import jax
import jax.numpy as jnp
from .constants import constants
from .local import local_processes
from .math import _dm_x, _ilamda_x
from .rounding import _op, _pow,real,_log10
from gpuwrf.backends.pallas import cuda_native


def _get(a,k):
    return jnp.sum(jnp.where(jnp.arange(a.shape[0],dtype=jnp.int32)==k,a,jnp.zeros_like(a)))


def _set(a,k,value):
    return jnp.where(jnp.arange(a.shape[0],dtype=jnp.int32)==k,value,a)


def _above(a):
    if not cuda_native(): return jnp.roll(a,-1)
    from gpuwrf.kernels.phys_thompson_sedimentation import _gather
    return _gather.bind(a,(jnp.arange(a.shape[0],dtype=jnp.int32)+1)%a.shape[0])


def _any(mask):
    return jnp.max(mask.astype(jnp.int32))!=jnp.int32(0)


def sediment(q, n, cat, eps_sedi, dmx, vmax, dmax, dt, de, ide, idz, dz, gam, z, c, levels):
    r=lambda x:real(x,q.dtype)
    suffix=('r','i','s','g','h')[cat-1]
    ck1=r(c['ckq'+suffix+'1']); ck2=r(c['ckq'+suffix+'2'])
    ck4=r(c[{'r':'icexr9','i':'ckqi4','s':'igs20','g':'ckqg4','h':'ckqh4'}[suffix]])
    cm=r(c['cm'+suffix]); bf=r(c['bf'+suffix]); invdm=_op('div',r(1.),dmx)
    # WRF finds the highest level below 20 km, then starts its species search ONE level lower.
    top_sedi=jnp.maximum(0,jnp.max(jnp.where(z<r(c['zmax_sedi']),levels,0)))
    top=jnp.maximum(0,jnp.max(jnp.where((levels<top_sedi)&(q>eps_sedi),levels,0)))
    active=_any((levels<top_sedi)&(q>eps_sedi))
    domain=levels<=top
    def speed(q,n):
        present=domain&(q>r(c['epsq']))&(n>r(c['epsn']))
        lam=_pow(_op('mul',_op('div',_op('mul',q,de),n),ck4),invdm)
        vv=_op('mul',_op('mul',gam,_pow(lam,bf)),ck1)
        return jnp.where(present,vv,r(0.))
    vv=speed(q,n)
    maxv=jnp.minimum(vmax,jnp.max(vv))
    passes=jnp.maximum(1,jnp.floor(_op('div',_op('mul',dt,maxv),_op('mul',r(.8),jnp.min(dz)))+r(.5)).astype(jnp.int32))
    dtx=_op('div',dt,passes.astype(q.dtype))
    def substep(_,carry):
        q,n,flux=carry
        vq=-speed(q,n)
        ratio=_op('div',ck2,ck1)
        if cat==5:
            tmp1=_pow(_op('div',_op('mul',_op('div',r(1.),cm),q),n),r(c['thrd']))
            mu=jnp.minimum(r(50.),_op('mul',r(.1),_op('mul',r(1000.),tmp1)))
            ratio=((r(3.)+mu)*(r(2.)+mu)*(r(1.)+mu))/((r(3.)+bf+mu)*(r(2.)+bf+mu)*(r(1.)+bf+mu))
        vn=jnp.where(vq!=r(0.),_op('mul',vq,ratio),r(0.))
        flux=_op('sub',flux,_op('mul',_op('mul',_get(vq,0),_get(de,0)),_get(q,0)))
        # WRF bottom-to-top in-place update sees old values at k+1.
        fq=_op('mul',_op('mul',de,q),vq); fn=_op('mul',n,vn)
        qnew=_op('add',q,_op('mul',_op('mul',_op('mul',dtx,ide),_op('add',-_above(fq),fq)),_above(idz)))
        nnew=_op('add',n,_op('mul',_op('mul',dtx,_op('add',-_above(fn),fn)),_above(idz)))
        q=jnp.where(domain,jnp.maximum(qnew,r(0.)),q)
        n=jnp.where(domain,jnp.maximum(nnew,r(0.)),n)
        def repair(n):
            def level(k,n):
                missing=(_get(q,k)>r(c['epsq']))&(_get(n,k)<r(c['epsn']))
                kk=jnp.min(jnp.where((levels>k)&(levels<=top)&(n>=r(c['epsn'])),levels,top+1))
                return jnp.where((levels==k)&missing,jnp.maximum(r(c['epsn']),_get(n,kk)),n)
            return jax.lax.fori_loop(0,top+1,level,n)
        n=jax.lax.cond(_any(domain&(q>r(c['epsq']))&(n<r(c['epsn']))),repair,lambda n:n,n)
        dx=_pow(_op('div',_op('mul',de,q),_op('mul',n,cm)),invdm)
        invmax=_op('div',r(1.),dmax)
        factor=_op('mul',jnp.maximum(dx,dmax),invmax)
        factor=_pow(factor,dmx)
        if cat==1:
            factor=jnp.where(dx>r(.003),jnp.maximum(r(1.)+r(2.e4)*(dx-r(.003))**2,(dx*invmax)**3),factor)
        n=jnp.where(domain&(q>r(c['epsq']))&(n>r(c['epsn'])),_op('mul',n,factor),n)
        return q,n,flux
    def solve(_):
        qo,no,flux=jax.lax.fori_loop(0,passes,substep,(q,n,r(0.)))
        return qo,no,_op('div',flux,passes.astype(q.dtype))
    zero_flux=_op('mul',r(0.),top.astype(q.dtype))
    return jax.lax.cond(active,solve,lambda _:(q,n,zero_flux),None)


def column(fields, pressure, ps, w, dt, *, nz=None):
    """One column: 14 REAL fields, six diameters, reflectivity, eleven m/s precip rates.

    Vertical order is WRF bottom-to-top. All pressure-based layer thicknesses,
    stale density/fallspeed factors, top-level exclusions and diagnostic index
    choices follow my2_mod, including its use of NK in surface type diagnostics.
    """
    fields=tuple(fields); dtype=fields[0].dtype; width=fields[0].shape[0]
    c=constants(str(dtype))
    r=lambda x:real(x,dtype)
    dt=r(dt); nk=width if nz is None else nz; levels=jnp.arange(width,dtype=jnp.int32)
    # Reproduce caller's two rounded operations, not a simplified pressure copy.
    pres=_op('mul',ps,_op('div',pressure,ps))
    initial_de=_op('div',pres,_op('mul',r(c['rgasd']),fields[0]))
    gam=jnp.sqrt(_op('div',r(c['deo']),initial_de))
    from gpuwrf.kernels.phys_thompson_sedimentation import _gather
    below=_gather.bind(pres,(levels-1)%width) if cuda_native() else jnp.roll(pres,1)
    dp=jnp.where(levels==0,_op('sub',ps,pres),_op('sub',below,pres))
    idp=_op('div',r(1.),dp)
    idz=_op('mul',_op('mul',initial_de,r(c['grav'])),idp)
    dz=_op('div',r(1.),idz)
    dz=jnp.where(levels<nk,dz,jnp.asarray(jnp.inf,dtype))
    # Cumulative addition follows WRF serial bottom-to-top order, not a tree sum.
    def height(k,carry):
        running,z=carry
        running=_op('add',running,_get(dz,k))
        return running,_set(z,k,running)
    _,z=jax.lax.fori_loop(0,nk,height,(r(0.),jnp.full_like(dz,jnp.inf)))
    local,de,ide=local_processes(tuple(fields),pres,w,dt,c,levels<nk-1)
    t,q,qc,qr,qi,qn,qg,qh,nc,nr,ny,nn,ng,nh=local
    sedimented=[]; fluxes=[]
    for cat,(qq,nnn),suf in zip(range(1,6),((qr,nr),(qi,ny),(qn,nn),(qg,ng),(qh,nh)),'risgh'):
        qq,nnn,flux=sediment(qq,nnn,cat,r(c['epsq'+suf+'_sedi']),r(c['dm'+suf]),r(c['v'+suf+'max']),
                            r(c['d'+suf+'max']),dt,de,ide,idz,dz,gam,z,c,levels)
        sedimented.extend((qq,nnn)); fluxes.append(flux)
    qr,nr,qi,ny,qn,nn,qg,ng,qh,nh=sedimented
    # WRF post-sedimentation snow intercept and slope constraints.
    mask=(qn>r(c['epsq']))&(nn>r(c['epsn']))
    invlam=jnp.maximum(r(c['ilammin2']),_ilamda_x(de,qn,r(1.)/nn,r(c['igs20']),r(c['idms'])))
    tmp=jnp.minimum(nn/invlam,r(c['no_s_max']))
    adjusted=_pow(tmp,r(c['dms'])/(r(1.)+r(c['dms'])))*_pow(r(c['igs20'])*de*qn,r(1.)/(r(1.)+r(c['dms'])))
    nn=jnp.where(mask,adjusted,nn)
    invlam=jnp.maximum(r(c['ilammin2']),_ilamda_x(de,qn,r(1.)/nn,r(c['igs20']),r(c['idms'])))
    lam=r(1.)/invlam; tmp4=r(.6)*r(c['lamdas_min']); tmp5=r(2.)*tmp4
    newlam=jnp.maximum(lam+tmp4*(jnp.maximum(r(0.),tmp5-lam)/tmp5)**r(2.),r(c['lamdas_min']))
    nn=jnp.where(mask,nn*_pow(newlam*invlam,r(c['dms'])),nn)
    rates=[r(0.) for _ in range(11)]
    for ix,flux in zip((0,4,5,6,7),fluxes): rates[ix]=_op('mul',flux,r(c['idew']))
    # The unmodified code deliberately uses NK here; preserve it rather than "fixing" WRF.
    fluxi=fluxes[1]*r(c['idei']); fluxg=fluxes[3]*r(c['ideg'])
    lams=r(1.)/_ilamda_x(_get(de,nk-1),_get(qn,nk-1),r(1.)/_get(nn,nk-1),r(c['igs20']),r(c['idms']))
    fluxs=jnp.where((_get(qn,nk-1)>r(c['epsq']))&(_get(nn,nk-1)>r(c['epsn']))&(fluxes[2]>r(0.)),
                    fluxes[2]*r(c['rfact_fvfm'])*lams**(r(c['dms'])-r(3.)),r(0.))
    total=_get(qr,nk-1)+_get(qi,nk-1)+_get(qn,nk-1)+_get(qg,nk-1)
    frac=jnp.where((_get(t,nk-1)>r(c['trpl']))&(total>r(c['epsq'])),_get(qr,nk-1)/total,r(0.))
    rates[10]=(r(1.)-frac)*(fluxi+fluxg+fluxs)+frac*(rates[4]+rates[5]+rates[6])
    de=_set(de,0,_get(pres,0)/(r(c['rgasd'])*_get(t,0)))
    dr=(_get(de,nk-1)*r(c['icmr'])*_get(qr,0)/_get(nr,nk-1))**r(c['thrd'])
    drizzle=(_get(qr,0)>r(c['epsq']))&(_get(nr,nk-1)>r(c['epsn']))&(dr>r(c['dr_large']))
    rates[1]=jnp.where(drizzle,rates[0],r(0.))
    rates[0]=jnp.where(drizzle,r(0.),rates[0])
    freezing=_get(t,nk-1)<r(c['trpl'])
    rates[2],rates[3]=(jnp.where(freezing,x,r(0.)) for x in rates[:2])
    rates[0],rates[1]=(jnp.where(freezing,r(0.),x) for x in rates[:2])
    hail=_get(t,nk-1)>(r(c['trpl'])+r(5.))
    rates[8]=jnp.where(hail,rates[7],r(0.))
    rates[7]=jnp.where(hail,r(0.),rates[7])
    dh=_dm_x(_get(de,0),_get(qh,0),r(1.)/_get(nh,0),r(c['icmh']),r(c['thrd']))
    rates[9]=jnp.where((_get(qh,0)>r(c['epsq']))&(dh>r(c['dh_large'])),rates[8],r(0.))
    q=jnp.maximum(q,r(0.))
    # Diagnostics recompute density, then the number mixing-ratio conversion recomputes its inverse.
    de=pres/(r(c['rgasd'])*t)
    refl=r(0.)
    for qq,nnn,g in zip((qr,qi,qn,qg,qh),(nr,ny,nn,ng,nh),(20.,20.,6.,20.,20.)):
        factor=r(c['icmr'])*r(c['icmr'])
        if g==6. or qq is not qr:
            factor=r(1.)/r(c['fdielec'])*r(c['icmr'])*r(c['icmr'])
        term=factor*r(g)*(de*de)*qq*qq/nnn
        if qq is not qr: term=jnp.where(t>r(c['trpl']),term*r(c['fdielec']),term)
        refl=refl+jnp.where((qq>r(c['epsq']))&(nnn>r(c['epsn'])),term,r(0.))
    zet=jnp.where(refl>r(0.),r(10.)*_log10(refl*r(c['zfact'])),r(c['minzet']))
    zet=jnp.maximum(zet,r(c['minzet']))
    diam=[]
    for qq,nnn,suf in zip((qc,qr,qi,qn,qg,qh),(nc,nr,ny,nn,ng,nh),'crisgh'):
        cs='r' if suf=='c' else suf
        d=_dm_x(de,qq,r(1.)/nnn,r(c['icm'+cs]),r(c['idms'] if cs=='s' else c['thrd']))
        diam.append(jnp.where((qq>r(c['epsq']))&(nnn>r(c['epsn'])),d,r(0.)))
    ide=(r(c['rgasd'])*t)/pres
    out=(t,q,qc,qr,qi,qn,qg,qh,nc*ide,nr*ide,ny*ide,nn*ide,ng*ide,nh*ide)
    zet=jnp.where(levels<nk,zet,r(c['minzet']))
    return out,tuple(rates),tuple(diam),zet,jnp.max(zet)


def columns(fields, pressure, ps, w, dt):
    """CPU qualification route; GPU dispatch uses per-column Pallas programs."""
    out,rates,diam,zet,zec=jax.vmap(column,in_axes=(0,0,0,0,None))(fields,pressure,ps,w,dt)
    return jnp.stack(out,axis=1),jnp.stack(rates,axis=1),jnp.stack(diam,axis=1),zet,zec
