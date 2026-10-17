"""Original ISHMAEL var_check2968..3070; qualification uses observed caller inputs."""
import jax.numpy as jnp
from .math import _real


def var_check(nu,ao,fourthirdspi,gammnu,qidum,dsdum,ani,cni,rbdum,nidum,aidum,cidum,
              gamma_tab,*,xp=jnp):
    qidum,r=_real(qidum,xp);W=xp.where
    pi=r(3.14159265);rho_ice=r(920.)
    low=dsdum<r(.55);high=dsdum>r(1.3)
    dsdum=W(low,r(.55),W(high,r(1.3),dsdum))
    a=xp.power(cni/xp.power(ao,r(1.)-dsdum),r(1.)/dsdum)
    ani=W(low,a,ani)
    c=xp.power(ao,r(1.)-dsdum)*xp.power(ani,dsdum)
    cni=W(high,c,cni)
    aidum=W(low|high,ani**2*cni*nidum,aidum)
    cidum=W(low|high,cni**2*ani*nidum,cidum)
    alphstr=xp.power(ao,r(1.)-dsdum)
    alphv=fourthirdspi*alphstr
    betam=r(2.)+dsdum
    argument=(nu+r(2.)+dsdum)*r(100000.)-r(355000.)+r(1.)
    # Fortran NINT rounds halfway away from zero; all reachable NU4 inputs
    # are positive, but retain the source rule before its original bounds.
    rounded=W(argument>=r(0.),xp.floor(argument+r(.5)),-xp.floor(-argument+r(.5)))
    gi=xp.minimum(xp.maximum(rounded.astype('int32'),1),505001)-1
    gt=(gamma_tab if getattr(gamma_tab,'_jensen_table_ref',False) else xp.asarray(gamma_tab,dtype=qidum.dtype))[gi]
    large_axis=ani>r(2.e-6)
    safe_axis=W(large_axis,ani,r(1.))
    denominator=nidum*alphv*xp.power(safe_axis,betam)*gt
    density=qidum*gammnu/W(large_axis,denominator,r(1.))
    rbdum=W(large_axis,density,rho_ice)
    high=rbdum>rho_ice;low=rbdum<r(50.)
    rbdum=W(high,rho_ice,W(low,r(50.),rbdum))
    a=xp.power((qidum*gammnu)/(rbdum*nidum*alphv*gt),r(1.)/betam)
    c=xp.power(ao,r(1.)-dsdum)*xp.power(a,dsdum)
    ani=W(high|low,a,ani);cni=W(high|low,c,cni)
    aidum=W(high|low,ani**2*cni*nidum,aidum)
    cidum=W(high|low,cni**2*ani*nidum,cidum)
    rni=xp.power(qidum*r(3.)/(nidum*rbdum*r(4.)*pi*(gt/gammnu)),r(.333333333333))
    small=rni<r(2.e-6)
    rni=W(small,r(2.e-6),rni)
    n=r(3.)*qidum*gammnu/(r(4.)*pi*rbdum*rni**3*gt)
    nidum=W(small,n,nidum)
    a=xp.power((qidum*gammnu)/(rbdum*nidum*alphv*gt),r(1.)/betam)
    c=xp.power(ao,r(1.)-dsdum)*xp.power(a,dsdum)
    ani=W(small,a,ani);cni=W(small,c,cni)
    aidum=W(small,ani**2*cni*nidum,aidum)
    cidum=W(small,cni**2*ani*nidum,cidum)
    large=xp.maximum(ani,cni)>r(1.e-3)
    largest_a=ani>=cni
    ani=W(large&largest_a,r(1.e-3),ani)
    cni=W(large&~largest_a,r(1.e-3),cni)
    a=xp.power(cni/xp.power(ao,r(1.)-dsdum),r(1.)/dsdum)
    ani=W(large&~largest_a,a,ani)
    n=qidum*gammnu/(fourthirdspi*rbdum*xp.power(ao,r(1.)-dsdum)*xp.power(ani,r(2.)+dsdum)*gt)
    nidum=W(large,n,nidum)
    c=xp.power(ao,r(1.)-dsdum)*xp.power(ani,dsdum)
    cni=W(large&largest_a,c,cni)
    aidum=W(large,ani**2*cni*nidum,aidum)
    cidum=W(large,cni**2*ani*nidum,cidum)
    radius=xp.power(qidum/(nidum*rbdum*fourthirdspi*(gt/gammnu)),r(.333333333333))
    rni=W(large,radius,rni)
    return dict(dsdum=dsdum,ani=ani,cni=cni,rni=rni,rbdum=rbdum,nidum=nidum,aidum=aidum,cidum=cidum,
                alphstr=alphstr,alphv=alphv,betam=betam)
