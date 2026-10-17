"""Original WRF TWOSTREAM canopy geometry choices."""
import jax.numpy as jnp


def canopy_gaps(cosz,vai,fveg,rc,hvt,hvb,*,opt_rad):
    """TWOSTREAM:3421-3453, with scalar branches represented by selects."""
    cosz,vai,fveg,rc,hvt,hvb=(jnp.asarray(v,jnp.float32) for v in (cosz,vai,fveg,rc,hvt,hvb))
    one,zero=jnp.ones_like(vai),jnp.zeros_like(vai)
    if opt_rad==2:
        gap=jnp.where(vai==0.,one,zero)
        return gap,gap,jnp.zeros_like(gap),jnp.zeros_like(gap)
    if opt_rad==3:
        gap=jnp.where(vai==0.,1.,1.-fveg)
        return gap,gap,jnp.zeros_like(gap),jnp.zeros_like(gap)
    if opt_rad!=1:raise ValueError("Noah-MP opt_rad must be 1, 2 or 3")
    # WRF enters the geometry equations only for daytime/nonzero canopy.
    active=(vai!=0.)&(cosz>0.)
    cosz=jnp.where(active,cosz,1.)
    rc=jnp.where(active,rc,1.)
    hd=jnp.where(active,hvt-hvb,1.)
    green=jnp.where(active,fveg,.5)
    pai=jnp.asarray(3.14159265,jnp.float32)
    den=-jnp.log(jnp.maximum(1.-green,.01))/(pai*rc**2)
    bb=.5*hd
    thetap=jnp.arctan(bb/rc*jnp.tan(jnp.arccos(jnp.maximum(.01,cosz))))
    bgap=jnp.exp(-den*pai*rc**2/jnp.cos(thetap))
    fa=vai/(1.33*pai*rc**3.0*(bb/rc)*den)
    newvai=hd*fa
    wgap=(1.-bgap)*jnp.exp(-.5*newvai/cosz)
    gap=jnp.minimum(1.-green,bgap+wgap)
    return (jnp.where(active,gap,one),jnp.where(active,jnp.asarray(.05,jnp.float32),one),
            jnp.where(active,bgap,0.),jnp.where(active,wgap,0.))
