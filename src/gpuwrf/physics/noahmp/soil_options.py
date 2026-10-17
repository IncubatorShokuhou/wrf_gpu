"""WRF Noah-MP soil composition pedotransfer (module_sf_noahmpdrv.F:1711)."""
import jax.numpy as jnp


OUTPUTS=('smcwlt','smcref','smcmax','bexp','psisat','dksat','quartz','dwsat','smcdry')


def pedotransfer_sr2006(sand,clay,orgm,coefficients):
    """Original Saxton–Rawls equations, conversion and source-imposed limits."""
    sand,clay,orgm=(jnp.asarray(v,jnp.float32) for v in (sand,clay,orgm))
    bad=(sand<=0.)|(clay<=0.)
    sand=jnp.where(bad,.41,sand);clay=jnp.where(bad,.18,clay);orgm=jnp.where(orgm<=0.,0.,orgm)
    p={k:jnp.asarray(v,jnp.float32) for k,v in coefficients.items()}
    def c(key):return p['sr2006_'+key]
    def mixture(key):
        return (c(key+'_a')*sand+c(key+'_b')*clay+c(key+'_c')*orgm+c(key+'_d')*sand*orgm+
                c(key+'_e')*clay*orgm+c(key+'_f')*sand*clay+c(key+'_g'))
    t1500=mixture('theta_1500t')
    theta1500=t1500+c('theta_1500_a')*t1500+c('theta_1500_b')
    t33=mixture('theta_33t')
    theta33=t33+c('theta_33_a')*t33*t33+c('theta_33_b')*t33+c('theta_33_c')
    ts33=mixture('theta_s33t')
    thetas33=ts33+c('theta_s33_a')*ts33+c('theta_s33_b')
    psi_et=(c('psi_et_a')*sand+c('psi_et_b')*clay+c('psi_et_c')*thetas33+c('psi_et_d')*sand*thetas33+
            c('psi_et_e')*clay*thetas33+c('psi_et_f')*sand*clay+c('psi_et_g'))
    psi_e=psi_et+c('psi_e_a')*psi_et*psi_et+c('psi_e_b')*psi_et+c('psi_e_c')
    smcmax=theta33+thetas33+c('smcmax_a')*sand+c('smcmax_b')
    bexp=jnp.asarray(3.816712826,jnp.float32)/(jnp.log(theta33)-jnp.log(theta1500))
    dksat=1930.*(smcmax-theta33)**(3.-1./bexp)
    psisat=.101997*jnp.maximum(.1,psi_e)
    dksat=dksat/3600000.
    dwsat=dksat*psisat*bexp/smcmax
    smcmax=jnp.maximum(.32,jnp.minimum(smcmax,.50))
    smcref=jnp.maximum(.17,jnp.minimum(theta33,smcmax))
    smcwlt=jnp.maximum(.01,jnp.minimum(theta1500,smcref))
    smcdry=jnp.maximum(.01,jnp.minimum(theta1500,smcref))
    return dict(zip(OUTPUTS,(smcwlt,smcref,smcmax,jnp.maximum(2.5,jnp.minimum(bexp,12.)),
        jnp.maximum(.03,jnp.minimum(psisat,1.)),jnp.maximum(5.e-7,jnp.minimum(dksat,1.e-5)),
        jnp.maximum(.05,jnp.minimum(sand,.95)),jnp.maximum(1.e-6,jnp.minimum(dwsat,3.e-5)),smcdry)))
