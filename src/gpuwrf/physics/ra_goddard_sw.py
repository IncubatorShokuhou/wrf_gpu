"""Explicit-DOUBLE Goddard option-5 SW core with the original REAL4 interface.

WRF uses overcast=True/fast_overcast=True. reduction_tau is explicit because
pristine WRF 4.7.1 reads an uninitialized local taux in its fast-overcast gas
reduction. The caller must choose its documented reference policy. This kernel
is not admitted operationally before the full scheme gate closes.
"""
from jax import lax
import jax.numpy as jnp
import numpy as np

from .ra_goddard_common import _host_tables
from .ra_goddard_lw import R


def delta_eddington(tau,ssc,g0,cza):
    ff=g0*g0;xx=1-ff*ssc
    taup=tau*xx;sscp=ssc*(1-ff)/xx;gp=g0/(1+g0)
    xx=3*gp
    gm1=(7-sscp*(4+xx))*R(.25)
    gm2=-(1-sscp*(4-xx))*R(.25)
    akk=jnp.sqrt((gm1+gm2)*(gm1-gm2))
    xx=akk*cza
    zth=jnp.where(jnp.abs((1-xx)*(1+xx))<R(1e-8),cza+R(.001),cza)
    xx=akk*zth;st7=1-xx;st8=1+xx;st3=st7*st8
    td=jnp.where(taup/zth<R(40),jnp.exp(-taup/zth),0.)
    gm3=(2-zth*3*gp)*R(.25)
    xx=gm1-gm2;alf1=gm1-gm3*xx;alf2=gm2+gm3*xx
    xx=akk*2
    all_=(gm3-alf2*zth)*xx*td
    bll=(1-gm3+alf1*zth)*xx
    xx=akk*gm3;cll=(alf2+xx)*st7;dll=(alf2-xx)*st8
    xx=akk*(1-gm3);fll=(alf1+xx)*st8;ell=(alf1-xx)*st7
    st2=jnp.where(akk*taup<R(40),jnp.exp(-akk*taup),0.)
    st4=st2*st2
    st1=sscp/((akk+gm1+(akk-gm1)*st4)*st3)
    rr=(cll-dll*st4-all_*st2)*st1
    tt=-((fll-ell*st4)*td-bll*st2)*st1
    return jnp.maximum(rr,0.),jnp.maximum(tt,0.)+td,td


def twostream_overcast(rr,tt,td,rs,ts,albedo_beam,albedo_diffuse,icb):
    """twostream_adding, ih=im=is=2 and cc=(1,1,1), preserving recurrences."""
    n=rr.shape[0];zeros=jnp.zeros((n,),jnp.float64)
    tda=zeros.at[0].set(td[0]);tta=zeros.at[0].set(tt[0]);rsa=zeros.at[0].set(rs[0])
    def forward(k,carry):
        a,b,c=carry
        denm=ts[k]/(1-c[k-1]*rs[k])
        aa=a[k-1]*td[k]
        # The low-cloud source orders the two reflectances differently.
        reflected=jnp.where(k+1>=icb,a[k-1]*rr[k]*c[k-1],a[k-1]*c[k-1]*rr[k])
        bb=a[k-1]*tt[k]+(reflected+b[k-1]-a[k-1])*denm
        cc=rs[k]+ts[k]*c[k-1]*denm
        return a.at[k].set(jnp.where(aa<R(1e-10),0.,aa)),b.at[k].set(jnp.where(bb<R(1e-10),0.,bb)),c.at[k].set(cc)
    tda,tta,rsa=lax.fori_loop(1,n,forward,(tda,tta,rsa))
    rra=jnp.zeros((n+1,),jnp.float64).at[n].set(albedo_beam)
    rxa=jnp.zeros((n+1,),jnp.float64).at[n].set(albedo_diffuse)
    def backward(k,carry):
        a,b=carry;k=n-1-k
        denm=ts[k]/(1-rs[k]*b[k+1])
        aa=rr[k]+(td[k]*a[k+1]+(tt[k]-td[k])*b[k+1])*denm
        bb=rs[k]+ts[k]*b[k+1]*denm
        return a.at[k].set(aa),b.at[k].set(bb)
    rra,rxa=lax.fori_loop(0,n,backward,(rra,rxa))
    denm=1/(1-rsa*rxa[1:])
    xx=tda*rra[1:];yy=tta-tda
    diffuse=(xx*rsa+yy)*denm
    up=(xx+yy*rxa[1:])*denm
    net=jnp.concatenate((jnp.reshape(1-rra[0],(1,)),tda+diffuse-up))
    down=jnp.concatenate((jnp.ones((1,),jnp.float64),tda+diffuse))
    return net,down,tda[-1],diffuse[-1]


def _accumulate(x):
    """WRF left-to-right column integral, including a zero TOA value."""
    out=jnp.zeros((x.shape[0]+1,),jnp.float64)
    return lax.fori_loop(0,x.shape[0],lambda k,a:a.at[k+1].set(a[k]+x[k]),out)


def _reduce_flux(swc,swh,u1,du,w1,dw,table):
    nu,nw=table.shape
    # The source's MIN assignment is overwritten by MAX on the raw input.
    # Keep that original extrapolation; clamp only the table indices.
    clog=jnp.maximum(swc,u1-R(.5)*du)
    wlog=jnp.maximum(swh,w1-R(.5)*dw)
    ic=jnp.clip(jnp.trunc((clog-(u1-R(.5)*du))/du+1).astype(jnp.int32),2,nu)-1
    iw=jnp.clip(jnp.trunc((wlog-(w1-R(.5)*dw))/dw+1).astype(jnp.int32),2,nw)-1
    dc=clog-(ic-1).astype(jnp.float64)*du-u1
    dd=wlog-(iw-1).astype(jnp.float64)*dw-w1
    x2=table[ic-1,iw-1]+(table[ic-1,iw]-table[ic-1,iw-1])/dw*dd
    return x2+(table[ic,iw-1]-table[ic-1,iw-1])/du*dc


def solve_sw_column(pl,ta,wa,oa,fcld,cosz,taucl,ssacl,asycl,taual,ssaal,asyal,albedo,icb,reduction_tau):
    """swrad core; net/down/up fractions and four surface direct/diffuse fractions.

    All inputs use original solver top-down ordering. cosz must be daytime;
    the goddardrad wrapper handles cosz<=REAL4(.0001) without calling swrad.
    """
    pl,ta,wa,oa,fcld,cosz,taucl,ssacl,asycl,taual,ssaal,asyal,albedo,reduction_tau=[jnp.asarray(x,jnp.float64) for x in (pl,ta,wa,oa,fcld,cosz,taucl,ssacl,asycl,taual,ssaal,asyal,albedo,reduction_tau)]
    host=_host_tables()
    tab={k:jnp.asarray(v) for k,v in host.items() if k.startswith(('sw_','sw_uvpar_','sw_ir_'))}
    n=ta.shape[0];dp=pl[1:]-pl[:-1]
    scal=dp*(R(.5)*(pl[:-1]+pl[1:])/R(300))**R(.8)
    wh=R(1.02)*wa*scal*(1+R(.00135)*(ta-R(240)))+R(1e-11)
    oh=R(1.02)*oa*dp*R(466.7)+R(1e-11)
    zero=jnp.zeros((n+1,),jnp.float64)
    def band(step,carry):
        net,down,surf=carry
        uv=step<8
        irband=jnp.clip((step-8)//10,0,2)
        b=jnp.where(uv,step,irband+8)
        uvband=jnp.minimum(step,7);ik=jnp.maximum(step-8,0)%10
        ray=jnp.where(uv,tab['sw_uvpar_ry'][uvband],tab['sw_ir_ry'][irband])*dp
        gas=jnp.where(uv,tab['sw_uvpar_zk'][uvband]*oh+tab['sw_uvpar_wk'][uvband]*wh,tab['sw_ir_xk'][ik]*wh)
        base=jnp.maximum(ray+gas+taual[:,b],R(1e-6))
        scatter=ssaal[:,b]*taual[:,b]+ray
        asym=asyal[:,b]*ssaal[:,b]*taual[:,b]
        tau=base+jnp.maximum(taucl[:,b],0.)
        ssa=jnp.clip((scatter+ssacl[:,b]*taucl[:,b])/tau,0.,R(.999999))
        gg=(asym+asycl[:,b]*ssacl[:,b]*taucl[:,b])/jnp.maximum(ssa*tau,float(np.finfo(np.float32).tiny))
        gg=jnp.clip(gg,jnp.where(uv,-1.,0.),jnp.where(uv,1.,R(.999999)))
        rr,tt,td=delta_eddington(tau,ssa,gg,cosz)
        rs,ts,_=delta_eddington(tau,ssa,gg,jnp.float64(R(.602)))
        fall,falld,direct,diffuse=twostream_overcast(rr,tt,td,rs,ts,albedo,albedo,icb)
        weight=jnp.where(uv,tab['sw_uvpar_hk'][uvband],tab['sw_ir_hk'][irband,ik])
        offset=jnp.where(uv,0,2)
        surf=surf.at[offset].set(surf[offset]+direct*weight)
        surf=surf.at[offset+1].set(surf[offset+1]+diffuse*weight)
        return net+fall*weight,down+falld*weight,surf
    net,down,surface=lax.fori_loop(0,38,band,(zero,zero,jnp.zeros((4,),jnp.float64)))
    snt=1/cosz
    oxygen=_accumulate(scal*(R(165.22)*snt))
    df=R(.0633)*(1-jnp.exp(-R(.000145)*jnp.sqrt(oxygen[1:])))
    co2=_accumulate((R(789)*(R(401.30e-6)*snt))*scal+R(1e-11))
    water=_accumulate(wh)
    df=df+_reduce_flux(jnp.log10(co2[1:]),jnp.log10(water[1:]*snt),R(-3),R(.15),R(-4),R(.15),tab['sw_cah'])
    df=df+_reduce_flux(jnp.full((n,),R(401.30e-6)*snt),jnp.log10(pl[1:]),R(.000250),R(.000050),R(-2),R(.05),tab['sw_coa'])
    df=jnp.concatenate((jnp.zeros((1,),jnp.float64),df))
    sub=jnp.maximum(df[1:]-df[:-1],0.)
    first=jnp.min(jnp.where(fcld>R(.01),jnp.arange(n),n))
    cloud_sum=lax.fori_loop(0,n,lambda k,a:a+jnp.where(k>first,reduction_tau[k],0.),jnp.float64(0))
    cloud_albedo=cloud_sum/(R(6.7)+cloud_sum)
    i_cos=jnp.clip(jnp.trunc(cosz*10).astype(jnp.int32),0,9)
    i_tau=jnp.clip(jnp.trunc(cloud_albedo*10).astype(jnp.int32),0,9)
    ratio=tab['sw_ratio'][i_tau,i_cos]
    sub=sub*jnp.where(jnp.arange(n)>=first,ratio,1.)
    reduction=_accumulate(sub)
    net=jnp.maximum(net-reduction,0.)
    down=jnp.maximum(down-reduction,0.)
    ir=surface[2]-reduction[-1]
    surface=surface.at[3].set(jnp.where(ir>=0,surface[3],jnp.maximum(0.,surface[3]+ir)))
    surface=surface.at[2].set(jnp.where(ir>=0,ir,0.))
    return net.astype(jnp.float32),down.astype(jnp.float32),(net-down).astype(jnp.float32),surface.astype(jnp.float32)
