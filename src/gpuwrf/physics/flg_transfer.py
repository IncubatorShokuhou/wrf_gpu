"""Literal FLG single-column transfer routines for embedding in a column kernel.

All vertical recurrences here are single-column; operational GPU callers must
embed them inside Pallas, never vmap a domain-wide level loop into the hot path.
"""
from __future__ import annotations

import jax.numpy as jnp
from jax import lax


def _source_small(x,threshold,replacement=None):
    """Source's signed small-denominator branch, with explicitly REAL values."""
    small=jnp.asarray(threshold if replacement is None else replacement,x.dtype)
    return jnp.where(jnp.abs(x)<threshold,jnp.where(x>=0.,small,-small),x)


def _mul(x,y):
    """Retain the WRF REAL product before a recurrence's subtract (CPU FMA)."""
    product=x*y
    return jnp.nextafter(product,product)


def cumulative_depth(depth):
    n = depth.shape[0]
    def body(i, values):
        return values.at[i].set(depth[i] + values[i - 1])
    return lax.fori_loop(1, n, body, jnp.zeros_like(depth).at[0].set(depth[0]))


def adjust(moments, omega, total_depth):
    """WRF adjust; D4S=true selects the fourth moment also in infrared."""
    f = moments[:, 3] / 9.
    fw = 1. - f * omega
    w1 = (moments[:, 0] - 3. * f) / (1. - f)
    w2 = (moments[:, 1] - 5. * f) / (1. - f)
    w3 = (moments[:, 2] - 7. * f) / (1. - f)
    w = (1. - f) * omega / fw
    difference = jnp.concatenate((total_depth[:1], total_depth[1:] - total_depth[:-1]))
    t = cumulative_depth(difference * fw)
    return jnp.stack((w1,w2,w3),axis=-1),w,t


def tridag(a,b,c,r):
    """WRF tridag/Thomas order. A zero pivot propagates nonfinite, never a clamp."""
    n = b.shape[0]
    u = jnp.zeros_like(b).at[0].set(r[0] / b[0])
    gam = jnp.zeros_like(b)
    def forward(j,carry):
        u,gam,bet=carry
        gamma = c[j-1] / bet
        bet = b[j] - _mul(a[j],gamma)
        value = (r[j] - _mul(a[j],u[j-1])) / bet
        return u.at[j].set(value),gam.at[j].set(gamma),bet
    u,gam,_ = lax.fori_loop(1,n,forward,(u,gam,b[0]))
    def backward(i,u):
        j=n-2-i
        return u.at[j].set(u[j] - _mul(gam[j+1],u[j+1]))
    return lax.fori_loop(0,n-1,backward,u)


def infrared_coefficients(w,w1,t0,t1,u0,f0):
    """WRF cfgts with HEMISP=true, QUADRA=EDDING=false, ib>6."""
    x = w * w1 / 3.
    gamma1 = 2. - w - x
    gamma2 = w - x
    lam = jnp.sqrt((gamma1 + gamma2) * (gamma1 - gamma2))
    gamma = gamma2 / (gamma1 + lam)
    fq = 1. / u0
    fw = 3.1415927 * f0
    exp = jnp.exp(-fq * (t1-t0))
    z = lam * lam - fq * fq
    z = _source_small(z,1.e-4)
    cadd0 = fw * ((gamma1-fq) + gamma2) / z
    cmin0 = fw * ((gamma1+fq) + gamma2) / z
    return lam,gamma,cadd0,cadd0*exp,cmin0,cmin0*exp,gamma1+gamma2


def infrared_two_stream(w1,w,t,u0,f0,asbs,ee):
    """WRF qccgts, retaining its 2*n tridiagonal equations and ordering."""
    n=w.shape[0]
    t0=jnp.concatenate((jnp.zeros_like(t[:1]),t[:-1]))
    wn=jnp.minimum(w,.999999)
    lam,gamma,ca0,ca,cm0,cm,g1g2 = infrared_coefficients(wn,w1,t0,t,u0,f0)
    expn=jnp.exp(-lam*(t-t0))
    xn=gamma*expn
    yn=(expn-gamma)/(xn-1.)
    zn=(expn+gamma)/(xn+1.)
    a,b,c,r=(jnp.zeros((n*2,),w.dtype) for _ in range(4))
    b=b.at[0].set(xn[0]+1.);c=c.at[0].set(xn[0]-1.);r=r.at[0].set(-cm0[0])
    # These are independent equation rows, not recurrences over levels.
    odd=jnp.arange(n-1)*2+1;even=odd+1
    a=a.at[odd].set(1.+xn[:-1]-yn[1:]*(gamma[:-1]+expn[:-1]))
    b=b.at[odd].set(1.-xn[:-1]-yn[1:]*(gamma[:-1]-expn[:-1]))
    c=c.at[odd].set(yn[1:]*(1.+xn[1:])-expn[1:]-gamma[1:])
    r=r.at[odd].set(ca0[1:]-ca[:-1]-yn[1:]*(cm0[1:]-cm[:-1]))
    a=a.at[even].set(gamma[:-1]-expn[:-1]-zn[:-1]*(1.-xn[:-1]))
    b=b.at[even].set(-1.-xn[1:]+zn[:-1]*(expn[1:]+gamma[1:]))
    c=c.at[even].set(zn[:-1]*(expn[1:]-gamma[1:])-xn[1:]+1.)
    r=r.at[even].set(cm0[1:]-cm[:-1]-zn[:-1]*(ca0[1:]-ca[:-1]))
    rsfc=1.-ee;ssfc=3.1415927*asbs
    wm1=1.-rsfc*gamma[-1];wm2=xn[-1]-rsfc*expn[-1]
    a=a.at[-1].set(wm1+wm2);b=b.at[-1].set(wm1-wm2);c=c.at[-1].set(0.)
    r=r.at[-1].set(rsfc*cm[-1]-ca[-1]+ssfc)
    solution=tridag(a,b,c,r)
    aa=solution[::2]+solution[1::2]
    bb=solution[::2]-solution[1::2]
    return lam,gamma,aa,bb,expn


def infrared_transfer(ee,bf,bs,moments,omega,total_depth):
    """WRF qftisf with the two source-function quadrature directions."""
    n=omega.shape[0]
    w123,w,t=adjust(moments,omega,total_depth)
    t0=jnp.concatenate((jnp.zeros_like(t[:1]),t[:-1]))
    q1=jnp.log(bf[1:]/bf[:-1])
    deltau=jnp.maximum(t-t0,1.e-12)
    q2=1./deltau
    f0=2.*(1.-w)*bf[:-1]
    # These are the original WRF degenerate-source branches.
    u0=jnp.where(jnp.abs(q1)<=1.e-10,-1.e10/q2,-1./(jnp.where(jnp.abs(q1)<=1.e-10,1.,q1)*q2))
    beta=-1./u0
    lam,gamma,aa,bb,expn=infrared_two_stream(w123[:,0],w,t,u0,f0,bs*ee,ee)
    xgy=lam*lam-beta*beta
    xgy=_source_small(xgy,1.e-4)
    x=2.*(1.-w)*w/xgy
    y1=w123[:,0]/3.
    y=2.*(1.-w*y1);z=-y1*beta
    fuq1=x*(y-z)+1.-w;fuq2=x*(y+z)+1.-w
    alfa=6.2832*bf
    y=gamma*(2.+lam);z=2.-lam
    fg=aa*z;fh=bb*y;fj=aa*y;fk=bb*z
    ug=jnp.asarray([.2113248,.7886752],w.dtype)
    ugwg=jnp.asarray([.105662,.394338],w.dtype)
    fx=jnp.exp(-(t-t0)[:,None]/ug)
    xx=lam[:,None]*ug
    plus=jnp.where(xx+1.==0.,1.e-4,xx+1.)
    minus=jnp.where(xx-1.==0.,1.e-4,xx-1.)
    fz1=(1.-fx*expn[:,None])/plus
    fz2=(fx-expn[:,None])/minus
    ub=ug*beta[:,None]+1.
    ub=jnp.where(ub==0.,1.e-4,ub)
    down=jnp.zeros((n+1,2),w.dtype)
    def descend(k,values):
        value=values[k]*fx[k]+fj[k]*fz1[k]+fk[k]*fz2[k]+1./ub[k]*(alfa[k+1]-alfa[k]*fx[k])*fuq2[k]
        return values.at[k+1].set(value)
    down=lax.fori_loop(0,n,descend,down)
    yy=ugwg[0]*down[-1,0]+ugwg[1]*down[-1,1]
    surface=yy*(1.-ee)*2.+6.2831854*ee*bs
    up=jnp.zeros((n+1,2),w.dtype).at[-1].set(surface)
    ub=ug*beta[:,None]
    ub=jnp.where(ub==1.,1.0001,ub)
    def ascend(i,values):
        k=n-1-i
        value=values[k+1]*fx[k]+fg[k]*fz2[k]+fh[k]*fz1[k]+1./(ub[k]-1.)*(alfa[k+1]*fx[k]-alfa[k])*fuq1[k]
        return values.at[k].set(value)
    up=lax.fori_loop(0,n,ascend,up)
    return up[:,0]*ugwg[0]+up[:,1]*ugwg[1],down[:,0]*ugwg[0]+down[:,1]*ugwg[1]


def solar_layer_coefficients(w,w1,w2,w3,t0,t1,u0,f0):
    """WRF coefft/coefft0 for a single solar layer, four-stream default."""
    dtype=w.dtype
    u=jnp.asarray([-.7886752,-.2113247,.2113247,.7886752],dtype)
    p1=jnp.asarray([-.788675,-.211325,.211325,.788675],dtype)
    p2=jnp.asarray([.433013,-.433013,-.433013,.433013],dtype)
    p3=jnp.asarray([-.043394,.293394,-.293394,.043394],dtype)
    p11=jnp.asarray([[.311004,.0833334,-.0833334,-.311004],
        [.0833334,.0223291,-.0223291,-.0833334],[-.0833334,-.0223291,.0223291,.0833334],
        [-.311004,-.0833334,.0833334,.311004]],dtype)
    p22=jnp.asarray([[.0937501,-.0937501,-.0937501,.0937501],
        [-.0937501,.0937501,.0937501,-.0937501],[-.0937501,.0937501,.0937501,-.0937501],
        [.0937501,-.0937501,-.0937501,.0937501]],dtype)
    p33=jnp.asarray([[.000941520,-.00636577,.00636577,-.000941520],
        [-.00636577,.0430400,-.0430400,.00636577],[.00636577,-.0430400,.0430400,-.00636577],
        [-.000941520,.00636577,-.00636577,.000941520]],dtype)
    dt=t1-t0

    def absorption(_):
        fk1=jnp.asarray(4.7320545,dtype);fk2=jnp.asarray(1.2679491,dtype)
        x=jnp.exp(-fk1*dt);y=jnp.exp(-fk2*dt)
        a1=jnp.eye(4,dtype=dtype)[:,::-1]
        top=jnp.zeros((4,4),dtype).at[0,3].set(y).at[1,2].set(x).at[2,1].set(1.).at[3,0].set(1.)
        bottom=jnp.zeros((4,4),dtype).at[0,3].set(1.).at[1,2].set(1.).at[2,1].set(x).at[3,0].set(y)
        z=jnp.zeros((4,),dtype)
        return top,bottom,jnp.zeros((4,2),dtype),a1,z,fk1,fk2

    def scattering(_):
        x=.5*w;w0=x;w1w=x*w1;w2w=x*w2;w3w=x*w3
        fw=u0*u0;q1=-w1w*u0;q2=w2w*(1.5*fw-.5);q3=-w3w*(2.5*fw-1.5)*u0
        c=.5*w0+w1w*p11[2:]+w2w*p22[2:]+w3w*p33[2:]
        c=(c-jnp.eye(4,dtype=dtype)[2:])/u[2:,None]
        c5=(w0+q1*p1+q2*p2+q3*p3)/u
        b=jnp.stack((jnp.stack((c[1,3]-c[1,0],c[1,3]+c[1,0],c5[3]-c5[0])),
                     jnp.stack((c[1,2]-c[1,1],c[1,2]+c[1,1],c5[2]-c5[1])),
                     jnp.stack((c[0,3]-c[0,0],c[0,3]+c[0,0],c5[2]+c5[1])),
                     jnp.stack((c[0,2]-c[0,1],c[0,2]+c[0,1],c5[3]+c5[0]))))
        fw1=b[0,0]*b[0,1];fw2=b[1,0]*b[2,1];fw3=b[2,0]*b[1,1];fw4=b[3,0]*b[3,1]
        a=jnp.zeros((2,2,2),dtype)
        a=a.at[1,1,0].set(fw1+fw2).at[1,0,0].set(b[0,0]*b[1,1]+b[1,0]*b[3,1])
        a=a.at[0,1,0].set(b[2,0]*b[0,1]+b[3,0]*b[2,1]).at[0,0,0].set(fw3+fw4)
        a=a.at[1,1,1].set(fw1+fw3).at[1,0,1].set(b[0,1]*b[1,0]+b[1,1]*b[3,0])
        a=a.at[0,1,1].set(b[2,1]*b[0,0]+b[3,1]*b[2,0]).at[0,0,1].set(fw2+fw4)
        d=jnp.stack((b[2,1]*b[3,2]+b[3,1]*b[2,2]+b[1,2]/u0,
                     b[0,1]*b[3,2]+b[1,1]*b[2,2]+b[0,2]/u0,
                     b[2,0]*b[0,2]+b[3,0]*b[1,2]+b[2,2]/u0,
                     b[0,0]*b[0,2]+b[1,0]*b[1,2]+b[3,2]/u0))
        x=u0*u0;b1=a[1,1,0]+a[0,0,0];c1=a[1,0,0]*a[0,1,0]-a[0,0,0]*a[1,1,0]
        z=jnp.stack((a[1,0,0]*d[2]+d[3]/x-a[0,0,0]*d[3],
                     a[0,1,0]*d[3]-a[1,1,0]*d[2]+d[2]/x,
                     a[1,0,1]*d[0]+d[1]/x-a[0,0,1]*d[1],
                     a[0,1,1]*d[1]-a[1,1,1]*d[0]+d[0]/x))
        x=jnp.sqrt(b1*b1+4.*c1)
        fk1=jnp.sqrt((b1+x)*.5);fk2=jnp.sqrt((b1-x)*.5)
        fw=u0*u0;x=1./(fw*fw)-b1/fw-c1
        x=_source_small(x,1.e-16,1.e-6)
        fw=.5*f0/x;z=fw*z
        z1=.5*jnp.stack((z[0]+z[2],z[1]+z[3],z[1]-z[3],z[0]-z[2]))
        a2=(fk1*fk1-a[1,1,0])/a[1,0,0];b2=(fk2*fk2-a[1,1,0])/a[1,0,0]
        x=b[0,0]*b[3,0]-b[2,0]*b[1,0];fw1=fk1/x;fw2=fk2/x
        y=fw2*(b2*b[1,0]-b[3,0]);zx=fw1*(a2*b[1,0]-b[3,0])
        row0=.5*jnp.stack((1.-y,1.-zx,1.+zx,1.+y))
        y=fw2*(b[2,0]-b2*b[0,0]);zx=fw1*(b[2,0]-a2*b[0,0])
        row1=.5*jnp.stack((b2-y,a2-zx,a2+zx,b2+y))
        a1=jnp.stack((row0,row1,row1[::-1],row0[::-1]))
        fq0=jnp.exp(-t0/u0);fq1=jnp.exp(-t1/u0)
        x=jnp.exp(-fk1*dt);y=jnp.exp(-fk2*dt)
        top=a1*jnp.stack((jnp.ones_like(x),jnp.ones_like(x),x,y))[None,:]
        bottom=a1*jnp.stack((y,x,jnp.ones_like(x),jnp.ones_like(x)))[None,:]
        return top,bottom,jnp.stack((z1*fq0,z1*fq1),axis=-1),a1,z1,fk1,fk2

    return lax.cond(w<=1.e-4,absorption,scattering,None)


def qcfel(ab,b):
    """WRF qcfel's sparse partial-pivot elimination, same row/column order.

    The native nested block/l loops are flattened; widths are exactly the
    original 14-l / 8-l (last block: 12-l / 4-l) sparsity bounds.
    """
    n4=b.shape[0]
    def column(col,carry):
        ab,b=carry
        pos=col%4
        last=col>=n4-4
        below=jnp.where(last,3-pos,5-pos)
        right=jnp.where(last,3-pos,7-pos)
        def pivot_body(row,carry):
            value,index=carry
            take=jnp.abs(ab[row,col])>jnp.abs(value)
            return jnp.where(take,ab[row,col],value),jnp.where(take,jnp.asarray(row,index.dtype),index)
        # Every valid matrix has a nonzero pivot. If singular, division by zero
        # remains nonfinite and is rejected; never replace it with a small pivot.
        _,pivot=lax.fori_loop(7,8+below,pivot_body,(jnp.asarray(0.,b.dtype),jnp.asarray(7,jnp.int32)))
        def swap(carry):
            ab,b=carry
            def entry(offset,ab):
                j=col+offset;r1=pivot+col-j;r2=7+col-j
                value=ab[r1,j]
                return ab.at[r1,j].set(ab[r2,j]).at[r2,j].set(value)
            ab=lax.fori_loop(0,right+1,entry,ab)
            row=pivot+col-7;value=b[row]
            b=b.at[row].set(b[col]).at[col].set(value)
            return ab,b
        ab,b=lax.cond(pivot!=7,swap,lambda x:x,(ab,b))
        yy=ab[7,col];ab=ab.at[7,col].set(1.)
        def normalize(offset,ab):
            j=col+offset;row=7-offset
            return ab.at[row,j].set(ab[row,j]/yy)
        ab=lax.fori_loop(1,right+1,normalize,ab)
        b=b.at[col].set(b[col]/yy)
        def eliminate(row,carry):
            ab,b=carry;xx=ab[row,col];ab=ab.at[row,col].set(0.)
            def entry(offset,ab):
                j=col+offset;dest=row-offset;src=7-offset
                return ab.at[dest,j].set(ab[dest,j]-ab[src,j]*xx)
            ab=lax.fori_loop(1,right+1,entry,ab)
            dest=row+col-7
            b=b.at[dest].set(b[dest]-b[col]*xx)
            return ab,b
        return lax.fori_loop(8,8+below,eliminate,(ab,b))
    ab,b=lax.fori_loop(0,n4-1,column,(ab,b))
    b=b.at[-1].set(b[-1]/ab[7,-1])
    x=jnp.zeros_like(b).at[-1].set(b[-1])
    def back(i,x):
        row=n4-2-i
        width=jnp.where(row>=n4-4,n4-1-row,7-row%4)
        def entry(offset,value):
            col=row+offset
            return value-ab[7-offset,col]*x[col]
        value=lax.fori_loop(1,width+1,entry,b[row])
        return x.at[row].set(value)
    return lax.fori_loop(0,n4-1,back,x)


def solar_four_stream(albedo,u0,f0,w123,w,t):
    """WRF qcfe's solar boundary and layer-continuity equations."""
    n=w.shape[0];n4=n*4
    ab=jnp.zeros((13,n4),w.dtype);rhs=jnp.zeros((n4,),w.dtype)
    a4=jnp.zeros((n,4,4),w.dtype);z4=jnp.zeros((n,4),w.dtype)
    eigen=jnp.zeros((n,2),w.dtype)
    bottom=jnp.zeros((4,4),w.dtype);zzbottom=jnp.zeros((4,),w.dtype)
    def layer(k,carry):
        ab,rhs,a4,z4,eigen,previous,previous_z=carry
        t0=jnp.where(k==0,0.,t[jnp.maximum(k-1,0)])
        wn=jnp.minimum(w[k],.999999)
        top,bottom,zz,a1,z1,fk1,fk2=solar_layer_coefficients(wn,*w123[k],t0,t[k],u0,f0)
        a4=a4.at[k].set(a1);z4=z4.at[k].set(z1);eigen=eigen.at[k].set(jnp.stack((fk1,fk2)))
        def top_rows(carry):
            ab,rhs=carry
            for i in range(2):
                rhs=rhs.at[i].set(-zz[i+2,0])
                for j in range(4):ab=ab.at[7+i-j,j].set(top[i+2,j])
            return ab,rhs
        def continuity(carry):
            ab,rhs=carry
            for i in range(4):
                row=4*k-2+i
                rhs=rhs.at[row].set(-previous_z[i]+zz[i,0])
                for j in range(4):
                    col=4*(k-1)+j
                    ab=ab.at[7+row-col,col].set(previous[i,j])
                    col=4*k+j
                    ab=ab.at[7+row-col,col].set(-top[i,j])
            return ab,rhs
        ab,rhs=lax.cond(k==0,top_rows,continuity,(ab,rhs))
        return ab,rhs,a4,z4,eigen,bottom,zz[:,1]
    ab,rhs,a4,z4,eigen,bottom,zzbottom=lax.fori_loop(0,n,layer,(ab,rhs,a4,z4,eigen,bottom,zzbottom))
    v1=.2113247*albedo;v2=.7886753*albedo;v3=albedo*u0*f0*jnp.exp(-t[-1]/u0)
    fw1=v1*zzbottom[2];fw2=v2*zzbottom[3]
    rhs=rhs.at[-2].set(-(zzbottom[0]-fw1-fw2-v3)).at[-1].set(-(zzbottom[1]-fw1-fw2-v3))
    for i in range(2):
        row=n4-2+i
        for j in range(4):
            col=n4-4+j
            fw1=v1*bottom[2,j];fw2=v2*bottom[3,j]
            ab=ab.at[7+row-col,col].set(bottom[i,j]-fw1-fw2)
    g4=qcfel(ab,rhs).reshape((n,4))
    return eigen,a4,z4,g4


def solar_transfer(albedo,u0,f0,moments,omega,total_depth):
    """WRF qfts: upward/downward, direct horizontal and diffuse flux profiles."""
    n=omega.shape[0]
    w123,w,t=adjust(moments,omega,total_depth)
    eigen,a4,z4,g4=solar_four_stream(albedo,u0,f0,w123,w,t)
    up=jnp.zeros((n+1,),w.dtype);down=jnp.zeros_like(up);direct=jnp.zeros_like(up)
    def interface(i,carry):
        up,down,direct=carry
        k=jnp.maximum(i-1,0)
        delta=jnp.where(k==0,t[0],t[k]-t[jnp.maximum(k-1,0)])
        x_top=jnp.stack((jnp.asarray(1.,w.dtype),jnp.asarray(1.,w.dtype),jnp.exp(-eigen[0,0]*t[0]),jnp.exp(-eigen[0,1]*t[0])))
        x_bottom=jnp.stack((jnp.exp(-eigen[k,1]*delta),jnp.exp(-eigen[k,0]*delta),jnp.asarray(1.,w.dtype),jnp.asarray(1.,w.dtype)))
        x=jnp.where(i==0,x_top,x_bottom)
        y=jnp.where(i==0,1.,jnp.exp(-t[k]/u0))
        fi=z4[k]*y
        for ii in range(4):
            product=g4[k,ii]*x[ii]
            fi=fi+a4[k,:,ii]*product
        fu=.6638961*fi[1]+2.4776962*fi[0]
        fd=.6638961*fi[2]+2.4776962*fi[3]+u0*3.14159*f0*y
        return up.at[i].set(fu),down.at[i].set(fd),direct.at[i].set(u0*3.14159*f0*y)
    up,down,direct=lax.fori_loop(0,n+1,interface,(up,down,direct))
    return up,down,direct,down-direct
