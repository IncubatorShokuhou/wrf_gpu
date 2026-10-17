"""WRF glacier hydrology/snow layers (:1998-2972), register-list form."""
from __future__ import annotations
import numpy as np
import jax.numpy as jnp
from .glacier_energy import at, put, TFRZ, HFUS, HSUB, CICE, CWAT, DENICE, DENH2O


def combo(d, l, i, t, d2, l2, i2, t2, xp):
    dc=d+d2;ic=i+i2;lc=l+l2
    h=(CICE*i+CWAT*l)*(t-TFRZ)+HFUS*l
    h2=(CICE*i2+CWAT*l2)*(t2-TFRZ)+HFUS*l2
    hc=h+h2;cap=CICE*ic+CWAT*lc
    cap=xp.where(cap!=0.,cap,1.)
    tc=xp.where(hc<0.,TFRZ+hc/cap,xp.where(hc<=HFUS*lc,TFRZ,TFRZ+(hc-HFUS*lc)/cap))
    return dc,lc,ic,tc


def snowfall(v,dz,rate,xp):
    n=v['isnow'];q=v['qsnow'];dt=v['dt']
    hit=(n==0)&(q>0.)
    v['snowh']=xp.where(hit,v['snowh']+rate*dt,v['snowh'])
    v['sneqv']=xp.where(hit,v['sneqv']+q*dt,v['sneqv'])
    new=(n==0)&(q>0.)&(v['snowh']>=np.float32(.05))
    v['isnow']=xp.where(new,-1,n)
    dz[2]=xp.where(new,v['snowh'],dz[2]);v['snowh']=xp.where(new,0.,v['snowh'])
    v['stc'][2]=xp.where(new,xp.minimum(TFRZ,v['sfctmp']),v['stc'][2])
    v['snice'][2]=xp.where(new,v['sneqv'],v['snice'][2]);v['snliq'][2]=xp.where(new,0.,v['snliq'][2])
    hit=(v['isnow']<0)&(~new)&(q>0.);idx=v['isnow']+3
    put(v['snice'],idx,at(v['snice'],idx,xp)+q*dt,hit,xp)
    put(dz,idx,at(dz,idx,xp)+rate*dt,hit,xp)


def compact(v,dz,imelt,xp):
    burden=xp.zeros_like(v['tg'])
    for k in range(3):
        active=k>=v['isnow']+3
        wx=v['snice'][k]+v['snliq'][k]
        depth=xp.where(active,dz[k],1.)
        fice=v['snice'][k]/xp.where(active,wx,1.)
        void=1.-(v['snice'][k]/DENICE+v['snliq'][k]/DENH2O)/depth
        hit=active&(void>np.float32(.001))&(v['snice'][k]>np.float32(.1))
        bi=v['snice'][k]/depth;td=xp.maximum(0.,TFRZ-v['stc'][k])
        dd1=-np.float32(2.5e-6)*xp.exp(-np.float32(.04)*td)
        dd1=xp.where(bi>100.,dd1*xp.exp(-np.float32(.046)*(bi-100.)),dd1)
        dd1=xp.where(v['snliq'][k]>np.float32(.01)*depth,dd1*2.,dd1)
        dd2=-(burden+np.float32(.5)*wx)*xp.exp(-np.float32(.08)*td-np.float32(.021)*bi)/np.float32(.8e6)
        dd3=xp.where(imelt[k]==1,-xp.maximum(0.,(v['ficeold'][k]-fice)/xp.maximum(np.float32(1.e-6),v['ficeold'][k]))/v['dt'],0.)
        change=xp.maximum(-.5,(dd1+dd2+dd3)*v['dt'])
        dz[k]=xp.where(hit,dz[k]*(1.+change),dz[k])
        burden=burden+xp.where(active,wx,0.)


def combine(v,dz,sice,mask,xp):
    oldn=v['isnow'];i=v['snice'];l=v['snliq'];t=v['stc']
    for j in range(3):
        hit=mask&(j>=oldn+3)&(i[j]<=np.float32(.1))
        if j<2:
            l[j+1]=xp.where(hit,l[j+1]+l[j],l[j+1]);i[j+1]=xp.where(hit,i[j+1]+i[j],i[j+1])
        else:
            deep=hit&(oldn<-1);shallow=hit&(~(oldn<-1))
            l[j-1]=xp.where(deep,l[j-1]+l[j],l[j-1]);i[j-1]=xp.where(deep,i[j-1]+i[j],i[j-1])
            v['ponding1']=xp.where(shallow,v['ponding1']+l[j],v['ponding1'])
            v['sneqv']=xp.where(shallow,i[j],v['sneqv']);v['snowh']=xp.where(shallow,dz[j],v['snowh'])
            l[j]=xp.where(shallow,0.,l[j]);i[j]=xp.where(shallow,0.,i[j]);dz[j]=xp.where(shallow,0.,dz[j])
        shift=hit&(j>v['isnow']+3)&(v['isnow']<-1)
        for k in range(j,0,-1):
            move=shift&(k>=v['isnow']+4)
            for a in (t,l,i,dz):
                a[k]=xp.where(move,a[k-1],a[k])
        v['isnow']=xp.where(hit,v['isnow']+1,v['isnow'])
    hit=mask&(sice[0]<0.)
    v['sh2o'][0]=xp.where(hit,v['sh2o'][0]+sice[0],v['sh2o'][0]);sice[0]=xp.where(hit,0.,sice[0])
    live=mask&(v['isnow']!=0)
    swe=xp.zeros_like(v['tg']);depth=xp.zeros_like(swe);ice=xp.zeros_like(swe);liq=xp.zeros_like(swe)
    for k in range(3):
        active=k>=v['isnow']+3
        swe=xp.where(active,swe+i[k]+l[k],swe);depth=xp.where(active,depth+dz[k],depth)
        ice=xp.where(active,ice+i[k],ice);liq=xp.where(active,liq+l[k],liq)
    v['sneqv']=xp.where(live,swe,v['sneqv']);v['snowh']=xp.where(live,depth,v['snowh'])
    thin=live&(v['snowh']<np.float32(.05))&(v['isnow']<0)
    v['isnow']=xp.where(thin,0,v['isnow']);v['sneqv']=xp.where(thin,ice,v['sneqv'])
    v['ponding2']=xp.where(thin,v['ponding2']+liq,v['ponding2'])
    v['snowh']=xp.where(thin&(v['sneqv']<=0.),0.,v['snowh'])
    oldn=v['isnow'];live=live&(oldn<-1);mssi=xp.zeros_like(oldn)
    for k in range(3):
        active=live&(k>=oldn+3)
        threshold=at([xp.full_like(swe,x) for x in (.045,.05,.2)],mssi,xp)
        hit=active&(dz[k]<threshold)
        if k==0:
            neighbor=xp.ones_like(oldn)
        elif k==2:
            neighbor=xp.ones_like(oldn)
        else:
            neighbor=xp.where(k==v['isnow']+3,k+1,xp.where(dz[k-1]+dz[k]<dz[k+1]+dz[k],k-1,k+1))
        j=xp.maximum(neighbor,k);other=xp.minimum(neighbor,k)
        vals=combo(at(dz,j,xp),at(l,j,xp),at(i,j,xp),at(t,j,xp),
                   at(dz,other,xp),at(l,other,xp),at(i,other,xp),at(t,other,xp),xp)
        for arr,value in zip((dz,l,i,t),vals):
            put(arr,j,value,hit,xp)
        for z in range(1,0,-1):
            shift=hit&(j-1>v['isnow']+3)&(z<=j-1)&(z>=v['isnow']+4)
            for arr in (t,i,l,dz):
                arr[z]=xp.where(shift,arr[z-1],arr[z])
        v['isnow']=xp.where(hit,v['isnow']+1,v['isnow'])
        live=live & (~(hit&(v['isnow']>=-1)))
        mssi=xp.where(active&(~hit),mssi+1,mssi)


def divide(v,dz,mask,xp):
    n=v['isnow'];m=-n;zero=xp.zeros_like(v['tg'])
    arrays=[[xp.where(j<m,at(arr,j+n+3,xp),zero) for j in range(3)] for arr in (dz,v['snliq'],v['snice'],v['stc'])]
    d,l,i,t=arrays
    hit=mask&(m==1)&(d[0]>.05);m=xp.where(hit,2,m)
    for arr in (d,i,l):
        arr[0]=xp.where(hit,arr[0]/2.,arr[0]);arr[1]=xp.where(hit,arr[0],arr[1])
    t[1]=xp.where(hit,t[0],t[1])
    hit=mask&(m>1)&(d[0]>.05)
    excess=d[0]-np.float32(.05);fraction=excess/xp.where(hit,d[0],1.)
    wi=fraction*i[0];wl=fraction*l[0];fraction=np.float32(.05)/xp.where(hit,d[0],1.)
    i[0]=xp.where(hit,fraction*i[0],i[0]);l[0]=xp.where(hit,fraction*l[0],l[0]);d[0]=xp.where(hit,.05,d[0])
    vals=combo(d[1],l[1],i[1],t[1],excess,wl,wi,t[0],xp)
    for arr,value in zip(arrays,vals):arr[1]=xp.where(hit,value,arr[1])
    split=hit&(m<=2)&(d[1]>.1);m=xp.where(split,3,m)
    gradient=(t[0]-t[1])/xp.where(split,(d[0]+d[1])/2.,1.)
    for arr in (d,i,l):
        arr[1]=xp.where(split,arr[1]/2.,arr[1]);arr[2]=xp.where(split,arr[1],arr[2])
    lower=t[1]-gradient*d[1]/2.
    t[2]=xp.where(split,xp.where(lower>=TFRZ,t[1],lower),t[2])
    t[1]=xp.where(split&(lower<TFRZ),t[1]+gradient*d[1]/2.,t[1])
    hit=mask&(m>2)&(d[1]>.2)
    excess=d[1]-np.float32(.2);fraction=excess/xp.where(hit,d[1],1.)
    wi=fraction*i[1];wl=fraction*l[1];fraction=np.float32(.2)/xp.where(hit,d[1],1.)
    i[1]=xp.where(hit,fraction*i[1],i[1]);l[1]=xp.where(hit,fraction*l[1],l[1]);d[1]=xp.where(hit,.2,d[1])
    vals=combo(d[2],l[2],i[2],t[2],excess,wl,wi,t[1],xp)
    for arr,value in zip(arrays,vals):arr[2]=xp.where(hit,value,arr[2])
    v['isnow']=xp.where(mask,-m,n)
    for k in range(3):
        hit=mask&(k>=v['isnow']+3)
        for arr,local in zip((dz,v['snliq'],v['snice'],v['stc']),arrays):
            arr[k]=xp.where(hit,at(local,k-v['isnow']-3,xp),arr[k])


def snow_h2o(v,dz,sice,rain,qf,qs,opt_gla,xp):
    empty=v['sneqv']==0.
    if opt_gla==1:
        sice[0]=xp.where(empty,sice[0]+(qf-qs)*v['dt']/(dz[3]*1000.),sice[0])
    else:
        v['fsh']=xp.where(empty,v['fsh']-(qf-qs)*HSUB,v['fsh'])
        qf=xp.where(empty,0.,qf);qs=xp.where(empty,0.,qs)
    shallow=(v['isnow']==0)&(v['sneqv']>0.)
    if opt_gla==1:
        old=v['sneqv'];new=old-qs*v['dt']+qf*v['dt']
        v['sneqv']=xp.where(shallow,new,old)
        v['snowh']=xp.where(shallow,xp.maximum(0.,new/xp.where(old!=0.,old,1.)*v['snowh']),v['snowh'])
    else:
        v['fsh']=xp.where(shallow,v['fsh']-(qf-qs)*HSUB,v['fsh'])
        qf=xp.where(shallow,0.,qf);qs=xp.where(shallow,0.,qs)
    negative=shallow&(v['sneqv']<0.)
    sice[0]=xp.where(negative,sice[0]+v['sneqv']/(dz[3]*1000.),sice[0])
    v['sneqv']=xp.where(negative,0.,v['sneqv']);v['snowh']=xp.where(negative,0.,v['snowh'])
    hit=shallow&(sice[0]<0.)
    v['sh2o'][0]=xp.where(hit,v['sh2o'][0]+sice[0],v['sh2o'][0]);sice[0]=xp.where(hit,0.,sice[0])
    empty=(v['snowh']<=np.float32(1.e-8))|(v['sneqv']<=np.float32(1.e-6))
    v['snowh']=xp.where(empty,0.,v['snowh']);v['sneqv']=xp.where(empty,0.,v['sneqv'])
    deep=v['isnow']<0;top=v['isnow']+3
    mass=at(v['snice'],top,xp)-qs*v['dt']+qf*v['dt']
    put(v['snice'],top,mass,deep,xp)
    combine(v,dz,sice,deep&(mass<np.float32(1.e-6)),xp)
    top=v['isnow']+3;deep=deep&(v['isnow']<0)
    put(v['snliq'],top,xp.maximum(0.,at(v['snliq'],top,xp)+rain*v['dt']),deep,xp)
    vi=[];vl=[];por=[]
    for k in range(3):
        active=k>=v['isnow']+3;depth=xp.where(active,dz[k],1.)
        vi.append(xp.minimum(1.,v['snice'][k]/(depth*DENICE)))
        por.append(1.-vi[k]);vl.append(xp.minimum(por[k],v['snliq'][k]/(depth*DENH2O)))
    qin=xp.zeros_like(v['tg']);qout=xp.zeros_like(qin)
    for k in range(3):
        active=k>=v['isnow']+3
        v['snliq'][k]=xp.where(active,v['snliq'][k]+qin,v['snliq'][k])
        out=xp.maximum(0.,(vl[k]-np.float32(.03)*por[k])*dz[k])
        if k<2:
            out=xp.where((por[k]<.05)|(por[k+1]<.05),0.,xp.minimum(out,(1.-vi[k+1]-vl[k+1])*dz[k+1]))
        qout=xp.where(active,out*1000.,qout)
        v['snliq'][k]=xp.where(active,v['snliq'][k]-qout,v['snliq'][k]);qin=xp.where(active,qout,qin)
    v['qsnbot']=qout/v['dt']
    return qf,qs


def water(v,dz,imelt,opt_gla,opt_snf,xp=jnp):
    zero=xp.zeros_like(v['tg']);dt=v['dt']
    sice=[xp.maximum(0.,s-l) for s,l in zip(v['smc'],v['sh2o'])]
    oldice=list(sice);oldliq=list(v['sh2o'])
    temp=v['sfctmp']
    if opt_snf in (1,4):
        fp=xp.where(temp>TFRZ+np.float32(2.5),0.,xp.where(temp<=TFRZ+np.float32(.5),1.,
                    xp.where(temp<=TFRZ+np.float32(2.),1.-(-np.float32(54.632)+np.float32(.2)*temp),np.float32(.6))))
    elif opt_snf==2:
        fp=xp.where(temp>=TFRZ+np.float32(2.2),np.float32(0.),np.float32(1.))
    elif opt_snf==3:
        fp=xp.where(temp>=TFRZ,np.float32(0.),np.float32(1.))
    else:
        # Pristine glacier WATER has no OPT_SNF=5 branch. Caller admission rejects it.
        raise ValueError('glacier OPT_SNF requires 1, 2, 3 or 4 in WRF 4.7.1')
    rain=v['prcp']*(1.-fp);v['qsnow']=v['prcp']*fp;v['fpice']=fp
    density=xp.minimum(120.,np.float32(67.92)+np.float32(51.25)*xp.exp((temp-TFRZ)/np.float32(2.59)))
    qf=xp.abs(xp.minimum(v['fgev']/HSUB,0.));qs=xp.maximum(v['fgev']/HSUB,0.)
    v['sneqvo']=v['sneqv'];v['edir']=qs-qf
    v['ponding1']=zero;v['ponding2']=zero
    snowfall(v,dz,v['qsnow']/density,xp)
    compact(v,dz,imelt,xp)
    layers=v['isnow']<0
    combine(v,dz,sice,layers,xp)
    divide(v,dz,layers,xp)
    for k in range(3):
        empty=k<v['isnow']+3
        for arr in (v['snice'],v['snliq'],v['stc'],dz,v['zsnso']):arr[k]=xp.where(empty,0.,arr[k])
    qf,qs=snow_h2o(v,dz,sice,rain,qf,qs,opt_gla,xp)
    flow=v['sneqv']>5000.
    bulk=v['snice'][2]/xp.where(flow,dz[2],1.);excess=v['sneqv']-5000.
    v['snice'][2]=xp.where(flow,v['snice'][2]-excess,v['snice'][2])
    dz[2]=xp.where(flow,dz[2]-excess/xp.where(flow,bulk,1.),dz[2])
    snowflow=xp.where(flow,excess/dt,0.)
    swe=zero
    for k in range(3):swe=xp.where(k>=v['isnow']+3,swe+v['snice'][k]+v['snliq'][k],swe)
    v['sneqv']=xp.where(v['isnow']!=0,swe,v['sneqv'])
    for k in range(3):dz[k]=xp.where(k>=v['isnow']+3,-dz[k],dz[k])
    for k in range(4):dz[k+3]=v['zsoil'][k]-(0. if k==0 else v['zsoil'][k-1])
    for k in range(7):
        z=dz[k] if k==0 else xp.where(k==v['isnow']+3,dz[k],v['zsnso'][k-1]+dz[k])
        v['zsnso'][k]=xp.where(k>=v['isnow']+3,z,v['zsnso'][k])
    for k in range(7):dz[k]=xp.where(k>=v['isnow']+3,-dz[k],dz[k])
    v['runsrf']=(v['ponding']+v['ponding1']+v['ponding2'])/dt
    v['runsrf']=v['runsrf']+v['qsnbot']+xp.where(v['isnow']==0,rain,0.)
    replacement=zero
    if opt_gla==1:
        for k in range(4):replacement=replacement+dz[k+3]*(sice[k]-oldice[k]+v['sh2o'][k]-oldliq[k])
        replacement=replacement*1000./dt
        sice=[xp.minimum(1.,x) for x in oldice]
    else:
        sice=[xp.ones_like(zero)]*4
        v['edir']=qs-qf;v['fgev']=v['edir']*HSUB
    v['sh2o']=[1.-x for x in sice];v['runsub']=snowflow+replacement if opt_gla==1 else snowflow
