"""Pristine Noah-MP simple and Hooghoudt drainage, without WRF-Hydro."""
from gpuwrf.physics.noahmp.land_columns import layer_row
from typing import NamedTuple
from jax import lax
import jax.numpy as jnp
from gpuwrf.physics.noahmp.land_columns import stack_layers,in_column_kernel


class TileParameters(NamedTuple):
    tdsmc_fac: object
    td_dc: object
    td_depth: object
    td_dcoef: object
    klat_fac: object
    td_adepth: object
    td_ddrain: object
    td_spac: object
    td_radi: object
    equivalent_depth: object


def equivalent_depth(depth,spacing,radius):
    """TD_EQUIVALENT_DEPTH:9058-9100, precomputed once from static tables."""
    depth,spacing,radius=(jnp.asarray(v,jnp.float32) for v in (depth,spacing,radius))
    pii=jnp.float32(22.)/jnp.float32(7.)
    x=(2.*pii*depth)/spacing
    def body(k,state):
        total,result,done=state
        i=jnp.asarray(2*k+1,jnp.float32)
        ex=jnp.exp(-2.*i*x);term=(4.*ex)/(i*(1.-ex))
        total=jnp.where(done,total,total+term)
        found=(term<1.e-6)&~done
        value=((pii*spacing)/8.)/(jnp.log(spacing/(pii*radius))+total)
        return total,jnp.where(found,value,result),done|found
    _,large,_=lax.fori_loop(0,23,body,(jnp.zeros_like(x),jnp.zeros_like(x),x<=.5))
    safe=jnp.where(x<1.e-8,1.,x)
    fx=((pii*pii)/(4.*safe))+jnp.log(safe/(2.*pii))
    small=((pii*spacing)/8.)/(jnp.log(spacing/(pii*radius))+fx)
    return jnp.where(x>.5,large,jnp.where(x<1.e-8,depth,small))


def simple_tile(p,sh2o,smc,sice,zsoil,smcref,dt,*,drain_layer_opt):
    """TILE_DRAIN:8688-8865; all six source layer selections."""
    dz=(stack_layers((-layer_row(zsoil,0),*(layer_row(zsoil,k-1)-layer_row(zsoil,k) for k in range(1,4)))) if in_column_kernel()
        else jnp.concatenate((-zsoil[:1],zsoil[:-1]-zsoil[1:]),axis=0))
    over=(sh2o-(p.tdsmc_fac*(smcref-sice)))*dz*1000.
    tddc=p.td_dc*dt/(jnp.float32(24.)*jnp.float32(3600.))
    if drain_layer_opt==0:
        selected=jnp.arange(1,5).reshape((4,)+(1,)*(sh2o.ndim-1))==p.td_depth
        amount=jnp.where(selected&(over>0.),jnp.minimum(over,tddc),0.)
        updated=jnp.where(selected&(over>0.),sh2o-amount/(dz*1000.),sh2o)
        updatedsmc=jnp.where(selected&(over>0.),updated+sice,smc)
        volume=jnp.zeros_like(dt)
        for k in range(4):volume=volume+layer_row(amount,k)
    else:
        indices={1:(0,1),2:(0,1,2),3:(1,2),4:(2,3),5:(0,1,2,3)}[drain_layer_opt]
        over=jnp.where(over<0.,0.,over);total=jnp.zeros_like(dt)
        for k in indices:total=total+layer_row(over,k)
        safe=jnp.where(total==0.,1.,total)
        fractions=[jnp.where(layer_row(over,k)!=0.,layer_row(over,k)/safe,0.) for k in range(4)]
        volume=jnp.where(total>tddc,tddc,total)
        layers=[];moist=[]
        for k in range(4):
            new=layer_row(sh2o,k)-(fractions[k]*volume)/(layer_row(dz,k)*1000.)
            active=(total>0.) if k in indices else jnp.zeros_like(total,dtype=bool)
            layers.append(jnp.where(active,new,layer_row(sh2o,k)));moist.append(jnp.where(active,new+layer_row(sice,k),layer_row(smc,k)))
        updated=stack_layers(layers);updatedsmc=stack_layers(moist)
        volume=jnp.where(total>0.,volume,0.)
    return updated,updatedsmc,volume/dt


def find_water_table(smc,smcref,smcwlt,zsoil):
    """TD_FINDZWAT:9104-9154; bottom-contiguous saturated layers only."""
    sat=jnp.zeros_like(layer_row(smc,0),dtype=jnp.int32)
    for k in range(3,-1,-1):
        selected=(layer_row(smc,k)>=layer_row(smcref,k))&(layer_row(smcref,k)>layer_row(smcwlt,k))
        selected=selected&((sat==k+2)|(k==3))
        sat=jnp.where(selected,jnp.int32(k+1),sat)
    zwt=-layer_row(zsoil,3)
    for k in range(1,4):zwt=jnp.where(sat==k+1,-layer_row(zsoil,k-1),zwt)
    return jnp.where(sat==1,0.,zwt)


def hooghoudt_tile(p,wcnd,sh2o,smc,sice,zsoil,smcref,smcwlt,dt,dx):
    """TILE_HOOGHOUDT:8869-9054, exact drainage order and INT drain count."""
    dz=(stack_layers((-layer_row(zsoil,0),*(layer_row(zsoil,k-1)-layer_row(zsoil,k) for k in range(1,4)))) if in_column_kernel()
        else jnp.concatenate((-zsoil[:1],zsoil[:-1]-zsoil[1:]),axis=0))
    zwt=find_water_table(smc,smcref,smcwlt,zsoil)
    yy=jnp.where(zwt>p.td_adepth,p.td_adepth,zwt)
    dtop=jnp.zeros_like(dt);depths=[];total=jnp.zeros_like(dt);lateral=jnp.zeros_like(dt)
    over=[];available=jnp.zeros_like(dt)
    for k in range(4):
        saturated=jnp.where(yy>(-1.*layer_row(zsoil,k)),0.,(-1.*layer_row(zsoil,k))-yy)
        xx=(-1.*layer_row(zsoil,k))-dtop
        saturated=jnp.where(saturated>xx,xx,saturated)
        depths.append(saturated);dtop=-1.*layer_row(zsoil,k)
        ov=(layer_row(sh2o,k)-(layer_row(smcref,k)-layer_row(sice,k)))*layer_row(dz,k)*1000.
        ov=jnp.where(ov<0.,0.,ov);over.append(ov);available=available+ov
    for k in range(4):
        klatk=layer_row(wcnd,k)*p.klat_fac*dt
        lateral=lateral+depths[k]*klatk;total=total+depths[k]
    total=jnp.where(total<.001,.001,total);lateral=jnp.where(lateral<.001,0.,lateral)
    klat=lateral/total;head=p.td_ddrain-yy
    flux=((8.*klat*p.equivalent_depth*head)+(4.*klat*head*head))/(p.td_spac*p.td_spac)
    flux=jnp.where(head<=0.,0.,flux)
    flux=flux*1000.
    tddc=p.td_dcoef*1000.*dt/(jnp.float32(24.)*jnp.float32(3600.))
    flux=jnp.where(flux<=0.,0.,flux);flux=jnp.where(flux>tddc,tddc,flux)
    ndrains=jnp.trunc(dx/p.td_spac).astype(jnp.int32)
    flux=flux*ndrains;flux=jnp.where(flux>available,available,flux)
    remaining=flux;done=jnp.zeros_like(flux,dtype=bool);layers=[];moist=[]
    for k in range(4):
        active=(remaining>0.)&(depths[k]>0.)&(layer_row(over,k)>0.)&~done
        rest=layer_row(over,k)-remaining
        baseline=layer_row(smcref,k)-layer_row(sice,k)
        updated=jnp.where(rest>0.,baseline+rest/(layer_row(dz,k)*1000.),baseline)
        layers.append(jnp.where(active,updated,layer_row(sh2o,k)));moist.append(jnp.where(active,updated+layer_row(sice,k),layer_row(smc,k)))
        remaining=jnp.where(active&(rest<=0.),remaining-layer_row(over,k),remaining)
        done=done|(active&(rest>0.))
    return stack_layers(layers),stack_layers(moist),flux/dt,zwt
