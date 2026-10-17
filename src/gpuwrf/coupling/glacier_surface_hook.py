"""Partition Noah-MP natural/glacier columns using the WRF caller predicate."""
from __future__ import annotations
import jax
import jax.numpy as jnp
from gpuwrf.contracts.noahmp_state import NoahMPFluxes
from gpuwrf.kernels.phys_noahmp_columns import column_call
from gpuwrf.physics.noahmp.glacier import glacier_column

_OUTPUTS=('isnow','sneqv','sneqvo','albold','cm','ch','smc','sh2o','zsnso','snowh','snice','snliq',
          'tg','stc','tauss','qsfc','qsnow','fsh','fgev','ssoil','trad','albedo','t2m','q2e','emissi',
          'runsrf','runsub','edir')
_HISTORY=('fsa','sag','fira','ch2b','qmelt','hcpct','ponding','ponding1','ponding2','fpice')


def route_glacier_with_lakes(step,land,forcing,static,dt,*,energy_params,rad_params,
                             history=False,lake_bundle=None,frozen_land=None):
    """Compose the existing non-lake partition with glacier/natural routing."""
    if lake_bundle is None or not lake_bundle.indices:
        return route_glacier_step(step,land,forcing,static,dt,energy_params=energy_params,
                                  rad_params=rad_params,history=history)
    from dataclasses import replace
    from gpuwrf.coupling.lake_surface_hook import noahmp_without_lakes
    positions={p:i for i,p in enumerate(lake_bundle.nonlake_indices)}
    plan=static.glacier
    if any(i not in positions for i in plan.indices):
        raise ValueError('input point classified as both lake and glacier')
    local=replace(plan,shape=(1,len(positions)),indices=tuple(positions[i] for i in plan.indices),
                  natural_indices=tuple(positions[i] for i in plan.natural_indices if i in positions))
    def nonlake(l,f,s,dt,**kwargs):
        return route_glacier_step(step,l,f,s.replace(glacier=local),dt,**kwargs)
    return noahmp_without_lakes(nonlake,land,forcing,static,dt,lake_bundle,
        energy_params=energy_params,rad_params=rad_params,history=history,frozen_land=frozen_land)


def _gather(tree, indices, shape):
    def leaf(a):
        if a is None or not hasattr(a,'shape') or a.ndim<2 or tuple(a.shape[-2:])!=shape:
            return a
        return a.reshape((*a.shape[:-2],-1))[...,indices].reshape((*a.shape[:-2],1,len(indices)))
    return jax.tree.map(leaf,tree)


def _scatter(a, b, indices, shape):
    if b is None:
        return a
    if not hasattr(b,'shape') or b.ndim<2:
        return b
    if a is None:
        a=jnp.zeros((*b.shape[:-2],*shape),b.dtype)
    return a.reshape((*a.shape[:-2],-1)).at[...,indices].set(
        b.reshape((*b.shape[:-2],-1)).astype(a.dtype)).reshape(a.shape)


def glacier_inputs(land,forcing,static,dt):
    """Exact noahmpdrv:1049 caller inputs; Q_ML is already specific humidity."""
    wx=land.snice+land.snliq
    fice=jnp.where(wx>0.,land.snice/jnp.where(wx>0.,wx,1.),0.)
    fields=dict(cosz=forcing.cosz,dt=jnp.full_like(land.tg,dt),sfctmp=forcing.sfctmp,
        sfcprs=forcing.sfcprs,uu=forcing.uu,vv=forcing.vv,q2=forcing.qair,soldn=forcing.soldn,
        prcp=forcing.prcpconv+forcing.prcpnonc,lwdn=forcing.lwdn,tbot=jnp.minimum(static.tbot,263.15),
        zlvl=forcing.zlvl,qsnow=land.glacier_qsnow,sneqvo=land.sneqvo,albold=land.albold,
        cm=land.cm,ch=land.ch,isnow=land.isnow,sneqv=land.sneqv,snowh=land.snowh,
        tg=land.tg,tauss=land.tauss,qsfc=land.qsfc,
        smc=list(land.smois),sh2o=list(land.sh2o),snice=list(land.snice),snliq=list(land.snliq),
        zsnso=list(land.zsnso),stc=[*land.tsno,*land.tslb],ficeold=list(fice),
        zsoil=[jnp.full_like(land.tg,z) for z in static.zsoil])
    return jax.tree.map(lambda a:jnp.asarray(a,jnp.int32 if getattr(a,'dtype',None)==jnp.int32 else jnp.float32),fields)


def glacier_kernel(inputs,config,*,history=False):
    # Separate WRF energy/phase/water column bodies avoid the measured
    # multi-minute Triton coalesce cliff of the single large expression graph.
    from gpuwrf.physics.noahmp.glacier_stages import staged_kernel
    return staged_kernel(inputs,config,history=history)


def single_glacier_kernel(inputs,config,*,history=False):
    """Retained same-work compiler comparison, never the operational route."""
    options={key:getattr(config,key) for key in ('opt_gla','opt_alb','opt_snf','opt_tbot','opt_stc')}
    keys=(*_OUTPUTS,*(_HISTORY if history else ()))
    def body(v):
        out=glacier_column(v,**options)
        return {key:out[key] for key in keys}
    return column_call(body,(inputs,),grid_shape=(1,len(config.indices)),name='noahmp_glacier_column',block=32,warps=1)


def route_glacier_step(step,land,forcing,static,dt,*,energy_params,rad_params,history=False):
    config=static.glacier;shape=config.shape
    ice=jnp.asarray(config.indices,jnp.int32);natural=jnp.asarray(config.natural_indices,jnp.int32)
    if land.glacier_qsnow is None:
        raise ValueError('active glacier restart lacks previous QSNOWXY (E78)')
    # Each WRF tile executes its own model; glacier columns never enter SFLX.
    if config.natural_indices:
        ns=static.replace(**{name:_gather(getattr(static,name),natural,shape)
            for name in ('ivgtyp','isltyp','xland','landmask','lakemask','lu_index','tbot','lat','shdmax','shdfac')},glacier=None)
        nl,nm=step(_gather(land,natural,shape),_gather(forcing,natural,shape),ns,dt,
            energy_params=_gather(energy_params,natural,shape),rad_params=_gather(rad_params,natural,shape),
            **({'history':True} if history else {}))
        result=jax.tree.map(lambda a,b:_scatter(a,b,natural,shape),land,nl)
        nm=jax.tree.map(lambda b:_scatter(None,b,natural,shape),nm)
    else:
        result=land;zero=jnp.zeros(shape,land.tg.dtype)
        nm=NoahMPFluxes(zero,zero,zero,zero,land.t_skin,land.qsfc,land.znt,land.emiss,land.albedo,zero,
                        t2=zero,t2mv=zero,t2mb=zero,q2=zero,history={} if history else None)
    gl=_gather(land,ice,shape);gf=_gather(forcing,ice,shape)
    gs=static.replace(tbot=_gather(static.tbot,ice,shape),glacier=None)
    v=glacier_kernel(glacier_inputs(gl,gf,gs,dt),config,history=history)
    updates=dict(tslb=jnp.stack(v['stc'][3:]),smois=jnp.stack(v['smc']),sh2o=jnp.stack(v['sh2o']),
        isnow=v['isnow'],tsno=jnp.stack(v['stc'][:3]),snice=jnp.stack(v['snice']),snliq=jnp.stack(v['snliq']),
        zsnso=jnp.stack(v['zsnso']),snowh=v['snowh'],sneqv=v['sneqv'],sneqvo=v['sneqvo'],tauss=v['tauss'],
        albold=v['albold'],tg=v['tg'],cm=v['cm'],ch=v['ch'],t_skin=v['trad'],qsfc=v['qsfc'],
        znt=jnp.full_like(v['tg'],.002),emiss=v['emissi'],glacier_qsnow=v['qsnow'],
        albedo=jnp.where(v['albedo']>-999.,v['albedo'],gl.albedo),
        sfcrunoff=gl.sfcrunoff+v['runsrf']*dt/1000.,udrunoff=gl.udrunoff+v['runsub']*dt/1000.)
    for name in ('tv','tah','eah','fwet','lai','sai'):
        updates[name]=jnp.full_like(v['tg'],-1.e36)
    for name in ('canliq','canice'):
        updates[name]=jnp.zeros_like(v['tg'])
    result=result.replace(**{k:_scatter(getattr(result,k),a,ice,shape) for k,a in updates.items()})
    fluxes=dict(hfx=v['fsh'],lh=v['fgev'],qfx=v['edir'],grdflx=v['ssoil'],
        tsk=v['trad'],qsfc=v['qsfc'],znt=updates['znt'],emiss=v['emissi'],albedo=updates['albedo'],chs=v['ch'],
        t2=v['t2m'],t2mv=jnp.full_like(v['tg'],-1.e36),t2mb=v['t2m'],q2=v['q2e']/(1.-v['q2e']))
    nm=nm._replace(**{k:_scatter(getattr(nm,k),a,ice,shape) for k,a in fluxes.items()})
    if history:
        from gpuwrf.runtime.history_accumulators import LAND_FLUX_FIELDS
        zero=jnp.zeros_like(v['tg']);undefined=jnp.full_like(zero,-1.e36)
        h={k:zero for k in LAND_FLUX_FIELDS}
        for k in ('APAR','PSN','CHV','CHLEAF','CHUC','CHV2','RSSUN','RSSHA','TGV','T2V'):h[k]=undefined
        h.update(SAG=v['sag'],FSA=v['fsa'],FIRA=v['fira'],EDIR=v['edir'],GRDFLX=v['ssoil'],TRAD=v['trad'],
            CHB=v['ch'],CHB2=v['ch2b'],IRB=v['fira'],SHB=v['fsh'],EVB=v['fgev'],GHB=v['ssoil'],
            T2B=v['t2m'],Q2V=jnp.full_like(zero,-1.),Q2B=fluxes['q2'],RUNSF=v['runsrf']*dt,RUNSB=v['runsub']*dt,
            SNOWC=jnp.ones_like(zero),TGB=v['tg'],SNOM_INCREMENT=v['qmelt']*dt+v['ponding']+v['ponding1']+v['ponding2'],
            QSNOWXY=v['qsnow'],QRAINXY=zero,FORCTLSM=gf.sfctmp,FORCQLSM=gf.qair,FORCPLSM=gf.sfcprs,
            FORCZLSM=gf.zlvl,FORCWLSM=jnp.sqrt(gf.uu**2+gf.vv**2))
        # HCPCT comes from the same glacier energy call, never a natural-soil re-solve.
        dz=[-v['zsnso'][0]]+[v['zsnso'][k-1]-v['zsnso'][k] for k in range(1,7)]
        soil=zero;snow=zero
        for k in range(7):
            term=dz[k]*v['hcpct'][k]*(v['stc'][k]-jnp.float32(273.16))*.001
            if k<3:snow=snow+jnp.where(k>=v['isnow']+3,term,0.)
            else:soil=soil+term
        h.update(SOILENERGY=soil,SNOWENERGY=snow)
        nm=nm._replace(history={k:_scatter(nm.history.get(k),a,ice,shape) for k,a in h.items()})
    return result,nm
