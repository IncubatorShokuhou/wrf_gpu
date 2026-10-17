"""Preserve optional natural-soil prognostics across WRF glacier partitions."""
import jax
import jax.numpy as jnp


def option_glacier_step(step,land,forcing,static,dt,*,energy_params,rad_params,
                        history=False,lake_bundle=None,frozen_land=None,**options):
    from gpuwrf.coupling.glacier_surface_hook import route_glacier_with_lakes,_gather,_scatter
    plan=static.glacier;shape=plan.shape
    nonlake=None if lake_bundle is None else set(lake_bundle.nonlake_indices)
    positions=tuple(i for i in plan.natural_indices if nonlake is None or i in nonlake)
    indices=jnp.asarray(positions,jnp.int32)
    selected={key:_gather(value,indices,shape) for key,value in options.items()}
    # The existing router calls the natural model once at trace time. Its
    # prognostic results remain device arrays, never host transfers.
    auxiliary=[]
    natural_history={}
    def natural(l,f,s,dt,**kwargs):
        # Layer-selected soil/hydrology parameters are per-cell fields; the
        # retained glacier router's original table parameters were uniform.
        s=s.replace(parameters=_gather(static.parameters,indices,shape))
        result=step(l,f,s,dt,**kwargs,**selected)
        auxiliary.extend(result[2:])
        natural_history.update(result[1].history or {})
        return result[:2]
    result=route_glacier_with_lakes(natural,land,forcing,static,dt,
        energy_params=energy_params,rad_params=rad_params,history=history,
        lake_bundle=lake_bundle,frozen_land=frozen_land)
    if history:
        # The glacier model has no BGAP/WGAP or carbon diagnostics. Its
        # original cold allocation holds zero while natural tiles update.
        routed,flux=result
        extra={key:_scatter(None,value,indices,shape) for key,value in natural_history.items()
               if key not in flux.history}
        result=routed,flux._replace(history={**flux.history,**extra})
    retained=[value[1] if key in ('irrigation','carbon') else value
              for key,value in options.items() if key!='options']
    if not positions:return (*result,*retained)
    if len(auxiliary)!=len(retained):raise ValueError('Noah-MP option tile carry ABI differs')
    restored=[jax.tree.map(lambda a,b:_scatter(a,b,indices,shape),old,new)
              for old,new in zip(retained,auxiliary,strict=True)]
    return (*result,*restored)
