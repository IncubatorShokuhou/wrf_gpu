"""CPU-only diagnostic returning actual final-RK native QC-PD operands.

No operator recomputation or callback in the step. Extra outputs change this
diagnostic program, so its recorded events must be compared with the OFF arm.
This helper is never imported by the forecast.
"""
import os
from dataclasses import replace

import jax
import jax.numpy as jnp


def make_step(om, namelist, clock_base):
    assert jax.devices()[0].platform=='cpu'
    if os.environ.get('GPUWRF_DYN_METRICS_FP32','0')=='1' and os.environ.get('GPUWRF_DYN_CARRY_FP32','0')=='1':
        from gpuwrf.kernels.dyn_rk_fp32 import real_metrics
        namelist=replace(namelist,metrics=real_metrics(namelist.metrics))
    assert namelist.run_physics

    def step(carry,index):
        def branch(run_radiation):
            def execute(c):
                captured={}
                original=om._rk_scan_step
                def observed(*args,**kwargs):
                    result=original(*args,**kwargs)
                    captured.update(kwargs['f2_pd_out'])
                    return result
                om._rk_scan_step=observed
                try:
                    result=om._physics_boundary_step(om._layout_pin('carry',c),namelist,index,
                        run_radiation=run_radiation,debug=False,clock_base=clock_base)
                finally:
                    om._rk_scan_step=original
                args=captured['native_args'];qc=captured['species'].index('qc')
                names=('ru','rv','rom','mut','muold','c1','c2','mx','my','rdzw','fzm','fzp')
                arrays=dict(zip(names,args[2:],strict=True))
                arrays.update(q=args[0][qc],qold=args[1][qc],mub=jnp.zeros_like(args[5]),
                              native_tendency=captured['native_tendency'][qc],
                              merged_tendency=captured['merged_tendency'],
                              update_mu=captured['mu_update'],before_flow_qc=captured['before_flow'])
                for name,leaf in (('ux','msfux'),('uy','msfuy'),('vx','msfvx'),('vy','msfvy')):
                    arrays[name]=jnp.asarray(getattr(namelist.metrics,leaf),jnp.float32)
                arrays.update({name:jnp.float32(captured['native_params'][name]) for name in ('rdx','rdy','dt')})
                return om._layout_pin('carry',result),arrays
            return execute
        radiation=(index-1)%int(namelist.radiation_cadence_steps)==0
        return jax.lax.cond(radiation,branch(True),branch(False),carry)
    return jax.jit(step)
