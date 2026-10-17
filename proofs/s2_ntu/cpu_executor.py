"""Validation-only NumPy execution of the shared NTU column equations.

The complete LLVM program exceeds the manager's 6 GiB CPU cap even with one
column. The existing NumPy reference backend executes the same column body
without that compilation. This proves caller/carry coupling, not JAX or GPU
execution. Original-WRF and native compiled column receipts are separate.
"""
import time
import resource
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp


class CPUExecutor:
    def __init__(self):
        self.receipts=[]

    def __enter__(self):
        from gpuwrf.physics.ntu import column
        assert jax.devices()[0].platform=='cpu'
        self.original=column.step
        column.step=self
        return self

    def __exit__(self,*args):
        from gpuwrf.physics.ntu import column
        column.step=self.original

    def __call__(self,fields,dt,*,ccnty=2,xp=jnp,output_names=None,observer=None):
        assert xp is jnp and observer is None
        values={n:np.asarray(a) for n,a in fields.items()}
        start=time.perf_counter()
        with np.errstate(all='ignore'):
            outputs=self.original(values,np.asarray(dt),ccnty=ccnty,xp=np,
                                  output_names=output_names)
        self.receipts.append(dict(backend='NumPy shared column body',CPU=True,
            GPU=False,JAX_NTU_execution=False,ccnty=ccnty,
            profile_shape=list(values['th'].shape),wall_s=time.perf_counter()-start,
            input_bytes=sum(a.nbytes for a in values.values()),
            output_bytes=sum(a.nbytes for a in outputs.values()),
            nonfinite=sum(int((~np.isfinite(a)).sum()) for a in outputs.values())))
        print('NTU_CPU_REFERENCE',self.receipts[-1],flush=True)
        with jax.enable_x64(True):
            return {n:jnp.asarray(a) for n,a in outputs.items()}


class CompiledCPUExecutor(CPUExecutor):
    """Unchanged JAX body for the required singleton, manager's 12 GiB scope.

    Dynamic microsteps remain lax loops; LLVM opt0 avoids the previously
    measured opt3 compile timeout. No NumPy NTU arithmetic is used here.
    """
    def __call__(self,fields,dt,*,ccnty=2,xp=jnp,output_names=None,observer=None):
        assert xp is jnp and observer is None
        original=self.original
        with jax.enable_x64(True),jax.disable_jit(False):
            fn=jax.jit(lambda f,d:original(f,d,ccnty=ccnty,xp=jnp,
                                         output_names=output_names))
            start=time.perf_counter();print('NTU_JAX_LOWER',flush=True)
            lowered=fn.lower(fields,dt)
            lower_s=time.perf_counter()-start;print('NTU_JAX_COMPILE',lower_s,flush=True)
            start=time.perf_counter()
            compiled=lowered.compile(compiler_options={'xla_backend_optimization_level':0})
            compile_s=time.perf_counter()-start
            start=time.perf_counter();outputs=compiled(fields,dt);jax.block_until_ready(outputs)
            row=dict(backend='JAX compiled shared column body',CPU=True,GPU=False,
                JAX_NTU_execution=True,ccnty=ccnty,profile_shape=list(fields['th'].shape),
                lower_s=lower_s,compile_s=compile_s,execute_s=time.perf_counter()-start,
                peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                compiler_options={'xla_backend_optimization_level':0})
            cgroup=Path('/proc/self/cgroup').read_text().split('0::',1)[1].strip()
            peak=Path('/sys/fs/cgroup')/cgroup.lstrip('/')/'memory.peak'
            if peak.exists():row['scope_peak_bytes']=int(peak.read_text())
            self.receipts.append(row);print('NTU_JAX_COMPLETED',row,flush=True)
            return outputs
