"""CPU-device memory hygiene only; no model, arithmetic or GPU-path changes."""
import ctypes,dataclasses,gc,json,os,resource
from pathlib import Path
EVENTS=[]
def release(jax,where):
    before=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    jax.clear_caches();gc.collect()
    before_trim=int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
    libc=ctypes.CDLL(None);libc.malloc_trim.argtypes=[ctypes.c_size_t];libc.malloc_trim.restype=ctypes.c_int
    trim_result=libc.malloc_trim(0)
    # Current RSS, not the monotonically increasing peak, measures retained state.
    rss=int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
    EVENTS.append(dict(where=where,peak_before_KiB=before,before_trim_RSS_bytes=before_trim,malloc_trim_result=trim_result,current_RSS_bytes=rss))
    Path('CPU_MEMORY_HYGIENE.json').write_text(json.dumps(dict(scope='CPU smoke only; math/model flags/cap unchanged; compiler cache/unused heap release',events=EVENTS),indent=2)+'\n')
def install(pipeline,jax):
    from gpuwrf.runtime import domain_tree as dt
    original=dt._prepare_operational_domain_tree_runtime
    def factory(*args,**kwargs):
        release(jax,'after_initialization_before_runtime')
        runtime=original(*args,**kwargs)
        def advance(name,*values):
            result=runtime.advance(name,*values);jax.block_until_ready(result)
            release(jax,'after_'+name)
            return result
        def lookup(name):
            fn=runtime.fused_cascade(name) if runtime.fused_cascade else None
            if fn is None:return None
            def fused(*values):
                result=fn(*values);jax.block_until_ready(result);release(jax,'after_fused_'+name);return result
            return fused
        return dataclasses.replace(runtime,advance=advance,fused_cascade=lookup if runtime.fused_cascade else None)
    dt._prepare_operational_domain_tree_runtime=factory;pipeline._prepare_operational_domain_tree_runtime=factory
    import atexit
    atexit.register(lambda:Path('CPU_MEMORY_HYGIENE.json').write_text(json.dumps(dict(scope='CPU smoke only; same math/model flags/cap; unused compiler caches cleared',events=EVENTS),indent=2)+'\n'))
