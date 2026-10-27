"""Diagnostic Python call accounting only during the existing CUDA capture.

The profiler changes host dispatch cost. Its wall is never a speed claim.
No array inspection, JAX call replacement, environment capture or GPU calls.
"""
import cProfile
import pstats


def install(h):
    original=h._cuda_profiler
    profiler=cProfile.Profile()
    active=False
    started=None

    def observe(event):
        nonlocal active,started
        if event=='cudaProfilerStart':
            assert not active,'duplicate host profile start'
            result=original(event)
            started=h._t();active=True;profiler.enable()
            return result
        if event=='cudaProfilerStop' and active:
            profiler.disable();ended=h._t();active=False
            stats=pstats.Stats(profiler)
            rows=[]
            for (file,line,fn),(primitive,calls,self_s,cumulative,_) in stats.stats.items():
                rows.append(dict(file=file,line=line,function=fn,primitive_calls=primitive,
                                 calls=calls,self_s=self_s,cumulative_s=cumulative))
            h.REC['w44_host_profile']={'start':started,'end':ended,'scope':__doc__,
                                      'total_calls':stats.total_calls,'total_s':stats.total_tt,
                                      'top_self':sorted(rows,key=lambda x:-x['self_s'])[:100],
                                      'top_cumulative':sorted(rows,key=lambda x:-x['cumulative_s'])[:100]}
        return original(event)

    h._cuda_profiler=observe
