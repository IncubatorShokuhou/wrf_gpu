"""CPU-only profiler forwarding/window control; no JAX/NVML import."""
import sys
import time
from types import SimpleNamespace
from host_observer import install

events=[]
sentinel=object()
def original(event):events.append(event);return sentinel
h=SimpleNamespace(REC={},_cuda_profiler=original,_t=time.perf_counter)
install(h)
assert h._cuda_profiler('cudaProfilerStart') is sentinel
def workload():return sum(range(1000))
assert workload()==499500
assert h._cuda_profiler('cudaProfilerStop') is sentinel
assert events==['cudaProfilerStart','cudaProfilerStop']
rec=h.REC['w44_host_profile']
assert rec['end']>=rec['start'] and rec['total_calls']>0
assert any(x['function']=='workload' and x['calls']==1 for x in rec['top_cumulative'])
assert h._cuda_profiler('other') is sentinel and events[-1]=='other'
assert 'jax' not in sys.modules
print('PASS host profile window, forwarding identity, captured call, no GPU import')
