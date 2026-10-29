"""Explicit NVTX per actual program call; no JIT/model code changes."""
import ctypes,dataclasses,hashlib,json,atexit
from pathlib import Path
LIB=Path(__file__).with_name('librc_nvtx.so')
assert hashlib.sha256(LIB.read_bytes()).hexdigest()=='548ea2b5c4bd674ab8dba0ac2db27f48270038f47c0e57430d30f1684ae4318a'
lib=ctypes.CDLL(str(LIB));lib.wrf_nvtx_initialize.argtypes=[];lib.wrf_nvtx_initialize.restype=None
lib.wrf_nvtx_push.argtypes=[ctypes.c_char_p];lib.wrf_nvtx_push.restype=ctypes.c_int
lib.wrf_nvtx_pop.argtypes=[];lib.wrf_nvtx_pop.restype=ctypes.c_int
lib.wrf_nvtx_initialize()
CALLS=[]
ROOT=0
def install(pipeline,jax):
 from gpuwrf.runtime import domain_tree as dt
 original=dt._prepare_operational_domain_tree_runtime
 def measure(name,fn,args,n_steps):
  global ROOT
  if name in ("d01","fused_d01"):ROOT+=int(n_steps)
  assert name in ('d01','d02','fused_d01','fused_d02'),name
  jax.block_until_ready(args)
  label='W45RC:step:'+name
  lib.wrf_nvtx_push(label.encode('ascii'))
  try:
   out=fn(*args);jax.block_until_ready(out)
  finally:lib.wrf_nvtx_pop()
  CALLS.append(dict(program=name,parent_steps=int(n_steps),root=ROOT))
  return out
 def factory(*args,**kwargs):
  runtime=original(*args,**kwargs)
  def advance(name,carry,start,n_steps):return measure(name,lambda *v:runtime.advance(name,*v),(carry,start,n_steps),n_steps)
  def lookup(name):
   fn=runtime.fused_cascade(name) if runtime.fused_cascade else None
   if fn is None:return None
   return lambda *values:measure('fused_'+name,fn,values,1)
  return dataclasses.replace(runtime,advance=advance,fused_cascade=lookup if runtime.fused_cascade else None)
 dt._prepare_operational_domain_tree_runtime=factory
 pipeline._prepare_operational_domain_tree_runtime=factory # by-name import too (E228)
 def save():
  Path('PROFILE_CALLS.json').write_text(json.dumps(dict(api='explicit NVTX3 push/pop',calls=CALLS,profile_only_extra_sync=True,scientific_timing='NOT S1; single-launch device durations under nsys'),indent=2)+'\n')
 atexit.register(save)
