from pathlib import Path
import importlib.util,json,os,subprocess,sys
R=Path('<USER_HOME>/wrf_gpu2_lanes/v45-rc');S=R/'archive/src';O=R/'TOKEN_FIX';sys.path.insert(0,str(S))
from gpuwrf.runtime.aot_cheap_key import global_trace_env_hash,trace_env_is_inert
original=os.environ.get('GPUWRF_GPU_LOCK_TOKEN');os.environ.pop('GPUWRF_GPU_LOCK_TOKEN',None);a=global_trace_env_hash();os.environ['GPUWRF_GPU_LOCK_TOKEN']='INERT_NON_SECRET_CAPABILITY';b=global_trace_env_hash();assert a==b and trace_env_is_inert('GPUWRF_GPU_LOCK_TOKEN')
if original is None:os.environ.pop('GPUWRF_GPU_LOCK_TOKEN',None)
else:os.environ['GPUWRF_GPU_LOCK_TOKEN']=original
spec=importlib.util.spec_from_file_location('private_gpu_preflight',S/'gpuwrf/runtime/gpu_preflight.py');mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
lock=O/'private.lock';holder=O/'private.holder';holder.write_text('INERT_NON_SECRET_CAPABILITY')
fd=os.open(lock,os.O_CREAT|os.O_RDWR,0o600)
try:
 e=dict(GPUWRF_GPU_LOCK_HELD='1',GPUWRF_GPU_LOCK_FD=str(fd),GPUWRF_GPU_LOCK_FILE=str(lock),GPUWRF_GPU_LOCK_TOKEN='INERT_NON_SECRET_CAPABILITY')
 before=mod.gpu_lock_status({k:v for k,v in e.items()if k!='GPUWRF_GPU_LOCK_TOKEN'},holder_file=holder,lock_file=lock);after=mod.gpu_lock_status(e,holder_file=holder,lock_file=lock)
 assert not before['ok'] and after['ok']
finally:os.close(fd)
(O/'PREFLIGHT_CONTROL.json').write_text(json.dumps(dict(PASS=True,source='immutable254c482374 archive',old_missing_token_refused=not before['ok'],fixed_capability_fd_holder_accepted=after['ok'],global_trace_env_hash_exact=a==b,lock_token_is_trace_inert=True,no_GPU_device_query_or_import=True,no_credentials_printed=True),indent=2)+'\n');print('CPU nested lock/preflight PASS; token excluded from actual global trace hash, paid Swiss identity carries')
