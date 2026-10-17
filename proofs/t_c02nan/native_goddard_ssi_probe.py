"""Actual native old/new Goddard columns, independently qualified CPU inputs."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import gzip
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('source', type=Path)
p.add_argument('label', choices=('original', 'candidate'))
args = p.parse_args()
assert os.environ.get('GPUWRF_GPU_LOCK_HELD') == '1'
fd = int(os.environ['GPUWRF_GPU_LOCK_FD'])
assert os.fstat(fd).st_ino == os.stat(os.environ['GPUWRF_GPU_LOCK_FILE']).st_ino
assert subprocess.run(['git', '-C', str(args.source), 'symbolic-ref', '--quiet', 'HEAD'], capture_output=True).returncode != 0
assert not subprocess.check_output(['git', '-C', str(args.source), 'status', '--porcelain'], text=True).strip()
sys.path.insert(0, str(args.source/'src'))
import gpuwrf
import jax
import jax.numpy as jnp
from gpuwrf.physics.goddard4ice.native import native_column
assert jax.devices()[0].platform == 'gpu', jax.devices()
root = Path('<DATA_ROOT>/wrf_gpu2/scratch/t-c02nan/TCN52_native_g4')/args.label
root.mkdir(exist_ok=True)
fixture = Path('<DATA_ROOT>/wrf_gpu2/scratch/t-c02nan/TCN42/source/proofs/t_c02nan/savepoints/goddard_c64_ssi.json')
data = json.loads(fixture.read_text())
captured = np.load('<DATA_ROOT>/wrf_gpu2/scratch/t-c02nan/TCN33_c64_postrk_cpu/caller_inputs.npz')
literal = np.load('<DATA_ROOT>/wrf_gpu2/scratch/t-c02nan/TCN33_c64_postrk_cpu/literal_outputs.npz')
names = ('theta','qv','qc','qr','qi','qs','qg','qh','rho','exner','pressure','dz','z','w')
keys = ('TH_IN','QV_IN','QC_IN','QR_IN','QI_IN','QS_IN','QG_IN','QH_IN','DEN','PII','P','DELZ','Z','W')
inputs = tuple(jnp.asarray(np.stack([captured[name], np.asarray(data['columns'][key], np.float32)], axis=-1)[:,None,:], jnp.float32) for name,key in zip(names,keys))
cfg = data['caller']
for k in ('dt','dx','xland','topo'):
    assert np.float32(captured[k]) == np.float32(cfg[k]), k
ac = {key:jnp.asarray(np.repeat(captured[key][:,None,None], 2, axis=2),jnp.float32) for key in captured.files if key.startswith('acphys')}
@jax.jit
def program(*operands):
    return native_column(*operands, np.float32(cfg['dt']), dx=np.float32(cfg['dx']),
                         xland=np.float32(cfg['xland']), topo=np.float32(cfg['topo']),
                         itimestep=cfg['itimestep'], diagnostics=ac, full=True)
start = time.monotonic()
lowered = program.lower(*inputs)
executable = lowered.compile()
compile_s = time.monotonic()-start
start = time.monotonic()
output = executable(*inputs)
jax.block_until_ready(output)
first_run_s = time.monotonic()-start
host = jax.device_get(output)
np.savez(root/'outputs.npz', **host)
mapping = dict(theta='TH_OUT',qv='QV_OUT',qc='QC_OUT',qr='QR_OUT',qi='QI_OUT',qs='QS_OUT',qg='QG_OUT',qh='QH_OUT')
mapping.update({'re_'+k+'_gsfc':'RE_'+k.upper() for k in ('cloud','rain','ice','snow','graupel','hail')})
mapping.update({k:k.upper() for k in ['phys'+x for x in 'cedsmf']+['acphys'+x for x in 'cedsmf']+['preci3d','precs3d','precg3d','prech3d','precr3d']})
rows={}
for name,key in mapping.items():
    actual=np.asarray(host[name])[:,0,:]
    reference=np.stack([literal[name],np.asarray(data['columns'][key],np.float32)],axis=-1)
    diff=np.abs(actual.astype(float)-reference.astype(float))
    rows[name]=dict(nonfinite_by_case=(~np.isfinite(actual)).sum(axis=0).tolist(),
                    BYTE_by_case=[actual[:,i].tobytes()==reference[:,i].tobytes() for i in range(2)],
                    maxabs_by_case=[float(np.nanmax(diff[:,i])) for i in range(2)])
bad={key:np.argwhere(~np.isfinite(value)).tolist() for key,value in host.items() if not np.isfinite(value).all()}
timings=[]
for _ in range(7):
    start=time.monotonic()
    for _ in range(20): output=executable(*inputs)
    jax.block_until_ready(output)
    timings.append((time.monotonic()-start)/20)
text=executable.as_text()
with gzip.open(root/'actual.optimized.hlo.gz','wt') as f:f.write(text)
receipt=dict(platform=jax.devices()[0].platform,device=str(jax.devices()[0]),
             GPU_lock_fd_valid=True,source_commit=subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip(),
             fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
             cases=['actualCPUpostRKcolumn; independent31fieldpristineBYTE', 'samecolumn TH[k26]+1float32ULP; independent31fieldpristineBYTE'],
             scope='actual GPU native Goddard component; CPU-derived inputs, NOT actual GPU preMP input byte claim; timings host+device, NOT forecast/S1',
             shape=[44,1,2],full=True,compile_s=compile_s,first_run_s=first_run_s,
             warm_host_device_percall_s=timings,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             nonfinite=bad,pristine_defined_fields=rows,field_count=len(host),
             optimized_HLO_sha256=hashlib.sha256(text.encode()).hexdigest(),
             finished_utc=datetime.now(timezone.utc).isoformat())
(root/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:receipt[k] for k in ('platform','source_commit','compile_s','first_run_s','warm_host_device_percall_s','peak_RSS_KiB','nonfinite','field_count')}),flush=True)
