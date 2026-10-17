"""Acquire one missing pre-repair census step from the completed Swiss smoke.

Reuses the exact full coupled restart; no repeated initialization or numerical
trajectory validation. This CPU diagnostic step is not a fidelity/speed arm.
"""
from pathlib import Path
import sys,json,time,hashlib,os

repo=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(repo/'scripts/v025'))
import cpu_guard  # noqa: E402,F401
sys.path.insert(0,str(repo/'proofs/o1_nlbind'))
import cpu_safe  # noqa: E402
cpu_safe.install()
import jax
import numpy as np
from gpuwrf.io.restart import read_restart
from gpuwrf.runtime import operational_mode as om
from gpuwrf.diagnostics.census import initial_census,EVENTS,GUARDS,F2_EVENTS

source,out=Path(sys.argv[1]),Path(sys.argv[2])
assert jax.devices()[0].platform=='cpu'
assert os.environ.get('GPUWRF_CENSUS')=='1'
carry,n,grid,step=read_restart(source)
with jax.enable_x64():
    carry=carry.replace(census=initial_census())
    clock=om.build_clock_base(n)
    start=time.perf_counter()
    with jax.disable_jit():
        carry=om._advance_chunk_fori(carry,n,step+1,clock,n_steps=1,cadence=int(n.radiation_cadence_steps))
        jax.block_until_ready(carry)
    counts=np.asarray(carry.census.guards)
    f2=np.asarray(carry.census.f2_counts)
    report={'backend':'cpu','restart_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'step':step+1,'wall_s':time.perf_counter()-start,
            'nonfinite':sum(int((~np.isfinite(np.asarray(v))).sum()) for v in jax.tree.leaves(carry)),
            'guard_total':int(counts.sum()),'guard_events':dict(zip(EVENTS,[int(x) for x in counts.sum(axis=0)])),
            'guard_rows':{name:counts[i].tolist() for i,name in enumerate(GUARDS) if counts[i].any()},
            'f2_events':dict(zip(F2_EVENTS,[int(x) for x in f2.sum(axis=(0,1))])),
            'scope':'One additional CPU diagnostic step from exact completed coupled restart; no GPU or numerical-fidelity claim'}
out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
assert report['nonfinite']==0
assert report['guard_total']==0
