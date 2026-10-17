"""Same canonical Swiss release jaxpr probe as proofs/o1_nlbind, lane-local IO."""
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time

repo=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(repo/'src'))
sys.path.insert(0,str(repo/'proofs/o1_nlbind'))
import jax
from jax.experimental import pallas as pl
import cpu_safe
cpu_safe.install()
from gpuwrf.contracts import state as state_contract
from gpuwrf.integration import nested_pipeline as pipeline
from gpuwrf.runtime import operational_mode as om

assert jax.devices()[0].platform=='cpu'
assert Path(om.__file__).is_relative_to(repo)
state_contract._gpu_device=lambda:jax.devices('cpu')[0]
lane=Path(sys.argv[1]).resolve()
lane.mkdir(parents=True,exist_ok=True)
case_source=Path('<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu')
original=pl.pallas_call
def interpret(*args,**kwargs):
    kwargs['interpret']=True
    kwargs.pop('compiler_params',None)
    return original(*args,**kwargs)
t0=time.perf_counter()
with tempfile.TemporaryDirectory(dir=lane) as temporary:
    case=Path(temporary)/'case'
    case.mkdir()
    for path in case_source.iterdir():
        if path.name != 'namelist.input':
            (case/path.name).symlink_to(path.resolve())
    (case/'namelist.input').write_text((case_source/'namelist.input').read_text())
    config=pipeline.NestedPipelineConfig(case,lane/'out',lane/'proof',1,1)
    pl.pallas_call=interpret
    try:
        _,bundles,_,_,_,carries=pipeline._load_domains(config,('d01',))
    finally:
        pl.pallas_call=original
nl,carry=bundles['d01'].namelist,carries['d01']
assert carry.timeseries is None
om._resolve_operational_suite(nl)
jax.clear_caches()
clock=om.build_clock_base(nl)
jx=jax.make_jaxpr(lambda c:om._advance_chunk_fori(c,nl,1,clock,n_steps=1,
                                              cadence=int(nl.radiation_cadence_steps)))(carry)
text=re.sub(r'name_and_src_info=[^\]\n]*','',str(jx))
text=re.sub(r'/[^ \n:]*\.py:\d+(:\d+)?','',text)
sha=hashlib.sha256(text.encode()).hexdigest()
receipt={'sha':sha,'expected_sha':'2bf12cb8374d3a84a440e47dea4b77b0c7436108fc2ee2616bba6ec79b17962a',
         'pass':sha.startswith('2bf12cb8'),'chars':len(text),'wall_s':time.perf_counter()-t0,
         'backend':jax.devices()[0].platform,'source':str(repo),'scope':'default no-tslist release jaxpr; no forecast/GPU claim'}
(lane/'identity.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
assert receipt['pass'],receipt
