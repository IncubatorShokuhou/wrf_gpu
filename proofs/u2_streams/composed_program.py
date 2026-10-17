"""Seal the full observer-ON program/constant bytes after another opt-in merge."""
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import time

repo=Path(sys.argv[1]).resolve()
output=Path(sys.argv[2]).resolve()
sys.path.insert(0,str(repo/'src'))
sys.path.insert(0,str(repo/'proofs/o1_nlbind'))
import jax
import numpy as np
from jax.experimental import pallas as pl
import cpu_safe
cpu_safe.install()
from gpuwrf.contracts import state as state_contract
from gpuwrf.integration.nested_pipeline import NestedPipelineConfig,_load_domains
from gpuwrf.runtime import operational_mode as om
from gpuwrf.streams.runtime import prepare_streams
assert jax.devices()[0].platform=='cpu'
state_contract._gpu_device=lambda:jax.devices('cpu')[0]
original=pl.pallas_call
def interpret(*args,**kwargs):
    kwargs['interpret']=True
    kwargs.pop('compiler_params',None)
    return original(*args,**kwargs)
source=Path('<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu')
started=time.perf_counter()
with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
    case=Path(temporary)
    for name in ('wrfinput_d01','wrfbdy_d01'):
        (case/name).symlink_to(source/name)
    (case/'namelist.input').write_bytes((source/'namelist.input').read_bytes())
    (case/'gpuwrf.streams.json').write_text(json.dumps({'version':1,'streams':[{
        'name':'airport','domain':'d01','interval_steps':2,'buffer_frames':16,
        'region':{'stations':[{'id':'ctr','i':20,'j':20},{'id':'west','i':9,'j':12}]},
        'variables':{'T2':['instant','min','max','mean'],'RAINNC':['instant','accumulation']}}]}))
    config=NestedPipelineConfig(case,case/'out',case/'proof',1,1)
    pl.pallas_call=interpret
    try:
        _,bundles,_,run_start,dt,carries=_load_domains(config,('d01',))
    finally:
        pl.pallas_call=original
    runtime,carries=prepare_streams(config,('d01',),carries,dt,run_start)
    runtime.close()
    nl=bundles['d01'].namelist
    om._resolve_operational_suite(nl)
    clock=om.build_clock_base(nl)
    jax.clear_caches()
    jx=jax.make_jaxpr(lambda c:om._advance_chunk_fori(c,nl,6,clock,n_steps=16,
                         cadence=int(nl.radiation_cadence_steps)))(carries['d01'])
    text=re.sub(r'name_and_src_info=[^\]\n]*','',str(jx))
    text=re.sub(r'/[^ \n:]*\.py:\d+(:\d+)?','',text)
    consts=hashlib.sha256()
    entries=[]
    for value in jx.consts:
        a=np.asarray(value)
        h=hashlib.sha256(a.tobytes()).hexdigest()
        entry={'shape':list(a.shape),'dtype':a.dtype.str,'sha256':h}
        consts.update(json.dumps(entry,sort_keys=True).encode())
        entries.append(entry)
    receipt={'source':str(repo),'backend':'cpu','program_sha256':hashlib.sha256(text.encode()).hexdigest(),
             'constant_manifest_sha256':consts.hexdigest(),'constants':entries,'chars':len(text),
             'wall_s':time.perf_counter()-started,'work':'Swiss42x42x44 dt18 FAST1 CENSUS1SLIM0 observerON points2 channels6 interval2 cap16 start6 steps16',
             'scope':'full ON jaxpr and captured constants only; no model/GPU numerical repeat'}
    output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='constants'}),flush=True)
