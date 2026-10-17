"""Regression-only binding of measured CPU-WRF prepass gates across opt-in main merges.

Uses the established release-probe setup, traces the actual operational backward setup,
and records complete seed-array bytes plus forward/default-diffusion program identities.
Usage: python probe_phase_identity.py IMMUTABLE_SOURCE OUT_JSON
"""
import hashlib
import json
from pathlib import Path
import sys
import subprocess

source,out=Path(sys.argv[1]),Path(sys.argv[2])
probe=source/'proofs/o1_nlbind/probe.py'
text=probe.read_text().split('out = {"src":',1)[0]
text=text.replace('LANE = Path("<USER_HOME>/wrf_gpu2_lanes/o1-nlbind")',
                  'LANE = Path("<USER_HOME>/wrf_gpu2_lanes/s1-dfi12")')
sys.path.insert(0,str(probe.parent))
ns={'__file__':str(probe),'__name__':'phase_identity_setup'}
exec(compile(text,str(probe),'exec'),ns)
jax,np,pl=ns['jax'],ns['np'],ns['pl']
pipeline,om=ns['pipeline'],ns['om']
assert Path(om.__file__).resolve().is_relative_to(source.resolve()),om.__file__
from dataclasses import replace
from gpuwrf.dfi import driver12
from gpuwrf.dfi.control import dfi_control_from_namelist
from gpuwrf.io.gen2_accessor import Gen2Run

case=Path('<USER_HOME>/wrf_gpu2_lanes/s1-dfi12/DF12-E3/case')
cfg=pipeline.NestedPipelineConfig(case,out.parent/'identity-out',out.parent/'identity-proof',hours=1,max_dom=1)
pl.pallas_call=ns['_interpret_pallas_call']
_h,bundles,_meta,_start,_dt,carries=pipeline._load_domains(cfg,('d01',))
pl.pallas_call=ns['_real_pallas_call']
carry,nl=carries['d01'],bundles['d01'].namelist
assert nl.mp_physics==8
result={'source':str(source),'regression_only':True,'arms':{},
        'source_commit':subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
        'src_tree':subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD:src/gpuwrf'],text=True).strip()}
def record(label,carry,nl,clock,step):
    jax.clear_caches()
    jx=jax.make_jaxpr(lambda c:om._advance_chunk_fori(c,nl,step,clock,n_steps=1,
                                                  cadence=int(nl.radiation_cadence_steps)))(carry)
    text=ns['canon'](jx)
    arrays=[]
    for path,leaf in jax.tree_util.tree_flatten_with_path((carry,nl,clock))[0]:
        value=np.asarray(leaf)
        arrays.append({'path':str(path),'dtype':str(value.dtype),'shape':list(value.shape),
                       'sha256':hashlib.sha256(value.tobytes()).hexdigest()})
    result['arms'][label]={'jaxpr_sha256':hashlib.sha256(text.encode()).hexdigest(),
                          'chars':len(text),'seed_arrays':arrays}
    out.write_text(json.dumps(result,indent=1)+'\n')
    print(label,result['arms'][label]['jaxpr_sha256'],len(arrays),flush=True)
record('forward',carry,nl,om.build_clock_base(nl),1)
nl0=replace(nl,diff_opt=0,km_opt=1)
record('forward_diff0',carry,nl0,om.build_clock_base(nl0),1)
captured={}
class Captured(Exception):pass
original=om._advance_chunk
def capture(carry,nl,step,clock,**kw):
    captured.update(carry=carry,nl=nl,step=step,clock=clock)
    raise Captured
om._advance_chunk=capture
try:
    driver12._backward_endpoint(carries['d01'],nl,dfi_control_from_namelist(Gen2Run(case).namelist),
                                Gen2Run(case),pipeline,{})
except Captured:
    pass
finally:
    om._advance_chunk=original
assert captured
record('backward',captured['carry'],captured['nl'],captured['clock'],captured['step'])
