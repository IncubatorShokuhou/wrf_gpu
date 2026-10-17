"""Three real production CPU steps, original-WRF scoring and resumed continuation.

This is a CPU legacy-dynamics smoke, not GPU performance qualification. Uses
the actual CLI domain loader and production step. Original WRF references and
the frozen D6 field bounds are read-only. The same artifact is used for restart
continuation, so no redundant second unbroken trajectory is computed.
"""
from dataclasses import replace
from datetime import timedelta
import hashlib,json,os,sys,time
from pathlib import Path
import jax
import netCDF4
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'proofs/o1_nlbind'))
import cpu_safe
cpu_safe.install()
from gpuwrf.contracts import state as sc
from gpuwrf.integration import nested_pipeline as pipeline
from gpuwrf.runtime import operational_mode as om
from gpuwrf.io.restart import write_restart,read_restart
from gpuwrf.io.restart_schema import validate_observations

assert jax.devices()[0].platform=='cpu'
sc._gpu_device=lambda:jax.devices('cpu')[0]


def main(case,out):
    out.mkdir(parents=True,exist_ok=True)
    cfg=pipeline.NestedPipelineConfig(case,out/'output',out/'proof',hours=.015,max_dom=1)
    t0=time.monotonic()
    _,bundles,_,_,_,carries=pipeline._load_domains(cfg,('d01',))
    nl=bundles['d01'].namelist
    nl=replace(nl,disable_guards=True)
    c=carries['d01'];clock=om.build_clock_base(nl)
    manifest=ROOT/'proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json'
    limits=json.loads(manifest.read_text())['fields']
    result=dict(source=os.environ.get('S1_OBS_SOURCE','unknown'),case=str(case),platform='cpu',
                fast_defaults=os.environ.get('GPUWRF_FAST_DEFAULTS'),
                gate_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                load_s=time.monotonic()-t0,steps=[])
    c_unbroken=None
    steps=int(os.environ.get('S1_OBS_STEPS','3'))
    resume=os.environ.get('S1_OBS_RESTART','1')=='1'
    for step in range(1,steps+1):
        start=time.monotonic()
        c=om._advance_chunk_fori(c,nl,step,clock,n_steps=1,cadence=int(nl.radiation_cadence_steps))
        jax.block_until_ready(c)
        leaves=[np.asarray(x) for x in jax.tree.leaves(c)]
        nf=sum(int((~np.isfinite(x)).sum()) for x in leaves if np.issubdtype(x.dtype,np.floating))
        assert nf==0
        label=(nl.time_utc+timedelta(seconds=step*float(nl.dt_s))).strftime('%Y-%m-%d_%H:%M:%S')
        fields={k:np.asarray(getattr(c.state,name)) for k,name in
                (('U','u'),('V','v'),('W','w'),('QVAPOR','qv'),('P','p_perturbation'),('PH','ph_perturbation'))}
        rvrd=461.6/287.
        fields['T']=np.asarray(c.state.theta)/(1.+rvrd*fields['QVAPOR'])-300.
        scores={}
        with netCDF4.Dataset(case/f'wrfout_d01_{label}') as original:
            for key,value in fields.items():
                truth=np.asarray(original[key][0],np.float64)
                diff=value-truth
                scores[key]=dict(rmse=float(np.sqrt(np.mean(diff*diff))),max_abs=float(np.max(np.abs(diff))),
                                 frozen_limits=limits[key])
        rec=dict(step=step,elapsed_s=time.monotonic()-start,nonfinite=nf,
                 guard_total=None if c.census is None else int(np.asarray(c.census.guards).sum()),scores=scores,
                 obs_weight_nonzero=[int(np.count_nonzero(np.asarray(x))) for x in c.obs.weights])
        result['steps'].append(rec)
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(rec),flush=True)
        if step==1 and resume:
            write_restart(c,nl,nl.grid,step,out/'restart.pkl')
        if step==2:c_unbroken=c
    # Resume the exact after-step1 carry and execute only step2. This checks
    # an ionf hold step against the already computed unbroken trajectory.
    if resume:
        restored=read_restart(out/'restart.pkl')
        rc,rnl,rgrid,rstep,*_=restored
        validate_observations(rc.obs,rnl)
        resumed=om._advance_chunk_fori(rc,rnl,2,om.build_clock_base(rnl),n_steps=1,
                                     cadence=int(rnl.radiation_cadence_steps))
        jax.block_until_ready(resumed)
        a,b=jax.tree.leaves(resumed),jax.tree.leaves(c_unbroken)
        assert len(a)==len(b)
        assert all(np.asarray(x).dtype==np.asarray(y).dtype and np.asarray(x).tobytes()==np.asarray(y).tobytes()
                   for x,y in zip(a,b,strict=True))
        result['restart_byte_exact_leaves']=len(a)
    result['bounded_field_failures']=[(row['step'],key) for row in result['steps']
        for key,score in row['scores'].items() if score['frozen_limits'].get('gate')=='hard_release_gate'
        and 'rmse' in score['frozen_limits'] and score['rmse']>score['frozen_limits']['rmse']]
    result['whole_wall_s']=time.monotonic()-t0
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    assert not result['bounded_field_failures'],result['bounded_field_failures']


if __name__=='__main__':main(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
