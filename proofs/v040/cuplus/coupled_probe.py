"""Three real Swiss production CPU steps with complete cuplus resident state.

Uses the production loader and step loop. The original case is read only.
A resolved selector override retains its original input files and selects the
CPU-qualified producer pairing without changing any timestep equation.
"""
from __future__ import annotations
import argparse,hashlib,json,os,resource,subprocess,sys,time
from dataclasses import replace
from datetime import datetime,timezone
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--case',type=Path,required=True)
    p.add_argument('--scheme',type=int,choices=(0,7,10,11),required=True)
    p.add_argument('--pbl',type=int,choices=(2,5,9),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--diagnostic-batch',action='store_true',
        help='three steps in one production loop, capture full F2 extrema; reuse prior restart gate')
    p.add_argument('--fixed-executable',action='store_true',
        help='reuse one compiled production loop for every compatible original/restart carry')
    args=p.parse_args();root=args.source.resolve()
    assert os.environ.get('JAX_PLATFORMS')=='cpu'
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    sys.path.insert(0,str(root/'src'))
    sys.path.insert(0,str(root/'proofs/o1_nlbind'))
    import jax
    import numpy as np
    import cpu_safe
    cpu_safe.install()
    from gpuwrf.contracts import state as state_contract
    state_contract._gpu_device=lambda:jax.devices('cpu')[0]
    # The established CPU loader proof keeps every Pallas operation interpreted;
    # no CPU probe may attempt to compile a CUDA executable.
    from jax.experimental import pallas as pl
    original_pallas=pl.pallas_call
    def interpreted(*a,**kw):
        kw['interpret']=True;kw.pop('compiler_params',None)
        return original_pallas(*a,**kw)
    pl.pallas_call=interpreted
    from gpuwrf.integration.nested_pipeline import NestedPipelineConfig,_load_domains
    from gpuwrf.runtime import operational_mode as om
    from gpuwrf.coupling.cuplus_config import resolved_errors
    from gpuwrf.runtime.finite_state_guard import assert_state_finite_at_boundary
    from gpuwrf.diagnostics.census import GUARDS,EVENTS
    assert all(d.platform=='cpu' for d in jax.devices())
    started=time.monotonic();record=dict(source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        started_utc=datetime.now(timezone.utc).isoformat(),scheme=args.scheme,pbl=args.pbl,
        case=str(args.case.resolve()),steps=[],cache='NONE GPUWRF_JAX_CACHE=0',gpu_executed=False,
        scope='real Swiss production CPU coupled smoke; not GPU/D6 forecast qualification; selector staged in resolved object only')
    def save():
        record.update(elapsed_s=time.monotonic()-started,maxrss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        args.output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    save()
    cfg=NestedPipelineConfig(args.case,args.output.parent/'unused_output',args.output.parent/'unused_proof',hours=.02,max_dom=1)
    _,bundles,_,_,_,loaded= _load_domains(cfg,('d01',))
    bundle=bundles['d01'];nml=bundle.namelist
    # Obtain the loaded time-n state from the normal loader's carry, then seed
    # the newly selected PBL/CU memory through the production initializer.
    state=loaded['d01'].state
    nml=replace(nml,cu_physics=args.scheme,bl_pbl_physics=args.pbl,
        sf_sfclay_physics=2 if args.pbl==2 else 5,rad_rk_tendf=1,
        cup_num_bins=5,cup_th_bin_size=.1,cup_r_bin_size=1e-4,
        cup_min_deep_freq=.1,cup_min_shallow_freq=.001)
    if args.pbl==2:
        # The retained Noah-MP surface adapter selects MYNN-SL internally;
        # it cannot supply the actual Janjic producer required by this pairing.
        nml=replace(nml,use_noahmp=False,sf_surface_physics=0)
    assert not resolved_errors(nml),resolved_errors(nml)
    from gpuwrf.coupling.cuplus import initial_carry
    from gpuwrf.coupling.scan_adapters import initial_camuw_carry
    carry=loaded['d01'].replace(cuplus=None if args.scheme==0 else initial_carry(state,args.scheme),
        cumulus_carry=None,cumulus_tendencies=None,
        camuw_pbl=initial_camuw_carry(state) if args.pbl==9 else None)
    if args.pbl==2:carry=carry.replace(noahmp_land=None,noahmp_rad=None,noahmp_precipitation=None)
    assert carry.census is not None,'GPUWRF_CENSUS=1 is required for pre-mask coupled evidence'
    clock=om.build_clock_base(nml)
    record['load_s']=time.monotonic()-started
    record['resolved_work']={k:getattr(nml,k) for k in ('dt_s','cu_physics','bl_pbl_physics','sf_sfclay_physics',
        'mp_physics','ra_sw_physics','ra_lw_physics','cumulus_cadence_steps','cudt_minutes','force_fp64')}
    record['shape']=list(state.theta.shape)
    record['input_sha256']={name:hashlib.sha256((args.case/name).read_bytes()).hexdigest() for name in
                            ('namelist.input','wrfinput_d01','wrfbdy_d01')}
    save();print('LOADED',record['load_s'],record['resolved_work'],flush=True)
    executable=None
    if args.fixed_executable:
        assert not om._CARRY_DONATE,'this CPU proof retains both continuation inputs'
        import jax.numpy as jnp
        tick=time.monotonic()
        executable=om._advance_chunk_fori.lower(carry,nml,jnp.int32(1),clock,
            n_steps=jnp.int32(1),cadence=jnp.int32(nml.radiation_cadence_steps)).compile()
        record['fixed_executable_compile_s']=time.monotonic()-tick
        record['fixed_executable_memory']=str(executable.memory_analysis())
        save()
    def advance(value,start,count):
        if executable is None:
            return om._advance_chunk(value,nml,start,clock,n_steps=count,cadence=int(nml.radiation_cadence_steps))
        return executable(value,nml,jnp.int32(start),clock,n_steps=jnp.int32(count),
            cadence=jnp.int32(nml.radiation_cadence_steps))
    for step in ((3,) if args.diagnostic_batch else (1,2,3)):
        tick=time.monotonic()
        carry=advance(carry,1 if args.diagnostic_batch else step,3 if args.diagnostic_batch else 1)
        jax.block_until_ready(carry)
        leaves=[np.asarray(x) for x in jax.tree.leaves(carry) if hasattr(x,'dtype') and np.issubdtype(x.dtype,np.floating)]
        nonfinite=sum(int(np.count_nonzero(~np.isfinite(x))) for x in leaves)
        assert_state_finite_at_boundary(carry.state,domain='d01',step=step,physics_carry=carry)
        counts={} if carry.census is None else {
            f'{guard}.{event}':int(np.asarray(carry.census.guards)[i,j]) for i,guard in enumerate(GUARDS)
            for j,event in enumerate(EVENTS)}
        events={} if carry.cuplus is None else {key:int(np.asarray(value).sum()) for key,value in carry.cuplus.held.items()
            if key in ('ENTROPY_FAILED','DLF_CLIPPED','VAPOR_FLOORED')}
        row=dict(step=step,wall_s=time.monotonic()-tick,nonfinite=nonfinite,guard_events=counts,
            source_events=events,negative_location_classes=None if carry.census is None else
            np.asarray(carry.census.f2_negative_location).tolist())
        from gpuwrf.diagnostics.census import f2_negative_summary
        row['f2_extrema']=f2_negative_summary(carry.census,field_shape=state.theta.shape)
        record['steps'].append(row);save();print('STEP',step,row['wall_s'],nonfinite,events,flush=True)
        assert nonfinite==0
        if step==2 and not args.diagnostic_batch:
            from gpuwrf.io.restart import write_restart,read_restart
            checkpoint=args.output.with_suffix('.restart.pkl')
            write_restart(carry,nml,nml.grid,step,checkpoint)
        if step==3 and not args.diagnostic_batch:
            restored,_,_,last=read_restart(checkpoint,namelist=nml)
            assert last==2
            resumed=advance(restored,3,1)
            jax.block_until_ready(resumed)
            a,b=jax.tree.leaves(carry),jax.tree.leaves(resumed)
            assert len(a)==len(b)
            for x,y in zip(a,b,strict=True):
                x,y=np.asarray(x),np.asarray(y)
                assert x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes(),'restart continuation differs'
            record['restart_continuation_byte_leaves']=len(a)
            checkpoint.unlink()
    record['complete']=True;save()


if __name__=='__main__':main()
