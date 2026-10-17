"""CPU coupled Swiss loader/step/restart check, no GPU/trajectory fidelity claim."""
from dataclasses import replace
from datetime import datetime,timedelta
from pathlib import Path
import json,sys,time,os,subprocess,hashlib
import jax
import numpy as np
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.contracts import state as sc
from gpuwrf.integration import nested_pipeline as pipeline
from gpuwrf.runtime import operational_mode as om
from gpuwrf.io.restart import write_restart,read_restart
from gpuwrf.assimilation.fdda_surface import SurfaceGridFddaInputs,validate_surface_restart

SOURCE=Path('<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu')
assert jax.devices()[0].platform=='cpu'
assert not jax.config.jax_disable_jit
sc._gpu_device=lambda:jax.devices('cpu')[0]


def prepare(out):
    """Build only private symlinks and paired files from original CPU-WRF fields."""
    out.mkdir(parents=True,exist_ok=True)
    for f in SOURCE.iterdir():
        if f.name not in ('namelist.input','wrffdda_d01','wrfsfdda_d01') and not (out/f.name).exists():
            (out/f.name).symlink_to(f)
    frames=[]
    for h in (0,1):
        with Dataset(SOURCE/f'wrfout_d01_2023-01-15_{h:02d}:00:00') as ds:
            g=lambda n:np.asarray(ds[n][0],np.float32)
            q=g('QVAPOR')
            th=(g('THM')+np.float32(300))/(np.float32(1)+np.float32(np.float32(461.6)/np.float32(287))*q)
            frames.append(dict(U=g('U'),V=g('V'),T=th-np.float32(300),Q=q,PH=g('PH'),MU=g('MU')))
    nz,ny,nx=frames[0]['T'].shape
    with Dataset(out/'wrffdda_d01','w') as ds:
        for n,size in (('Time',1),('DateStrLen',19),('x',nx),('y',ny),('z',nz),('xs',nx+1),('ys',ny+1),('zs',nz+1)):
            ds.createDimension(n,size)
        ds.createVariable('Times','S1',('Time','DateStrLen'))[0]=np.frombuffer(b'2023-01-15_00:00:00',dtype='S1')
        for n,dims in (('U',('z','y','xs')),('V',('z','ys','x')),('T',('z','y','x')),('Q',('z','y','x')),
                       ('PH',('zs','y','x')),('MU',('y','x'))):
            for suffix,frame in zip(('OLD','NEW'),frames):
                ds.createVariable(n+'_NDG_'+suffix,'f4',('Time',*dims))[0]=frame[n]
    with Dataset(out/'wrfsfdda_d01','w') as ds:
        for n,size in (('Time',1),('DateStrLen',19),('x',nx),('y',ny),('xs',nx+1),('ys',ny+1)):
            ds.createDimension(n,size)
        ds.createVariable('Times','S1',('Time','DateStrLen'))[0]=np.frombuffer(b'2023-01-15_00:00:00',dtype='S1')
        for suffix,h in zip(('OLD','NEW'),(0,1)):
            with Dataset(SOURCE/f'wrfout_d01_2023-01-15_{h:02d}:00:00') as src:
                for name,dims in (('U10',('y','xs')),('V10',('ys','x')),('TH2',('y','x')),('Q2',('y','x')),
                                  ('ODIS',('y','x')),('T2',('y','x')),('RH',('y','x')),('PSL',('y','x')),
                                  ('PS',('y','x')),('TOB',('y','x'))):
                    a=(np.asarray(src['PSFC' if name=='PS' else name][0],np.float32)
                       if name in ('U10','V10','TH2','Q2','T2','PS') else np.zeros((ny,nx),np.float32))
                    if name=='U10':a=np.pad(a,((0,0),(0,1)),mode='edge')
                    if name=='V10':a=np.pad(a,((0,1),(0,0)),mode='edge')
                    ds.createVariable(name+'_NDG_'+suffix,'f4',('Time',*dims))[0]=a
    # Real WRF refuses/ignores aux fields without its V4 header and FieldType.
    # Only our newly written files are opened for append, never input symlinks.
    with Dataset(SOURCE/'wrfinput_d01') as original:
        attrs={k:original.getncattr(k) for k in original.ncattrs() if k!='TITLE'}
    for name in ('wrffdda_d01','wrfsfdda_d01'):
        assert not (out/name).is_symlink()
        with Dataset(out/name,'a') as ds:
            for old,new in (('x','west_east'),('y','south_north'),('z','bottom_top'),
                            ('xs','west_east_stag'),('ys','south_north_stag'),('zs','bottom_top_stag')):
                if old in ds.dimensions:ds.renameDimension(old,new)
            ds.setncatts(attrs);ds.TITLE='WRF V4.7.1-compatible analyses assembled from original Swiss CPU-WRF fields'
            for key,var in ds.variables.items():
                if key=='Times':continue
                var.FieldType=104
                var.MemoryOrder='XYZ' if var.ndim==4 else 'XY'
                var.stagger={'U':'X','V':'Y','PH':'Z','U10':'X','V10':'Y'}.get(key.split('_NDG_')[0],'')


def main(out,code):
    out.mkdir(parents=True,exist_ok=True);case=out/'case';prepare(case)
    text=(SOURCE/'namelist.input').read_text()
    import re
    text=re.sub(r'&fdda\b.*?\n\s*/','',text,flags=re.S|re.I)
    text+='\n&fdda\n grid_fdda=1, gfdda_interval_m=60, gfdda_end_h=1,\n guv=0.0003,gt=0.0003,gq=0.0003,\n'
    text+=f' grid_sfdda={code},sgfdda_inname="wrfsfdda_d<domain>",sgfdda_interval_m=60,sgfdda_end_h=1,\n'
    text+=' guv_sfc=0.0003,gt_sfc=0.0003,gq_sfc=0.0003,rinblw=100.,fgdt=0.9,\n/\n'
    (case/'namelist.input').write_text(text)
    cfg=pipeline.NestedPipelineConfig(case,out/'output',out/'proof',hours=.015,max_dom=1)
    start=time.monotonic()
    _,bundles,_,_,_,carries=pipeline._load_domains(cfg,('d01',))
    nl=bundles['d01'].namelist;c=carries['d01'];clock=om.build_clock_base(nl)
    assert isinstance(nl.data_assimilation,SurfaceGridFddaInputs)==bool(code)
    if code:assert c.fdda.surface is not None
    source_root=Path(__file__).resolve().parents[2]
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source_root,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=source_root,text=True).strip()
    from gpuwrf.diagnostics.census import GUARDS,EVENTS
    result=dict(source_commit=source_commit,platform='cpu',mode=code,jit_enabled=True,
                step_entrypoint='production _advance_chunk wrapper -> jitted _advance_chunk_fori',load_s=time.monotonic()-start,steps=[],
                configuration=repr(nl.data_assimilation.options),gpu_qualified=False,
                fast_defaults=os.environ.get('GPUWRF_FAST_DEFAULTS','1'),census=os.environ.get('GPUWRF_CENSUS','0'))
    original_case=Path(os.environ['SF_ORIGINAL_CASE']) if os.environ.get('SF_ORIGINAL_CASE') else None
    limits=json.loads((Path(__file__).parents[2]/'proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json').read_text())['fields']
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    for step in (1,2):
        t=time.monotonic()
        c=om._advance_chunk(c,nl,step,clock,n_steps=1,cadence=int(nl.radiation_cadence_steps))
        jax.block_until_ready(c)
        leaves=[np.asarray(x) for x in jax.tree.leaves(c)]
        nonfinite=sum(int((~np.isfinite(x)).sum()) for x in leaves if x.dtype.kind in 'fc')
        assert nonfinite==0
        validate_surface_restart(c.fdda,nl)
        row=dict(step=step,wall_s=time.monotonic()-t,nonfinite=nonfinite,
            held_absmax=[float(np.abs(np.asarray(a)).max()) for a in c.fdda.tendencies],
            guards=None if c.census is None else np.asarray(c.census.guards).tolist())
        if c.census is not None:
            guards=np.asarray(c.census.guards)
            assert guards.shape==(len(GUARDS),len(EVENTS))
            row.update(guard_shape=list(guards.shape),guard_counters=int(guards.size),
                       guard_event_total=int(guards.sum()))
        if original_case is not None:
            label=(nl.time_utc+timedelta(seconds=step*float(nl.dt_s))).strftime('%Y-%m-%d_%H:%M:%S')
            with Dataset(original_case/f'wrfout_d01_{label}') as cpu:
                fields={key:np.asarray(getattr(c.state,name)) for key,name in
                        (('U','u'),('V','v'),('W','w'),('QVAPOR','qv'),('P','p_perturbation'),('PH','ph_perturbation'))}
                fields['T']=np.asarray(c.state.theta)/(1.+(461.6/287.)*fields['QVAPOR'])-300.
                row['scores']={}
                for key,value in fields.items():
                    truth=np.asarray(cpu[key][0],np.float64);diff=value-truth
                    assert np.isfinite(truth).all() and value.shape==truth.shape
                    row['scores'][key]=dict(rmse=float(np.sqrt(np.mean(diff*diff))),max_abs=float(np.abs(diff).max()),frozen_limits=limits[key])
                row['bounded_fields']=[k for k,v in row['scores'].items() if v['frozen_limits'].get('gate')=='hard_release_gate']
                row['report_only_fields']=[k for k,v in row['scores'].items() if v['frozen_limits'].get('gate')=='critical_report_only']
                assert set(row['bounded_fields'])=={'T','U','V','W','QVAPOR'}
                assert set(row['report_only_fields'])=={'P','PH'}
                for score in row['scores'].values():assert np.isfinite([score['rmse'],score['max_abs']]).all()
                row['held_rate_scores']={}
                for key,value in zip(('RUNDGDTEN','RVNDGDTEN','RTHNDGDTEN','RQVNDGDTEN'),c.fdda.tendencies):
                    if key in cpu.variables:
                        diff=np.asarray(value)-np.asarray(cpu[key][0])
                        row['held_rate_scores'][key]=dict(rmse=float(np.sqrt(np.mean(diff*diff))),max_abs=float(np.abs(diff).max()))
        result['steps'].append(row)
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        if step==1:write_restart(c,nl,nl.grid,step,out/'restart.pkl')
    rc,rnl,_,_=read_restart(out/'restart.pkl')
    # Static JIT identity can otherwise recompile the complete CPU model after
    # pickle reconstruction. Prove every static value equal, including dates,
    # then reuse the same compiled program for the actual restored carry.
    from gpuwrf.runtime.aot_cheap_key import canonical_digest
    assert canonical_digest(nl)==canonical_digest(rnl)
    assert nl.time_utc==rnl.time_utc and nl.noahmp_julian==rnl.noahmp_julian and nl.noahmp_yearlen==rnl.noahmp_yearlen
    result['restart_namelist_digest']=canonical_digest(rnl)
    resumed=om._advance_chunk(rc,nl,2,clock,n_steps=1,cadence=int(nl.radiation_cadence_steps))
    jax.block_until_ready(resumed)
    aa,ta=jax.tree.flatten(c);bb,tb=jax.tree.flatten(resumed)
    assert ta==tb
    digests=[hashlib.sha256(),hashlib.sha256()]
    for a,b in zip(aa,bb,strict=True):
        a,b=np.asarray(a),np.asarray(b)
        assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
        for h,value in zip(digests,(a,b)):
            h.update(str((value.dtype.str,value.shape)).encode());h.update(value.tobytes())
    result['compiled_carry_sha256']=digests[0].hexdigest()
    result['compiled_resumed_carry_sha256']=digests[1].hexdigest()
    assert result['compiled_carry_sha256']==result['compiled_resumed_carry_sha256']
    result['compiled_jit_cache_entries']=int(om._advance_chunk_fori._cache_size())
    assert result['compiled_jit_cache_entries']>0
    result['restart_exact_leaves']=len(aa);result['wall_s']=time.monotonic()-start
    result['guard_names']=list(GUARDS);result['guard_event_names']=list(EVENTS)
    result['guard_counters_per_step']=[row.get('guard_counters') for row in result['steps']]
    result['bounded_fields']=sorted(result['steps'][0].get('bounded_fields',[]))
    result['report_only_fields']=sorted(result['steps'][0].get('report_only_fields',[]))
    result['hfx_fdda_absmax']=float(np.max(np.abs(np.asarray(c.fdda.hfx_fdda))))
    result['hard_field_failures']=[(row['step'],key) for row in result['steps'] for key,s in row.get('scores',{}).items()
        if s['frozen_limits'].get('gate')=='hard_release_gate' and 'rmse' in s['frozen_limits'] and s['rmse']>s['frozen_limits']['rmse']]
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    assert not result['hard_field_failures'],result['hard_field_failures']


if __name__=='__main__':main(Path(sys.argv[1]),int(sys.argv[2]))
