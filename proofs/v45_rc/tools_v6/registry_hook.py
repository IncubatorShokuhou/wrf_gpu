"""One registry START+terminal END for each actual arm invocation; no JAX import."""
import importlib.util,json,os,signal,time,uuid,sys
sys.dont_write_bytecode=True  # -I ignores PYTHONDONTWRITEBYTECODE; keep pinned source/cache-free
from pathlib import Path
def load_hooks(smoke):
    import sys
    script=Path(os.environ['W45R_PINNED_SOURCE'])/'scripts/alisios_registry.py'
    spec=importlib.util.spec_from_file_location('v045_registry_hooks',script);hooks=importlib.util.module_from_spec(spec);sys.modules[spec.name]=hooks;spec.loader.exec_module(hooks)
    atlas=Path(os.environ['W45R_PRIVATE_ATLAS']) if smoke else None
    if smoke:hooks.PRODUCTION_ATLAS=atlas.parent/'production_atlas_standin'
    return hooks.Registry(atlas_dir=atlas,smoke=smoke)
def record_deleted(out):
    out=Path(out);start=json.loads((out/'REGISTRY_START.json').read_text());end=json.loads((out/'REGISTRY_END.json').read_text())
    registry=load_hooks(os.environ.get('WRF_GPU2_SEQ_SMOKE')=='1')
    metadata=dict(run_id=start['run_id'],by='v45-rc',raw_deleted=True,extract=[str(out/'WRFOUT_FILES.tsv')+'::'+__import__('hashlib').sha256((out/'WRFOUT_FILES.tsv').read_bytes()).hexdigest()],notes='SHA/size receipt preserved before exact output deletion')
    registry.update(metadata,authorization='Manager task v45-rc real-world registry hooks; 2026-10-28T12:30:54Z')
    end['raw_deleted']=True;(out/'REGISTRY_END.json').write_text(json.dumps(end,indent=1)+'\n')
def begin(command):
    positions=[i for i,v in enumerate(command)if Path(v).name=='arm.sh']
    if len(positions)!=1:raise ValueError('registry requires exactly one sealed arm.sh command')
    i=positions[0];envfile,name,case,out=command[i+1:i+5];out=Path(out);out.mkdir(parents=True,exist_ok=True)
    root=Path(os.environ['W45R_PINNED_SOURCE']);script=root/'scripts/alisios_registry.py'
    spec=importlib.util.spec_from_file_location('v045_registry_hooks',script);hooks=importlib.util.module_from_spec(spec)
    import sys;sys.modules[spec.name]=hooks;spec.loader.exec_module(hooks)
    smoke=os.environ.get('WRF_GPU2_SEQ_SMOKE')=='1'
    if smoke:
        atlas=Path(os.environ['W45R_PRIVATE_ATLAS'])
        hooks.PRODUCTION_ATLAS=atlas.parent/'production_atlas_standin'
    else:atlas=None
    registry=hooks.Registry(atlas_dir=atlas,smoke=smoke)
    manifest=json.loads(Path(os.environ['W45R_PIN_MANIFEST']).read_text())
    matches=[k for k,v in manifest['inputs'].items()if Path(v).resolve()==Path(case).resolve()]
    if len(matches)!=1:raise ValueError('arm input has no unique sealed case identity')
    meta=manifest['input_metadata'][matches[0]]
    args=command[i+5:];hours=float(args[args.index('--hours')+1])if '--hours'in args else 0
    hours=int(hours)if hours==int(hours)else hours
    run_id='v045rc:'+manifest['pin'][:12]+':'+name+':'+uuid.uuid4().hex
    start=dict(run_id=run_id,by='v45-rc',producer='wrf_gpu2',purpose=('CPU_SEQ_SMOKE PLAN_ONLY'if smoke and matches[0]!='SWISS'else'CPU_SEQ_SMOKE first root'if smoke else'v0.4.5 RW '+name),family='release',forcing=meta['forcing'],engine='wrf_gpu',nests=meta['nests'],lead_h=hours,issue=meta['issue'],run_dir=str(out),retention='raw_keep')
    registry.start(start)
    (out/'REGISTRY_START.json').write_text(json.dumps(start,indent=1)+'\n')
    os.environ['ALISIOS_WRF_RUN_ID']=run_id
    if smoke:os.environ['ALISIOS_ATLAS_DIR']=str(atlas)
    return registry,start,out
def finish(handle,receipt):
    registry,start,out=handle;rc=receipt.get('rc',1)
    status='ABORTED'if receipt.get('timed_out')or receipt.get('signal')or rc in(124,130,137,143)else'DONE_HELD'if rc==0 else'FAILED'
    end=dict(run_id=start['run_id'],by='v45-rc',rc=rc,status=status,corpus='held',raw_deleted=False,notes='actual arm terminal event after descendant teardown; no backfill')
    registry.end(end);(out/'REGISTRY_END.json').write_text(json.dumps(end,indent=1)+'\n')

def close_open_after_teardown(rc):
    """The entry supervisor survives an owned-scope OOM; end its started arms after reaping.
    Live terminal handling, not a backfill. Closed arms and unstarted slots are untouched.
    """
    manifest_path=os.environ.get('W45R_MANIFEST_PATH')
    if not manifest_path:return []  # standalone supervisor controls without a model entry
    manifest=json.loads(Path(manifest_path).read_text());mode=os.environ['W45R_MODE']
    os.environ['W45R_PINNED_SOURCE']=manifest['snapshot']
    os.environ['W45R_PRIVATE_ATLAS']='<USER_HOME>/wrf_gpu2_lanes/v45-rc/smoke_v6/private_atlas'
    roots=[Path(os.environ['W45R_RUN'])]
    if os.environ.get('W45R_ENTRY_NAME')!='rw3':
        base=Path(os.environ['W45R_SMOKE_ROOT']) if mode=='smoke' else Path('<DATA_ROOT>/wrf_gpu2/scratch/v45-rc')
        tag=os.environ.get('W45R_SMOKE_SESSION_TAG','v6')
        roots.append(base/('session_'+manifest['pin'][:12]+'_'+mode+'_'+tag))
    closed=[]
    for root in roots:
        for startfile in sorted((root/'runs').glob('*/REGISTRY_START.json')):
            out=startfile.parent
            if (out/'REGISTRY_END.json').exists():continue
            start=json.loads(startfile.read_text())
            assert start['run_id'].startswith('v045rc:'+manifest['pin'][:12]+':')
            assert start['run_dir']==str(out),'receipt must describe this actual arm'
            registry=load_hooks(mode=='smoke')
            finish((registry,start,out),dict(rc=rc if rc else 125,timed_out=rc==124,signal=15))
            closed.append(dict(run_id=start['run_id'],status='ABORTED',reason='entry supervisor terminal event after all owned children reaped'))
    return closed
