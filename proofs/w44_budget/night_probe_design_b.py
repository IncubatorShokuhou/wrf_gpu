"""Manager-only GPU protocol for the composed, opt-in diagnostic twin.

One GPU process: real day/night two-step components, prime OFF/ON into one
cache, then both three-nest 3 h runs. An EDMF-off prime/replay qualifies the
master switch. Every physical carry is supplied by unchanged production code.
"""
import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import runpy
import subprocess
import sys
import time


HERE=Path(__file__).resolve().parent


def CASES(manifest):
    """(edmf, period) component cases; production default = all four."""
    return [tuple(c) for c in manifest.get('component_cases',[[1,'day'],[1,'night'],[0,'day'],[0,'night']])]


def atomic(path,record):
    temp=path.with_suffix('.pending');temp.write_text(json.dumps(record,indent=2)+'\n');os.replace(temp,path)


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def prereq():
    manifest=json.loads((HERE/'manifest.json').read_text())
    model=Path(manifest['model']);pin=manifest['model_pin']
    assert subprocess.check_output(['git','-C',str(model),'rev-parse','HEAD'],text=True).strip()==pin
    assert not subprocess.check_output(['git','-C',str(model),'status','--porcelain','--untracked-files=no'],text=True).strip()
    assert (model/'data/wrf_pristine').exists()
    old=json.loads(Path(manifest['input_receipt']).read_text())
    case=Path(manifest['case'])
    for name,item in old['input_files'].items():assert sha(case/name)==item['sha256'],name
    return manifest,old['input_files']


class PrimeComplete(BaseException):pass


def aot_domains_ok(actual,resolved):
    """Stepping programs in domain_tree.nested_aot_report()['domains'], read as the product names them.

    Nothing is re-spelled here (manager 02:22Z): per-domain programs are the report keys equal to the
    product's resolved domain names; every other key is a nested (e.g. fused cascade) program, taken
    verbatim. The root is the resolved domain with the largest time step. Require the root program plus
    a nested program (or every domain separately), and a recorded load/compile source for every entry."""
    names=set(actual);domains=set(resolved)
    per_domain=names&domains;nested=names-domains
    root=max(domains,key=lambda n:float(resolved[n]['dt'])) if domains else None
    return (root in per_domain and (bool(nested) or domains<=names)
            and all(isinstance(v,dict) and v.get('source') for v in actual.values()))


def configure_loaded(values,on,edmf,*,seed_history,config_of,protect,validate):
    """Host-only selection hook; existing loaded arrays/static operands stay put."""
    bundles=dict(values[1]);carries=dict(values[-1]);resolved={}
    for name,bundle in bundles.items():
        nl=replace(bundle.namelist,tke_budget=int(on),bl_mynn_output=int(on),bl_mynn_edmf=edmf)
        validate(nl);bundles[name]=replace(bundle,namelist=nl)
        carry=carries[name]
        carry=carry.replace(mynn_budget=seed_history(carry.state.theta.shape,config_of(nl)))
        carries[name]=protect(carry,nl)
        resolved[name]=dict(dt=float(nl.dt_s),cadence=int(nl.radiation_cadence_steps),
                           shape=list(carry.state.theta.shape),**config_of(nl))
    return (values[0],bundles,*values[2:-1],carries),resolved


def profiler(event):
    if os.environ.get('BUDGET_CPU_ALL_PHASES')=='1':
        assert os.environ.get('JAX_PLATFORMS')=='cpu'
        return False  # profiler API is part of the CPU-only device substitution
    import ctypes
    for name in ('libcudart.so.12','libcudart.so'):
        try:return getattr(ctypes.CDLL(name),event)()==0
        except OSError:continue
    return False


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--dry-run',action='store_true')
    args=p.parse_args();manifest,inputs=prereq()
    if args.dry_run:
        print(json.dumps(dict(model=manifest['model_pin'],input_files=len(inputs),gpu_started=False,
            order=['components_'+'_'.join(f'{p}_edmf{e}' for e,p in CASES(manifest))]+(
                ['prime_off','prime_on','replay_off','replay_on','prime_edmf0','replay_edmf0'] if manifest.get('coupled',True) else []))))
        return
    if os.environ.get('BUDGET_CPU_LAUNCH_SMOKE')=='1':
        # This branch is admitted only by the CPU device shim. All physical
        # flags are the real launcher flags; no GPU executable is exported.
        assert os.environ.get('JAX_PLATFORMS')=='cpu'
        model=Path(manifest['model']);sys.path[:0]=[str(model/'src'),str(model/'tests')]
        import jax
        import numpy as np
        from gpuwrf.coupling.mynn_budget_adapter import mynn_sources_with_history
        from gpuwrf.diagnostics.mynn_budget import configuration,history_fields
        module_spec=importlib.util.spec_from_file_location('budget_smoke_real_cloud',model/'tests/v025/fid_cloud/test_sgs_carry.py')
        fixture=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(fixture)
        assert jax.devices()[0].platform=='cpu'
        state=fixture._cloud_state()
        config=configuration(dict(tke_budget=1,bl_mynn_output=1))
        start=time.perf_counter()
        program=jax.jit(lambda s:mynn_sources_with_history(s,18.,budget_config=config,restart=True)).lower(state).compile()
        physical,history=jax.block_until_ready(program(state))
        assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves((physical,history)))
        assert len(history_fields(history))==21
        args.out.mkdir(parents=True,exist_ok=False)
        atomic(args.out/'result.json',dict(verdict='LAUNCH_SMOKE_MODEL_REACHED',actual_platform='cpu',
            gpu_executed=False,source_commit=manifest['model_pin'],model='actual MYNN production caller + diagnostic twin',
            shape=list(state.theta.shape),history_fields=21,compile_first_s=time.perf_counter()-start,
            note='CPU device substitution; no physical BYTE, GPU E41 or full forecast pass claimed'))
        return
    device_smoke=os.environ.get('BUDGET_CPU_ALL_PHASES')=='1'
    assert os.environ.get('GPUWRF_GPU_LOCK_HELD')=='1'
    platform='cpu' if device_smoke else 'gpu'
    assert (os.environ.get('JAX_PLATFORMS')=='cpu')==device_smoke
    model=Path(manifest['model']);sys.path[:0]=[str(model/'src'),str(HERE)]
    import jax
    import jax.numpy as jnp
    import numpy as np
    from gpuwrf import cli
    from gpuwrf.integration import nested_pipeline as pipeline
    from gpuwrf.runtime import domain_tree as D,operational_mode as O
    from gpuwrf.diagnostics.mynn_budget import seed,configuration,history_fields
    assert jax.devices()[0].platform==platform
    assert Path(pipeline.__file__).resolve().is_relative_to(model)
    args.out.mkdir(parents=True,exist_ok=False)
    path=args.out/'result.json'
    record=dict(schema='wrf_gpu2.w44_budget.night_design_b.v1',platform=platform,source_commit=manifest['model_pin'],
        scope='compiled GPU regression and cost; pristine term oracles run on the CPU after GPU unlock',
        input_files=inputs,cache=os.environ['GPUWRF_JAX_CACHE_DIR'],components={},phases={},
        profiler_events=[],verdict='STARTED')
    inventory={}
    def save_inventory():atomic(args.out/'compiled_program_inventory.json',dict(metadata=dict(nested_aot=dict(domains=inventory))))
    atomic(path,record)
    for edmf,period in CASES(manifest):
        name=f'wn3_d03_{period}_edmf{edmf}';out=args.out/'components'/name
        out.parent.mkdir(exist_ok=True)
        previous=sys.argv
        sys.argv=[str(HERE/'twin_probe.py'),'--source',str(model),'--out',str(out),
            '--case','wn3_d03_'+period,'--edmf',str(edmf),'--platform',platform,
            '--columns',str(manifest.get('component_columns',32)),
            '--export-cache',os.environ['GPUWRF_JAX_CACHE_DIR']]
        try:runpy.run_path(str(HERE/'twin_probe.py'),run_name='__main__')
        finally:sys.argv=previous
        component=json.loads((out/'result.json').read_text())
        assert component['platform']==platform and component['byte_fields']==component['total_fields']==50
        record['components'][name]=component
        for arm in component['arms'].values():inventory[arm['program_name']]=arm['aot_export']
        save_inventory();atomic(path,record)
    if not manifest.get('coupled',True):
        # Component-only CUDA mini (manifest): same builders/env/E41/checker; no coupled replay claim.
        record['verdict']='COMPONENTS_BYTE';atomic(path,record);return
    original_load,original_segment=pipeline._load_domains,pipeline.run_operational_domain_tree
    current={};final={};prime_arrays={}
    def loaded(*args,**kwargs):
        values,resolved=configure_loaded(original_load(*args,**kwargs),current['on'],current['edmf'],
            seed_history=seed,config_of=configuration,protect=O._protect_initial_carry_seeds,validate=O._resolve_operational_suite)
        current['resolved']=resolved;current['root_dt']=resolved['d01']['dt']
        return values
    def segment(*args,**kwargs):
        if current['prime']:kwargs=dict(kwargs,root_steps=2)
        capture=(not current['prime'] and current['edmf']==1 and len(current['segments'])==1)
        if capture:record['profiler_events'].append(dict(phase=current['label'],start_ok=profiler('cudaProfilerStart')))
        start=time.perf_counter()
        result=original_segment(*args,**kwargs);jax.block_until_ready(result.carries)
        if capture:record['profiler_events'][-1]['stop_ok']=profiler('cudaProfilerStop')
        roots=int(kwargs['root_steps']);seconds=roots*current['root_dt']
        current['segments'].append(dict(root_steps=roots,wall_s=time.perf_counter()-start,
            simulated_s=seconds,own_steps=dict(result.own_steps)))
        final.clear();final.update(result.carries)
        actual=D.nested_aot_report().get('domains',{})
        assert aot_domains_ok(actual,current['resolved']),actual
        inventory.update(actual);save_inventory()
        if current['prime']:raise PrimeComplete()
        return result
    def digest(carries,keep=False):
        rows,arrays={},{}
        for domain,carry in carries.items():
            if current['on']:
                fields=history_fields(carry.mynn_budget);assert len(fields)==21
                assert any(np.count_nonzero(np.asarray(fields[n])) for n in ('qSHEAR','Dtke')),'history was not updated'
                for name,value in fields.items():
                    value=np.asarray(value);assert value.dtype==np.float32 and np.isfinite(value).all(),name
                    if current['edmf']==0 and name.startswith(('edmf_','sub_','det_')):assert not np.count_nonzero(value),name
            carry=carry.replace(mynn_budget=None)
            for key,value in jax.tree.flatten_with_path(carry)[0]:
                value=np.asarray(value);assert np.isfinite(value).all(),(domain,key)
                key=domain+'/'+str(key)
                rows[key]=dict(shape=list(value.shape),dtype=str(value.dtype),sha256=hashlib.sha256(value.tobytes()).hexdigest())
                if keep:arrays[key]=value.copy()
        return rows,arrays
    def files(directory):
        from netCDF4 import Dataset
        result={}
        for file in sorted(directory.glob('wrfout_d*')):
            if not file.is_file():continue
            with Dataset(file) as dataset:
                dataset.set_auto_mask(False)
                result[file.name]={}
                for name,v in dataset.variables.items():
                    value=np.asarray(v[:])
                    if value.dtype.kind in 'f':assert np.isfinite(value).all(),(file,name)
                    result[file.name][name]=dict(shape=list(value.shape),dtype=str(value.dtype),sha256=hashlib.sha256(value.tobytes()).hexdigest())
        return result
    pipeline._load_domains,pipeline.run_operational_domain_tree=loaded,segment
    off_rows=off_files=None
    try:
        for label,on,prime,edmf in [('prime_off',False,True,1),('prime_on',True,True,1),
                ('replay_off',False,False,1),('replay_on',True,False,1),
                ('prime_edmf0',True,True,0),('replay_edmf0',True,False,0)]:
            current.clear();current.update(label=label,on=on,prime=prime,edmf=edmf,segments=[])
            final.clear();out=args.out/label;start=time.perf_counter()
            try:
                rc=cli.main(['run','--input-dir',manifest['case'],'--output-dir',str(out/'wrfout'),
                    '--scratch-dir',str(out/'scratch'),*manifest.get('duration_args',['--hours','3']),
                    '--max-dom',str(manifest.get('max_dom',3))])
                assert not prime and rc==0,(label,rc)
            except PrimeComplete:assert prime
            rows,arrays=digest(final,keep=prime and edmf==1)
            written=files(out/'wrfout')
            arm=dict(wall_s=time.perf_counter()-start,segments=current['segments'],resolved=current['resolved'],
                terminal_leaf_sha256=rows,history_files=written,rss_max_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                device_memory_stats=jax.devices()[0].memory_stats())
            record['phases'][label]=arm
            if label=='prime_off':prime_rows,prime_arrays=rows,arrays
            if label=='prime_on':
                assert rows.keys()==prime_rows.keys()
                differences={}
                for key,value in arrays.items():
                    if rows[key]!=prime_rows[key]:
                        delta=prime_arrays[key].astype(float)-value.astype(float)
                        differences[key]=dict(max_abs=float(np.abs(delta).max()),rms=float(np.sqrt(np.mean(delta*delta))))
                arm['old_carry_differences']=differences;prime_arrays.clear();arrays.clear()
                if differences:
                    record['verdict']='STOP_NONBYTE_PRIME';atomic(path,record);raise SystemExit(2)
            if label=='replay_off':
                assert len(written)==manifest.get('expected_history_files',9);off_rows,off_files=rows,written
            if label=='replay_on':
                assert len(written)==manifest.get('expected_history_files',9) and written.keys()==off_files.keys()
                from gpuwrf.io.mynn_budget_history import METADATA
                for name,expected in off_files.items():
                    assert {k:v for k,v in written[name].items() if k not in METADATA}=={
                        k:v for k,v in expected.items() if k not in METADATA},(name,'old nonbudget wrfout variable changed')
                    assert set(written[name])==set(expected)|set(METADATA),(name,'unexpected variable selection')
                    assert set(METADATA)<=written[name].keys(),name
                if rows!=off_rows:
                    record['verdict']='STOP_NONBYTE_3H';atomic(path,record);raise SystemExit(2)
                arm['terminal_byte_equal']=arm['old_wrfout_arrays_byte_equal']=True
            if label=='replay_edmf0':assert len(written)==manifest.get('expected_history_files',9)
            atomic(path,record);print(label,json.dumps(dict(wall_s=arm['wall_s'],files=len(written))),flush=True)
    finally:pipeline._load_domains,pipeline.run_operational_domain_tree=original_load,original_segment
    record['verdict']='FIRST_STEP_AND_REPLAY_BYTE';record['duration_args']=manifest.get('duration_args',['--hours','3']);atomic(path,record)


if __name__=='__main__':main()
