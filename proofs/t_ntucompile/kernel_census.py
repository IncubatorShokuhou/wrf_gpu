"""Kernel census of an actual composite step program (t-ntucompile; from t_lowersweep/lower_case.py).

Traces the native CUDA step program like lower_case.py, then lowers every unique Pallas
kernel to TTIR (files + tensor memory-op counts). No XLA compile, no GPU.
Original: CPU-only CUDA preflight of one immutable-source, actual composite input.

Initialization uses the portable facade. Tracing restores native Pallas and
the product tridiagonal solver. No forecast or GPU-fidelity claim is made.
"""
import argparse
import faulthandler
import gzip
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import resource
import re
import shutil
import subprocess
import sys
import time
import traceback

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--case', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--target', type=Path, required=True)
p.add_argument('--compile-under-bytes', type=int, default=4000000)
a = p.parse_args()
assert os.environ['JAX_PLATFORMS'] == 'cpu'
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
assert not Path('/tmp/wrf_gpu2_quiet').exists()
assert subprocess.run(['git','-C',str(a.source),'symbolic-ref','--quiet','HEAD'],capture_output=True).returncode != 0
assert not subprocess.check_output(['git','-C',str(a.source),'status','--porcelain','--untracked-files=no'])
a.out.mkdir(parents=True, exist_ok=False)
stack_file = (a.out/'slow_phase_stacks.txt').open('w')
faulthandler.dump_traceback_later(120,repeat=True,file=stack_file)
started = time.monotonic()
rec = dict(case=a.case.name, source_commit=subprocess.check_output(
    ['git','-C',str(a.source),'rev-parse','HEAD'], text=True).strip(),
    physical_backend='cpu', lowering_platform='cuda', gpu_executed=False,
    scope='native CUDA lowering / optional AUTOTUNE0 deviceless prescreen, not fidelity or speed',
    lower='NOT_STARTED', compile='NOT_ATTEMPTED', programs=[],
    cache='private per-arm scratch directories; managed GPUWRF cache off; cleaned by supervisor',
    target_sha256=hashlib.sha256(a.target.read_bytes()).hexdigest(),
    harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    environment={k:v for k,v in os.environ.items() if k.startswith('GPUWRF_') or k in
                 ('JAX_PLATFORMS','CUDA_VISIBLE_DEVICES','JAX_PALLAS_USE_MOSAIC_GPU','XLA_FLAGS',
                  'JAX_COMPILATION_CACHE_DIR','TRITON_CACHE_DIR','CUDA_CACHE_PATH','TMPDIR')})
rec['pid']=os.getpid()
rec['boot_id']=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
rec['cgroup']=Path('/proc/self/cgroup').read_text().strip()
rec['systemd_scope']=next((part for part in rec['cgroup'].split('/') if part.endswith('.scope')),None)

def save():
    rec['wall_s'] = time.monotonic()-started
    rec['peak_rss_KiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (a.out/'receipt.json').write_text(json.dumps(rec,indent=2,default=str)+'\n')
    print(a.case.name,rec['lower'],rec['compile'],round(rec['wall_s'],2),flush=True)

save()
try:
    inputs = json.loads((a.case/'inputs.json').read_text())
    rec['matrix_status'] = inputs['status']
    rec['input_sha256'] = {f.name:hashlib.sha256(f.read_bytes()).hexdigest()
                          for f in (a.case/'input').iterdir() if f.is_file()}
    assert rec['input_sha256'] == inputs['input_sha256'], 'actual input hash differs from sealed manifest'
    rec['phase'] = 'input_hashes_verified'; save()
    sys.path[:0] = [str(a.source/'src'),str(a.source/'proofs/o1_nlbind'),str(a.source/'scripts')]
    import gpuwrf
    import jax
    import jax.numpy as jnp
    import jaxlib._jax as x
    from jax.experimental import pallas as pl, topologies
    from jax._src.pallas import pallas_call as pc
    import cpu_safe
    original_tridiag = jax.lax.linalg.tridiagonal_solve
    cpu_safe.install()
    from gpuwrf.backends import pallas as facade, platform as capability
    physical_resolver = capability.runtime_backend
    physical_backend = jax.default_backend
    from gpuwrf.contracts import state as sc
    from gpuwrf.integration import nested_pipeline as pipeline
    from gpuwrf.runtime import operational_mode as op, domain_tree as dt
    from gpuwrf.io.gen2_accessor import Gen2Run
    assert all(d.platform == 'cpu' for d in jax.devices())
    rec['jax_version'] = jax.__version__
    rec['jax_enable_x64_before_native_trace'] = bool(jax.config.jax_enable_x64)
    rec['jax_use_simplified_jaxpr_constants']=getattr(jax.config,'jax_use_simplified_jaxpr_constants',None)
    sc._gpu_device = lambda: jax.devices('cpu')[0]
    assert not capability.cuda_native()
    names = tuple(inputs['domains'])
    cfg = pipeline.NestedPipelineConfig(a.case/'input',a.out/'unused_output',a.out/'unused_proof',
        hours=inputs['hours'],max_dom=len(names))
    t = time.monotonic()
    rec['phase'] = 'CPU_loader_initialization'; save()
    hierarchy,bundles,meta,run_start,dts,carries = pipeline._load_domains(cfg,names)
    cadence_run = Gen2Run(a.case/'input')
    # These CLI post-loader seeds change the carry ABI used by the step.
    if (a.case/'input/tslist').is_file():
        from gpuwrf.timeseries.runtime import prepare_time_series
        _,carries = prepare_time_series(cfg,names,bundles,carries,dts,run_start,cadence_run.namelist)
        rec['timeseries_seeded'] = True
    if ((a.case/'input/gpuwrf.streams.json').is_file() or
            any(getattr(c,'regional_streams',None) is not None for c in carries.values())):
        from gpuwrf.streams.runtime import prepare_streams
        _,carries = prepare_streams(cfg,names,carries,dts,run_start,resumed=False,
                                   restart_store=None,restart_identity=None,saved=None)
        rec['regional_streams_seeded'] = True
    from gpuwrf.dfi.control import dfi_control_from_namelist
    control = dfi_control_from_namelist(cadence_run.namelist)
    rec['dfi_prepass'] = 'ADDITIONAL_STAGE_PREFLIGHT_REQUIRED' if control.active else 'INACTIVE'
    rec['init_s'] = time.monotonic()-t
    rec['phase'] = 'child_force_initialization'; save()
    nesting = meta.get('nesting',dict(feedback=False,spec_zone=1,smooth_option=1))
    tree = dt.DomainTree.from_domains(hierarchy,bundles,feedback_enabled=nesting['feedback'],
        feedback_spec_zone=nesting['spec_zone'],feedback_smooth_option=nesting['smooth_option'])
    cadences,_ = pipeline._output_cadence_steps_by_domain(cadence_run,names,dts)
    # E156: seed children using the real force-down, not initial placeholders.
    for parent,edges in tree.edges.items():
        for edge in edges:
            carries[edge.child] = dt._operational_force(edge,carries[parent],carries[edge.child])
    topology = topologies.get_topology_desc(platform='cuda',target_config=a.target.read_text(),topology='1x1x1')
    device = topology.devices[0]
    assert isinstance(device.client,x.CompileOnlyPyClient)
    sharding = jax.sharding.SingleDeviceSharding(device)
    def abstract(v):
        if hasattr(v,'shape') and hasattr(v,'dtype'):
            return jax.ShapeDtypeStruct(v.shape,v.dtype,weak_type=bool(getattr(v,'weak_type',False)),
                sharding=sharding if getattr(v,'committed',False) else None)
        return v
    original_call = pc.pallas_call
    def native_call(*args,**kwargs):
        return original_call(*args,**{**kwargs,'interpret':False})
    if a.case.name != 'c48':
        jax.clear_caches()  # Backend selectors below are trace-time globals.
    # Patch the shared resolver, not just the facade attribute: kernels import
    # cuda_native aliases directly, and MCICA/tridiagonal ask platform itself.
    # Leaving the function objects intact also preserves imports made in init.
    capability.runtime_backend = lambda: 'cuda'
    assert facade.cuda_native() and capability.cuda_native()
    jax.default_backend = lambda: 'gpu'
    pc.pallas_call = pl.pallas_call = native_call
    jax.lax.linalg.tridiagonal_solve = original_tridiag
    from gpuwrf.kernels import rad_mcica
    assert rad_mcica.pl_cuda_native()
    rec['native_trace'] = dict(facade=True,capability_resolver='cuda',mcica_native=True,
                              python_backend='gpu',pallas_interpret=False,
                              product_tridiagonal=True,virtual_target=str(device))
    jobs = []
    for name in names:
        nl = bundles[name].namelist
        op._resolve_operational_suite(nl)
        # A lone root advances one history segment; nested parents advance one
        # step, leaves advance the parent ratio (split at output alarms).
        count = cadences[name] if len(names)==1 else 1
        if len(names)>1 and not hierarchy.children(name):
            count = next(int(e.parent_grid_ratio) for es in tree.edges.values() for e in es if e.child==name)
        args = (carries[name],nl,jnp.asarray(1,jnp.int32),op.build_clock_base(nl))
        jobs.append((name,op._advance_chunk_fori,args,dict(n_steps=count,cadence=int(nl.radiation_cadence_steps))))
    # Capture the product's existing fused closure rather than reimplementing it.
    fused = dt._operational_fused_cascade_factory(tree)(names[0]) if len(names)>1 else None
    if fused is not None:
        closure = inspect.getclosurevars(fused).nonlocals
        fn = closure['fused_jit']; make_args = closure['_args']
        children = tuple(e.child for e in dt._fusable_parent(tree,names[0]))
        args = make_args(carries[names[0]],tuple(carries[c] for c in children),1,tuple(1 for _ in children))
        jobs.append(('fused_'+names[0],fn,args,{}))
    rec['phase'] = 'kernel_census'; save()
    from jax._src.pallas.triton import lowering as tlow
    seen = {}
    def visit(value, kernels):
        if hasattr(value,'jaxpr'):visit(value.jaxpr,kernels)
        elif hasattr(value,'eqns'):
            for eq in value.eqns:
                if eq.primitive.name=='pallas_call':kernels.append(eq.params)
                visit(eq.params,kernels)
        elif isinstance(value,dict):
            for child in value.values():visit(child,kernels)
        elif isinstance(value,(tuple,list)):
            for child in value:visit(child,kernels)
    census=[]
    kdir=a.out/'kernels'; kdir.mkdir()
    for name,fn,args,kw in jobs:
        traced = fn.trace(*jax.tree.map(abstract,args),**kw)
        kernels=[]; visit(traced.jaxpr,kernels)
        for params in kernels:
            kname=str(params.get('name_and_src_info',params.get('name','?')))
            try:
                module=tlow.lower_jaxpr_to_triton_module(params['jaxpr'],params['grid_mapping'],'cuda',120)
                text=str(module.module)
            except Exception as exc:
                census.append(dict(program=name,kernel=kname,error=repr(exc)[:300]));continue
            digest=hashlib.sha256(text.encode()).hexdigest()
            if digest in seen:seen[digest]['count']+=1;continue
            path=kdir/f'{len(seen):03d}.ttir'; path.write_text(text)
            cp=params.get('compiler_params') or {}
            warps=getattr(cp.get('triton',None),'num_warps',None) if isinstance(cp,dict) else getattr(cp,'num_warps',None)
            seen[digest]=row=dict(program=name,kernel=kname,file=str(path),bytes=len(text),
                tensor_loads=len(re.findall(r'tt\.load[^\n]*tensor<[^\n]*!tt\.ptr',text)),
                tensor_stores=len(re.findall(r'tt\.store[^\n]*tensor<[^\n]*!tt\.ptr',text)),
                num_warps=warps,count=1,sha256=digest)
            census.append(row)
        rec['kernel_census']=census; save()
    rec['phase']='census_done'; save()
except BaseException as exc:
    rec['error']=repr(exc); rec['traceback']=traceback.format_exc(); save(); raise
finally:
    stack_file.close()
os._exit(0)
