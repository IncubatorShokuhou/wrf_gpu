"""Deviceless E41 prescreen of the PROD d01 stepping program (E156 recipe).

CPU init through the portable facade, then native (Triton) trace + CUDA lowering of
_advance_chunk_fori for d01 (n_steps=1, nested parent), CompileOnly compile for the
captured sm_120 target (autotune 0), optimized-HLO kInput/all transpose-fusion scan.
Derived source-exact from proofs/v05_diffusion/deviceless.py; W2 research binder only.
Older trees predate the portable facade: backend patching is conditional; CPU
Thomas init and the native product tridiagonal are common to both versions. No GPU execution, no speed claim.
Usage: dl_prod_d01.py --source TREE --case CASE_DIR --target target.pbtxt --out OUT
"""
import argparse
import gzip
import hashlib
import inspect
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--case', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--target', type=Path, required=True)
p.add_argument('--controls', default='off', help='off, all, or comma-separated W2 names')
p.add_argument('--lower-only', action='store_true')
p.add_argument('--domain', default='d01')  # d01 | d02 | fused_d02 ...
a = p.parse_args()
assert a.out.resolve().is_relative_to(Path('<USER_HOME>/wrf_gpu2_lanes/v45-lsmparams'))
assert os.environ['JAX_PLATFORMS'] == 'cpu'
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
assert os.environ['JAX_ENABLE_X64'].lower() == 'true'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
assert subprocess.run(['git', '-C', str(a.source), 'symbolic-ref', '--quiet', 'HEAD'], capture_output=True).returncode != 0
dirty = subprocess.check_output(['git', '-C', str(a.source), 'status', '--porcelain', '--untracked-files=no'])
a.out.mkdir(parents=True, exist_ok=False)
for key, name in (('TMPDIR', 'tmp'), ('TRITON_CACHE_DIR', 'triton'), ('CUDA_CACHE_PATH', 'cuda')):
    (a.out / name).mkdir()
    os.environ[key] = str(a.out / name)
started = time.monotonic()
rec = dict(source=str(a.source), source_commit=subprocess.check_output(
    ['git', '-C', str(a.source), 'rev-parse', 'HEAD'], text=True).strip(), source_dirty=dirty.decode(),
    case=str(a.case), domain=a.domain, physical_backend='cpu', lowering_platform='cuda', gpu_executed=False,
    scope='deviceless CompileOnly prescreen (autotune 0); actual GPU E41 scan stays authoritative',
    target_sha256=hashlib.sha256(a.target.read_bytes()).hexdigest(),
    harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    environment={k: os.environ.get(k) for k in ('GPUWRF_FAST_DEFAULTS','GPUWRF_JAX_CACHE','GPUWRF_NESTED_AOT',
                 'JAX_PLATFORMS','JAX_ENABLE_X64','CUDA_VISIBLE_DEVICES','XLA_FLAGS')}, phase='start')


def save():
    rec['wall_s'] = time.monotonic() - started
    rec['peak_rss_KiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (a.out / 'receipt.json').write_text(json.dumps(rec, indent=2, default=str) + '\n')
    print(rec['phase'], round(rec['wall_s'], 1), rec['peak_rss_KiB'], flush=True)


save()
rc = 1
try:
    inputs = json.loads((a.case / 'inputs.json').read_text())
    rec['input_sha256'] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                           for f in (a.case / 'input').iterdir() if f.is_file()}
    assert rec['input_sha256'] == inputs['input_sha256'], 'input hash differs from sealed manifest'
    sys.path[:0] = [str(a.source / 'src'), str(Path(__file__).resolve().parents[2] / 'proofs/o1_nlbind'), str(Path(__file__).resolve().parents[2] / 'scripts')]
    import gpuwrf
    assert Path(gpuwrf.__file__).resolve().is_relative_to(a.source.resolve()), gpuwrf.__file__
    import jax
    import jax.numpy as jnp
    import jaxlib._jax as x
    from jax.experimental import pallas as pl, topologies
    from jax._src.pallas import pallas_call as pc
    original_pallas_call = pc.pallas_call

    def cpu_call(*args, **kwargs):
        return original_pallas_call(*args, **{**kwargs, 'interpret': True})

    pc.pallas_call = pl.pallas_call = cpu_call
    import cpu_safe
    original_tridiag = jax.lax.linalg.tridiagonal_solve
    cpu_safe.install()
    try:
        from gpuwrf.backends import pallas as facade, platform as capability
    except ImportError:
        facade = capability = None  # v0.3.2 predates the portable facade
    from gpuwrf.contracts import state as sc
    from gpuwrf.integration import nested_pipeline as pipeline
    from gpuwrf.runtime import operational_mode as op, domain_tree as dt
    assert all(d.platform == 'cpu' for d in jax.devices())
    rec['gpuwrf_file'] = gpuwrf.__file__
    sc._gpu_device = lambda: jax.devices('cpu')[0]
    names = tuple(inputs['domains'])
    cfg = pipeline.NestedPipelineConfig(a.case / 'input', a.out / 'unused_output', a.out / 'unused_proof',
                                        hours=inputs['hours'], max_dom=len(names))
    rec['phase'] = 'cpu_init'; save()
    hierarchy, bundles, meta, run_start, dts, carries = pipeline._load_domains(cfg, names)
    nesting = meta.get('nesting', dict(feedback=False, spec_zone=1, smooth_option=1))
    tree_options = dict(feedback_enabled=nesting['feedback'], feedback_spec_zone=nesting['spec_zone'])
    if 'feedback_smooth_option' in inspect.signature(dt.DomainTree.from_domains).parameters:
        tree_options['feedback_smooth_option'] = nesting['smooth_option']
    else:
        assert not nesting['feedback'], 'old schema only screened for one-way PROD'
    import dataclasses
    from gpuwrf.surface_params import controls as sc
    from gpuwrf.io.noahmp_options import NoahMPOptions
    selected = [] if a.controls == 'off' else sorted(sc.LSM_CONTROLS)
    if a.controls not in ('off', 'all'):
        selected = a.controls.split(',')
    # Explicit Jarvis2 research context in both OFF and ON, for all nine controls.
    bundles = {n: dataclasses.replace(b, namelist=dataclasses.replace(b.namelist,
        noahmp_options=dataclasses.replace(b.namelist.noahmp_options or NoahMPOptions(),opt_crs=2))) for n,b in bundles.items()}
    rec['selected'] = selected
    assert sc.UNAVAILABLE_SURFACE_CONTROLS == frozenset(sc.LSM_CONTROLS)
    if selected:
        sc.UNAVAILABLE_SURFACE_CONTROLS = frozenset()  # this research process only
        values = {n: sc.prior_controls(b.namelist, selected, land_state=carries[n].noahmp_land) for n,b in bundles.items()}
        bundles = {n: dataclasses.replace(b, namelist=sc.bind_domain_controls(b.namelist,values,n,
                   land_state=carries[n].noahmp_land)) for n,b in bundles.items()}
    tree = dt.DomainTree.from_domains(hierarchy, bundles, **tree_options)
    for parent, edges in tree.edges.items():  # E156 (3): real force-down seeds children
        for edge in edges:
            carries[edge.child] = dt._operational_force(edge, carries[parent], carries[edge.child])
    rec['phase'] = 'init_done'; save()
    topology = topologies.get_topology_desc(platform='cuda', target_config=a.target.read_text(), topology='1x1x1')
    device = topology.devices[0]
    assert isinstance(device.client, x.CompileOnlyPyClient)
    sharding = jax.sharding.SingleDeviceSharding(device)

    def abstract(v):
        if hasattr(v, 'shape') and hasattr(v, 'dtype'):
            return jax.ShapeDtypeStruct(v.shape, v.dtype, weak_type=bool(getattr(v, 'weak_type', False)),
                                        sharding=sharding if getattr(v, 'committed', False) else None)
        return v

    original_call = original_pallas_call

    def native_call(*args, **kwargs):
        return original_call(*args, **{**kwargs, 'interpret': False})

    jax.clear_caches()
    if capability is not None:
        capability.runtime_backend = lambda: 'cuda'
        assert facade.cuda_native() and capability.cuda_native()
    jax.default_backend = lambda: 'gpu'
    pc.pallas_call = pl.pallas_call = native_call
    jax.lax.linalg.tridiagonal_solve = original_tridiag
    name = a.domain
    if name.startswith('fused_'):  # the product's fused parent+children closure (lower_case.py)
        import inspect
        parent = name[len('fused_'):]
        for n in names:
            op._resolve_operational_suite(bundles[n].namelist)
        fused = dt._operational_fused_cascade_factory(tree)(parent)
        assert fused is not None, 'no fused program for ' + parent
        closure = inspect.getclosurevars(fused).nonlocals
        fn, make_args = closure['fused_jit'], closure['_args']
        children = tuple(e.child for e in dt._fusable_parent(tree, parent))
        args = make_args(carries[parent], tuple(carries[c] for c in children), 1, tuple(1 for _ in children))
        kw = {}
        rec['fused_children'] = children
    else:
        fn = op._advance_chunk_fori
        nl = bundles[name].namelist
        op._resolve_operational_suite(nl)
        if len(names) == 1:  # a lone root advances one history segment (lower_case.py)
            from gpuwrf.io.gen2_accessor import Gen2Run
            cadences, _ = pipeline._output_cadence_steps_by_domain(Gen2Run(a.case / 'input'), names, dts)
            n_steps = int(cadences[name])
        elif hierarchy.children(name):
            n_steps = 1
        else:
            n_steps = next(int(e.parent_grid_ratio) for es in tree.edges.values() for e in es if e.child == name)
        kw = dict(n_steps=n_steps, cadence=int(nl.radiation_cadence_steps))
        args = (carries[name], nl, jnp.asarray(1, jnp.int32), op.build_clock_base(nl))
    rec['static_kwargs'] = kw
    rec['phase'] = 'trace'; save()
    traced = fn.trace(*jax.tree.map(abstract, args), **kw)
    flags = []

    def visit(value):
        if hasattr(value, 'jaxpr'):
            visit(value.jaxpr)
        elif hasattr(value, 'eqns'):
            for eq in value.eqns:
                if eq.primitive.name == 'pallas_call':
                    flags.append(bool(eq.params['interpret']))
                visit(eq.params)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)

    visit(traced.jaxpr)
    assert not any(flags), 'interpreted Pallas in native trace'
    rec['traced_pallas_calls'] = len(flags)
    rec['phase'] = 'lower'; save()
    from off_identity_probe import KernelCensus, pallas_stats, name_selfcheck, hlo_hashes
    with KernelCensus(a.domain) as census:
        lowered = traced.lower(lowering_platforms=('cuda',))
    stats = pallas_stats(traced.jaxpr)
    rec['pallas'] = stats
    rec['native_bodies'] = census.kernels
    rec['census_errors'] = name_selfcheck(stats, census.kernels)
    assert stats['interpreted'] == 0 and not rec['census_errors']
    rec.update(hlo_hashes(lowered.compiler_ir('hlo')))
    raw = lowered.as_text().encode()
    rec['stablehlo_sha256'] = hashlib.sha256(raw).hexdigest()
    rec['stablehlo_bytes'] = len(raw)
    del raw
    proto = lowered.compiler_ir('hlo').as_serialized_hlo_module_proto()
    rec['full_constant_raw_proto_sha256'] = hashlib.sha256(proto).hexdigest()
    (a.out / 'raw.hlo.pb').write_bytes(proto)
    del proto
    if a.lower_only:
        rec['phase'] = 'lower_done'; save()
        sys.stdout.flush(); os._exit(0)
    rec['phase'] = 'compile'; save()
    from jax._src import compiler as jax_compiler
    jax_compiler._get_cross_compile_backend = lambda _: None
    t = time.monotonic()
    compiled = lowered.compile(compiler_options={'xla_gpu_autotune_level': 0})
    rec['compile_s'] = time.monotonic() - t
    executable = compiled.runtime_executable()
    module = executable.hlo_modules()[0]
    hlo = module.to_string()
    (a.out / 'optimized.hlo').write_text(hlo)
    try:
        (a.out / 'optimized.hlo.pb').write_bytes(module.as_serialized_hlo_module_proto())
    except Exception as exc:  # frames are optional for the gate
        rec['proto_error'] = repr(exc)
    rec['compiled_memory'] = {k: getattr(compiled.memory_analysis(), k) for k in ('argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes', 'alias_size_in_bytes', 'generated_code_size_in_bytes')}
    rec['optimized_hlo_sha256'] = hashlib.sha256(hlo.encode()).hexdigest()
    rec['phase'] = 'scan'; save()
    from e41 import hlo_scan
    scan = hlo_scan.scan_text(hlo)
    rows = scan['transpose_hero_fusions']
    rec['n_fusions'] = scan['n_fusions']
    rec['transpose_fusions'] = len(rows)
    rec['kinput_transpose_over500'] = [(r['fusion'], r['n_instr']) for r in rows
                                       if r['kind'] == 'kInput' and r['n_instr'] > 500]
    rec['any_transpose_over500'] = [(r['fusion'], r['kind'], r['n_instr']) for r in rows if r['n_instr'] > 500]
    rec['top_transpose'] = sorted(((r['n_instr'], r['kind'], r['fusion']) for r in rows), reverse=True)[:12]
    blob = executable.serialize()
    cache = a.out / 'scan_cache/aot/captured'
    cache.mkdir(parents=True)
    (cache / f'{name}.xlaexec').write_bytes(blob)
    import wave_cubin_gate
    gate = wave_cubin_gate.run_gate(a.out / 'scan_cache', a.out / 'e41')
    rec['native_gate'] = gate.get('e41_gate')
    rec['hard_stack_gate'] = gate.get('hard_stack_gate')
    rec['max_stack_B'] = max(e['max_stack_B'] for r in gate['entries'] for e in r['elfs'])
    rec['kernels'] = sum(e['kernels'] for r in gate['entries'] for e in r['elfs'])
    (cache / f'{name}.xlaexec').unlink()
    rec['phase'] = 'done'
    rc = 0
except BaseException as exc:
    rec['exception'] = repr(exc)
    rec['traceback'] = traceback.format_exc()
finally:
    import shutil
    for d in ('unused_output', 'unused_proof', 'tmp', 'triton', 'cuda', 'scan_cache'):
        if (a.out / d).exists():
            shutil.rmtree(a.out / d)
    save()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(rc)
