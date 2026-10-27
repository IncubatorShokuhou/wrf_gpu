"""o5-perf H-CB deviceless A/B (OFF/V1/V2 compile options on ONE lowered step program) + memory probe: deviceless compile + compiled memory stats (+ optional XLA buffer-assignment dump).
Derived from w44-perf deviceless.py (E41 prescreen of the PROD d01 stepping program, E156 recipe); E41 gate removed.

CPU init through the portable facade, then native (Triton) trace + CUDA lowering of
_advance_chunk_fori for d01 (n_steps=1, nested parent), CompileOnly compile for the
captured sm_120 target (autotune 0), optimized-HLO kInput/all transpose-fusion scan.
Derived from t-cfix-mynn/e41/dl_prod_d01.py; w44 version comparison.
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
p.add_argument('--domain', default='d01')  # d01 | d02 | fused_d02 ...
a = p.parse_args()
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
    environment={k: v for k, v in os.environ.items() if k.startswith('GPUWRF_') or k in
                 ('JAX_PLATFORMS', 'CUDA_VISIBLE_DEVICES', 'XLA_FLAGS')}, phase='start')


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
    sys.path[:0] = [str(a.source / 'src'), '<DATA_ROOT>/wrf_gpu2/worktrees/o5-perf/proofs/o1_nlbind', '<DATA_ROOT>/wrf_gpu2/worktrees/o5-perf/scripts']
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
    lowered = traced.lower(lowering_platforms=('cuda',))
    raw = lowered.as_text().encode()
    rec['stablehlo_sha256'] = hashlib.sha256(raw).hexdigest()
    rec['stablehlo_bytes'] = len(raw)
    del raw
    rec['phase'] = 'compile'; save()
    from jax._src import compiler as jax_compiler
    jax_compiler._get_cross_compile_backend = lambda _: None
    import zstandard, importlib.util as ilu
    from gpuwrf.runtime import aot_executable as ax, aot_slim_module as asm
    gspec = ilu.spec_from_file_location('gate', '<DATA_ROOT>/wrf_gpu2/worktrees/o5-perf/scripts/wave_cubin_gate.py')
    gate = ilu.module_from_spec(gspec); sys.modules['gate'] = gate; gspec.loader.exec_module(gate)
    F = lambda d: list(ax._proto_fields(d))
    def dbg_fields(blob):
        gpu = gate._gpu_payload(blob, ax, asm)
        if ax._gpu_blob_format(blob) == ax.GPU_THUNK_FORMAT:
            _, info, pos = asm._chunks(gpu)[-1]
            data, _ = asm._read_logical(gpu, pos, info['data_size'])
            (gpu,) = asm._decode_simple_chunk(data, info['num_records'])
        hwc = next(v for f, w, v in F(gpu) if f == 1); cfg = next(v for f, w, v in F(hwc) if f == 2)
        dbg = next(v for f, w, v in F(cfg) if f == 14)
        vals = {}
        for f, w, v in F(dbg):
            vals.setdefault(f, []).append(v.hex() if isinstance(v, bytes) else v)
        return {k: vals.get(k) for k in (208, 258, 435)}
    V2_TYPES = 'FUSION,CUBLAS,CUBLASLT,CUSTOM_CALL,CUDNN,DYNAMIC_SLICE_FUSION,CONDITIONAL'
    variants = [('OFF', {}), ('V1', {'xla_gpu_graph_min_graph_size': 1}), ('V2', {'xla_gpu_enable_command_buffer': V2_TYPES})]
    rec['variants'] = {}
    for vname, extra in variants:
        t = time.monotonic()
        opts = {'xla_gpu_autotune_level': 0, 'xla_gpu_experimental_aot_compiled_thunks': True, **extra}
        try:
            compiled = lowered.compile(compiler_options=opts)
        except Exception as exc:
            rec['variants'][vname] = {'options': {k: str(v) for k, v in opts.items()}, 'error': repr(exc)}; save(); continue
        v = {'options': {k: str(v) for k, v in opts.items()}, 'compile_s': time.monotonic() - t}
        executable = compiled.runtime_executable()
        hlo = executable.hlo_modules()[0].to_string()
        v['optimized_hlo_sha256'] = hashlib.sha256(hlo.encode()).hexdigest()
        ma = compiled.memory_analysis()
        v['memory'] = {k: getattr(ma, k, None) for k in ('argument_size_in_bytes', 'output_size_in_bytes', 'alias_size_in_bytes', 'temp_size_in_bytes')}
        blob = executable.serialize()
        v['blob_format'] = ax._gpu_blob_format(blob)
        try:
            v['records_sha256'] = {str(i): hashlib.sha256(gate._record(blob, i, ax, asm)).hexdigest() for i in (0, 1, 2)}
        except Exception as exc:
            v['records_error'] = repr(exc)
        elfs = list(gate._elfs(blob))
        v['elf_count'] = len(elfs); v['elf_concat_sha256'] = hashlib.sha256(b''.join(elfs)).hexdigest()
        try:
            v['debug_208_258_435'] = dbg_fields(blob)
        except Exception as exc:
            v['debug_error'] = repr(exc)
        (a.out / f'{vname}.xlaexec').write_bytes(blob)
        rec['variants'][vname] = v
        rec['phase'] = 'compiled_' + vname; save()
        del compiled, executable, hlo, blob, elfs
    ref = rec['variants'].get('OFF', {})
    rec['identity_vs_OFF'] = {n: {k: rec['variants'][n].get(k) == ref.get(k) for k in ('optimized_hlo_sha256', 'records_sha256', 'elf_concat_sha256', 'memory')}
                              for n in rec['variants'] if n != 'OFF'}
    rec['phase'] = 'done'
    rc = 0
except BaseException as exc:
    rec['exception'] = repr(exc)
    rec['traceback'] = traceback.format_exc()
finally:
    import shutil
    for d in ('unused_output', 'unused_proof', 'tmp', 'triton', 'cuda'):
        if (a.out / d).exists():
            shutil.rmtree(a.out / d)
    save()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(rc)
