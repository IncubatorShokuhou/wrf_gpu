"""Compile the operational NTU mp56 / Jensen mp55 Pallas kernel on CPU (SM120 target).

Isolated per-kernel prescreen (E137/E157 recipe): trace the production
full_column entry with the operational adapter outputs at 44 levels, lower to
TTIR, run the normal TTIR pipeline, then TTGIR/LLIR/PTX/cubin with per-stage
timing, and scan the cubin (scripts/e41/cubin_scan.py). No CUDA device is used.
Not an actual GPU/E41/forecast result.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trace(scheme, nz):
    import jax
    import jax.numpy as jnp
    s = jax.ShapeDtypeStruct((1, nz), jnp.float32)
    col = jax.ShapeDtypeStruct((1,), jnp.float32)
    if scheme == 'ntu':
        from gpuwrf.kernels.phys_ntu import full_column
        from gpuwrf.physics.ntu.adapter import OUTPUT_NAMES
        from gpuwrf.physics.ntu.column import PROGNOSTICS
        fields = {n: s for n in ('th', 'p', 'pii', 'dz', *PROGNOSTICS)}
        fields['w'] = jax.ShapeDtypeStruct((1, nz + 1), jnp.float32)
        fields.update({n: col for n in ('rainnc', 'snownc', 'grapnc', 'hailnc')})
        fn = lambda x, d: full_column(x, d, ccnty=2, output_names=OUTPUT_NAMES)
        args = (fields, jax.ShapeDtypeStruct((), jnp.float32))
    else:
        from gpuwrf.kernels.phys_jensen import full_column
        from gpuwrf.physics.jensen_ishmael.registry import output_mapping
        from gpuwrf.contracts.ice_diagnostics import ice_diagnostic_mapping
        mapping = output_mapping()
        names = [n for n in mapping if n not in ('effc', 'effi')] + ['th', 'p', 'dz']
        fields = {n: s for n in names}
        mapping.update(ice_diagnostic_mapping(55))
        outs = (*mapping, 'th')
        fn = lambda x, d, i: full_column(x, d, i, output_names=outs)
        args = (fields, jax.ShapeDtypeStruct((), jnp.float32), jax.ShapeDtypeStruct((), jnp.int32))
    with jax.enable_x64(True):
        program = jax.make_jaxpr(fn)(*args).jaxpr
    return next(e.params for e in program.eqns if e.primitive.name == 'pallas_call')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scheme', choices=('ntu', 'jensen'), required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--nz', type=int, default=44)
    p.add_argument('--warps', type=int, default=None)
    p.add_argument('--stop', choices=('ttir', 'ttgir', 'all'), default='all')
    a = p.parse_args()
    assert os.environ['JAX_PLATFORMS'] == 'cpu'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    import gpuwrf  # noqa: F401
    import jax
    from jax._src.pallas.triton import lowering
    from jax.experimental import pallas as pl
    import gpuwrf.backends.pallas as facade
    assert jax.devices()[0].platform == 'cpu'
    root = Path(__file__).resolve().parents[2]
    facade.cuda_native = lambda: True
    facade.pallas_call = pl.pallas_call
    arch = 120
    a.out.mkdir(parents=True, exist_ok=False)
    if (root / '.git').exists():
        source = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.run(['git', '-C', str(root), 'status', '--porcelain', 'src'],
                               capture_output=True, text=True).stdout
    else:  # immutable lane snapshot (tools/mksnap.sh): HEAD + code-only srcsha
        source = Path(str(root) + '.head').read_text().strip()
        dirty = 'snapshot srcsha ' + Path(str(root) + '.srcsha').read_text().strip()
    report = {'source': source, 'src_dirty': dirty, 'snapshot': str(root), 'scheme': a.scheme, 'nz': a.nz,
              'scope': 'CPU isolated native kernel compile (SM120 target); not actual GPU/E41/forecast',
              'arch': arch, 'stages': {}}

    def save():
        (a.out / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')

    started = time.monotonic()
    params = trace(a.scheme, a.nz)
    report['trace_s'] = time.monotonic() - started
    warps = a.warps or 4  # both production kernels use CompilerParams(num_warps=4)
    report['num_warps'] = warps
    report['compiler_params'] = str(params['compiler_params'])
    t = time.monotonic()
    module = lowering.lower_jaxpr_to_triton_module(params['jaxpr'], params['grid_mapping'], 'cuda', arch)
    raw = str(module.module)
    report['lower_s'] = time.monotonic() - t
    ttir = a.out / 'native.ttir'
    ttir.write_text(raw)
    report.update(ttir_bytes=len(raw), ttir_lines=raw.count('\n'), ttir_sha256=sha(ttir),
                  tt_load=raw.count('tt.load'), tt_store=raw.count('tt.store'),
                  inline_asm=raw.count('elementwise_inline_asm'), scf_while=raw.count('scf.while'),
                  scf_for=raw.count('scf.for'), scf_if=raw.count('scf.if'), f64_tokens=raw.count('f64'))
    save()
    print('LOWER', report['trace_s'], report['lower_s'], report['ttir_bytes'], 'loads', report['tt_load'],
          'stores', report['tt_store'], flush=True)
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import CUDABackend
    from triton.compiler.compiler import IRSource, make_backend
    from triton._C.libtriton import ir as triton_ir
    target = GPUTarget('cuda', arch, 32)
    backend = make_backend(target)
    context = triton_ir.context()
    source = IRSource(str(ttir), context, backend)
    source.module.context = context
    options = backend.parse_options({'num_warps': warps, 'enable_fp_fusion': True})
    t = time.monotonic()
    optimized = backend.make_ttir(source.module, {}, options, arch)
    report['stages']['ttir'] = time.monotonic() - t
    opt_text = str(optimized)
    ottir = a.out / 'optimized.ttir'
    ottir.write_text(opt_text)
    report.update(optimized_ttir_bytes=len(opt_text), optimized_ttir_sha256=sha(ottir),
                  opt_tt_load=opt_text.count('tt.load'), opt_tt_store=opt_text.count('tt.store'))
    save()
    print('DONE normalTTIR', report['stages']['ttir'], len(opt_text), 'loads', report['opt_tt_load'], flush=True)
    if a.stop == 'ttir':
        os._exit(0)
    original = CUDABackend.add_stages

    def instrument(self, stages, options, language):
        original(self, stages, options, language)
        for name, fn in list(stages.items()):
            def timed(src, metadata, fn=fn, name=name):
                t0 = time.monotonic()
                print('START', name, flush=True)
                result = fn(src, metadata)
                report['stages'][name] = time.monotonic() - t0
                save()
                print('DONE', name, report['stages'][name], flush=True)
                if name == 'ttgir' and a.stop == 'ttgir':
                    os._exit(0)
                return result
            stages[name] = timed

    CUDABackend.add_stages = instrument  # timing only
    t = time.monotonic()
    kernel = triton.compile(str(ottir), target=target, options={'num_warps': warps, 'enable_fp_fusion': True})
    report['compile_s'] = time.monotonic() - t
    cubin = a.out / 'native.cubin'
    cubin.write_bytes(kernel.asm['cubin'])
    (a.out / 'native.ptx').write_text(kernel.asm['ptx'])
    report.update(status='COMPILED', cubin_sha256=sha(cubin), cubin_bytes=cubin.stat().st_size,
                  shared_bytes=kernel.metadata.shared, n_regs=getattr(kernel, 'n_regs', None),
                  n_spills=getattr(kernel, 'n_spills', None),
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    spec = importlib.util.spec_from_file_location('native_scan', root / 'scripts/e41/cubin_scan.py')
    scan = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scan)
    report['native_resources'] = scan.scan(cubin, '/usr/local/cuda/bin/cuobjdump')
    save()
    print('COMPILED', report['compile_s'], report['native_resources'], flush=True)
    os._exit(0)


if __name__ == '__main__':
    main()
