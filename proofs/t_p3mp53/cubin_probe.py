"""Compile the operational MP53 Pallas kernel on CPU with a captured target."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--warps', type=int, choices=(2, 4), default=2)
    a = p.parse_args()
    assert os.environ['JAX_PLATFORMS'] == 'cpu'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    import gpuwrf  # noqa: F401
    import jax
    import jax.numpy as jnp
    from jax._src.pallas.triton import lowering
    from jax.experimental import pallas as pl
    import gpuwrf.backends.pallas as facade
    from gpuwrf.kernels.phys_p3 import full_column
    from gpuwrf.physics.p3.adapter import output_mapping
    assert jax.devices()[0].platform == 'cpu'
    root = Path(__file__).resolve().parents[2]
    assert subprocess.run(['git', '-C', str(root), 'symbolic-ref', '--quiet', 'HEAD'],
                          capture_output=True).returncode != 0
    facade.cuda_native = lambda: True
    facade.pallas_call = pl.pallas_call
    target = a.target.read_text()
    capability = re.search(r'cuda_compute_capability\s*\{([^}]+)\}', target)[1]
    major = re.search(r'major:\s*(\d+)', capability)
    minor = re.search(r'minor:\s*(\d+)', capability)
    arch = int(major[1]) * 10 + (int(minor[1]) if minor else 0)
    assert arch == 120
    a.out.mkdir(parents=True, exist_ok=False)
    names = ('qv', 'qc', 'qr', 'qi', 'ni', 'nr', 'nc', 'qir', 'bir',
             'th_old', 'qv_old', 'th', 'p', 'dz', 'qzi')
    fields = {n: jax.ShapeDtypeStruct((1, 44), jnp.float32) for n in names}
    outputs = (*output_mapping(53), 'th', 'rainncv', 'snowncv', 'ze')
    report = {'source': subprocess.check_output(
        ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
        'scope': 'CPU native operational MP53 compile; not actual GPU/E41/forecast',
        'target_sha256': sha(a.target), 'arch': arch, 'num_warps': a.warps,
        'inputs': names, 'outputs': outputs, 'stages': {}}

    def save():
        (a.out / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')

    started = time.monotonic()
    with jax.enable_x64(True):
        program = jax.make_jaxpr(lambda x, d, i: full_column(
            x, d, i, mp=53, output_names=outputs))(
                fields, jax.ShapeDtypeStruct((), jnp.float32),
                jax.ShapeDtypeStruct((), jnp.int32)).jaxpr
    params = next(e.params for e in program.eqns if e.primitive.name == 'pallas_call')
    module = lowering.lower_jaxpr_to_triton_module(
        params['jaxpr'], params['grid_mapping'], 'cuda', arch)
    ttir = a.out / 'native.ttir'
    raw = str(module.module)
    for line in raw.splitlines():
        if 'elementwise_inline_asm' in line and 'ld.global' in line:
            assert 'xf64>' not in line, 'P3 inline load operands must use typed REAL4 fill'
    ttir.write_text(raw)
    report.update(lower_s=time.monotonic() - started,
                  ttir_bytes=ttir.stat().st_size, ttir_sha256=sha(ttir),
                  compiler_params=str(params['compiler_params']))
    save()
    print('LOWER', report['lower_s'], report['ttir_bytes'], flush=True)
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import CUDABackend
    from triton.compiler.compiler import IRSource, make_backend
    from triton._C.libtriton import ir as triton_ir
    # File-input compilation starts after make_ttir. Run the normal TTIR
    # pipeline explicitly, as JAX does, before the TTGIR/backend stages.
    target_device = GPUTarget('cuda', arch, 32)
    backend = make_backend(target_device)
    context = triton_ir.context()
    source = IRSource(str(ttir), context, backend)
    source.module.context = context
    options = backend.parse_options({'num_warps': a.warps, 'enable_fp_fusion': True})
    started = time.monotonic()
    optimized = backend.make_ttir(source.module, {}, options, arch)
    report['stages']['ttir'] = time.monotonic() - started
    ttir = a.out / 'optimized.ttir'
    ttir.write_text(str(optimized))
    report['optimized_ttir_sha256'] = sha(ttir)
    save()
    print('DONE normalTTIR', report['stages']['ttir'], flush=True)
    original = CUDABackend.add_stages

    def instrument(self, stages, options, language):
        original(self, stages, options, language)
        for name, fn in list(stages.items()):
            def timed(src, metadata, fn=fn, name=name):
                t = time.monotonic()
                print('START', name, flush=True)
                result = fn(src, metadata)
                report['stages'][name] = time.monotonic() - t
                save()
                print('DONE', name, report['stages'][name], flush=True)
                return result
            stages[name] = timed

    CUDABackend.add_stages = instrument  # Timing only; no pass modification.
    started = time.monotonic()
    kernel = triton.compile(str(ttir), target=GPUTarget('cuda', arch, 32),
                            options={'num_warps': a.warps, 'enable_fp_fusion': True})
    report['compile_s'] = time.monotonic() - started
    cubin = a.out / 'native.cubin'
    cubin.write_bytes(kernel.asm['cubin'])
    (a.out / 'native.ptx').write_text(kernel.asm['ptx'])
    report.update(status='COMPILED', cubin_sha256=sha(cubin),
                  cubin_bytes=cubin.stat().st_size,
                  shared_bytes=kernel.metadata.shared,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    spec = importlib.util.spec_from_file_location('native_scan', root / 'scripts/e41/cubin_scan.py')
    scan = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scan)
    report['native_resources'] = scan.scan(cubin, '/usr/local/cuda/bin/cuobjdump')
    save()
    print('COMPILED', report['compile_s'], report['native_resources'], flush=True)
    os._exit(0)  # No CUDA module/function is ever instantiated.


if __name__ == '__main__':
    main()
