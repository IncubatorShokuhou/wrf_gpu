"""Compile a QNSE column on CPU against the captured c26 GPU target.

This measures compiler readiness, not GPU performance or WRF fidelity.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import re
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tree', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--scheme', choices=('sfc', 'pbl', 'mf'), required=True)
    p.add_argument('--levels', type=int, default=44)
    p.add_argument('--target', type=Path, default=Path(
        '<USER_HOME>/wrf_gpu2_lanes/v-prep/c26/target_config.pbtxt'))
    a = p.parse_args()
    assert os.environ['JAX_PLATFORMS'] == 'cpu'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    assert subprocess.run(['git', '-C', str(a.tree), 'symbolic-ref', '--quiet', 'HEAD'],
                          capture_output=True).returncode != 0
    sys.path.insert(0, str(a.tree / 'src'))
    import gpuwrf  # noqa: F401
    import jax
    import jax.numpy as jnp
    from jax._src.pallas.triton import lowering
    from jax.experimental import pallas
    import gpuwrf.backends.pallas as facade
    from gpuwrf.kernels.phys_qnse_columns import qnse_columns
    assert jax.devices()[0].platform == 'cpu'
    facade.cuda_native = lambda: True
    facade.pallas_call = pallas.pallas_call
    capability = re.search(r'cuda_compute_capability\s*\{([^}]+)\}', a.target.read_text())[1]
    major = int(re.search(r'major:\s*(\d+)', capability)[1])
    minor = re.search(r'minor:\s*(\d+)', capability)
    arch = 10 * major + (int(minor[1]) if minor else 0)
    assert arch == 120
    profiles = {
        'sfc': 'u v temperature theta qv qc p_mid dz q2',
        'pbl': 'u v temperature theta qv qc p_mid p_int exner dz tke EXCH_H EXCH_M RTHBLTEN RQVBLTEN',
        'mf': 'theta qv p_mid exner dz rho tke u v rc_mf',
    }[a.scheme].split()
    scalars = {
        'sfc': 'tsk xland z0base psfc znt ustar mavail qsfc ht dt itimestep hfx qfx scm_force_flux thz0 qz0 uz0 vz0 akhs akms',
        'pbl': 'xland ustar hfx elflx qsfc corf snow sice chklowq dt stepbl ht akhs akms thz0 qz0 uz0 vz0 ct',
        'mf': 'hfx qfx dt stepbl',
    }[a.scheme].split()
    fields = {n: jax.ShapeDtypeStruct((a.levels + (n == 'p_int'), 1764), jnp.float32)
              for n in profiles}
    fields.update({n: jax.ShapeDtypeStruct((1764,), jnp.float32) for n in scalars})
    requested = ('RTHBLTEN', 'RQVBLTEN', 'rc_mf') if a.scheme == 'mf' else None
    a.out.mkdir(parents=True, exist_ok=False)
    report = {
        'source': subprocess.check_output(['git', '-C', str(a.tree), 'rev-parse', 'HEAD'], text=True).strip(),
        'scope': 'CPU Triton compile; operational QNSE column inputs; not GPU/E41/forecast',
        'scheme': a.scheme, 'levels': a.levels, 'ncol': 1764, 'block': 64,
        'requested': requested, 'stages': {}, 'platform': 'cpu',
        'target_sha256': hashlib.sha256(a.target.read_bytes()).hexdigest(), 'arch': arch,
    }

    def save():
        (a.out / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')

    t = time.monotonic()
    program = jax.make_jaxpr(lambda x: qnse_columns(
        a.scheme, x, interpret=False, requested=requested))(fields).jaxpr
    params = next(e.params for e in program.eqns if e.primitive.name == 'pallas_call')
    module = lowering.lower_jaxpr_to_triton_module(params['jaxpr'], params['grid_mapping'], 'cuda', arch)
    ttir = a.out / 'native.ttir'
    ttir.write_text(str(module.module))
    report.update(lower_s=time.monotonic()-t, ttir_bytes=ttir.stat().st_size,
                  ttir_sha256=hashlib.sha256(ttir.read_bytes()).hexdigest())
    save()
    print('LOWER', report['scheme'], report['lower_s'], report['ttir_bytes'], flush=True)
    import triton
    report.update(triton_version=triton.__version__, triton_path=triton.__file__)
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import CUDABackend
    original = CUDABackend.add_stages

    def instrument(self, stages, options, language):
        original(self, stages, options, language)
        for name, fn in list(stages.items()):
            def timed(src, metadata, fn=fn, name=name):
                t = time.monotonic()
                print('START', name, flush=True)
                result = fn(src, metadata)
                report['stages'][name] = time.monotonic()-t
                save()
                print('DONE', name, report['stages'][name], flush=True)
                return result
            stages[name] = timed

    CUDABackend.add_stages = instrument  # Timing only, no pass changes.
    t = time.monotonic()
    kernel = triton.compile(str(ttir), target=GPUTarget('cuda', arch, 32),
                            options={'num_warps': 4, 'enable_fp_fusion': True})
    report['compile_s'] = time.monotonic()-t
    cubin = a.out / 'native.cubin'
    cubin.write_bytes(kernel.asm['cubin'])
    (a.out / 'native.ptx').write_text(kernel.asm['ptx'])
    report.update(status='COMPILED', cubin_sha256=hashlib.sha256(cubin.read_bytes()).hexdigest(),
                  cubin_bytes=cubin.stat().st_size, shared_bytes=kernel.metadata.shared,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    spec = importlib.util.spec_from_file_location('native_scan', a.tree / 'scripts/e41/cubin_scan.py')
    scan = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scan)
    report['native_resources'] = scan.scan(cubin, '/usr/local/cuda/bin/cuobjdump')
    save()
    print('COMPILED', report['compile_s'], report['native_resources'], flush=True)
    os._exit(0)  # Never instantiate CUDA modules/functions.


if __name__ == '__main__':
    main()
