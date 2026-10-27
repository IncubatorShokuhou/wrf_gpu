"""Measured-process GPU assertion and compile accounting around the real CLI.

The compiler observer forwards the original callable/arguments unchanged. It
counts actual JAX backend compilations, not tracing or persistent-cache loads.
"""
import functools
import json
import os
from pathlib import Path
import resource
import sys
import time
from launch_preflight import asserted_device,cpu_mini


def main():
    case, output = map(Path, sys.argv[1:3])
    phase = sys.argv[3]
    assert phase in ('cold', 'warm')
    device,dry=asserted_device()
    output.mkdir(exist_ok=False, parents=True)
    if dry:
        cpu_mini(Path('<USER_HOME>/wrf_gpu2_lanes/w44-warmstart/WS11_mynn'),output,'forecast')
        return 0
    import jax
    from jax._src import compiler
    (output/'PLATFORM.json').write_text(json.dumps({'platform':device.platform,'device_kind':device.device_kind})+'\n')
    original = compiler.backend_compile_and_load
    compiles = []
    @functools.wraps(original)
    def observe(backend, module, *args, **kwargs):
        started = time.monotonic()
        row = {'platform': backend.platform, 'module': str(module.operation.attributes['sym_name'])}
        compiles.append(row)
        try:
            return original(backend, module, *args, **kwargs)
        finally:
            row['seconds'] = time.monotonic()-started
    compiler.backend_compile_and_load = observe
    rc = 2
    start = time.monotonic()
    try:
        from gpuwrf.cli import main as cli_main
        argv = ['run', '--input-dir', str(case), '--namelist', str(case/'namelist.input'),
                '--output-dir', str(output/'wrfout'), '--proof-dir', str(output/'proofs'),
                '--scratch-dir', str(output/'scratch'), '--max-dom', '3', '--hours', '3', '--emit-initial-history']
        rc = cli_main(argv)
    finally:
        compiler.backend_compile_and_load = original
        report = {'phase': phase, 'rc': rc, 'platform': device.platform, 'device_kind': device.device_kind,
            'compile_count': len(compiles), 'gpu_compile_count': sum(r['platform'] in ('gpu','cuda','rocm') for r in compiles),
            'compiles': compiles, 'runner_wall_s': time.monotonic()-start,
            'rss_peak_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'device_memory_stats': device.memory_stats(),
            'flags': {k:v for k,v in os.environ.items() if k.startswith(('GPUWRF_', 'JAX_', 'XLA_'))},
            'compile_observer': 'JAX compiler.backend_compile_and_load forwarding wrapper; no graph transformation',
            'timing_scope': 'runner excludes interpreter import; supervisor elapsed includes process start through last file'}
        (output/'MEASUREMENT.json').write_text(json.dumps(report,indent=2)+'\n')
    if rc != 0:
        return rc
    proof = json.loads((output/'proofs/nested_pipeline_run.json').read_text())
    assert proof['hours'] == 3 and proof['max_dom'] == 3
    assert proof['all_domains_finite'] and proof['all_outputs_present']
    assert len(proof['wrfout_files']) == 12
    if phase == 'warm':
        assert report['compile_count'] == 0, report['compiles']
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
