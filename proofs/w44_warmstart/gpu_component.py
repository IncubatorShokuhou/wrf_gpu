"""Run the fixed WS11 original-WRF witness in the measured GPU process."""
import json
import os
from pathlib import Path
import runpy
import resource
import sys
import time
from launch_preflight import asserted_device,cpu_mini


def main():
    device,dry=asserted_device()
    output, oracle = map(Path, sys.argv[1:3])
    output.mkdir(parents=True, exist_ok=False)
    if dry:
        cpu_mini(oracle,output,'component')
        return 0
    import jax
    (output / 'PLATFORM.json').write_text(json.dumps({
        'platform': device.platform, 'device_kind': device.device_kind,
        'jax': jax.__version__, 'scope': 'measured process, not parent attestation',
        'flags': {k:v for k,v in os.environ.items() if k.startswith('GPUWRF_MYNN_')},
    }, indent=2)+'\n')
    probe = Path(__file__).with_name('compare_mynn_init.py')
    sys.argv = [str(probe), str(oracle), str(output / 'result.json')]
    start = time.monotonic()
    try:
        runpy.run_path(str(probe), run_name='__main__')
    finally:
        stats = device.memory_stats()
        (output / 'RESOURCES.json').write_text(json.dumps({
            'compile_execute_wall_s': time.monotonic()-start,
            'rss_peak_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'device_memory_stats': stats,
            'scope': '40-column init component; not forecast timing or full-model VRAM',
        }, indent=2)+'\n')


if __name__ == '__main__':
    main()
