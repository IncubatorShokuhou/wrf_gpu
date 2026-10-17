"""Score a private MP52 GPU attempt against the retained ZERO WRF reference."""
import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case', type=Path, required=True)
    p.add_argument('--zero', type=Path, required=True)
    p.add_argument('--tools', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    assert os.environ['JAX_PLATFORMS'] == 'cpu'
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    lane = Path('<USER_HOME>/wrf_gpu2_lanes/t-p3fix').resolve()
    assert a.out.resolve().is_relative_to(lane)
    assert a.case.resolve().is_relative_to(lane)
    zero = json.loads((a.zero / 'receipt.json').read_text())
    gpu = json.loads((a.case / 'gpu_run.json').read_text())
    assert zero['status'] == 'COMPLETE' and zero['frames'] == 25
    assert gpu['platform'] == 'gpu'
    assert zero['input_sha256'] == gpu['actual_input_sha256']
    sealed_tools = json.loads((a.tools / 'TOOLS.json').read_text())['files']
    assert all(sha(a.tools / name) == digest for name, digest in sealed_tools.items())
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'input').mkdir()
    with (a.out / 'input/namelist.input').open('xb') as stream:
        stream.write((a.case / 'input/namelist.input').read_bytes())
    for target, source in [('cpu_run.json', 'receipt.json'), ('CPU_OUTPUTS.json', 'outputs.json')]:
        with (a.out / target).open('xb') as stream:
            stream.write((a.zero / source).read_bytes())
    (a.out / 'cpu').symlink_to(a.zero.resolve())
    (a.out / 'gpu').symlink_to((a.case / 'gpu').resolve())
    path = a.tools / 'proofs/v040/final_validation/score.py'
    spec = importlib.util.spec_from_file_location('frozen_score', path)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    import netCDF4
    from gpuwrf.io.netcdf_lock import Dataset
    netCDF4.set_chunk_cache(1 << 20, 521, .75)
    start = time.monotonic()
    with ExitStack() as stack:
        handles = {}

        class Held:
            def __init__(self, ds):
                self.ds = ds

            def __enter__(self):
                return self.ds

            def __exit__(self, *exc):
                return False

        def readonly(path, mode='r', *args, **kwargs):
            assert mode == 'r' and not args and not kwargs
            key = str(path)
            if key not in handles:
                handles[key] = stack.enter_context(Dataset(key, 'r'))
            return Held(handles[key])

        # IO lifetime only: the sealed scientific scorer and bounds are exact.
        scorer.Dataset = readonly
        result = scorer.score(a.tools, a.out, 'd01')
    report = {'scope': 'actual MP52 GPU vs original ZERO CPU-WRF, frozen D6',
              'source': gpu['source_commit'], 'gpu_receipt_sha256': sha(a.case / 'gpu_run.json'),
              'zero_receipt_sha256': sha(a.zero / 'receipt.json'),
              'zero_executable_sha256': zero['source_executable_sha256'],
              'manifest_sha256': result['manifest_sha256'],
              'score_source_sha256': sha(path), 'tool_sha256': sealed_tools,
              'wall_s': time.monotonic() - start, 'verdict': result['verdict'],
              'paired_frames': result['paired_frames'],
              'bounded_misses': len(result['failures']), 'nonfinite': len(result['nonfinite']),
              'missing': sorted({n for row in result['by_frame'] for n in row['missing_vs_cpu']})}
    (a.out / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
