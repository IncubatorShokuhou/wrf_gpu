"""Score the single paid c26 arm with the sealed, unchanged D6 consumer."""
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
    p.add_argument('--tools', type=Path, required=True)
    a = p.parse_args()
    assert os.environ['JAX_PLATFORMS'] == 'cpu'
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    assert a.case.resolve().is_relative_to(Path('<USER_HOME>/wrf_gpu2_lanes/t-qnsecompile'))
    gpu = json.loads((a.case / 'gpu_run.json').read_text())
    cpu = json.loads((a.case / 'cpu_run.json').read_text())
    assert gpu['platform'] == 'gpu' and cpu['status'] == 'COMPLETE'
    assert gpu['actual_input_sha256'] == cpu['actual_input_sha256']
    tools = json.loads((a.tools / 'TOOLS.json').read_text())['files']
    assert all(sha(a.tools / name) == digest for name, digest in tools.items())
    path = a.tools / 'proofs/v040/final_validation/score.py'
    spec = importlib.util.spec_from_file_location('frozen_score', path)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    import netCDF4
    from gpuwrf.io.netcdf_lock import Dataset
    netCDF4.set_chunk_cache(1 << 20, 521, .75)
    t = time.monotonic()
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

        # Identical equations/bounds, only bounded read-only IO lifetime.
        scorer.Dataset = readonly
        result = scorer.score(a.tools, a.case, 'd01')
    report = dict(scope='actual c26 QNSE GPU vs original CPU-WRF, frozen D6',
                  source=gpu['source_commit'], gpu_receipt_sha256=sha(a.case / 'gpu_run.json'),
                  cpu_receipt_sha256=sha(a.case / 'cpu_run.json'), tools=tools,
                  manifest_sha256=result['manifest_sha256'], wall_s=time.monotonic()-t,
                  verdict=result['verdict'], paired_frames=result['paired_frames'],
                  coverage_complete=result['coverage_complete'],
                  bounded_misses=len(result['failures']), nonfinite=len(result['nonfinite']),
                  missing=sorted({n for row in result['by_frame'] for n in row['missing_vs_cpu']}))
    (a.case / 'D6_SUMMARY.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
