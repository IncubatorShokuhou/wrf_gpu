"""Score a short GPU arm against a sealed prefix of a full original CPU run.

The CPU reference is never modified. The derived receipt explicitly describes
a prefix view, not a new CPU forecast. The frozen scorer and D6 bounds remain
unchanged, and this result cannot qualify the full 24-hour release gate.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('source',type=Path)
    p.add_argument('reference',type=Path)
    p.add_argument('arm',type=Path)
    p.add_argument('view',type=Path)
    p.add_argument('--hours',type=int,default=2)
    p.add_argument('--scorer',type=Path,required=True)
    a = p.parse_args()
    assert os.environ.get('JAX_PLATFORMS') == 'cpu'
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    receipt = json.loads((a.reference/'cpu_run.json').read_text())
    full = a.reference/'CPU_OUTPUTS.json'
    assert sha(full) == receipt['output_manifest_sha256']
    rows = sorted((r for r in json.loads(full.read_text()) if r['file'].startswith('wrfout_d01_')),
                  key=lambda r:r['times'][0])[:a.hours+1]
    assert len(rows)==a.hours+1
    a.view.mkdir(parents=True,exist_ok=True)
    (a.view/'cpu').mkdir(exist_ok=True)
    for row in rows:
        original = a.reference/'cpu'/row['file']
        assert sha(original)==row['sha256']
        (a.view/'cpu'/row['file']).symlink_to(original.resolve())
    (a.view/'input').symlink_to((a.reference/'input').resolve())
    (a.view/'gpu').symlink_to((a.arm/'gpu').resolve())
    subset=a.view/'CPU_OUTPUTS.json'
    subset.write_text(json.dumps(rows,indent=2)+'\n')
    derived = dict(output_manifest_sha256=sha(subset),
        scope=f'Prefix view of completed original 24h CPU-WRF run, leads 0..{a.hours}; no new CPU run',
        original_cpu_run_path=str(a.reference/'cpu_run.json'),
        original_cpu_run_sha256=sha(a.reference/'cpu_run.json'),
        original_full_manifest_path=str(full),original_full_manifest_sha256=sha(full),
        frozen_scorer_sha256=sha(a.scorer),source=str(a.source),gpu_arm=str(a.arm))
    (a.view/'cpu_run.json').write_text(json.dumps(derived,indent=2)+'\n')
    sys.path.insert(0,str(a.source/'src'))
    spec=importlib.util.spec_from_file_location('frozen_prefix_scorer',a.scorer)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result=module.score(a.source.resolve(),a.view.resolve(),'d01',hours=a.hours)
    print(json.dumps(dict(verdict=result['verdict'],frames=result['paired_frames'],
        scope=derived['scope'],failures=len(result['failures']),nonfinite=len(result['nonfinite']))))
    raise SystemExit(0 if result['verdict']=='PASS' else 1)


if __name__=='__main__':
    main()
