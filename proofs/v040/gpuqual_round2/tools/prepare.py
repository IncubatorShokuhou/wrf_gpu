"""Freeze matched original-WRF inputs without modifying reference directories."""
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path('<USER_HOME>/wrf_gpu2_lanes/v-run')
SOURCE = ROOT / 'snapshot'
OLD = Path('<USER_HOME>/wrf_gpu2_lanes/s1-gpuqual')
NAMES = ('release_default', 'mp16', 'mp24', 'mp26', 'mp40', 'bl9',
         'fdda1', 'mp18', 'mp18_cu5', 'mp28')
assert os.environ['JAX_PLATFORMS'] == 'cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

revision = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
assert not subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain'], text=True).strip()
configs = []
for name in NAMES:
    case = ROOT / name
    case.mkdir()
    inp = case / 'input'
    inp.mkdir()
    if name == 'release_default':
        ref = Path('<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu')
        original_input = ref
    elif name == 'fdda1':
        ref = Path('<USER_HOME>/wrf_gpu2_lanes/t-fdda/TF02/full24')
        original_input = ref
    else:
        ref = OLD / name / 'cpu'
        original_input = OLD / name / 'input'
    assert len(list(ref.glob('wrfout_d01_*'))) == 25, ref
    (case / 'cpu').symlink_to(ref, target_is_directory=True)
    files = ['namelist.input', 'wrfinput_d01', 'wrfbdy_d01']
    if name == 'fdda1':
        files.append('wrffdda_d01')
    hashes = {}
    for file in files:
        origin = original_input / file
        assert origin.is_file(), origin
        (inp / file).symlink_to(origin.resolve())
        actual = sha(inp / file)
        expected = sha(ref / file)
        assert actual == expected, (name, file, actual, expected)
        hashes[file] = {'CPU_sha256': expected, 'GPU_sha256': actual, 'equal': True,
                        'original_input': str(origin.resolve())}
    mp_re = '1'  # release default; activation predicate excludes mp16/26/40
    config = {'config': name, 'source_rev': revision, 'source': str(SOURCE),
              'reference': str(ref.resolve()), 'matched_inputs': hashes,
              'hours': 24, 'dt_s': 18, 'history_minutes': 60,
              'GPUWRF_RRTMG_MP_RE': mp_re,
              'reference_reuse': 'Original CPU-WRF; unchanged matched namelist and inputs; '
                  + ('t-fdda corrected FieldType=104 analysis ingestion reference.' if name == 'fdda1'
                     else 'No new CPU model execution required.')}
    (case / 'inputs.json').write_text(json.dumps(config, indent=2) + '\n')
    configs.append(config)
tools = Path(__file__).resolve().parent
for name in ('gpu_cli.py', 'batch.py', 'consume.py', 'vrun_queue.py'):
    shutil.copyfile(tools / name, ROOT / 'tools' / name)
shutil.copyfile(SOURCE / 'proofs/v040/gpuqual/tools/score.py', ROOT / 'tools/score.py')
(ROOT / 'PLAN.json').write_text(json.dumps({'source_rev': revision, 'configs': configs,
    'groups': [list(NAMES[i:i+2]) for i in range(0, len(NAMES), 2)],
    'tools_sha256': {p.name: sha(p) for p in (ROOT / 'tools').glob('*.py')},
    'GPU_window_UTC': '11:30-19:00, hard end 19:15',
    'GPU_arm_timeout_s': 1800, 'CPU_core': 9,
    'timing_scope': 'cold qualification with concurrent Swiss runs and CPU scoring; not S1'}, indent=2) + '\n')
print('Prepared', revision, len(configs), 'matched references, 25 frames each')
