"""Bind paid fix-lane evidence at its actual source; preserve every predecessor.

This does no forecast, compilation or numerical rescore. Current-main program
identity and uncomputed diagnostics remain explicit independent obligations.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

assert os.environ['JAX_PLATFORMS'] == 'cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
ROOT = Path('<USER_HOME>/wrf_gpu2_lanes/v-prep')
PROOFS = Path(__file__).resolve().parent
MAIN = Path('<USER_HOME>/src/wrf_gpu2/.agent/kernel')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def sealed(path):
    return {'path': str(path), 'sha256': sha(path)}


ident = sys.argv[1]
own = ROOT / ident
if ident == 'c02':
    producer = Path('<DATA_ROOT>/wrf_gpu2/scratch/t-c02nan/TCN23_c02')
    runpath, scorepath = producer / 'receipt.json', producer / 'D6_d01.json'
    pins, e41 = producer / 'e41/PIN_MAP.json', producer / 'e41/wave_gate.json'
elif ident == 'c26':
    producer = Path('<USER_HOME>/wrf_gpu2_lanes/t-qnsecompile/QC05/c26')
    runpath, scorepath = producer / 'gpu_run.json', producer / 'D6_d01.json'
    paid = Path('<USER_HOME>/wrf_gpu2_lanes/t-qnsecompile/QC06/paid')
    pins, e41 = paid / 'PIN.json', paid / 'e41/wave_gate.json'
elif ident == 'c23':
    producer = Path('<USER_HOME>/wrf_gpu2_lanes/t-p3fix/TP56/c23')
    runpath = producer / 'gpu_run.json'
    scorepath = Path('<USER_HOME>/wrf_gpu2_lanes/t-p3fix/TP61/D6_d01.json')
    paid = Path('<USER_HOME>/wrf_gpu2_lanes/t-p3fix/TP60')
    pins, e41 = paid / 'PIN.json', paid / 'e41/wave_gate.json'
elif ident == 'c24':
    # Manager: reuse the one reviewed MP53 arm only after its fix is merged.
    subprocess.run(['git', 'merge-base', '--is-ancestor',
        'f552e5fb2d1cca53ce513cbd2ef66b1c69eb3ac2', 'main'],
        cwd=MAIN.parents[1], check=True)
    producer = Path('<USER_HOME>/wrf_gpu2_lanes/t-p3mp53/MP5312/c24')
    runpath = producer / 'gpu_run.json'
    scorepath = Path('<USER_HOME>/wrf_gpu2_lanes/t-p3mp53/MP5314/D6_d01.json')
    paid = Path('<USER_HOME>/wrf_gpu2_lanes/t-p3mp53/MP5313')
    pins, e41 = paid / 'PIN.json', paid / 'e41/wave_gate.json'
else:
    raise ValueError(ident)
run, score = read(runpath), read(scorepath)
assert run['platform'] == 'gpu' and run['rc'] == 0
assert score['paired_frames'] == 25 and not score['failures'] and not score['nonfinite']
assert score['manifest_sha256'] == '9df4fab8514523286f07b127eb8bf57a7545459c4448fef721ac44c0ecc351dc'
actual = run.get('actual_input_sha256', run.get('input_sha256'))
assert {p.name: sha(p) for p in (own / 'input').iterdir() if p.is_file()} == actual
assert {p.name: sha(p) for p in (producer / 'input').iterdir() if p.is_file()} == actual
reference = None
if ident in ('c23', 'c24'):
    index = read(Path('<USER_HOME>/wrf_gpu2_lanes/t-p3fix/ZERO_REFERENCE_INDEX.json'))['cases'][ident]
    assert index['input_sha256'] == actual
    zero_receipt = Path(index['receipt'])
    assert sha(zero_receipt) == index['receipt_sha256']
    assert index['source_executable_sha256'] == '9a67a93dd891115c095f64a95ce8f50e7ff0942ef1abf4185faf20bd7065742a'
    reference = dict(policy='Manager-authorized zero-init original CPU-WRF; raw original preserved', **index)
else:
    for name in ('cpu_run.json', 'CPU_OUTPUTS.json'):
        assert (own / name).read_bytes() == (producer / name).read_bytes(), name
    reference = dict(policy='BYTE-identical own original CPU-WRF run and complete output manifest',
                     cpu_run=sealed(own / 'cpu_run.json'), manifest=sealed(own / 'CPU_OUTPUTS.json'))
    if ident == 'c26':
        assert run['original_cpu_receipt_sha256'] == sha(own / 'cpu_run.json')
pin_data = read(pins)
pin_rows = pin_data['pins'] if isinstance(pin_data, dict) else pin_data
assert len(pin_rows) == 2
for pin in pin_rows:
    assert sha(Path(pin['pin'])) == pin['sha256'], pin['pin']
for frame in score['by_frame']:
    assert Path(frame['gpu_file']).resolve() == (producer / 'gpu' / Path(frame['gpu_file']).name).resolve()
    assert Path(frame['cpu_file']).is_file()
gpu_files = {p.name: sha(p) for p in sorted((producer / 'gpu').glob('wrfout*'))}
assert len(gpu_files) == 25
reason = 'before_paid_c24_mp53_reuse' if ident == 'c24' else 'before_paid_fix_reuse_0710'
subprocess.run(['python', str(PROOFS / 'archive_attempt.py'), reason, ident], check=True)
target = own / 'gpu'
if target.is_symlink():
    assert target.resolve().is_relative_to(Path('<DATA_ROOT>/wrf_gpu2/scratch/v-prep/arms') / ident)
    assert not list(target.iterdir())
    target.unlink()
assert not target.exists()
target.symlink_to(producer / 'gpu', target_is_directory=True)
shutil.copyfile(scorepath, own / 'D6_d01.json')
e41verdict = read(e41)['e41_gate']
receipt = dict(schema='v040-gpu-arm-reuse-v1', utc=datetime.now(timezone.utc).isoformat(),
    case=ident, producer_case=str(producer), producer_receipt=str(runpath),
    producer_receipt_sha256=sha(runpath), source_commit=run['source_commit'],
    actual_input_sha256=actual, matched_inputs_sha256=sha(own / 'inputs.json'),
    original_cpu_receipt_sha256=sha(own / 'cpu_run.json'), reference_of_record=reference,
    source_D6=sealed(scorepath), d6=score['verdict'], coverage_complete=score['coverage_complete'],
    paired_frames=25, bounded_misses=0, nonfinite=0, gpu_files=gpu_files,
    e41=e41verdict, external_e41=str(e41), external_e41_sha256=sha(e41),
    external_pins=str(pins), external_pins_sha256=sha(pins), all_actual_pin_SHA_verified=True,
    scope='Reuse exact paid GPU/input/CPU-reference evidence at its measured source. No later-main identity or new compiler realization inferred. Original attempts and raw original CPU references preserved.',
    current_main_identity='PENDING separate selected-program/source proof; no redundant forecast queued for c23/c26',
    F2='OPEN; producer counters retained, no justification inferred')
(own / 'GPU_REUSE.json').write_text(json.dumps(receipt, indent=2) + '\n')
if ident in ('c23', 'c24'):
    (own / 'REFERENCE_OF_RECORD.json').write_text(json.dumps(reference, indent=2) + '\n')
(PROOFS / f'{ident.upper()}_PAID_FIX_REUSE.json').write_text(json.dumps(receipt, indent=2) + '\n')
with (MAIN / 'EXPERIMENTS.tsv').open('a') as f:
    f.write(f"{receipt['utc']}\tv-prep\tGPT Sol\tVP172-{ident}\tReuse fix-lane exact paid forecast/input/reference instead of rerun\t[M]25paired/0bounded/0NF/D6{score['verdict']}/E41{e41verdict}/2actualPINsSHA\tKEEP measured-source evidence; later-main identity separate\tartifact={PROOFS}/{ident.upper()}_PAID_FIX_REUSE.json;cacheNONE;workCPU16NI19JAXcpuQUIETtimeout180;noGPU/norescore\n")
print(ident, score['verdict'], '25 frames', 'E41', e41verdict, 'paid source', run['source_commit'])
