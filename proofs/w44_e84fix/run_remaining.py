"""Serial CPU-only product/baseline continuation; one capped job at a time."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

LANE = Path('<USER_HOME>/wrf_gpu2_lanes/w44-e84fix')
SNAPSHOT = Path('<DATA_ROOT>/wrf_gpu2/scratch/w44-e84fix/snapshot')
BASELINE = Path('<DATA_ROOT>/wrf_gpu2/scratch/w44-e84fix/baseline')
TOOL = SNAPSHOT / 'proofs/w44_e84fix/native_gate.py'
FUSED_TOOL = Path(__file__).resolve().parent / 'native_gate.py'
TARGET = Path('<USER_HOME>/wrf_gpu2_lanes/w44-perf/target_config.pbtxt')
PROD = Path('<USER_HOME>/wrf_gpu2_lanes/w44-perf/prod_case')
WN3 = Path('<USER_HOME>/wrf_gpu2_lanes/o5-perf/mem/wn3_case')
assert os.environ['JAX_PLATFORMS'] == 'cpu'
assert subprocess.check_output(['git', '-C', str(BASELINE), 'rev-parse', 'HEAD:src'], text=True).strip() == '70fa8384672257d3457cf8a9863b34dac0a115b5'
tool_sha = hashlib.sha256(TOOL.read_bytes()).hexdigest()
while True:
    receipt = LANE / 'E8409_wn3_d01/receipt.json'
    if receipt.exists() and json.loads(receipt.read_text()).get('phase') == 'done':
        break
    time.sleep(10)
jobs = [
    ('E8410b_wn3_fused_d02', SNAPSHOT, WN3, 'fused_d02'),
    ('E8411_base_prod_d01', BASELINE, PROD, 'd01'),
    ('E8412_base_prod_d02', BASELINE, PROD, 'd02'),
    ('E8413_base_wn3_d01', BASELINE, WN3, 'd01'),
    ('E8414_base_wn3_fused_d02', BASELINE, WN3, 'fused_d02'),
]
for label, source, case, domain in jobs:
    selected_tool = FUSED_TOOL if domain.startswith('fused') else TOOL
    expected_tool_sha = hashlib.sha256(selected_tool.read_bytes()).hexdigest()
    assert hashlib.sha256(TOOL.read_bytes()).hexdigest() == tool_sha
    out = LANE / label
    if out.exists():
        rec = json.loads((out / 'receipt.json').read_text())
        assert rec['phase'] == 'done' and rec['harness_sha256'] == expected_tool_sha, (label, rec.get('phase'))
        print('REUSE DONE', label, flush=True)
        continue
    while any(Path(f).exists() for f in ('/tmp/wrf_gpu2_quiet', '/tmp/wrf_gpu2_gpu_window')):
        time.sleep(10)
    pattern = 'jit_fused_jit' if domain.startswith('fused') else 'jit__advance_chunk_fori'
    command = [str(SNAPSHOT / 'scripts/with_ram_slot.sh'), '--label', 'w44-e84fix-' + label,
        '--max', '12G', '--timeout', '3600', '--', 'taskset', '-c', '8,9,12', 'nice', '-n', '19',
        'timeout', '1800', 'bash', '-c',
        'if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; exec "$@"', 'e84-child',
        'env', 'JAX_PLATFORMS=cpu', 'CUDA_VISIBLE_DEVICES=', 'JAX_ENABLE_X64=true',
        'GPUWRF_JAX_CACHE=0', 'JAX_ENABLE_COMPILATION_CACHE=false',
        'XLA_FLAGS=--xla_dump_to=' + str(out / 'dump') + ' --xla_dump_hlo_module_re=' + pattern,
        '<USER_HOME>/miniconda3/bin/python', str(selected_tool), '--source', str(source), '--case', str(case),
        '--out', str(out), '--target', str(TARGET), '--domain', domain]
    print('START', label, time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), flush=True)
    with (LANE / (label + '.log')).open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=5500)
    assert result.returncode == 0, (label, result.returncode)
    rec = json.loads((out / 'receipt.json').read_text())
    assert rec['phase'] == 'done' and rec['native_gate'] == 'PASS', (label, rec)
    print('DONE', label, 'kernels', rec['kernels'], 'temp', rec['memory']['temp_size_in_bytes'], flush=True)
