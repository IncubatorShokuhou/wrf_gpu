"""Join actual QNSE native symbols to rich-HLO Pallas custom-call sites."""
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

assert os.environ['JAX_PLATFORMS'] == 'cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
root = Path(__file__).resolve().parents[3]
out = Path('<DATA_ROOT>/wrf_gpu2/scratch/t-e41opt/EO46_c26_qnse')
packet_commit = '17943b8b2'


def receipt(name):
    return json.loads(subprocess.check_output([
        'git', 'show', packet_commit + ':proofs/t_qnsecompile/receipts/' + name], text=True))


gate = receipt('ACTUAL_E41.json')
pin = next(row for row in receipt('PINS.json') if row['kind'] == 'jax_cache')
raw = Path(pin['pin']).read_bytes()
assert hashlib.sha256(raw).hexdigest() == pin['sha256']
import zstandard
blob = zstandard.ZstdDecompressor().decompress(raw, max_output_size=1 << 30)[4:]
entry = next(row for row in gate['entries'] if row['source_kind'] == 'jax_cache')
assert hashlib.sha256(blob).hexdigest() == entry['blob_sha256']
sys.path.insert(0, str(root / 'scripts'))
import wave_cubin_gate as wave
from gpuwrf.runtime import aot_executable as ax, aot_slim_module as asm
text = wave._optimized_hlo(blob, ax, asm)
spec = importlib.util.spec_from_file_location('frozen_map', root / 'proofs/t_e41opt/map_flags.py')
mapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mapper)
chain = mapper.source_stacks(text)
flags = {}
for kernel, stack in entry['elfs'][0]['stack_over256']:
    sites = []
    counts = Counter()
    for line in text.splitlines():
        if 'custom-call(' not in line or kernel not in line:
            continue
        name = re.match(r'\s*(?:ROOT )?%?([\w.\-]+) = ', line)[1]
        frame = re.search(r'stack_frame_id=(\d+)', line)
        frames = [frame for frame in chain(frame[1]) if '/gpuwrf/' in frame[0]] if frame else []
        assert frames, (kernel, name, 'no rich source stacks')
        counts[frames[0]] += 1
        sites.append({'hlo_instruction': name, 'source_stack': frames,
                      'call_line_sha256': hashlib.sha256(line.encode()).hexdigest()})
    assert sites, (kernel, 'no matching Pallas custom call')
    flags[kernel] = {'stack_B': stack, 'kind': 'Pallas native kernel',
                     'instructions': None, 'custom_call_sites': sites,
                     'sources': [{'file': file, 'function': function, 'line': line,
                                  'custom_call_sites': count}
                                 for (file, function, line), count in counts.most_common()],
                     'scope': 'opaque Pallas body; source counts refer to custom-call sites, not HLO body instructions'}
out.mkdir(exist_ok=True)
(out / 'actual.optimized.hlo.gz').write_bytes(gzip.compress(text.encode(), mtime=0))
report = {'case': 'c26', 'producer': 't-qnsecompile', 'packet_commit': packet_commit,
          'paid_source': '4dba72ce4c2998d02c67be1abc1d5e3a5a689be9',
          'pin': pin['pin'], 'pin_sha256': pin['sha256'], 'blob_sha256': entry['blob_sha256'],
          'hlo_sha256': hashlib.sha256(text.encode()).hexdigest(), 'flags': flags,
          'scope': 'actual paid rich-HLO custom-call/source/native-symbol join; no fusion-body or runtime-cost inference'}
(out / 'source_map.json').write_text(json.dumps(report, indent=2) + '\n')
for name, row in flags.items():
    print(name, row['stack_B'], row['sources'], 'custom_call_sites', len(row['custom_call_sites']))
