"""Compare exact old/new STACK fusion source families and radar attribution."""
from collections import Counter
from functools import lru_cache
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess

root = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('frozen_mapper', root / 'proofs/t_e41opt/map_flags.py')
mapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mapper)
rows = []
for case_name, map_path in (
    ('mp18-old', Path('<DATA_ROOT>/wrf_gpu2/scratch/t-e41opt/EO31_mp18/source_map.json')),
    ('mp18-r3', Path('<DATA_ROOT>/wrf_gpu2/scratch/t-e41opt/EO42_mp18_r3/source_map.json')),
    ('mp18_cu5-r3', Path('<DATA_ROOT>/wrf_gpu2/scratch/t-e41opt/EO43_mp18_cu5_r3/source_map.json')),
):
    mapped = json.loads(map_path.read_text())
    text = gzip.decompress((map_path.parent / 'actual.optimized.hlo.gz').read_bytes()).decode()
    assert hashlib.sha256(text.encode()).hexdigest() == mapped['hlo_sha256']
    chain = lru_cache(None)(mapper.source_stacks(text))
    lines = text.splitlines()
    ranges = {}
    begin, comp = None, None
    calls = {}
    for index, line in enumerate(lines):
        start = re.match(r'^%?([\w.\-]+) .*\{$', line)
        if start:
            begin, comp = index, start[1]
        elif line.strip() == '}' and comp is not None:
            ranges[comp] = (begin + 1, index)
            begin, comp = None, None
        fusion = re.match(r'\s*(?:ROOT )?%?([\w.\-]+) = .* fusion\(', line)
        if fusion:
            calls[fusion[1].replace('.', '_')] = re.search(r'calls=%?([\w.\-]+)', line)[1]
    case = Path(mapped['case'])
    gate = json.loads((case / 'e41/wave_gate.json').read_text())
    jit = next(entry for entry in gate['entries'] if entry['source_kind'] == 'jax_cache')
    label = hashlib.sha256(jit['path'].encode()).hexdigest()[:12]
    resources = {}
    for index, _ in enumerate(jit['elfs']):
        for resource in json.loads((case / f'e41/cubin_scan_{label}_e{index}.json').read_text()):
            resources[resource['kernel']] = resource
    census = {}
    radar_fusions = []
    for fusion, comp in calls.items():
        start, end = ranges[comp]
        files, functions = Counter(), Counter()
        radar = 0
        for line in lines[start:end]:
            frame = re.search(r'stack_frame_id=(\d+)', line)
            project = [f for f in chain(frame[1]) if '/gpuwrf/' in f[0]] if frame else []
            if project:
                files[project[0][0]] += 1
                functions[project[0][:2]] += 1
            # diagnostics.py also contains effective-radius work; only the
            # radardd02 function/ancestor identifies the radar computation.
            if any('radardd02' in fn for _, fn, _ in project):
                radar += 1
        resource = resources.get(fusion)
        row = {'fusion': fusion, 'body_instructions': end - start,
               'attributed_instructions': sum(files.values()),
               'radar_stack_instructions': radar,
               'file_counts': dict(files),
               'function_counts': [{'file': file, 'function': fn, 'instructions': n}
                                   for (file, fn), n in functions.most_common(12)],
               'native_STACK_B': resource['stack'] if resource else None}
        if fusion in mapped['flags']:
            census[fusion] = row
        if radar:
            radar_fusions.append(row)
    stack_name = max((name for name, flag in mapped['flags'].items() if 'stack_B' in flag),
                     key=lambda name: mapped['flags'][name]['stack_B'])
    assert stack_name in census
    rows.append({'case': case_name, 'map_path': str(map_path),
                 'map_sha256': hashlib.sha256(map_path.read_bytes()).hexdigest(),
                 'blob_sha256': mapped['blob_sha256'], 'max_STACK_fusion': census[stack_name],
                 'all_flagged_fusions': census, 'radar_attributed_fusions': radar_fusions,
                 'radar_native_matched_count': sum(row['native_STACK_B'] is not None for row in radar_fusions),
                 'radar_native_max_STACK_B': max((row['native_STACK_B'] for row in radar_fusions
                                                 if row['native_STACK_B'] is not None), default=None)})
core = []
for name in ('gs_a.py', 'gs_b.py', 'gs_c.py', 'mathfun.py', 'diagnostics.py'):
    path = 'src/gpuwrf/physics/nssl2mom/' + name
    hashes = [subprocess.check_output(['git', 'rev-parse', commit + ':' + path], text=True).strip()
              for commit in ('8afb83d7d', '5abde3224')]
    assert hashes[0] == hashes[1]
    core.append({'file': path, 'git_blob': hashes[0], 'old_new_BYTE': True})
report = {'scope': 'exact rich-HLO source census and native resource join; no runtime attribution or causal A/B',
          'program_max_STACK_growth_B': 1336 - 504, 'program_max_STACK_ratio': 1336 / 504,
          'core_files_byte_unchanged': core, 'records': rows,
          'causal_limit': 'Different source/coupling graphs and different flagged fusions; no radar-only native ablation was measured.'}
(out / 'STACK_COMPARISON.json').write_text(json.dumps(report, indent=2) + '\n')
for row in rows:
    maximum = row['max_STACK_fusion']
    print(row['case'], maximum['fusion'], maximum['native_STACK_B'], maximum['file_counts'],
          'radar_in_flag', maximum['radar_stack_instructions'],
          'radar_fusions', len(row['radar_attributed_fusions']),
          'radar_native_max', row['radar_native_max_STACK_B'])
