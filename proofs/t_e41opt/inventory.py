"""Freeze opt-in E41 failures and producer evidence without rerunning forecasts."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

out = Path(sys.argv[1])
records = []
for root in (Path('<USER_HOME>/wrf_gpu2_lanes/v-run'), Path('<USER_HOME>/wrf_gpu2_lanes/v-prep')):
    for path in sorted(root.glob('*/e41/wave_gate.json')):
        gate = json.loads(path.read_text())
        if gate.get('e41_gate') != 'FAIL':
            continue
        raw = (path.parent.parent/'inputs.json').read_bytes()
        inputs = json.loads(raw)
        flags = {}
        for entry in gate['entries']:
            for elf in entry['elfs']:
                for name, value in elf['stack_over256']:
                    flags.setdefault(name.replace('.', '_'), {})['stack_B'] = value
                if elf['nonleaf_over1k']:
                    raise ValueError('new non-leaf failure requires explicit disclosure')
            for name, value in entry.get('kinput_transpose_over500') or []:
                flags.setdefault(name.replace('.', '_'), {})['kinput_HLO'] = value
        native_resources, resource_scans = {}, []
        for entry in gate['entries']:
            label = hashlib.sha256(entry['path'].encode()).hexdigest()[:12]
            for index, elf in enumerate(entry['elfs']):
                scan_path = path.parent/f'cubin_scan_{label}_e{index}.json'
                resources = json.loads(scan_path.read_text())
                resource_scans.append({'path':str(scan_path),'sha256':hashlib.sha256(scan_path.read_bytes()).hexdigest()})
                for row in resources:
                    if row['kernel'] in flags:
                        old = native_resources.setdefault(row['kernel'], row['stack'])
                        assert old == row['stack'], (path,row['kernel'],'distinct native resources')
        assert set(native_resources) == set(flags), (path,'missing native flag resource')
        for name, row in flags.items():
            row['stack_B'] = native_resources[name]
        assert flags, path
        case = path.parent.parent
        maps = list(Path('<USER_HOME>/wrf_gpu2_lanes/t-e41opt').glob(f'*_{case.name}/source_map.json'))
        attribution = json.loads(maps[0].read_text()) if len(maps) == 1 else None
        if attribution:
            jit = next(e for e in gate['entries'] if e['source_kind'] == 'jax_cache')
            assert attribution['blob_sha256'] == jit['blob_sha256'], (case,'stale source map')
            for name, row in flags.items():
                mapped = attribution['flags'][name]
                if 'kinput_HLO' in row:
                    assert row['kinput_HLO'] == mapped['kinput_HLO'], (case,name,'HLO count changed')
                row['sources'] = mapped['sources'][:8]
                row['instructions'] = mapped['instructions']
        d6path = case/('grid_gate.json' if root.name == 'v-run' else 'D6_d01.json')
        d6 = json.loads(d6path.read_text()) if d6path.exists() else {}
        run = json.loads((case/'gpu_run.json').read_text())
        options = inputs.get('options')
        if options is None:
            namelist = (case/'input/namelist.input').read_text()
            options = {}
            for key in ('mp_physics','cu_physics','bl_pbl_physics','sf_sfclay_physics',
                        'sf_surface_physics','ra_lw_physics','ra_sw_physics'):
                match = re.search(r'(?im)^\s*'+key+r'\s*=\s*(\d+)',namelist)
                assert match, (case,key)
                options[key] = int(match[1])
        records.append({'producer': root.name, 'case': case.name,
                        'options': options,
                        'source_commit': run.get('source_commit',run.get('source_rev',inputs.get('source_rev'))),
                        'gate_path': str(path), 'gate_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'inputs_path': str(case/'inputs.json'), 'inputs_sha256': hashlib.sha256(raw).hexdigest(),
                        'flags': flags, 'attribution_path': str(maps[0]) if attribution else None,
                        'program_max_stack_B': max(elf['max_stack_B'] for e in gate['entries'] for elf in e['elfs']),
                        'owner': 't-c03rain' if case.name == 'c03' else 't-e41opt disclosure',
                        'D6_verdict': d6.get('verdict','INCOMPLETE'),
                        'D6_path': str(d6path), 'D6_paired_frames': d6.get('paired_frames'),
                        'D6_sha256': hashlib.sha256(d6path.read_bytes()).hexdigest() if d6path.exists() else None,
                        'gpu_run_path': str(case/'gpu_run.json'),
                        'gpu_run_sha256': hashlib.sha256((case/'gpu_run.json').read_bytes()).hexdigest(),
                        'native_resource_scans': resource_scans,
                        'runtime_impact': 'UNKNOWN; no exact-artifact per-kernel profile',
                        'whole_run_wall_s': run.get('monotonic_wall_s',run.get('wall_s')),
                        'timing_scope': 'concurrent cold qualification, not S1 or isolated E41 impact'})
report = {'cutoff_utc': datetime.now(timezone.utc).isoformat(), 'records': records,
          'scope': 'actual retained opt-in E41 failures only; defaults retain hard gate; D6 and F2 separate'}
out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(len(records), 'configuration failures,',sum(len(r['flags']) for r in records),'unique per-config flags')
