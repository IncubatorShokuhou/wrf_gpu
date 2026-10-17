"""Append current NSSL radar-carry receipts without modifying older decisions."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

root = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent
prior = root / 'proofs/t_e41opt_updates/post_reboot'
inventory = json.loads((prior / 'INVENTORY.json').read_text())
sources = json.loads((prior / 'SOURCE_MAPS.json').read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


for number, original_name in ((42, 'mp18'), (43, 'mp18_cu5')):
    name = original_name + '-r3'
    case = Path('<USER_HOME>/wrf_gpu2_lanes/v-run/round3') / original_name
    mapped = Path(f'<DATA_ROOT>/wrf_gpu2/scratch/t-e41opt/EO{number}_{original_name}_r3/source_map.json')
    mapping = json.loads(mapped.read_text())
    gate_path = case / 'e41/wave_gate.json'
    gate = json.loads(gate_path.read_text())
    run = json.loads((case / 'gpu_run.json').read_text())
    d6 = json.loads((case / 'grid_gate.json').read_text())
    summary = json.loads((case / 'RESULT.json').read_text())
    writer = json.loads((case / 'WRITER_AUDIT.json').read_text())
    assert run['platform'] == 'gpu' and run['rc'] == 0
    assert run['source_rev'] == '5abde3224a50e5f67674bb1865aa0cc6d1ff1937'
    assert d6['verdict'] == summary['verdict'] == 'PASS' and d6['coverage_complete']
    assert d6['paired_frames'] == summary['paired_frames'] == writer['paired_frames'] == 25
    assert summary['bounded_D6'] == 'PASS' and not summary['missing_CPU_fields']
    assert not d6['failures'] and not d6['raw_gpu_nonfinite'] and d6['first_failure'] is None
    assert writer['variable_metadata_mismatch_count'] == 0
    assert gate['e41_gate'] == gate['hard_stack_gate'] == 'FAIL'
    assert gate['coverage']['status'] == 'PASS'
    assert gate['coverage']['n_discovered'] == gate['coverage']['n_scanned'] == 2
    jit = next(entry for entry in gate['entries'] if entry['source_kind'] == 'jax_cache')
    assert jit['blob_sha256'] == mapping['blob_sha256']
    assert sha(Path(mapping['pin'])) == mapping['pin_sha256']
    flags, scans, resources = {}, [], {}
    for entry in gate['entries']:
        label = hashlib.sha256(entry['path'].encode()).hexdigest()[:12]
        for index, elf in enumerate(entry['elfs']):
            assert elf['max_stack_B'] == 1336 and not elf['nonleaf_over1k'] and not elf['callgraph_unknown']
            for fusion, size in elf['stack_over256']:
                assert flags.setdefault(fusion, {}).setdefault('stack_B', size) == size == 1336
            scan = case / f'e41/cubin_scan_{label}_e{index}.json'
            scans.append({'path': str(scan), 'sha256': sha(scan)})
        for fusion, count in entry['kinput_transpose_over500']:
            assert flags.setdefault(fusion.replace('.', '_'), {}).setdefault('kinput_HLO', count) == count
    for scan in scans:
        for resource in json.loads(Path(scan['path']).read_text()):
            if resource['kernel'] in flags:
                assert resources.setdefault(resource['kernel'], resource['stack']) == resource['stack']
    assert set(flags) == set(mapping['flags']) == set(resources)
    assert len(flags) == (7 if original_name == 'mp18' else 8)
    for fusion, flag in flags.items():
        source = mapping['flags'][fusion]
        assert source['kinput_HLO'] == flag['kinput_HLO']
        flag.update(stack_B=resources[fusion], sources=source['sources'][:8], instructions=source['instructions'])
        if flag['stack_B'] > 256:
            assert flag['stack_B'] == 1336 and flag['kinput_HLO'] == 1172
    inputs_path = case / 'inputs.json'
    inputs = json.loads(inputs_path.read_text())
    namelist = (case / 'input/namelist.input').read_text()
    options = {key: int(re.search(r'(?im)^\s*' + key + r'\s*=\s*(\d+)', namelist)[1])
               for key in ('mp_physics', 'cu_physics', 'bl_pbl_physics', 'sf_sfclay_physics',
                           'sf_surface_physics', 'ra_lw_physics', 'ra_sw_physics')}
    row = {'producer': 'v-run', 'case': name, 'producer_case': original_name, 'round': 3,
           'options': options, 'source_commit': run['source_rev'], 'flags': flags,
           'program_max_stack_B': 1336, 'nonleaf_over1k': 0, 'attribution_path': str(mapped),
           'owner': 't-e41opt disclosure; no numerical code edits',
           'runtime_impact': 'UNKNOWN; no exact-artifact per-kernel profile',
           'timing_scope': 'cold/shared qualification, not S1 or isolated radar overhead',
           'whole_run_wall_s': run['wall_s'] if 'wall_s' in run else run['monotonic_wall_s'],
           'gate_path': str(gate_path), 'gate_sha256': sha(gate_path),
           'inputs_path': str(inputs_path), 'inputs_sha256': sha(inputs_path),
           'D6_verdict': 'PASS', 'D6_paired_frames': 25, 'bounded_D6': 'PASS', 'missing_CPU_fields': [],
           'D6_path': str(case / 'grid_gate.json'), 'D6_sha256': sha(case / 'grid_gate.json'),
           'gpu_run_path': str(case / 'gpu_run.json'), 'gpu_run_sha256': sha(case / 'gpu_run.json'),
           'producer_summary_path': str(case / 'RESULT.json'), 'producer_summary_sha256': sha(case / 'RESULT.json'),
           'writer_audit_path': str(case / 'WRITER_AUDIT.json'), 'writer_audit_sha256': sha(case / 'WRITER_AUDIT.json'),
           'native_resource_scans': scans, 'radar_tolerance_scope': 'REFL_10CM inventoried/finite, REPORT_ONLY under frozen manifest',
           'F2_scope': 'negative phase-event justification remains OPEN; D6 PASS is not an F2 waiver'}
    inventory['records'].append(row)
    packed = {key: mapping[key] for key in ('pin', 'pin_sha256', 'blob_sha256', 'hlo_sha256')}
    packed.update(case=name, producer='v-run', map_path=str(mapped), map_sha256=sha(mapped),
                  flags={fusion: {key: value for key, value in source.items() if key not in ('call', 'stacks')}
                         for fusion, source in mapping['flags'].items()})
    for flag in packed['flags'].values():
        flag['sources'] = flag['sources'][:10]
    sources.append(packed)
assert len(inventory['records']) == len(sources) == 22
assert sum(len(row['flags']) for row in inventory['records']) == 77
assert sum(row.get('nonleaf_over1k', 0) for row in inventory['records']) == 3
inventory.update(cutoff_utc=datetime.now(timezone.utc).isoformat(),
                 scope='22 source/selector receipts / 77 flagged fusions + 3 nonleaf callee findings; NSSL round3 full D6/coverage PASS')
(out / 'INVENTORY.json').write_text(json.dumps(inventory, indent=2, allow_nan=False) + '\n')
(out / 'SOURCE_MAPS.json').write_text(json.dumps(sources, indent=2, allow_nan=False) + '\n')
comparison = json.loads((out / 'STACK_COMPARISON.json').read_text())
for record in comparison['records'][1:]:
    assert record['max_STACK_fusion']['native_STACK_B'] == 1336
    assert record['max_STACK_fusion']['radar_stack_instructions'] == 0
    assert record['radar_native_max_STACK_B'] == 0
    assert len(record['radar_attributed_fusions']) == record['radar_native_matched_count']
doc = root / 'docs/release/E41_OPT_IN_v0.4.md'
text = subprocess.check_output(['git', 'show', '29d0b8ef8:docs/release/E41_OPT_IN_v0.4.md'], text=True)
text = re.sub(r'Inventory cutoff: \*\*.*?\*\*', 'Inventory cutoff: **' + inventory['cutoff_utc'] + '**', text, count=1)
config_rows, flag_rows = [], []
for row in inventory['records'][20:]:
    options = '/'.join(str(row['options'][key]) for key in ('mp_physics', 'cu_physics', 'bl_pbl_physics',
                         'sf_sfclay_physics', 'sf_surface_physics', 'ra_lw_physics', 'ra_sw_physics'))
    config_rows.append(f'| {row["case"]} | {options} | `{row["source_commit"][:10]}` | PASS | 25 | 1336 | {len(row["flags"])} |')
    for name, flag in row['flags'].items():
        sites = '; '.join(f'`{Path(s["file"]).name}:{s["line"]}` ({s["function"]})' for s in flag['sources'][:2])
        flag_rows.append(f'| {row["case"]} | `{name}` | {flag["stack_B"]} | {flag["kinput_HLO"]} | {sites} | — |')
text = text.replace('\n## Support-matrix disclosure rows', '\n'.join(config_rows) + '\n\n## Support-matrix disclosure rows', 1)
text = text.replace('\n## Remedy disposition and resource consequences', '\n'.join(flag_rows) + '\n\n## Remedy disposition and resource consequences', 1)
text = text.replace('Active inventory/maps are in `proofs/t_e41opt_updates/post_reboot/`;',
                    'Active inventory/maps are in `proofs/t_e41opt_updates/nssl_round3/`;')
text = text.replace('Current scope: 20 source/selector receipts, 62 flagged fusions and 3',
                    'The post-reboot update covered 20 source/selector receipts, 62 flagged fusions and 3')
text += ('\n## NSSL round3: radar carry and STACK comparison\n\n'
         'Both actual `5abde3224` arms have full original CPU-WRF D6 and coverage\n'
         'PASS: 25 frames, 372 CPU fields, 62 bounded fields and zero raw nonfinite\n'
         'values. REFL_10CM is inventoried and finite but REPORT_ONLY under the\n'
         'frozen manifest. F2 negative-event justification remains OPEN. These\n'
         'facts do not qualify a different source or waive E41.\n\n'
         'E41 FAIL remains in both arms: maximum STACK 1,336 B, seven kInput\n'
         'failures for MP18 and eight for MP18+CU5 (maximum 1,172 HLO), nonleaf\n'
         'zero. The STACK fusion is itself the 1,172-HLO kInput failure and is\n'
         'counted once per executable. Older round2 records/maps remain unchanged.\n\n'
         '**Measured increase:** MP18 program maximum STACK rose from 504 B to\n'
         '1,336 B: +832 B, or 2.65 times. MP18+CU5 previously had maximum 16 B\n'
         'and no STACK failure; it now also has 1,336 B. These are different\n'
         'fusions: the earlier maximum was `input_and_multiply_power_reduce_select_fusion`\n'
         '(chiefly `mathfun.gamxinfdp`), while the new maximum is\n'
         '`input_add_and_divide_maximum_minimum_multiply_select_subtract_transpose_fusion`.\n\n'
         '**Radar attribution:** the new 1,172-instruction fusion has 580\n'
         'instructions attributed to `gs_c.py`, 132 to `gs_a.py` and 126 to\n'
         '`gs_b.py`, in both arms. None of its 880/879 attributed source stacks\n'
         'contains `radardd02`. Direct `radardd02` attribution identifies nine\n'
         'MP18 and eleven MP18+CU5 fusions; all match actual native kernels with\n'
         'STACK 0 B, and their largest HLO body is 400 instructions.\n\n'
         'These are static fusion counts, not launches per timestep.\n\n'
         'Thus the measured large frame belongs to fused NSSL process work,\n'
         'rather than a directly radar-attributed kernel. An indirect fusion\n'
         'change caused by added radar/carry outputs is possible [I], but is\n'
         'not isolated: the coupling graphs differ and no radar-only native A/B\n'
         'was measured. `gs_a.py`, `gs_b.py`, `gs_c.py`, `mathfun.py` and\n'
         '`diagnostics.py` are byte-identical between the old and new sources.\n'
         'Static source counts are not runtime weights or a speed/memory ratio.\n'
         '`STACK_COMPARISON.json` seals the complete census and byte proof.\n\n'
         'Current scope: 22 source/selector receipts, 77 flagged fusions and\n'
         'three nonleaf callee findings (the latter remain the earlier C15 arm).\n'
         'No GPU, forecast, numerical validation or native compile was repeated\n'
         'for this disclosure. No remedy or firewall is admitted.\n')
doc.write_text(text)
seal = {'verdict': 'DISCLOSURE_UPDATE_READY', 'parent_packet_commit': '29d0b8ef8',
        'parent_qualification_sha256': sha(prior / 'QUALIFICATION.json'),
        'source_commit': '5abde3224a50e5f67674bb1865aa0cc6d1ff1937',
        'active_receipts': 22, 'flagged_fusions': 77, 'nonleaf_callee_findings': 3,
        'production_code_changed': False, 'new_GPU_runs': 0, 'new_forecasts': 0,
        'E41': 'both new arms FAIL retained; no waiver',
        'causality': 'program STACK growth measured; no direct radar source in failing fusion; indirect cause not isolated',
        'files': {str(path.relative_to(root)): sha(path)
                  for path in (out / 'INVENTORY.json', out / 'SOURCE_MAPS.json', out / 'STACK_COMPARISON.json',
                               out / 'analyze_stack.py', Path(__file__), doc)}}
(out / 'QUALIFICATION.json').write_text(json.dumps(seal, indent=2) + '\n')
print('PASS: 22receipts/77flaggedfusions/3nonleaf; prior20rows/maps exact; NSSL bothD6PASS/E41FAIL retained')
