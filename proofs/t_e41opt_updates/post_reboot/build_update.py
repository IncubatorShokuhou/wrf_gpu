"""Disclose new actual C61, failed short C08, and C15 native artifacts."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

root = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent
prior = root / 'proofs/t_e41opt_updates/c17'
inventory = json.loads((prior / 'INVENTORY.json').read_text())
sources = json.loads((prior / 'SOURCE_MAPS.json').read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


for name, producer, case, mapped in (
    ('c61', 'v-prep', Path('<USER_HOME>/wrf_gpu2_lanes/v-prep/c61'),
     Path('<DATA_ROOT>/wrf_gpu2/scratch/t-e41opt/EO39_c61/source_map.json')),
    ('c08-short', 't-c02nan', Path('<DATA_ROOT>/wrf_gpu2/scratch/t-c02nan/TCN24_c08'),
     Path('<DATA_ROOT>/wrf_gpu2/scratch/t-c02nan/TCN24_c08/e41/attribution/source_map.json')),
    ('c15', 'v-prep', Path('<USER_HOME>/wrf_gpu2_lanes/v-prep/c15'),
     Path('<DATA_ROOT>/wrf_gpu2/scratch/t-e41opt/EO40_c15/source_map.json')),
):
    mapping = json.loads(mapped.read_text())
    gate_path = case / 'e41/wave_gate.json'
    gate = json.loads(gate_path.read_text())
    run_path = case / ('receipt.json' if name == 'c08-short' else 'gpu_run.json')
    run = json.loads(run_path.read_text())
    assert run['platform'] == 'gpu' and run['rc'] == (1 if name == 'c08-short' else 0)
    assert gate['e41_gate'] == 'FAIL' and gate['coverage']['status'] == 'PASS'
    assert gate['coverage']['n_discovered'] == gate['coverage']['n_scanned'] == 2
    jit = next(entry for entry in gate['entries'] if entry['source_kind'] == 'jax_cache')
    assert jit['blob_sha256'] == mapping['blob_sha256']
    assert sha(Path(mapping['pin'])) == mapping['pin_sha256']
    flags = {fusion: {'sources': source['sources'][:8], 'instructions': source['instructions']}
             for fusion, source in mapping['flags'].items()}
    scans, resources, nonleaf_by_entry = [], {}, []
    for entry in gate['entries']:
        label = hashlib.sha256(entry['path'].encode()).hexdigest()[:12]
        nonleaf = []
        for index, elf in enumerate(entry['elfs']):
            assert not elf['callgraph_unknown']
            path = case / f'e41/cubin_scan_{label}_e{index}.json'
            scans.append({'path': str(path), 'sha256': sha(path)})
            scan = json.loads(path.read_text())
            for resource in scan:
                if resource['kernel'] in flags:
                    assert resources.setdefault(resource['kernel'], resource['stack']) == resource['stack']
            for truncated_name, size in elf['stack_over256']:
                matches = [fusion for fusion in flags if fusion == truncated_name or fusion.startswith(truncated_name)]
                assert len(matches) == 1, (truncated_name, matches)
                fusion = matches[0]
                assert resources[fusion] == size
                if fusion != truncated_name:
                    flags[fusion]['native_reported_stack_name'] = truncated_name
            nonleaf.extend(elf['nonleaf_over1k'])
        for fusion, count in entry['kinput_transpose_over500']:
            fusion = fusion.replace('.', '_')
            assert flags[fusion].setdefault('kinput_HLO', count) == count
            assert mapping['flags'][fusion]['kinput_HLO'] == count
        nonleaf_by_entry.append(nonleaf)
    assert set(resources) == set(flags)
    assert nonleaf_by_entry[0] == nonleaf_by_entry[1]
    nonleaf = nonleaf_by_entry[0]
    for callee in nonleaf:
        assert callee['kernel'] in flags and callee['instructions'] > 1000 and callee['nested_calls'] > 0
        flags[callee['kernel']].setdefault('nonleaf_callees', []).append(callee)
    for fusion, flag in flags.items():
        flag['stack_B'] = resources[fusion]
        if 'stack_B' in mapping['flags'][fusion]:
            assert flag['stack_B'] == mapping['flags'][fusion]['stack_B']
    max_stack = max(elf['max_stack_B'] for entry in gate['entries'] for elf in entry['elfs'])
    expected = {'c61': (1, 8, 0), 'c08-short': (6, 408, 0), 'c15': (1, 5920, 3)}[name]
    assert (len(flags), max_stack, len(nonleaf)) == expected
    mapper_receipt = mapped.parent / 'mapper_receipt.json'
    if name == 'c15':
        audit = json.loads(mapper_receipt.read_text())
        assert audit['frozen_tool_sha256'] == sha(root / 'proofs/t_e41opt/map_flags.py')
    namelist_path = case / 'input/namelist.input'
    namelist = namelist_path.read_text()
    options = {key: int(re.search(r'(?im)^\s*' + key + r'\s*=\s*(\d+)', namelist)[1])
               for key in ('mp_physics', 'cu_physics', 'bl_pbl_physics', 'sf_sfclay_physics',
                           'sf_surface_physics', 'ra_lw_physics', 'ra_sw_physics')}
    input_sha256 = run.get('actual_input_sha256', run.get('input_sha256'))
    assert sha(namelist_path) == input_sha256['namelist.input']
    row = {'case': name, 'producer': producer, 'producer_case': case.name, 'options': options,
           'source_commit': run['source_commit'], 'flags': flags, 'program_max_stack_B': max_stack,
           'gate_path': str(gate_path), 'gate_sha256': sha(gate_path), 'attribution_path': str(mapped),
           'gpu_run_path': str(run_path), 'gpu_run_sha256': sha(run_path), 'GPU_rc': run['rc'],
           'namelist_path': str(namelist_path), 'namelist_sha256': sha(namelist_path),
           'input_sha256': input_sha256, 'native_resource_scans': scans, 'nonleaf_over1k': len(nonleaf),
           'final_run_proof_available': bool(gate['coverage']['run_proofs']),
           'owner': 't-e41opt disclosure; numerical modules retain their owners',
           'runtime_impact': 'UNKNOWN; no exact-artifact per-kernel profile',
           'whole_run_wall_s': run.get('monotonic_wall_s', run.get('wall_s')),
           'timing_scope': 'original pre-reboot cold/shared qualification; not S1 or later-main qualification'}
    if name == 'c15':
        row.update(mapper_receipt_path=str(mapper_receipt), mapper_receipt_sha256=sha(mapper_receipt))
    if name == 'c08-short':
        error_path = Path('<USER_HOME>/wrf_gpu2_lanes/t-c02nan/TCN24/gpu.log')
        assert 'NonFiniteStateError' in error_path.read_text() and 'step=200 sim_time_s=3600' in error_path.read_text()
        frames = list((case / 'gpu').glob('wrfout_d01_*'))
        assert len(frames) == 1 and not row['final_run_proof_available']
        row.update(D6_verdict='NOT_RUN', D6_paired_frames=None, D6_path=None,
                   forecast_status='FAIL: NonFiniteStateError at step200/sim3600; intended2h arm',
                   retained_GPU_frames=1, error_path=str(error_path), error_sha256=sha(error_path))
    else:
        d6_path = case / 'D6_d01.json'
        d6 = json.loads(d6_path.read_text())
        assert d6['paired_frames'] == 25 and not d6['failures'] and not d6['nonfinite']
        assert d6['verdict'] == ('PASS' if name == 'c61' else 'INCOMPLETE')
        assert d6['coverage_complete'] == (name == 'c61')
        missing = sorted({field for frame in d6['by_frame'] for field in frame['missing_vs_cpu']})
        assert missing == ([] if name == 'c61' else ['REFD_MAX', 'REFL_10CM'])
        row.update(D6_verdict=d6['verdict'], D6_paired_frames=25, D6_path=str(d6_path),
                   D6_sha256=sha(d6_path), missing_CPU_fields=missing, bounded_D6='PASS')
        inputs_path = case / 'inputs.json'
        row.update(inputs_path=str(inputs_path), inputs_sha256=sha(inputs_path))
    inventory['records'].append(row)
    packed = {key: mapping[key] for key in ('pin', 'pin_sha256', 'blob_sha256', 'hlo_sha256')}
    packed.update(case=name, producer=producer, map_path=str(mapped), map_sha256=sha(mapped),
                  flags={fusion: {key: value for key, value in source.items() if key not in ('call', 'stacks')}
                         for fusion, source in mapping['flags'].items()})
    for flag in packed['flags'].values():
        flag['sources'] = flag['sources'][:10]
    sources.append(packed)

assert len(inventory['records']) == len(sources) == 20
assert sum(len(row['flags']) for row in inventory['records']) == 62
inventory.update(cutoff_utc=datetime.now(timezone.utc).isoformat(),
                 scope='20 source/selector receipts / 62 flagged fusions + 3 nonleaf callee findings; includes failed short C08')
(out / 'INVENTORY.json').write_text(json.dumps(inventory, indent=2, allow_nan=False) + '\n')
(out / 'SOURCE_MAPS.json').write_text(json.dumps(sources, indent=2, allow_nan=False) + '\n')
doc = root / 'docs/release/E41_OPT_IN_v0.4.md'
text = subprocess.check_output(['git', 'show', '0479d8efb:docs/release/E41_OPT_IN_v0.4.md'], text=True)
text = re.sub(r'Inventory cutoff: \*\*.*?\*\*', 'Inventory cutoff: **' + inventory['cutoff_utc'] + '**', text, count=1)
text = text.replace('The complete producer scans report no non-leaf violation.',
                    'C15 has three non-leaf callee violations; other listed receipts have none.')
text = text.replace('**experimental: E41 kInput** as indicated by its measured flag.',
                    '**experimental: E41 kInput** and/or **experimental: E41 nonleaf** as measured.')
config_rows, flag_rows = [], []
for row in inventory['records'][17:]:
    options = '/'.join(str(row['options'][key]) for key in ('mp_physics', 'cu_physics', 'bl_pbl_physics',
                         'sf_sfclay_physics', 'sf_surface_physics', 'ra_lw_physics', 'ra_sw_physics'))
    config_rows.append(f'| {row["case"]} | {options} | `{row["source_commit"][:10]}` | {row["D6_verdict"]} | '
                       f'{row["D6_paired_frames"] or "not scored"} | {row["program_max_stack_B"]} | {len(row["flags"])} |')
    for name, flag in row['flags'].items():
        sites = '; '.join(f'`{Path(s["file"]).name}:{s["line"]}` ({s["function"]})' for s in flag['sources'][:2])
        flag_rows.append(f'| {row["case"]} | `{name}` | {flag["stack_B"]} | {flag.get("kinput_HLO", "—")} | {sites} |')
text = text.replace('\n## Support-matrix disclosure rows', '\n'.join(config_rows) + '\n\n## Support-matrix disclosure rows', 1)
text = text.replace('\n## Remedy disposition and resource consequences', '\n'.join(flag_rows) + '\n\n## Remedy disposition and resource consequences', 1)
start = text.index('## Support-matrix disclosure rows')
end = text.index('## Remedy disposition and resource consequences')
section = text[start:end]
lines = []
for line in section.splitlines():
    if line.startswith('| Case |'):
        line += ' Nonleaf callee SASS >1000 |'
    elif line.startswith('|---'):
        line += '---|'
    elif line.startswith('|'):
        line += ' 10240 / 8899 / 13137 |' if line.startswith('| c15 |') else ' — |'
    lines.append(line)
text = text[:start] + '\n'.join(lines) + '\n\n' + text[end:]
text = text.replace('Active inventory/maps are in `proofs/t_e41opt_updates/c17/`;',
                    'Active inventory/maps are in `proofs/t_e41opt_updates/post_reboot/`;')
text = text.replace('The active inventory has 17 source/selector receipts and 54 E41 flags.',
                    'The C17 update brought the inventory to 17 source/selector receipts and 54 flagged fusions.')
text = text.replace('| Case | Options | Source | D6 | Frames | Program max STACK B | Flags |',
                    '| Case | Options | Source | D6 | Paired frames | Program max STACK B | Flagged fusions |')
text += ('\n## Post-reboot disclosure reconciliation\n\n'
         'C02 is already disclosed: six kInput 570/549/661/659/586/632 failures,\n'
         'maximum STACK 8 B and nonleaf zero PASS, full original CPU-WRF D6 PASS\n'
         'over 25 frames.\n'
         'No C02 forecast or numerical gate was repeated.\n\n'
         'C61 (`d2782882bd`, MP97/PBL0/SF91) has full D6 PASS over 25 frames with\n'
         'zero bounded failures/nonfinite values. E41 FAIL is solely kInput 527 HLO,\n'
         'in the Goddard `_saticel_cell` source family. Maximum STACK 8 B and\n'
         'nonleaf zero pass. Its artifact remains distinct from later source trees.\n\n'
         'C08-short is `TCN24_c08` / `5c0e7c11e`, a failed intended 2h arm. It\n'
         'terminated with NonFiniteStateError at step 200 (3,600 s) and retained\n'
         'only lead 0 output. No D6 scoring or complete forecast proof exists. Its\n'
         'actual JIT+AOT scans show STACK 408 B and six kInput sizes\n'
         '1320/1334/843/1630/974/734 HLO instructions.\n'
         'The first two source stacks resolve to SF1 table lookup; the other four\n'
         'to Morrison. The earlier C08 arm remains unchanged. Neither native\n'
         'properties nor the nonfinite cause are transferred between sources.\n\n'
         'C15 (`afcdb5072c`, MP21) completed 25 frames with bounded D6 PASS and\n'
         'zero nonfinite values, but full coverage is INCOMPLETE: REFD_MAX and\n'
         'REFL_10CM are missing. E41 FAIL includes STACK 5,920 B and three nonleaf\n'
         'callees of 10,240/8,899/13,137 SASS instructions, with 14/8/16 nested calls.\n'
         'The parent fusion resolves to NSSL source sites. The native\n'
         'STACK label is truncated; its unique full cubin kernel and rich HLO\n'
         'fusion name are joined explicitly in the sealed evidence. The source\n'
         'sites describe the parent fusion; individual native callees are not\n'
         'assigned source or runtime weights by this mapping.\n\n'
         'Current scope: 20 source/selector receipts, 62 flagged fusions and 3\n'
         'nonleaf callee findings. Runtime impact remains unknown. No remedy or\n'
         'firewall is admitted. All work used retained pre-reboot artifacts;\n'
         'there was no GPU work, forecast rerun, compile cache or snapshot.\n')
doc.write_text(text)
seal = {'verdict': 'DISCLOSURE_UPDATE_READY', 'parent_packet_commit': '0479d8efb',
        'parent_qualification_sha256': sha(prior / 'QUALIFICATION.json'),
        'active_receipts': 20, 'flagged_fusions': 62, 'nonleaf_callee_findings': 3,
        'production_code_changed': False, 'new_GPU_runs': 0, 'new_forecasts': 0,
        'E41': 'all failures retained, no waiver',
        'CPU_resource_scope': 'main with_ram_slot.sh --label t-e41opt; one job/lane,6G/noSwap,global15GB; CPU12,13,28,29nice19/JAXcpu/QUIET/timeout',
        'files': {str(path.relative_to(root)): sha(path)
                  for path in (out / 'INVENTORY.json', out / 'SOURCE_MAPS.json', Path(__file__), out / 'map_long_native_name.py', doc)}}
(out / 'QUALIFICATION.json').write_text(json.dumps(seal, indent=2) + '\n')
print('PASS: prior17rows/maps exact;20receipts62flaggedfusions+3nonleafcallee findings; noGPU/forecast/code changes')
