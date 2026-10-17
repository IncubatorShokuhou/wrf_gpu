"""Append final C02 and MP28 receipts without changing earlier sealed packets."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

root = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent
prior = root / 'proofs/t_e41opt_updates/mp18'
inventory = json.loads((prior / 'INVENTORY.json').read_text())
sources = json.loads((prior / 'SOURCE_MAPS.json').read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


for name, producer, case, mapped in (
    ('c02', 't-c02nan', Path('<DATA_ROOT>/wrf_gpu2/scratch/t-c02nan/TCN23_c02'),
     Path('<DATA_ROOT>/wrf_gpu2/scratch/t-c02nan/TCN23_c02/e41/attribution/source_map.json')),
    ('mp28', 'v-run', Path('<USER_HOME>/wrf_gpu2_lanes/v-run/mp28'),
     Path('<DATA_ROOT>/wrf_gpu2/scratch/t-e41opt/EO34_mp28/source_map.json')),
):
    mapping = json.loads(mapped.read_text())
    gate_path = case / 'e41/wave_gate.json'
    gate = json.loads(gate_path.read_text())
    run_path = case / ('receipt.json' if name == 'c02' else 'gpu_run.json')
    d6_path = case / ('D6_d01.json' if name == 'c02' else 'grid_gate.json')
    run = json.loads(run_path.read_text())
    d6 = json.loads(d6_path.read_text())
    assert gate['e41_gate'] == 'FAIL' and run['rc'] == 0
    assert gate['hard_stack_gate'] == 'PASS' and gate['coverage']['status'] == 'PASS'
    assert gate['coverage']['n_discovered'] == gate['coverage']['n_scanned'] == 2
    assert d6['paired_frames'] == 25 and not d6['failures']
    if name == 'c02':
        assert run['platform'] == 'gpu'
        assert d6['verdict'] == 'PASS' and d6['coverage_complete'] and not d6['nonfinite']
        missing = []
        assert not d6['incomplete']
    else:
        summary = json.loads((case / 'RESULT.json').read_text())
        assert d6['verdict'] == summary['verdict'] == 'INCOMPLETE'
        assert summary['bounded_D6'] == 'PASS' and not d6['raw_gpu_nonfinite']
        missing = summary['missing_CPU_fields']
        assert set(missing) == {'QNBCA', 'QNIFA2D', 'QNWFA2D', 'TAOD5502D'}
    jit = next(e for e in gate['entries'] if e['source_kind'] == 'jax_cache')
    assert mapping['blob_sha256'] == jit['blob_sha256']
    assert sha(Path(mapping['pin'])) == mapping['pin_sha256']
    flags, scans, resources = {}, [], {}
    for entry in gate['entries']:
        label = hashlib.sha256(entry['path'].encode()).hexdigest()[:12]
        for index, elf in enumerate(entry['elfs']):
            assert not elf['stack_over256'] and not elf['nonleaf_over1k'] and not elf['callgraph_unknown']
            scan = case / f'e41/cubin_scan_{label}_e{index}.json'
            scans.append({'path': str(scan), 'sha256': sha(scan)})
        for fusion, count in entry['kinput_transpose_over500']:
            old = flags.setdefault(fusion.replace('.', '_'), {'kinput_HLO': count})
            assert old['kinput_HLO'] == count
    for scan in scans:
        for resource in json.loads(Path(scan['path']).read_text()):
            if resource['kernel'] in flags:
                assert resources.setdefault(resource['kernel'], resource['stack']) == resource['stack']
    assert set(flags) == set(resources) == set(mapping['flags'])
    assert len(flags) == (6 if name == 'c02' else 4)
    for fusion, flag in flags.items():
        source = mapping['flags'][fusion]
        assert flag['kinput_HLO'] == source['kinput_HLO']
        flag.update(stack_B=resources[fusion], sources=source['sources'][:8], instructions=source['instructions'])
    namelist_path = case / 'input/namelist.input'
    namelist = namelist_path.read_text()
    options = {key: int(re.search(r'(?im)^\s*' + key + r'\s*=\s*(\d+)', namelist)[1])
               for key in ('mp_physics', 'cu_physics', 'bl_pbl_physics', 'sf_sfclay_physics',
                           'sf_surface_physics', 'ra_lw_physics', 'ra_sw_physics')}
    stack_max = max(elf['max_stack_B'] for entry in gate['entries'] for elf in entry['elfs'])
    assert stack_max == (8 if name == 'c02' else 248)
    row = {'producer': producer, 'case': name, 'producer_case': case.name, 'options': options,
           'source_commit': run.get('source_commit', run.get('source_rev')),
           'gate_path': str(gate_path), 'gate_sha256': sha(gate_path),
           'flags': flags, 'attribution_path': str(mapped), 'program_max_stack_B': stack_max,
           'owner': 't-e41opt disclosure; numerical files retain their owners',
           'D6_verdict': d6['verdict'], 'D6_path': str(d6_path), 'D6_sha256': sha(d6_path),
           'D6_paired_frames': 25, 'bounded_D6': 'PASS', 'missing_CPU_fields': missing,
           'gpu_run_path': str(run_path), 'gpu_run_sha256': sha(run_path),
           'namelist_path': str(namelist_path), 'namelist_sha256': sha(namelist_path),
           'native_resource_scans': scans, 'nonleaf_over1k': 0,
           'runtime_impact': 'UNKNOWN; no exact-artifact per-kernel profile',
           'whole_run_wall_s': run.get('wall_s', run.get('monotonic_wall_s')),
           'timing_scope': 'completed after GPU RESUME; cold qualification, not S1'}
    if name == 'mp28':
        row.update(producer_summary_path=str(case / 'RESULT.json'), producer_summary_sha256=sha(case / 'RESULT.json'),
                   inputs_path=str(case / 'inputs.json'), inputs_sha256=sha(case / 'inputs.json'))
    else:
        row.update(pin_manifest_path=str(case / 'e41/PIN_MAP.json'), pin_manifest_sha256=sha(case / 'e41/PIN_MAP.json'),
                   input_sha256=run['input_sha256'], sampled_process_vram_MiB=run['sampled_process_peak_vram_MiB'])
        assert row['input_sha256']['namelist.input'] == sha(namelist_path)
    inventory['records'].append(row)
    packed = {key: mapping[key] for key in ('pin', 'pin_sha256', 'blob_sha256', 'hlo_sha256')}
    packed.update(case=name, producer=producer, map_path=str(mapped), map_sha256=sha(mapped),
                  flags={fusion: {key: value for key, value in flag.items() if key not in ('call', 'stacks')}
                         for fusion, flag in mapping['flags'].items()})
    for flag in packed['flags'].values():
        flag['sources'] = flag['sources'][:10]
    sources.append(packed)

assert len(inventory['records']) == 16 and sum(len(row['flags']) for row in inventory['records']) == 53
inventory.update(cutoff_utc=datetime.now(timezone.utc).isoformat(),
                 scope='16 source/selector receipts / 53 active E41 flags; final C02 D6 PASS, MP28 incomplete aerosol coverage')
(out / 'INVENTORY.json').write_text(json.dumps(inventory, indent=2, allow_nan=False) + '\n')
(out / 'SOURCE_MAPS.json').write_text(json.dumps(sources, indent=2, allow_nan=False) + '\n')
aliases = []
for alias, producer_case in (('c13', 'mp18_cu5'), ('c19', 'mp28')):
    case = Path('<USER_HOME>/wrf_gpu2_lanes/v-prep') / alias
    reuse_path = case / 'GPU_REUSE.json'
    reuse = json.loads(reuse_path.read_text())
    producer_row = next(row for row in inventory['records'] if row['case'] == producer_case)
    assert reuse['producer_receipt_sha256'] == producer_row['gpu_run_sha256']
    assert reuse['external_e41_sha256'] == producer_row['gate_sha256']
    assert reuse['e41'] == 'FAIL' and reuse['d6'] == 'INCOMPLETE'
    d6_path = case / 'D6_d01.json'
    d6 = json.loads(d6_path.read_text())
    assert d6['verdict'] == 'INCOMPLETE' and d6['paired_frames'] == 25
    assert not d6['failures'] and not d6['nonfinite']
    aliases.append({'case': alias, 'producer': 'v-prep', 'uses_native_case': producer_case,
                    'reuse_path': str(reuse_path), 'reuse_sha256': sha(reuse_path),
                    'D6_path': str(d6_path), 'D6_sha256': sha(d6_path),
                    'scope': 'same actual native executable; not counted as a new realization or added flags'})
(out / 'ALIAS_RECEIPTS.json').write_text(json.dumps(aliases, indent=2) + '\n')
doc = root / 'docs/release/E41_OPT_IN_v0.4.md'
previous = subprocess.check_output(['git', 'show', 'b79a93a80:docs/release/E41_OPT_IN_v0.4.md'], text=True)
tail = previous[previous.index('## Final C03 successor'):]
tail = tail.replace('had 12 configurations and 38 flags; it now has 14 selector/configuration receipts and 43 flags.',
                    'had 12 configurations and 38 flags; the MP18 update increased that to 14 receipts and 43 flags.')
renderer = root / 'proofs/t_e41opt/render_inventory.py'
sys.argv = [str(renderer), str(out / 'INVENTORY.json'), str(doc)]
exec(compile(renderer.read_text(), str(renderer), 'exec'), {'__file__': str(renderer), '__name__': '__main__'})
text = doc.read_text().replace('The compact evidence files are in `proofs/t_e41opt/`.',
    'Active inventory/maps are in `proofs/t_e41opt_updates/c02_mp28/`; earlier sealed packets remain unchanged.')
text = text.replace('C03 Eta/mixed-surface remedies belong to `t-c03rain`; SF1/PBL work belongs\nto `t-c02nan`. Their files were not edited.',
    'C03 E41 disclosure/remedies belong to `t-e41opt` by manager transfer.\nSF1/PBL numerical files retain their owners; they were not edited.')
text = text.replace('`SOURCE_MAPS.json` and `CPU_NEUTRALITY.json` preserve the compact evidence.\nFull maps remain under `<USER_HOME>/wrf_gpu2_lanes/t-e41opt/EO*`.\nProducer references and artifacts were not changed. Disposable lane\nsnapshots were removed; no lane JAX/native cache is retained.',
    '`SOURCE_MAPS.json` preserves the current compact attribution evidence;\n'
    '`proofs/t_e41opt/CPU_NEUTRALITY.json` retains the original MP24/26 byte checks.\n'
    'Full maps reside at the exact paths recorded in the inventory and source maps,\n'
    'including `<DATA_ROOT>/wrf_gpu2/scratch/`. Producer artifacts were read-only.\n'
    'Separate firewall candidates and their pre-screen scratch have their own\n'
    'receipts; no candidate is admitted or included in this disclosure change.')
text += '\n' + tail + '\n## Final C02 and MP28 successor\n\n'
text += ('Final C02 is the `5c0e7c11e` / `TCN23_c02` corrected GPU arm, with original\n'
         'CPU-WRF D6 PASS over all 25 frames and complete coverage. E41 still FAILs:\n'
         'six kInput fusions have 570/549/661/659/586/632 optimized HLO instructions.\n'
         'Its maximum STACK is 8 B (PASS), and nonleaf-over-1k count is zero. The\n'
         'earlier nonfinite C02 arm and its 752 B STACK frame remain historical;\n'
         'their properties are not transferred to the corrected executable.\n\n'
         'MP28 (`8afb83d7d`) retains bounded D6 PASS for 25 frames and zero raw\n'
         'nonfinite values, but full coverage is INCOMPLETE: `QNBCA`, `QNIFA2D`,\n'
         '`QNWFA2D`, and `TAOD5502D` are missing. Its four kInput fusions have\n'
         '833/827/1175/739 HLO instructions. Maximum STACK 248 B and nonleaf zero\n'
         'pass. Both arms completed after GPU RESUME; neither is an S1 benchmark.\n\n'
         'The active inventory now has 16 source/selector receipts and 53 flags.\n'
         'No SF1/PBL numerical code was edited. No E41 waiver or runtime-cost\n'
         'claim follows from source attribution, and no firewall is shipped.\n')
text += ('\nV-prep C13 and C19 reuse these exact MP18+CU5 and MP28 native receipts,\n'
         'respectively. Their own matched original CPU-WRF rescoring is also\n'
         'INCOMPLETE over 25 frames with no bounded failures or nonfinite values.\n'
         '`ALIAS_RECEIPTS.json` binds both reuse decisions and D6 reports. They\n'
         'retain the corresponding E41 FAILs and add no new native realization\n'
         'or duplicated flags to the count above.\n')
text += ('\n## Separate MP24 firewall pre-screen disposition\n\n'
         'The separate `lane/t-e41opt-firewall` candidate is NOT ADMITTED. Its\n'
         'captured-RTX-target CPU native pre-screen reproduced baseline STACK\n'
         '1,216 B, then measured candidate STACK 9,616 B and 20 nonleaf-over-1k\n'
         'findings. Cold compile elapsed was 237.250 s versus 740.405 s; native\n'
         'size was 23,837,107 B versus 32,000,643 B. Both static XLA allocation\n'
         'reports gave 271,095,137 B; this is not peak VRAM. These are measured\n'
         'CPU pre-screen properties, not GPU throughput or actual MP options\n'
         'replay: the paired control used captured CAM options with autotune zero.\n'
         'Semantic options were byte-equal after removing two private path fields.\n\n'
         'The negative packet is retained only on that candidate branch at\n'
         '`39abddfd4`, in `proofs/t_e41opt_firewall/native_prescreen/`. No GPU arm\n'
         'was queued and no MP26 native arm was spent on this failed combined\n'
         'candidate. Its caches and detached snapshot were cleaned after E41.\n'
         'The separate C03 Lin/GSFC candidate remains CPU-neutrality-only and\n'
         'NOT ADMITTED; native/default/actual-GPU resource gates remain pending.\n')
doc.write_text(text)
seal = {'verdict': 'DISCLOSURE_UPDATE_READY', 'parent_packet_commit': 'b79a93a80',
        'parent_qualification_sha256': sha(prior / 'QUALIFICATION.json'),
        'active_receipts': 16, 'active_native_flags': 53, 'new_GPU_runs': 0,
        'production_code_changed': False, 'E41': 'all failures retained; no waiver',
        'files': {str(path.relative_to(root)): sha(path)
                  for path in (out / 'INVENTORY.json', out / 'SOURCE_MAPS.json', out / 'ALIAS_RECEIPTS.json', Path(__file__), doc)}}
(out / 'QUALIFICATION.json').write_text(json.dumps(seal, indent=2) + '\n')
print('PASS: 16 receipts / 53 E41 flags; final C02 STACK8 and MP28 STACK248 pass independently')
