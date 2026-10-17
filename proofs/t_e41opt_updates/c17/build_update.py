"""Append the exact C17 MP24/CU93 executable; preserve earlier packets."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

root = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent
prior = root / 'proofs/t_e41opt_updates/c02_mp28'
case = Path('<USER_HOME>/wrf_gpu2_lanes/v-prep/c17')
mapped = Path('<DATA_ROOT>/wrf_gpu2/scratch/t-e41opt/EO36_c17/source_map.json')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


inventory = json.loads((prior / 'INVENTORY.json').read_text())
sources = json.loads((prior / 'SOURCE_MAPS.json').read_text())
mapping = json.loads(mapped.read_text())
gate = json.loads((case / 'e41/wave_gate.json').read_text())
run = json.loads((case / 'gpu_run.json').read_text())
inputs = json.loads((case / 'inputs.json').read_text())
d6 = json.loads((case / 'D6_d01.json').read_text())
assert gate['e41_gate'] == 'FAIL' and gate['hard_stack_gate'] == 'PASS'
assert gate['coverage']['status'] == 'PASS'
assert gate['coverage']['n_discovered'] == gate['coverage']['n_scanned'] == 2
assert run['platform'] == 'gpu' and run['rc'] == 0
assert d6['verdict'] == 'PASS' and d6['coverage_complete'] and d6['paired_frames'] == 25
assert not d6['failures'] and not d6['nonfinite'] and not d6['incomplete']
assert run['actual_input_sha256'] == inputs['input_sha256']
assert sha(case / 'input/namelist.input') == inputs['input_sha256']['namelist.input']
jit = next(entry for entry in gate['entries'] if entry['source_kind'] == 'jax_cache')
assert mapping['blob_sha256'] == jit['blob_sha256']
assert sha(Path(mapping['pin'])) == mapping['pin_sha256']
name = 'input_compare_reduce_select_fusion'
assert set(mapping['flags']) == {name} and mapping['flags'][name]['kinput_HLO'] == 844
scans, resources = [], []
for entry in gate['entries']:
    assert entry['kinput_transpose_over500'] == [[name, 844]]
    label = hashlib.sha256(entry['path'].encode()).hexdigest()[:12]
    for index, elf in enumerate(entry['elfs']):
        assert elf['max_stack_B'] == 32 and not elf['stack_over256']
        assert not elf['nonleaf_over1k'] and not elf['callgraph_unknown']
        path = case / f'e41/cubin_scan_{label}_e{index}.json'
        scans.append({'path': str(path), 'sha256': sha(path)})
        resources.append(next(row['stack'] for row in json.loads(path.read_text()) if row['kernel'] == name))
assert len(resources) == 2 and resources[0] == resources[1] and resources[0] <= 32
source = mapping['flags'][name]
row = {'producer': 'v-prep', 'case': 'c17', 'options': inputs['options'],
       'source_commit': run['source_commit'], 'program_max_stack_B': 32,
       'flags': {name: {'kinput_HLO': 844, 'stack_B': resources[0],
                        'sources': source['sources'][:8], 'instructions': source['instructions']}},
       'attribution_path': str(mapped), 'owner': 't-e41opt disclosure',
       'gate_path': str(case / 'e41/wave_gate.json'), 'gate_sha256': sha(case / 'e41/wave_gate.json'),
       'inputs_path': str(case / 'inputs.json'), 'inputs_sha256': sha(case / 'inputs.json'),
       'D6_verdict': 'PASS', 'D6_paired_frames': 25, 'D6_path': str(case / 'D6_d01.json'),
       'D6_sha256': sha(case / 'D6_d01.json'), 'gpu_run_path': str(case / 'gpu_run.json'),
       'gpu_run_sha256': sha(case / 'gpu_run.json'), 'native_resource_scans': scans,
       'nonleaf_over1k': 0, 'whole_run_wall_s': run['monotonic_wall_s'],
       'sampled_process_vram_MiB': run['sampled_process_vram_mib_peak'],
       'runtime_impact': 'UNKNOWN; no exact-artifact per-kernel profile',
       'timing_scope': 'concurrent cold qualification, completed before 17:30Z thrashing; not S1'}
inventory['records'].append(row)
inventory.update(cutoff_utc=datetime.now(timezone.utc).isoformat(),
                 scope='17 source/selector receipts / 54 active E41 flags; C17 MP24/CU93 D6 PASS, kInput 844 FAIL')
packed = {key: mapping[key] for key in ('pin', 'pin_sha256', 'blob_sha256', 'hlo_sha256')}
packed.update(case='c17', producer='v-prep', map_path=str(mapped), map_sha256=sha(mapped),
              flags={name: {key: value for key, value in source.items() if key not in ('call', 'stacks')}})
packed['flags'][name]['sources'] = packed['flags'][name]['sources'][:10]
sources.append(packed)
assert len(inventory['records']) == len(sources) == 17
assert sum(len(row['flags']) for row in inventory['records']) == 54
(out / 'INVENTORY.json').write_text(json.dumps(inventory, indent=2, allow_nan=False) + '\n')
(out / 'SOURCE_MAPS.json').write_text(json.dumps(sources, indent=2, allow_nan=False) + '\n')
doc = root / 'docs/release/E41_OPT_IN_v0.4.md'
text = subprocess.check_output(['git', 'show', '92e4ec473:docs/release/E41_OPT_IN_v0.4.md'], text=True)
text = re.sub(r'Inventory cutoff: \*\*.*?\*\*', 'Inventory cutoff: **' + inventory['cutoff_utc'] + '**', text, count=1)
options = '/'.join(str(inputs['options'][key]) for key in ('mp_physics', 'cu_physics', 'bl_pbl_physics',
                         'sf_sfclay_physics', 'sf_surface_physics', 'ra_lw_physics', 'ra_sw_physics'))
config = f'| c17 | {options} | `{run["source_commit"][:10]}` | PASS | 25 | 32 | 1 |\n'
text = text.replace('\n## Support-matrix disclosure rows', config + '\n## Support-matrix disclosure rows', 1)
sites = '; '.join(f'`{Path(s["file"]).name}:{s["line"]}` ({s["function"]})' for s in source['sources'][:2])
flag = f'| c17 | `{name}` | {resources[0]} | 844 | {sites} |\n'
text = text.replace('\n## Remedy disposition and resource consequences', flag + '\n## Remedy disposition and resource consequences', 1)
text = text.replace('The active inventory now has 16 source/selector receipts and 53 flags.',
                    'The C02/MP28 update brought the inventory to 16 source/selector receipts and 53 flags.')
text = text.replace('Active inventory/maps are in `proofs/t_e41opt_updates/c02_mp28/`; earlier sealed packets remain unchanged.',
                    'Active inventory/maps are in `proofs/t_e41opt_updates/c17/`; earlier sealed packets remain unchanged.')
text = text.replace('`ALIAS_RECEIPTS.json`', '`proofs/t_e41opt_updates/c02_mp28/ALIAS_RECEIPTS.json`')
text += ('\n## C17 MP24/CU93 successor\n\n'
         'C17 is the actual `3fd61797e` MP24/CU93 executable: full original\n'
         'CPU-WRF D6 PASS over 25 frames, zero bounded failures and nonfinite\n'
         'values. E41 FAIL is solely kInput 844 HLO in\n'
         '`input_compare_reduce_select_fusion`; the rich source stack resolves\n'
         'to WSM7 `wsm7_run` / `_wsm7_minor_loop`. Program maximum STACK 32 B\n'
         'and nonleaf zero pass. This distinct source/selector executable does\n'
         'not inherit the 1,216 B frame of the earlier MP24/CU0 program. No\n'
         'firewall candidate or native pre-screen property is substituted for\n'
         'this actual GPU result. Runtime impact is unknown.\n\n'
         'The active inventory has 17 source/selector receipts and 54 E41 flags.\n'
         'No production code or producer reference was changed. The one new\n'
         'CPU attribution arm ran under a 6 GiB memory cap with zero swap,\n'
         'on lane cores with JAX CPU only; it created no compile cache or snapshot.\n')
doc.write_text(text)
seal = {'verdict': 'DISCLOSURE_UPDATE_READY', 'parent_packet_commit': '92e4ec473',
        'parent_qualification_sha256': sha(prior / 'QUALIFICATION.json'),
        'active_receipts': 17, 'active_native_flags': 54, 'production_code_changed': False,
        'new_GPU_runs': 0, 'E41': 'all failures retained; no waiver',
        'CPU_resource_scope': 'systemd user scope MemoryMax=6G MemorySwapMax=0; CPU12,13,28,29 nice19 timeout300; JAXcpu',
        'files': {str(path.relative_to(root)): sha(path)
                  for path in (out / 'INVENTORY.json', out / 'SOURCE_MAPS.json', Path(__file__), doc)}}
(out / 'QUALIFICATION.json').write_text(json.dumps(seal, indent=2) + '\n')
print('PASS: 17 receipts / 54 flags; C17 D6PASS25 STACK32/nonleaf0 PASS, kInput844 FAIL')
