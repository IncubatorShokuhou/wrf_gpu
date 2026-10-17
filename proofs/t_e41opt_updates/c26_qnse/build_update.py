"""Add actual C26 QNSE Pallas STACK flags with paid-source provenance."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

root = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent
prior = root / 'proofs/t_e41opt_updates/nssl_round3'
reviewed = '17943b8b2d3fb56a91fa6c175033900e6fac1fed'
paid = '4dba72ce4c2998d02c67be1abc1d5e3a5a689be9'
case = Path('<USER_HOME>/wrf_gpu2_lanes/t-qnsecompile/QC05/c26')
map_path = Path('<DATA_ROOT>/wrf_gpu2/scratch/t-e41opt/EO46_c26_qnse/source_map.json')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_bytes(path, commit=reviewed):
    return subprocess.check_output(['git', 'show', commit + ':' + path])


def receipt(name):
    return json.loads(git_bytes('proofs/t_qnsecompile/receipts/' + name))


producer_raw = git_bytes('proofs/t_qnsecompile/QUALIFICATION.json')
producer = json.loads(producer_raw)
assert producer['paid_source'] == paid
assert producer['d6']['verdict'] == 'PASS' and producer['bounded_fields_pass'] == 62
assert producer['full_inventory_cpu_fields'] == 360
for name, expected in producer['receipts'].items():
    assert hashlib.sha256(git_bytes('proofs/t_qnsecompile/receipts/' + name)).hexdigest() == expected, name
assert not subprocess.check_output(['git', 'diff', paid, reviewed, '--', 'src/gpuwrf'])
gate = receipt('ACTUAL_E41.json')
resources = receipt('ACTUAL_RESOURCES.json')
run = receipt('GPU_RUN.json')
raw_evidence = receipt('RAW_EVIDENCE.json')
for evidence in raw_evidence.values():
    assert sha(Path(evidence['path'])) == evidence['sha256']
d6_path = Path(raw_evidence['D6_d01.json']['path'])
d6 = json.loads(d6_path.read_text())
assert run['platform'] == 'gpu' and run['rc'] == 0 and run['source_commit'] == paid
assert d6['verdict'] == 'PASS' and d6['coverage_complete'] and d6['paired_frames'] == 25
assert not d6['failures'] and not d6['nonfinite']
assert all(frame['cpu_field_count'] == 360 for frame in d6['by_frame'])
assert gate['e41_gate'] == gate['hard_stack_gate'] == 'FAIL'
assert gate['coverage']['status'] == 'PASS'
assert gate['coverage']['n_discovered'] == gate['coverage']['n_scanned'] == 2
assert len({elf['sha256'] for entry in gate['entries'] for elf in entry['elfs']}) == 1
assert set(resources['actual_elfs']) == {elf['sha256'] for entry in gate['entries'] for elf in entry['elfs']}
for entry in gate['entries']:
    assert not entry['kinput_transpose_over500']
    for elf in entry['elfs']:
        assert dict(elf['stack_over256']) == {'qnse_pbl_columns': 7232, 'qnse_mf_columns': 2952}
        assert not elf['nonleaf_over1k'] and not elf['callgraph_unknown']
for elf in resources['actual_elfs'].values():
    assert sha(Path(elf['path'])) in resources['actual_elfs']
mapping = json.loads(map_path.read_text())
assert mapping['paid_source'] == paid and set(mapping['flags']) == {'qnse_pbl_columns', 'qnse_mf_columns'}
kernel_path = 'src/gpuwrf/kernels/phys_qnse_columns.py'
kernel_source = git_bytes(kernel_path, paid).decode()
kernel_ast = ast.parse(kernel_source)
imports = {alias.name: node.module for node in kernel_ast.body if isinstance(node, ast.ImportFrom)
           for alias in node.names}
assert imports['qnsepbl_column'] == 'gpuwrf.physics.pbl_qnse'
assert imports['mfshconvpbl_column'] == 'gpuwrf.physics.qnse_edmf'
driver = next(node for node in kernel_ast.body if isinstance(node, ast.FunctionDef) and node.name == 'qnse_columns')
binding = next(node for node in ast.walk(driver) if isinstance(node, ast.Dict)
               and any(isinstance(key, ast.Constant) and key.value == 'pbl' for key in node.keys))
fn_by_scheme = {key.value: value.elts[-1].id for key, value in zip(binding.keys, binding.values)}
assert fn_by_scheme['pbl'] == 'qnsepbl_column' and fn_by_scheme['mf'] == 'mfshconvpbl_column'
call = next(node for node in ast.walk(driver) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and node.func.attr == 'pallas_call')
name_expression = next(keyword.value for keyword in call.keywords if keyword.arg == 'name')
assert ast.unparse(name_expression) == "'qnse_' + scheme + '_columns'"
bindings = []
flags = {}
native_rows = {row['kernel']: row for row in resources['qnse_rows']}
for scheme, source_path, function in (
    ('pbl', 'src/gpuwrf/physics/pbl_qnse.py', 'qnsepbl_column'),
    ('mf', 'src/gpuwrf/physics/qnse_edmf.py', 'mfshconvpbl_column'),
):
    kernel = 'qnse_' + scheme + '_columns'
    source = git_bytes(source_path, paid)
    node = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == function)
    binding_proof = {'scheme': scheme, 'native_kernel': kernel, 'function': function,
                     'file': source_path, 'line': node.lineno,
                     'git_blob': subprocess.check_output(['git', 'rev-parse', paid + ':' + source_path], text=True).strip(),
                     'file_sha256': hashlib.sha256(source).hexdigest(),
                     'attribution_kind': 'paid-source AST driver binding, not a kernel-body HLO stack frame'}
    bindings.append(binding_proof)
    mapped = mapping['flags'][kernel]
    resource = native_rows[kernel]
    assert mapped['stack_B'] == resource['stack'] and not resource['calls'] and not resource['nonleaf_over1k']
    assert len(mapped['custom_call_sites']) == 1
    assert mapped['sources'][0]['file'].endswith('/kernels/phys_qnse_columns.py')
    assert mapped['sources'][0]['line'] == call.lineno
    flags[kernel] = {'stack_B': resource['stack'], 'kind': 'Pallas native kernel', 'instructions': None,
                     'sources': mapped['sources'] + [binding_proof],
                     'source_measure': 'custom-call site plus verified AST binding; no HLO body count',
                     'native_registers': resource['regs'], 'native_SASS_instructions': resource['sass'],
                     'native_calls': 0}
inventory = json.loads((prior / 'INVENTORY.json').read_text())
sources = json.loads((prior / 'SOURCE_MAPS.json').read_text())
namelist_path = case / 'input/namelist.input'
namelist = namelist_path.read_text()
assert sha(namelist_path) == run['actual_input_sha256']['namelist.input']
options = {key: int(re.search(r'(?im)^\s*' + key + r'\s*=\s*(\d+)', namelist)[1])
           for key in ('mp_physics', 'cu_physics', 'bl_pbl_physics', 'sf_sfclay_physics',
                       'sf_surface_physics', 'ra_lw_physics', 'ra_sw_physics')}
row = {'case': 'c26-qnse', 'producer': 't-qnsecompile', 'producer_case': 'c26',
       'source_commit': paid, 'reviewed_packet_commit': reviewed, 'options': options,
       'producer_qualification_sha256': hashlib.sha256(producer_raw).hexdigest(),
       'flags': flags, 'program_max_stack_B': 7232, 'nonleaf_over1k': 0, 'kinput_transpose_over500': 0,
       'gate_receipt_path_in_commit': 'proofs/t_qnsecompile/receipts/ACTUAL_E41.json',
       'gate_sha256': producer['receipts']['ACTUAL_E41.json'],
       'resources_receipt_sha256': producer['receipts']['ACTUAL_RESOURCES.json'],
       'native_elfs': resources['actual_elfs'], 'attribution_path': str(map_path),
       'D6_verdict': 'PASS', 'D6_paired_frames': 25, 'bounded_D6_fields': 62, 'CPU_inventory_fields': 360,
       'D6_path': str(d6_path), 'D6_sha256': sha(d6_path), 'raw_evidence': raw_evidence,
       'gpu_run_receipt_sha256': producer['receipts']['GPU_RUN.json'],
       'namelist_path': str(namelist_path), 'namelist_sha256': sha(namelist_path),
       'input_sha256': run['actual_input_sha256'], 'pins': receipt('PINS.json'),
       'whole_run_wall_s': run['monotonic_wall_s'], 'sampled_process_vram_MiB': run['sampled_process_vram_mib_peak'],
       'runtime_impact': 'UNKNOWN; no matched native runtime baseline or per-kernel profile',
       'timing_scope': 'cold/shared qualification, not S1', 'F2_scope': 'negative7854866/guardrepairs170344 remain OPEN'}
inventory['records'].append(row)
packed = {key: mapping[key] for key in ('pin', 'pin_sha256', 'blob_sha256', 'hlo_sha256')}
packed.update(case='c26-qnse', producer='t-qnsecompile', map_path=str(map_path), map_sha256=sha(map_path),
              flags=mapping['flags'], paid_source_bindings=bindings)
sources.append(packed)
assert len(inventory['records']) == len(sources) == 23
assert sum(len(record['flags']) for record in inventory['records']) == 79
assert sum(record.get('nonleaf_over1k', 0) for record in inventory['records']) == 3
inventory.update(cutoff_utc=datetime.now(timezone.utc).isoformat(),
                 scope='23 source/selector receipts / 79 flagged kernels or fusions + 3 nonleaf callee findings; C26 Pallas native STACK FAIL')
(out / 'INVENTORY.json').write_text(json.dumps(inventory, indent=2, allow_nan=False) + '\n')
(out / 'SOURCE_MAPS.json').write_text(json.dumps(sources, indent=2, allow_nan=False) + '\n')
doc = root / 'docs/release/E41_OPT_IN_v0.4.md'
text = git_bytes('docs/release/E41_OPT_IN_v0.4.md', 'fa5902d9b').decode()
text = re.sub(r'Inventory cutoff: \*\*.*?\*\*', 'Inventory cutoff: **' + inventory['cutoff_utc'] + '**', text, count=1)
text = text.replace('| Flagged fusions |', '| Flagged kernels/fusions |').replace('| Case | Fusion |', '| Case | Kernel/fusion |')
option_text = '/'.join(str(options[key]) for key in ('mp_physics', 'cu_physics', 'bl_pbl_physics',
                            'sf_sfclay_physics', 'sf_surface_physics', 'ra_lw_physics', 'ra_sw_physics'))
text = text.replace('\n## Support-matrix disclosure rows',
                    f'| c26-qnse | {option_text} | `{paid[:10]}` | PASS | 25 | 7232 | 2 |\n\n## Support-matrix disclosure rows', 1)
flag_rows = []
for kernel, flag in flags.items():
    sites = '; '.join(f'`{Path(source["file"]).name}:{source["line"]}` ({source["function"]})' for source in flag['sources'])
    flag_rows.append(f'| c26-qnse | `{kernel}` | {flag["stack_B"]} | — | {sites} | — |')
text = text.replace('\n## Remedy disposition and resource consequences',
                    '\n'.join(flag_rows) + '\n\n## Remedy disposition and resource consequences', 1)
text = text.replace('sites are recovered from rich HLO stack frames, not inferred from names.',
                    'Fusion sites come from rich HLO stack frames. Opaque Pallas sites show\nactual custom-call stacks and explicitly verified paid-source driver bindings.')
text = text.replace('Active inventory/maps are in `proofs/t_e41opt_updates/nssl_round3/`;',
                    'Active inventory/maps are in `proofs/t_e41opt_updates/c26_qnse/`;')
text = text.replace('Current scope: 22 source/selector receipts, 77 flagged fusions and',
                    'The NSSL round3 update covered 22 source/selector receipts, 77 flagged fusions and')
text = text.replace('`STACK_COMPARISON.json`', '`proofs/t_e41opt_updates/nssl_round3/STACK_COMPARISON.json`')
text += ('\n## C26 QNSE actual native resources\n\n'
         'The reviewed `t-qnsecompile` packet is `17943b8b2`; the actual paid GPU\n'
         'source is `4dba72ce4`, with the entire model tree byte-identical between\n'
         'them. C26 has full original CPU-WRF D6 PASS over 25 frames, 62 bounded\n'
         'fields and 360 fields in the complete CPU inventory: zero bounded\n'
         'misses, nonfinite values or missing fields. No numerical gate was repeated.\n\n'
         'E41 remains FAIL on the actual paid JIT+AOT executable (same native ELF):\n'
         '`qnse_pbl_columns` STACK **7,232 B**, `qnse_mf_columns` STACK **2,952 B**.\n'
         'QNSE surface STACK is 0 B. Nonleaf and kInput-over-500 counts are zero.\n'
         'These are opaque Pallas native kernels, so their STACK values are not\n'
         'optimized HLO fusion instruction counts. The actual rich-HLO custom calls\n'
         'map both symbols to `phys_qnse_columns.qnse_columns`; the paid-source AST\n'
         'binds PBL to `pbl_qnse.qnsepbl_column` and MF to\n'
         '`qnse_edmf.mfshconvpbl_column`. The table distinguishes this source\n'
         'binding from a kernel-body HLO stack frame. Standalone pre-screen\n'
         'STACK 9,608/2,584 B is not substituted for the actual values above.\n\n'
         'The compiler-stall repair is independently qualified; its actual\n'
         'resource failures remain experimental opt-in disclosures. Cold/shared\n'
         '24h wall 519.776 s and sampled peak VRAM 7,054 MiB are qualification\n'
         'observations, not S1 or isolated kernel overhead. Runtime spill cost and\n'
         'occupancy impact remain unknown without a matched native profile.\n'
         'F2 negative-event and guard-repair justification remains OPEN.\n\n'
         'Current scope: 23 source/selector receipts and 79 flagged native kernels\n'
         'or HLO fusions, with three earlier C15 nonleaf callee findings. Default\n'
         'E41 limits remain unchanged. This update changes no production code and\n'
         'runs no GPU, forecast, native compile or repeated numerical validation.\n')
doc.write_text(text)
seal = {'verdict': 'DISCLOSURE_UPDATE_READY', 'parent_packet_commit': 'fa5902d9b',
        'parent_qualification_sha256': sha(prior / 'QUALIFICATION.json'),
        'producer_packet_commit': reviewed, 'paid_source_commit': paid,
        'active_receipts': 23, 'flagged_kernels_or_fusions': 79,
        'opaque_Pallas_flags': 2, 'nonleaf_callee_findings': 3,
        'new_GPU_runs': 0, 'new_forecasts': 0, 'production_code_changed': False,
        'E41': 'actual STACK FAIL retained, no waiver',
        'files': {str(path.relative_to(root)): sha(path) for path in (
            out / 'INVENTORY.json', out / 'SOURCE_MAPS.json', out / 'map_custom_calls.py', Path(__file__), doc)}}
(out / 'QUALIFICATION.json').write_text(json.dumps(seal, indent=2) + '\n')
print('PASS C26 actual PallasSTACK7232/2952,D6PASS25/62/360;23receipts79flags;prior22records/maps unchanged')
