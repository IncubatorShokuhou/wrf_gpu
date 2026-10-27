"""Assemble reviewable gate results only after every required arm completes."""
import argparse
import ast
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import re

p = argparse.ArgumentParser()
p.add_argument('--source', required=True, type=Path)
p.add_argument('--artifacts', required=True, type=Path)
a = p.parse_args()
root, lane = a.source, a.artifacts
dest = root / 'proofs/w44_e84fix/receipts'
dest.mkdir(parents=True, exist_ok=True)
C = 'src/gpuwrf/coupling/physics_couplers.py'
O = 'src/gpuwrf/physics/wrf_cam_ozone.py'
def git(*args):
    return subprocess.check_output(['git', '-C', str(root), *args])
changed = git('diff', '--name-only', '8f31d181ba', 'HEAD', '--', 'src').decode().splitlines()
assert set(changed) == {C, O}, changed
scope = {}
for file in changed:
    patch = git('diff', '8f31d181ba', 'HEAD', '--', file)
    patch_id = subprocess.check_output(['git', 'patch-id', '--stable'], input=patch).decode().split()[0]
    source = (root / file).read_bytes()
    measured = git('show', 'cb98b44524:' + file)
    if file == C:
        assert source == measured
        reuse = 'BYTE identical to measured cb98b44524'
    else:
        trees = [ast.parse(value) for value in (source, measured)]
        for tree in trees:
            helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_real4_div_rn')
            helper.body[0].value.value = '<DOCSTRING>'
        assert ast.dump(trees[0]) == ast.dump(trees[1])
        reuse = 'AST identical to measured cb98b44524 except one helper docstring; source locations unchanged'
    scope[file] = dict(sha256=hashlib.sha256(source).hexdigest(), per_file_patch_id=patch_id,
                       measured_numerical_reuse=reuse)

def copy_json(source, name):
    value = json.loads(source.read_text())
    (dest / name).write_text(json.dumps(value, indent=2) + '\n')
    return value
component = copy_json(lane / 'E8407/component.json', 'COMPONENT.json')
quotient = copy_json(lane / 'E8402/quotient.json', 'OPERAND_RANGES.json')
mutants = copy_json(lane / 'E8405b/mutations.json', 'MUTATIONS.json')
assert component['original_solar_rows_frozen_bounds_PASS'] == 504
assert all(v['all_fields_BYTE'] for v in component['base_product_BYTE'].values())
assert sum(r['clocks'] for r in component['cohorts']) == 677
assert all(r['protected_quotient_max_ULP'] == 0 and not any(r['predicate_flips'])
           and r['ozone_profile_max_ULP'] <= 1 for r in component['cohorts'])
assert quotient['total_values'] == 14398452 and quotient['exact']
assert mutants['genuine_AssertionError_kills'] == 16 and mutants['setup_errors'] == 0
candidates = {
    'Swiss d01': 'E8408_swiss_d01', 'PROD d01': 'E8404_prod_d01',
    'PROD d02': 'E8406_prod_d02', 'WN3 d01': 'E8409_wn3_d01',
    'WN3 fused d02+d03': 'E8410b_wn3_fused_d02',
}
baselines = {'PROD d01': 'E8411_base_prod_d01', 'PROD d02': 'E8412_base_prod_d02',
             'WN3 d01': 'E8413_base_wn3_d01', 'WN3 fused d02+d03': 'E8414_base_wn3_fused_d02'}
programs = {}
for name, directory in candidates.items():
    receipt = copy_json(lane / directory / 'receipt.json', directory + '.json')
    rn = copy_json(lane / directory / 'RN_AUDIT.json', directory + '_RN.json')
    assert receipt['phase'] == 'done' and receipt['native_gate'] == 'PASS'
    assert receipt['max_stack_B'] <= 256 and not receipt['kinput_transpose_over500']
    assert rn['PTX_site_proofs'] and all(rn['family_counts'].get(f, 0) for f in rn['required_families'])
    site_kernels = {}
    for proof in rn['PTX_site_proofs']:
        for family in proof['families']:
            entry = site_kernels.setdefault(family, dict(scalar=set(), field=set()))
            kind = 'scalar' if not proof['shape'] else 'field'
            entry[kind].update(p['kernel'] for p in proof['PTX'])
    for alias in rn['CSE_aliases']:
        site_kernels[alias['family']] = site_kernels[alias['shared_with']]
    site_kernels = {f:{k:sorted(v) for k,v in s.items()} for f,s in site_kernels.items()}
    scalar_fusions = {}
    for item in rn['RN_fusion_inventory']:
        if item['shape']:
            continue
        for family in item['families']:
            scalar_fusions.setdefault(family, set()).update(b['fusion'] for b in item['fusion_bindings'])
    for alias in rn['CSE_aliases']:
        scalar_fusions[alias['family']] = scalar_fusions[alias['shared_with']]
    program = dict(candidate_directory=directory, native_kernels=receipt['kernels'],
        fusions=receipt['n_fusions'], traced_Pallas=receipt['traced_pallas_calls'],
        compiled_memory=receipt['memory'], E41='PASS', max_stack_B=receipt['max_stack_B'],
        RN_f64_divides=rn['f64_divides'], RN_f64_operand_converts=rn['f64_operand_converts'],
        RN_f32_result_converts=rn['f32_result_converts'], RN_f64_divide_elements=rn['f64_divide_elements'],
        per_site_kernel_bindings=site_kernels, full_constant_proto_sha256=rn['full_constant_proto_sha256'],
        per_site_scalar_fusions={f:dict(count=len(v), fusions=sorted(v)) for f,v in scalar_fusions.items()},
        static_scheduled_kernel_references=rn['static_scheduled_kernel_references'],
        candidate_compile_s=receipt['compile_s'])
    if name in baselines:
        base_dir = baselines[name]
        base = copy_json(lane / base_dir / 'receipt.json', base_dir + '.json')
        assert base['phase'] == 'done' and base['native_gate'] == 'PASS'
        for key in ('harness_sha256', 'input_sha256', 'target_sha256', 'static_kwargs', 'traced_pallas_calls'):
            assert base.get(key) == receipt.get(key), (name, key, base.get(key), receipt.get(key))
        for key in ('argument_size_in_bytes', 'output_size_in_bytes', 'alias_size_in_bytes'):
            assert base['memory'][key] == receipt['memory'][key], (name, key)
        kernel_delta = receipt['kernels'] - base['kernels']
        temp_delta = receipt['memory']['temp_size_in_bytes'] - base['memory']['temp_size_in_bytes']
        base_refs = 0
        for f in list((lane/base_dir/'dump').glob('*.thunk_sequence.txt')) + list((lane/base_dir/'dump').glob('*.thunk_sequence.txt.gz')):
            contents = gzip.decompress(f.read_bytes()).decode() if f.suffix == '.gz' else f.read_text()
            base_refs += len(re.findall(r'kernel=([^,]+), profile_annotation=([^,]+),', contents))
        refs_delta = rn['static_scheduled_kernel_references'] - base_refs
        charged_launches = max(kernel_delta, refs_delta, 0)
        program['matched_baseline'] = dict(directory=base_dir, native_kernels=base['kernels'],
            native_kernel_delta=kernel_delta, native_kernel_delta_pct=100*kernel_delta/base['kernels'],
            temp_bytes=base['memory']['temp_size_in_bytes'], temp_delta_bytes=temp_delta,
            temp_delta_pct=100*temp_delta/base['memory']['temp_size_in_bytes'],
            fusions=base['n_fusions'], fusions_delta=receipt['n_fusions']-base['n_fusions'],
            argument_output_alias_bytes_exact=True, baseline_compile_s=base['compile_s'],
            same_tool_options_case=True,
            baseline_static_kernel_references=base_refs, static_kernel_reference_delta=refs_delta,
            launch_cost_scenario_I_us_per_radiation_call=[charged_launches*2, charged_launches*10],
            launch_cost_scenario_assumption='[I] 2–10 us per added static scheduled kernel reference (or unique native kernel if larger), charged once per refresh; scenario, NOT measured GPU time or a bound on field-divide time')
    programs[name] = program
packet = dict(source_commit=git('rev-parse', 'HEAD').decode().strip(),
    source_tree=git('rev-parse', 'HEAD:src').decode().strip(), measured_numeric_source='cb98b44524',
    baseline_source='8f31d181ba / src70fa838467', source_scope=scope,
    gate1='PASS manager amendment2026-10-25T18:09:44Z', gate2='PASS <=1ULP original-WRF profiles',
    gate3='PASS CUDA CompileOnly only; actual composed RC GPU remains required',
    gate4='PASS14deletion+2semantic genuineASSERT/0setup', gate5='PASS scoped source/E41/matched cost disclosure',
    physical_backend='cpu', GPU_executed=False, programs=programs,
    open_release_evidence=['Actual composed-RC GPU E41/clock check', 'RW1–RW4 including original-CPU-WRF72h',
        'GPU timing, VRAM and readiness; scenario estimates do not replace measurement'],
    known_issues='proofs/w44_e84fix/KNOWN_ISSUES_E84.md', S2_inventory='proofs/w44_e84fix/S2_ISLANDS.md')
(dest / 'GATE_PACKET.json').write_text(json.dumps(packet, indent=2)+'\n')
print(json.dumps({n: dict(kernels=p['native_kernels'], RN_divides=p['RN_f64_divides'],
    baseline=p.get('matched_baseline')) for n,p in programs.items()}, indent=2))
