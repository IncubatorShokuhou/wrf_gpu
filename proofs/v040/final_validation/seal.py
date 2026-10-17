"""Publish compact preparation manifests; never infer a completed GPU gate."""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
from prepare import ROOT, SLOTS, sha, write_json


def verified_namelist_sha(path, expected, context):
    actual = sha(path)
    assert actual == expected, f'{context}: current namelist differs from sealed CPU input'
    return actual


def implemented_catalog(source):
    path = source / 'src/gpuwrf/io/scheme_catalog.py'
    tree = ast.parse(path.read_text())
    node = next(n.value for n in tree.body if isinstance(n, ast.AnnAssign)
                and isinstance(n.target, ast.Name) and n.target.id == '_IMPLEMENTED')
    assert isinstance(node, ast.Dict)
    result = {}
    for key, value in zip(node.keys, node.values, strict=True):
        assert isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == 'frozenset'
        result[ast.literal_eval(key)] = sorted(ast.literal_eval(value.args[0]))
    selectors = {}
    for name in ('_DYNAMICS_IMPLEMENTED', '_IMPLEMENTED_FEATURE_VALUES', '_IMPLEMENTED_CONTROLS'):
        value = next(n.value for n in tree.body if isinstance(n, ast.AnnAssign)
                     and isinstance(n.target, ast.Name) and n.target.id == name)
        allowed = (ast.Expression, ast.Dict, ast.Constant, ast.Call, ast.Name, ast.Load,
                   ast.Store, ast.Set, ast.Tuple, ast.List, ast.DictComp, ast.comprehension)
        assert all(isinstance(n, allowed) for n in ast.walk(value))
        assert all(isinstance(n.func, ast.Name) and n.func.id == 'frozenset'
                   for n in ast.walk(value) if isinstance(n, ast.Call))
        mapping = eval(compile(ast.Expression(value), '<closed catalog sets>', 'eval'),
                       {'__builtins__': {}, 'frozenset': frozenset})
        selectors.update({k: sorted(v) for k,v in mapping.items()})
    controls = next(n.value for n in tree.body if isinstance(n,ast.AnnAssign)
                    and isinstance(n.target,ast.Name) and n.target.id=='_RECOGNIZED_CONTROLS')
    for call in controls.elts:
        values = call.args[2]
        if isinstance(values,ast.Call) and isinstance(values.func,ast.Name) and values.func.id=='frozenset' and values.args:
            options = ast.literal_eval(values.args[0])
            if len(options)>1:
                selectors[ast.literal_eval(call.args[0])] = sorted(options)
    noah_path=source/'src/gpuwrf/io/noahmp_options.py'
    if noah_path.exists():
        # Host-only module, executed from the caller's frozen source revision.
        # SUPPORTED is extended after its initial assignment, so a first-AST-
        # literal reader would silently miss all nondefault land controls.
        spec=importlib.util.spec_from_file_location('vprep_noah_controls',noah_path)
        module=importlib.util.module_from_spec(spec)
        import sys
        sys.modules[spec.name]=module
        spec.loader.exec_module(module)
        selectors.update({k:sorted(v) for k,v in module.SUPPORTED.items()})
        selectors['sst_skin']=[0,1]
    return result, selectors, sha(path)


def seal(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    matrix = json.loads((ROOT / 'MATRIX.json').read_text())
    catalog, selectors, catalog_sha = implemented_catalog(source)
    from gpuwrf.io.namelist_parser import parse_namelist
    from gpuwrf.io.namelist_binding import namelist_value
    from gpuwrf.io.namelist_binding import namelist_honour_report
    from gpuwrf.io.namelist_check import _coerce_config
    from gpuwrf.io.wrf_registry_defaults import WRF_REGISTRY_DEFAULTS
    selected = {key:set() for key in selectors}
    rows, references, pending, failed, cli_blocked = [], [], [], [], []
    for old in matrix['cases']:
        row = dict(old)
        case = ROOT / row['id']
        inp = case / 'input'
        # A CPU run directory may symlink its namelist back to input/. Neither
        # current path is independent authority: the sealed run SHA is.
        current_namelist_sha = verified_namelist_sha(
            inp / 'namelist.input', row['input_sha256']['namelist.input'], row['id'])
        if row.get('gpu_eligible',True) and not row['admission_errors']:
            try:
                errors, warnings = namelist_honour_report(
                    _coerce_config(inp / 'namelist.input'), row['domains'], replay=False)
                row['actual_cli_admission'] = dict(errors=errors, warnings=warnings)
            except Exception as exc:
                row['actual_cli_admission'] = dict(exception=repr(exc))
            if row['actual_cli_admission'].get('errors') or row['actual_cli_admission'].get('exception'):
                cli_blocked.append(row['id'])
            nl = parse_namelist(inp / 'namelist.input')
            for key in selectors:
                lookup = 'rk_ord' if key=='rk_order' else key
                for domain in row['domains']:
                    value = namelist_value(nl, lookup, domain)
                    if value is None:
                        for group in nl.values():
                            if lookup in group:
                                raw = group[lookup]
                                value = raw[min(int(domain[1:])-1,len(raw)-1)] if isinstance(raw,list) else raw
                                break
                    if value is None:
                        default = WRF_REGISTRY_DEFAULTS.get(lookup)
                        value = default[3] if default is not None else 0
                    if key=='mfshconv' and row['options']['bl_pbl_physics']!=4:
                        value=0  # pristine WRF resets MFSHCONV outside QNSE
                    if value is not None:
                        selected[key].add(value)
        target = destination / 'namelists' / row['id']
        target.mkdir(parents=True, exist_ok=True)
        for p in inp.iterdir():
            # NetCDF forcing/analysis arrays remain external and hash-bound.
            # Publish only reviewable text controls, never hundreds of MB of FDDA.
            if p.is_file() and (p.name in ('namelist.input','tslist') or
                               p.suffix in ('.txt','.tbl','.json') or p.name.startswith('OBS_DOMAIN')):
                shutil.copyfile(p, target / p.name)
        # The preparation schema uses source_commit for the PORT recipe, not
        # the WRF executable. Its separate binary/source-body provenance is
        # authoritative and is never substituted by that port revision.
        result = dict(id=row['id'], gpu_eligible=row.get('gpu_eligible', True),
                      preparation_port_commit=matrix['source_commit'],
                      actual_input_dir=str(inp), input_sha256=row['input_sha256'],
                      current_namelist_sha256=current_namelist_sha)
        receipt_path = case / 'cpu_run.json'
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            result.update(status=receipt['status'], receipt=str(receipt_path),
                receipt_sha256=sha(receipt_path), wall_s=receipt['monotonic_wall_s'],
                original_executable=receipt['executable'], frame_count=receipt['frame_count'],
                output_manifest=receipt['output_manifest'],
                output_manifest_sha256=receipt['output_manifest_sha256'])
            assert receipt['actual_input_sha256'] == row['input_sha256']
            cpu_namelist = case / 'cpu' / 'namelist.input'
            if cpu_namelist.exists():
                verified_namelist_sha(cpu_namelist,
                    receipt['actual_input_sha256']['namelist.input'], row['id'] + ' CPU path')
            assert sha(case / 'CPU_OUTPUTS.json') == receipt['output_manifest_sha256']
            if receipt['status'] != 'COMPLETE' and result['gpu_eligible']:
                failed.append(row['id'])
        else:
            result['status'] = 'PENDING'
            if result['gpu_eligible']:
                pending.append(row['id'])
        row['cpu_reference'] = result['status']
        references.append(result)
        rows.append(row)
    active = [r for r in rows if r.get('gpu_eligible', True)]
    coverage = {key: sorted({r['options'][key] for r in active if not r['admission_errors']}) for key in SLOTS}
    uncovered = {key: sorted(set(values) - set(coverage.get(key, []))) for key, values in catalog.items()}
    api_only = {'sf_surface_physics': [1, 2, 7]}
    cpu_only = {'ra_lw_physics':[7,99],'ra_sw_physics':[7,99]}
    owner_slot_exclusions={'mp_physics':[15]}
    actionable = {key: sorted(set(values) - set(api_only.get(key, []))-set(cpu_only.get(key,[]))-set(owner_slot_exclusions.get(key,[]))) for key, values in uncovered.items()}
    uncovered_selectors = {key:sorted(set(values)-selected[key]) for key,values in selectors.items()}
    owner_exclusions={'km_opt':[3],'opt_crop':[1],'opt_tdrn':[1,2]}
    # aer_init_opt is WRF-derived (Registry, not a namelist entry). The CLI
    # rejects an explicit request; its recognized API value1 is outside scope.
    api_selector_exclusions={'aer_init_opt':[1]}
    gate_uncovered_selectors={key:sorted(set(values)-set(owner_exclusions.get(key,[]))-set(api_selector_exclusions.get(key,[])))
                              for key,values in uncovered_selectors.items()}
    prepared=not pending and not failed and not any(actionable.values()) and not any(gate_uncovered_selectors.values())
    manifest = dict(schema='v040-final-validation-preparation-v1',
        sealed_utc=datetime.now(timezone.utc).isoformat(),
        catalog_source_commit=subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
        catalog_sha256=catalog_sha, implemented=catalog, coverage=coverage,
        uncovered=uncovered, uncovered_cli=actionable,
        api_only_exclusions=api_only, api_scope_decision='APPROVED Owner chat2026-10-14: CLI-admitted only; no GPU API-only options',
        cpu_only_exclusions=cpu_only,
        owner_approved_slot_exclusions=owner_slot_exclusions,
        non_mp_selectors=selectors, selector_coverage={k:sorted(v) for k,v in selected.items()},
        uncovered_non_mp_selectors=uncovered_selectors,
        owner_approved_selector_exclusions=owner_exclusions,
        api_only_selector_exclusions=api_selector_exclusions,
        uncovered_gate_selectors=gate_uncovered_selectors,
        active_composite_count=len(active), retained_reference_control_count=len(rows)-len(active),
        cases=rows, cpu_references=references, pending_references=pending, failed_references=failed,
        cpu_preparation_complete=prepared,
        ready_for_gpu=prepared and not cli_blocked,
        actual_native_cli_admission_blocked=cli_blocked,
        actual_native_cli_admission_scope='Exact CLI _coerce_config(Path), replay=False; CPU references and selector coverage are separate gates',
        qualification='Preparation gate only; GPU batches running under Owner authorization. No matrix-wide D6/E41/station PASS inferred.',
        final_rc_inventory_refresh_required=True,
        estimated_gpu_minutes_inferred=[400, 600], measured_final_rc_gpu_minutes=None)
    write_json(destination / 'MATRIX.json', manifest)
    write_json(destination / 'CPU_REFERENCES.json', references)
    if (ROOT / 'ZERO_BUILD_PIN.json').exists():
        shutil.copyfile(ROOT / 'ZERO_BUILD_PIN.json', destination / 'ZERO_BUILD_PIN.json')
        shutil.copyfile(ROOT / 'ZERO_BUILD.json', destination / 'ZERO_BUILD.json')
    lines = ['| Case | Geography | MP/CU/PBL/SF/LSM/LW/SW | CPU reference | Features |',
             '|---|---|---|---|---|']
    for row in active:
        lines.append('| ' + ' | '.join((row['id'], row['geography'],
            '/'.join(str(row['options'][k]) for k in SLOTS), row['cpu_reference'],
            ', '.join(row.get('features', [])))) + ' |')
    (destination / 'MATRIX.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(active=len(active), complete=sum(r['status']=='COMPLETE' and r['gpu_eligible'] for r in references),
        pending=pending, failed=failed, uncovered_cli=actionable), indent=2))


if __name__ == '__main__':
    assert os.environ.get('JAX_PLATFORMS') == 'cpu'
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    p = argparse.ArgumentParser()
    p.add_argument('catalog_source', type=Path)
    p.add_argument('destination', type=Path)
    args = p.parse_args()
    seal(args.catalog_source.resolve(), args.destination.resolve())
