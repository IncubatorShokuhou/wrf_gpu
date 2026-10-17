"""Host-only identity/reduction proof; reuse accepted physical executions.

Replay saved BEM03 candidate arrays through the stricter comparison helper.
No model, oracle build, JIT or forecast is executed.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from conditioned_gate import comparison, column_budget, load_budget

REVIEWED = '0feac83ce84816904e983b172aacb062d7049ce4'
PROOF = Path(__file__).resolve().parent
ROOT = PROOF.parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(artifacts, out):
    qualified = json.loads((PROOF / 'evidence/qualification.json').read_text())
    exact = {}
    for name, expected in qualified['source_sha256'].items():
        if '/urban_' in name or Path(name).name in ('pbl_ysu.py', 'pbl_boulac.py', 'sfclay_revised_mm5.py'):
            exact[name] = digest(ROOT / name)
            assert exact[name] == expected, name
    assets = {}
    for name, expected in qualified['assets']['sha256'].items():
        assets[name] = digest(PROOF / name)
        assert assets[name] == expected, name
    seals = {}
    for name in ('roundoff_budget.py', 'evidence/ROUND01.npz', 'evidence/ROUND01.json',
                 'evidence/ROUND01_APPROVAL.json', 'evidence/WRF_flux_operands.json',
                 'evidence/BEP26.npz', '../urban_slucm/savepoints/URBPARM.TBL.default'):
        path = (PROOF / name).resolve()
        original = subprocess.check_output(['git', 'show', REVIEWED + ':' + str(path.relative_to(ROOT))])
        seals[name] = digest(path)
        assert seals[name] == hashlib.sha256(original).hexdigest(), name
    functions = {}
    for name in ('bep_driver_parity.py', 'bem_driver_parity.py'):
        original = subprocess.check_output(['git', 'show', REVIEWED + ':' +
                                            str((PROOF / name).relative_to(ROOT))], text=True)
        before = ast.parse(original)
        after = ast.parse((PROOF / name).read_text())
        for function in ('unpack', 'packed'):
            get = lambda tree: next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
            assert ast.dump(get(before)) == ast.dump(get(after)), (name, function)
            functions[name + ':' + function] = hashlib.sha256(ast.dump(get(after)).encode()).hexdigest()
    rows = json.loads((artifacts / 'BEM03.json').read_text())
    actual = np.load(artifacts / 'BEM03_actual.npz')
    budgets = load_budget(PROOF / 'evidence/ROUND01.npz')
    replayed = 0
    raw_counts = {}
    for arm, record in rows.items():
        precision, ac = arm.split('_', 1)
        data = np.load(PROOF / f'savepoints/urban3_{precision}_{ac}.npz')
        data64 = np.load(PROOF / f'savepoints/urban3_fp64_{ac}.npz')
        for row in record['fields']:
            name, step = row['field'], row['step']
            candidate = actual[arm + '__' + name][:, step - 1]
            index = step if data[name].shape[1] == 7 else step - 1
            reference = data[name][:, index]
            if candidate.ndim == 2 and reference.shape[-1] == 9 and candidate.shape[-1] == 8:
                reference = reference[:, :8]
            if precision == 'fp64':
                limit = 2e-11 + 2e-12 * np.abs(reference)
                extra = 0.
            else:
                ref64 = data64[name][:, index]
                if ref64.shape != reference.shape: ref64 = ref64[:, :8]
                scale = 1. if name in ('TSK', 'TS_URB', 'TRB_URB4D', 'TW1_URB4D', 'TW2_URB4D', 'TGB_URB4D') else 1.e-3
                limit = 2 * np.abs(reference - ref64) + 8 * np.finfo(np.float32).eps * np.maximum(np.abs(reference), scale)
                extra = column_budget(budgets, 'urban3_' + ac, name, step, candidate.shape, data[name].shape[1])
            result = dict(step=step, field=name, **comparison(candidate, reference, limit, extra))
            assert result == row, (arm, name, step, result, row)
            replayed += 1
        raw_counts[arm] = sum(row['raw_violations'] for row in record['fields'])
    receipt = dict(reviewed=REVIEWED, snapshot=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                   unchanged_numerical_sources=exact, unchanged_assets=assets, unchanged_seals=seals,
                   unchanged_driver_preparation_functions=functions,
                   BEM03_saved_comparison_rows_exact=replayed, raw_violations_retained=raw_counts,
                   actual_arrays_sha256=digest(artifacts / 'BEM03_actual.npz'),
                   physical_execution_repeated=False, no_tolerance_changes=True)
    out.write_text(json.dumps(receipt, indent=2) + '\n')
    print(len(exact), 'numerical sources BYTE;', len(assets), 'assets BYTE;', replayed, 'comparison rows EXACT')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('artifacts', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    run(args.artifacts, args.out)
