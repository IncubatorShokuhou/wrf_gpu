"""Delete each admission/version fix in memory; its actual pytest must fail."""
import ast
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

assert os.environ['JAX_PLATFORMS'] == 'cpu'
ROOT = Path(os.environ['V05_AOTFP_SOURCE'])
OUT = Path(os.environ['V05_AOTFP_ARTIFACT_DIR'])


def worker(name):
    sys.path.insert(0, str(ROOT / 'scripts/v025'))
    import cpu_guard  # noqa: F401 - establish the CPU guard before JAX imports
    import pytest
    from gpuwrf.runtime import aot_executable as ax, aot_precompile as ap
    from gpuwrf import cli
    module, symbol, target = {
        'required_fields': (ax, 'fingerprint_matches', 'tests/test_aot_fingerprint.py::test_empty_fingerprint_refused'),
        'named_opt_in': (ax, 'load', 'tests/test_aot_fingerprint.py::test_internal_loaders_require_named_legacy_opt_in[False]'),
        'prefetch_policy': (ap, '_load_domain_blob_prefetched', 'tests/test_aot_fingerprint.py::test_unnamed_bypass_cannot_consume_prefetch'),
        'version': (cli, 'build_parser', 'tests/test_cli_version.py'),
    }[name]
    tree = ast.parse(inspect.getsource(getattr(module, symbol)))
    func = tree.body[0]
    def remove(node):
        source = ast.unparse(node)
        return ((name == 'required_fields' and isinstance(node, ast.If) and source.startswith('if not allow_legacy_fingerprint:'))
            or (name == 'named_opt_in' and isinstance(node, ast.If) and source.startswith('if not check_fingerprint and'))
            or (name == 'prefetch_policy' and isinstance(node, ast.If) and source.startswith('if allow_legacy_fingerprint or not check_fingerprint:'))
            or (name == 'version' and isinstance(node, ast.Expr) and 'parser.add_argument(\'--version\'' in source))
    class DeleteFix(ast.NodeTransformer):
        count = 0

        def visit(self, node):
            if remove(node):
                self.count += 1
                return None
            return super().visit(node)

    deletion = DeleteFix()
    tree = deletion.visit(tree)
    assert deletion.count == 1, deletion.count
    namespace = dict(module.__dict__)
    exec('from __future__ import annotations\n' + ast.unparse(tree), namespace)
    setattr(module, symbol, namespace[symbol])
    return pytest.main(['-q', target, '--basetemp', str(OUT / ('mutation_pytest_' + name))])


if len(sys.argv) > 1:
    sys.exit(worker(sys.argv[1]))
rows = []
for name in ['required_fields', 'named_opt_in', 'prefetch_policy', 'version']:
    path = OUT / ('mutation_' + name + '.log')
    with path.open('w') as log:
        result = subprocess.run([sys.executable, __file__, name], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=120)
    log_text = path.read_text()
    assert result.returncode == 1 and 'FAILED tests/' in log_text and '1 failed' in log_text, (name, result.returncode, log_text[-2000:])
    rows.append(dict(mutation=name, deleted_fix_blocks=1, pytest_rc=result.returncode, verdict='KILLED', log=str(path)))
(OUT / 'MUTATIONS.json').write_text(json.dumps(rows, indent=2) + '\n')
print(json.dumps(rows, indent=2))
