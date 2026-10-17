"""Seal K0 identity vs a specified base on an immutable committed checkout."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

base, output = sys.argv[1:]


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


head = git("rev-parse", "HEAD")
protected = ["dynamics", "physics", "kernels", "runtime", "integration", "nesting", "coupling", "contracts", "io"]
trees = {key: [git("rev-parse", f"{rev}:src/gpuwrf/{key}") for rev in (base, head)] for key in protected}
assert all(old == new for old, new in trees.values()), trees
old_cli = ast.parse(git("show", f"{base}:src/gpuwrf/cli.py"))
new_cli = ast.parse(Path("src/gpuwrf/cli.py").read_text())
old_functions = {n.name: ast.dump(n, include_attributes=False) for n in old_cli.body if isinstance(n, ast.FunctionDef)}
new_functions = {n.name: ast.dump(n, include_attributes=False) for n in new_cli.body if isinstance(n, ast.FunctionDef)}
same = {k: old_functions[k] == new_functions[k] for k in old_functions if k != "build_parser"}
assert all(same.values()), same
old_parser = next(n for n in old_cli.body if isinstance(n, ast.FunctionDef) and n.name == "build_parser")
new_parser = next(n for n in new_cli.body if isinstance(n, ast.FunctionDef) and n.name == "build_parser")
new_parser.body = [n for n in new_parser.body if not any(isinstance(x, ast.Name) and x.id == "post" for x in ast.walk(n))]
assert ast.dump(old_parser, include_attributes=False) == ast.dump(new_parser, include_attributes=False)
payload = {"base": git("rev-parse", base), "head": head,
           "source_tree": git("rev-parse", "HEAD:src/gpuwrf"),
           "post_tree": git("rev-parse", "HEAD:src/gpuwrf/post"),
           "cli_blob": git("rev-parse", "HEAD:src/gpuwrf/cli.py"),
           "tests_blob": git("rev-parse", "HEAD:tests/test_post.py"),
           "fixtures_tree": git("rev-parse", "HEAD:tests/fixtures/post"),
           "protected_tree_ids": trees, "existing_cli_functions_identical": same,
           "parser_without_post_identical": True, "swiss_release_jaxpr_reused": "2bf12cb8",
           "identity_basis": "Existing model/coupler/IO code and every pre-existing CLI command are source-exact. No model numerical re-run required for K0.",
           "diff_paths": git("diff", "--name-only", base, head).splitlines()}
Path(output).write_text(json.dumps(payload, indent=2)+"\n")
print(json.dumps({"head": head, "protected_trees_exact": len(trees), "existing_cli_functions_exact": len(same)}))
