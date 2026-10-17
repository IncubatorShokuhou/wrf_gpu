"""Audit the final output fix composed with reviewed main host-only changes."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[2]).resolve()
measured = "45e3ec82b2d16324a65a354a3f78cfeb0304d93c"
upstream = "489a914837303cf0c76644cff919f49363f39e5c"
model = "src/gpuwrf/runtime/operational_mode.py"
daily = "src/gpuwrf/integration/daily_pipeline.py"


def git(*args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def blobs(commit):
    return {line.split()[3]: line.split()[2]
            for line in git("ls-tree", "-r", commit, "--", "src").decode().splitlines()}


def node(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


before, after, main = blobs(measured), blobs("HEAD"), blobs(upstream)
changes = sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p))
host = ["src/gpuwrf/__init__.py", "src/gpuwrf/cli.py", "src/gpuwrf/optimize/__init__.py",
        "src/gpuwrf/optimize/calibration.py", "src/gpuwrf/optimize/cli.py",
        "src/gpuwrf/optimize/data.py", "src/gpuwrf/optimize/study.py"]
assert changes == sorted([*host, model]), changes
assert all(after[p] == main[p] for p in host)
assert after[daily] == before[daily]
old = ast.parse(git("show", f"{measured}:{model}"))
new = ast.parse((root / model).read_bytes())
main_tree = ast.parse(git("show", f"{upstream}:{model}"))
guard = node(new, "_protect_initial_carry_seeds")
assert ast.dump(guard) == ast.dump(node(main_tree, guard.name))
new.body.remove(guard)
initial = node(new, "_initial_carry_for_run")
assert ast.unparse(initial.body[-1]) == "return _protect_initial_carry_seeds(result, namelist)"
initial.body[-1] = ast.Return(value=ast.Name(id="result", ctx=ast.Load()))
assert ast.dump(old) == ast.dump(new), "Numerical model changed beyond reviewed host seed guard"
result = dict(candidate=git("rev-parse", "HEAD").decode().strip(), measured_output_source=measured,
    integrated_main=upstream, changed_src_paths=changes,
    unchanged_existing_src_blobs=sum(before[p] == after.get(p) for p in before),
    daily_output_source_byte_exact=True,
    daily_sha256=hashlib.sha256((root / daily).read_bytes()).hexdigest(),
    numerical_model_AST_exact_after_removing_reviewed_host_seed_guard=True,
    ownership_guard_AST_exact_main=True, upstream_host_blobs_exact_main=host,
    reused_release_sha256="2bf12cb8374d3a84a440e47dea4b77b0c7436108fc2ee2616bba6ec79b17962a",
    scope="Only upstream host initialization guard and optimizer/version/CLI code differ from measured output fix. "
          "Observation seed precedes ownership guard; no timestep, physics, restart or output arithmetic change.",
    performance="No added timestep kernels/temps/transfers. None-safe dispatcher uses existing output work; "
                "present history avoids bulk recomputation. Main ownership guard copies shared seeds once at init.")
Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
