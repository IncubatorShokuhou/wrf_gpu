"""Bind the host-only daily fix to the critic-accepted scientific source."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else Path(__file__).resolve().parents[3]
ACCEPTED = "a8dd517f08db56d5f6942f1f87a5c09b57f4f034"
DAILY = "src/gpuwrf/integration/daily_pipeline.py"


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def blobs(commit):
    return {line.split()[3]: line.split()[2]
            for line in git("ls-tree", "-r", commit, "--", "src").decode().splitlines()}


def nodes(source):
    return {n.name: n for n in ast.parse(source).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


before, after = blobs(ACCEPTED), blobs("HEAD")
changed = sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p))
assert changed == [DAILY], changed
old = nodes(git("show", f"{ACCEPTED}:{DAILY}"))
new = nodes((ROOT / DAILY).read_bytes())
unchanged_functions = ["_build_real_case", "_default_forecast_fn", "_segmented_forecast_fn",
                       "_capture_boundary_leaves", "_rewindow_boundary_leaves"]
for name in unchanged_functions:
    assert ast.dump(old[name]) == ast.dump(new[name]), name
for name in ("__init__", "__call__"):
    a = next(n for n in old["_RucDailyStepper"].body if getattr(n, "name", None) == name)
    b = next(n for n in new["_RucDailyStepper"].body if getattr(n, "name", None) == name)
    assert ast.dump(a) == ast.dump(b), name
result = dict(
    accepted=ACCEPTED, candidate=git("rev-parse", "HEAD").decode().strip(),
    changed_src_paths=changed, unchanged_src_blobs=len(before) - 1,
    unchanged_daily_functions=unchanged_functions,
    unchanged_existing_stepper_methods=["__init__", "__call__"],
    daily_sha256=hashlib.sha256((ROOT / DAILY).read_bytes()).hexdigest(),
    reused_release_jaxpr_sha256="2bf12cb8374d3a84a440e47dea4b77b0c7436108fc2ee2616bba6ec79b17962a",
    reuse="Critic accepted operators, native coupled smokes, full restart and release trace. "
          "All model/compiler/restart source bytes unchanged; only daily host orchestration changed.",
    performance="No new timestep kernels, allocations or transfers. Existing bounded chunk runner; "
                "one initialization per daily run, held Noah-MP diagnostics and full carry retained. "
                "Checkpoint host copies occur only at the requested output boundary. "
                "GPU timing/memory and long-forecast qualification remain unmeasured.",
)
Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
