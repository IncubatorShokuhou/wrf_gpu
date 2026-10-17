"""Compare real release/no-hail traces with the exact pre-fix functions.

The baseline is source-scoped regression evidence, never WRF truth. All
functions come verbatim from the pinned parent commit, with current interfaces.
"""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT, OUTPUT = (Path(x).resolve() for x in sys.argv[1:3])
MODE, CASE = sys.argv[3:5]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "proofs/o1_nlbind")]
from gpuwrf.integration import nested_pipeline as pipeline
from gpuwrf.backends import pallas as facade
from gpuwrf.runtime import operational_mode as om
from gpuwrf.coupling import physics_couplers as pc
from gpuwrf.kernels import phys_hydrostatic_real as native

if MODE == "parent":
    for module, names in ((om, ("_psfc_from_state", "_acoustic_core_state_from_prep")),
                          (pc, ("_wrf_hydrostatic_pressure_profiles_from_state",)),
                          (native, ("_sweep", "hydrostatic_profiles"))):
        relative = Path(module.__file__).relative_to(ROOT)
        source = subprocess.check_output(["git", "show", "ad8b0c729:" + str(relative)], cwd=ROOT, text=True)
        tree = ast.parse(source)
        namespace = module.__dict__.copy()
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(relative), "exec"), namespace)
                setattr(module, node.name, namespace[node.name])
elif MODE != "candidate":
    raise ValueError(MODE)

load = pipeline._load_domains
def load_cpu(*args, **kw):
    facade.cuda_native = lambda: False
    try:
        return load(*args, **kw)
    finally:
        facade.cuda_native = lambda: True
pipeline._load_domains = load_cpu

os.environ["O1_CASE"] = CASE
arms = sys.argv[5:] or ["release"]
probe = ROOT / "proofs/o1_nlbind/probe.py"
source = probe.read_text().replace('LANE = Path("<USER_HOME>/wrf_gpu2_lanes/o1-nlbind")',
                                 'LANE = Path("<USER_HOME>/wrf_gpu2_lanes/t-qhmass")')
sys.argv = [str(probe), str(ROOT), str(OUTPUT), *arms]
exec(compile(source, str(probe), "exec"), {"__file__": str(probe), "__name__": "__main__"})
report = json.loads(OUTPUT.read_text())
report.update(mode=MODE, baseline_source="ad8b0c729", scope="CPU-only native-facade program identity; regression only")
OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
for name, row in report["arms"].items():
    assert "error" not in row and row.get("sha"), (name, row)
