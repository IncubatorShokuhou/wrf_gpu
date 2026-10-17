"""Delete the CAM-specific forwarding at the actual operand-builder source site."""
import ast
import inspect
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/v025"))
import cpu_guard  # noqa: E402,F401
import pytest
from gpuwrf.coupling import cam_radiation as cam

tree = ast.parse(inspect.getsource(cam.cam_operands))
changed = 0
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "_rrtmg_column_inputs":
        before = len(node.keywords)
        node.keywords = [k for k in node.keywords if k.arg not in ("use_mynn_sgs_cloud", "cloud_mp_physics")]
        changed += before - len(node.keywords)
assert changed == 2
namespace = dict(cam.__dict__)
exec(compile(ast.fix_missing_locations(tree), "CAM_CLOUD_FORWARDING_DELETED", "exec"), namespace)
cam.cam_operands = namespace["cam_operands"]
raise SystemExit(pytest.main(__import__("sys").argv[1:]))
