"""Seal the narrow CAM-only change and reuse unchanged pristine physics gates."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root, base, output = Path(sys.argv[1]).resolve(), sys.argv[2], Path(sys.argv[3]).resolve()


def git(*args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


paths = git("diff", "--name-only", base, "HEAD", "--", "src").splitlines()
assert set(paths) == {"src/gpuwrf/coupling/cam_radiation.py", "src/gpuwrf/runtime/operational_mode.py"}
current = ast.parse((root / paths[1]).read_text())
old = ast.parse(git("show", base + ":" + paths[1]))
removed = 0
for node in ast.walk(current):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in (
            "cam_surface_forcing", "cam_sw_theta_tendency", "cam_lw_theta_tendency"):
        old_n = len(node.keywords)
        node.keywords = [kw for kw in node.keywords if not (
            kw.arg is None and isinstance(kw.value, ast.Call) and isinstance(kw.value.func, ast.Name)
            and kw.value.func.id == "_rrtmg_pbl_cloud_kwargs")]
        removed += old_n - len(node.keywords)
assert removed == 3 and ast.dump(current) == ast.dump(old)
physics_tree = git("rev-parse", "HEAD:src/gpuwrf/physics")
assert physics_tree == git("rev-parse", base + ":src/gpuwrf/physics")
oracle = json.loads((root / "proofs/t_camuw/cloud_oracle.json").read_text())
source = Path("<USER_HOME>/src/wrf_pristine/WRF/phys/module_radiation_driver.F").read_bytes()
assert hashlib.sha256(source).hexdigest() == oracle["source_sha256"]
start = source.index(b"   SUBROUTINE cal_cldfra1(")
end = source.index(b"   END SUBROUTINE cal_cldfra1", start) + len(b"   END SUBROUTINE cal_cldfra1")
assert hashlib.sha256(source[start:end] + b"\n").hexdigest() == oracle["subroutine_sha256"]
row = dict(source=git("rev-parse", "HEAD"), base=base, changed_product_files=paths,
           operational_ast_after_removing_only_3_cam_kwargs="EXACT", unchanged_physics_tree=physics_tree,
           original_cloud_source_sha256=oracle["source_sha256"], original_cloud_body_sha256=oracle["subroutine_sha256"],
           reuse="All spectral CAM LW/abs/SW code and WRF cloud-fraction code/fixtures unchanged; existing numerical gates retained",
           new_baked_inputs=[], new_carry_fields=[], gpu_evidence="PENDING manager c28 final-candidate rerun")
output.write_text(json.dumps(row, indent=2)+"\n")
print(json.dumps(row))
