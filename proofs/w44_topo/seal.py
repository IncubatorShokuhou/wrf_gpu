"""Small source/data seal; no JAX import or model execution."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
out = Path(__import__("sys").argv[1])


def function_ast(source):
    return ast.dump(next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                         and n.name == "topographic_shadow"), include_attributes=False)


old = subprocess.check_output(["git", "show", "338e2d4d60:src/gpuwrf/physics/topographic_shading.py"], cwd=ROOT, text=True)
new = (ROOT / "proofs/w44_topo/operator_probe.py").read_text()
assert function_ast(old) == function_ast(new)
changed = subprocess.check_output(["git", "diff", "--name-only", "748191e1eb", "HEAD", "--", "src"], cwd=ROOT, text=True).splitlines()
assert changed == ["src/gpuwrf/io/scheme_catalog.py"], changed
old_cat = subprocess.check_output(["git", "show", "748191e1eb:src/gpuwrf/io/scheme_catalog.py"], cwd=ROOT, text=True)
new_cat = (ROOT / changed[0]).read_text()
# Only the text explaining an INVALID slope_rad value may change in product AST.
def without_reason(source):
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.AnnAssign)
                and isinstance(n.target, ast.Name) and n.target.id == "_IMPLEMENTED_CONTROL_REASON")
    for key, value in zip(node.value.keys, node.value.values):
        if key.value == "slope_rad":
            value.elts[1] = ast.Constant(value="<invalid-slope-mode-explanation>")
    return ast.dump(tree, include_attributes=False)
assert without_reason(old_cat) == without_reason(new_cat)
receipt = {"status": "PASS", "prototype_function_ast_identical_to_TP03": True,
           "changed_product_files": changed, "product_ast_only_invalid_slope_reason_changed": True,
           "forecast_runtime_source_byte_identical": True,
           "default_release_identities": {"Swiss": "b977712f", "PROD": "321aa651", "basis": "carried forward; no forecast-source change"}}
for arm in ("TP03", "TP04"):
    path = ROOT / "proofs/w44_topo" / (arm + "_receipt.json")
    receipt[arm + "_receipt_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
out.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
