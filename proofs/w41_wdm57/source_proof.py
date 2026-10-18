"""Prove the only numerical-source edits are two identity-firewall operands."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

lane = Path("<USER_HOME>/wrf_gpu2_lanes/w41-wdm57")
base, new = lane / "baseline", lane / "candidate"
relative = "src/gpuwrf/coupling/scan_adapters.py"
old = ast.parse((base / relative).read_text())
tree = ast.parse((new / relative).read_text())


class RemoveOnlyTwoFences(ast.NodeTransformer):
    count = 0

    def visit_FunctionDef(self, node):
        if node.name not in ("wdm5_adapter", "wdm7_adapter"):
            return node
        return self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "_small_grid_firewall":
            assert ast.unparse(node) == "_small_grid_firewall(mp(state.Nn), (ny, nx))"
            self.count += 1
            return node.args[0]
        return self.generic_visit(node)


transform = RemoveOnlyTwoFences()
normalized = ast.dump(transform.visit(tree), include_attributes=False)
assert transform.count == 2
assert ast.dump(old, include_attributes=False) == normalized
changed = subprocess.check_output([
    "git", "-C", str(new), "diff", "--name-only", "47c50d376b", "HEAD", "--", "src"], text=True).splitlines()
assert changed == [relative]
result = {
    "base": subprocess.check_output(["git", "-C", str(base), "rev-parse", "HEAD"], text=True).strip(),
    "candidate": subprocess.check_output(["git", "-C", str(new), "rev-parse", "HEAD"], text=True).strip(),
    "changed_source_files": changed,
    "identity_firewall_operands": transform.count,
    "normalized_whole_module_AST_exact": True,
    "normalized_AST_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
    "source_sha256": {name: hashlib.sha256((root / relative).read_bytes()).hexdigest()
                      for name, root in (("base", base), ("candidate", new))},
    "scope": "source containment, not CPU-WRF/GPU numerical fidelity or efficiency proof",
}
(lane / "source_proof.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
