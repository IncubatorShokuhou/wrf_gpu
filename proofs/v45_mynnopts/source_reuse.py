"""Bind reused original-WRF column fixtures to exact operator source."""
from pathlib import Path
import ast
import hashlib
import json
import subprocess
import sys


def digest(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def check(root):
    reference = "e22a3fcaa5757c33cfd2ca8f0c197d46d77bf4a7"
    files = {
        "src/gpuwrf/physics/mynn_sgs_cloud.py": (
            "esat_blend", "_alp_bet", "_case0_sigma", "_case1_sigma", "_sommeria_deardorff_cloud",
            "_cloudpdf01_columns", "mym_condensation_cloudpdf01"),
        "src/gpuwrf/physics/mynn_pbl.py": ("_mym_predict_tsq_cov",),
    }
    records = []
    for file, names in files.items():
        before = ast.parse(subprocess.check_output(["git", "show", reference + ":" + file], text=True))
        after = ast.parse((root / file).read_text())
        for name in names:
            old = next(node for node in before.body if isinstance(node, ast.FunctionDef) and node.name == name)
            new = next(node for node in after.body if isinstance(node, ast.FunctionDef) and node.name == name)
            exact = digest(old) == digest(new)
            barrier_only = False
            if name == "esat_blend" and not exact:
                import copy
                class RemoveBarrier(ast.NodeTransformer):
                    def visit_Call(self, node):
                        node = self.generic_visit(node)
                        if isinstance(node.func, ast.Attribute) and node.func.attr == "optimization_barrier":
                            assert len(node.args) == 1 and not node.keywords
                            return node.args[0]
                        return node
                barrier_only = digest(old) == digest(RemoveBarrier().visit(copy.deepcopy(new)))
            records.append(dict(file=file, name=name, ast_sha256=digest(new), exact=exact,
                                approved_barrier_only=barrier_only))
        if file.endswith("mynn_sgs_cloud.py"):
            # Full literal coefficient arrays and their REAL-fold construction,
            # not a raw-HLO string that hides large constants (E224).
            def constants(tree):
                return [node for node in tree.body if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id.startswith(("_F4", "_R_", "_CP", "_XL", "_EP_", "_T0", "_TICE", "_B2", "_RR", "_ESAT"))
                                for target in node.targets)]
            old, new = constants(before), constants(after)
            records.append(dict(file=file, name="REAL coefficients", ast_sha256=hashlib.sha256(
                "".join(map(digest, new)).encode()).hexdigest(), exact=list(map(digest, old)) == list(map(digest, new))))
    return dict(reference_commit=reference,
                source_exact=all(record["exact"] for record in records), operators=records,
                unchanged_or_barrier_only=all(record["exact"] or record.get("approved_barrier_only") for record in records),
                scope="IEEE divisor barrier changes ESAT: condensation is freshly recaptured; other operator/constants exact. Init recaptured separately.")


if __name__ == "__main__":
    result = check(Path(__file__).resolve().parents[2])
    Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + "\n")
    assert result["unchanged_or_barrier_only"], result
