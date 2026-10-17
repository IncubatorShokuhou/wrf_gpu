"""Prove optional mp38 hooks specialize to the previous shared function AST.

This is source regression evidence, not numerical qualification.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess


class Specialize(ast.NodeTransformer):
    bindings = {"hail_context": None, "working": None, "working_rho": None,
                "return_rates": False, "wrf_hail": False, "hail_mode": None}

    def known(self, node):
        if isinstance(node, ast.Name) and node.id in self.bindings:
            return True, self.bindings[node.id]
        if isinstance(node, ast.Constant):
            return True, node.value
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            ka, a = self.known(node.left)
            kb, b = self.known(node.comparators[0])
            if ka and kb:
                op = node.ops[0]
                if isinstance(op, (ast.Is, ast.Eq)):
                    return True, a == b
                if isinstance(op, (ast.IsNot, ast.NotEq)):
                    return True, a != b
        return False, None

    def visit_If(self, node):
        known, value = self.known(node.test)
        if known:
            result = []
            for item in node.body if value else node.orelse:
                child = self.visit(item)
                result.extend(child if isinstance(child, list) else [child])
            return result
        return self.generic_visit(node)

    def visit_IfExp(self, node):
        known, value = self.known(node.test)
        return self.visit(node.body if value else node.orelse) if known else self.generic_visit(node)

    def visit_FunctionDef(self, node):
        node = self.generic_visit(node)
        args = node.args
        kept = [(a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults) if a.arg not in self.bindings]
        args.kwonlyargs = [a for a, _ in kept]
        args.kw_defaults = [d for _, d in kept]
        # Ignore annotations/docstrings, which do not alter execution.
        for a in args.args + args.kwonlyargs:
            a.annotation = None
        node.returns = None
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:]
        return node


def function(source, name):
    found = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.dump(Specialize().visit(found), include_attributes=False)


def main():
    root = Path(__file__).resolve().parents[4]
    baseline = "0eac91890268c6d5e8572a56298fc979440dbd14"
    paths = {
        "src/gpuwrf/physics/thompson_column.py": ["_wrf_mixed_phase_sources", "_fall_speeds", "_snow_moment", "_snow_moments", "_rain_distribution", "_ice_distribution", "_wrf_warm_rain_rates"],
        "src/gpuwrf/physics/thompson_aero_column.py": ["_entry_cloud_number", "_ice_koop", "_eff_aero", "_saturation_adjustment_aero", "_rain_evaporation_aero", "_sed_cloud_water_aero", "_cloud_water_fall_speeds_aero", "_finish_aero"],
    }
    rows = []
    for path, names in paths.items():
        before = subprocess.check_output(["git", "show", f"{baseline}:{path}"], cwd=root, text=True)
        after = (root/path).read_text()
        for name in names:
            a, b = function(before, name), function(after, name)
            rows.append(dict(path=path, function=name, exact=a == b, before_sha256=hashlib.sha256(a.encode()).hexdigest(), after_sha256=hashlib.sha256(b.encode()).hexdigest()))
    result = dict(scope="Default None/False optional-hook AST specialization; regression only", baseline=baseline, candidate=subprocess.check_output(["git", "rev-parse", "HEAD"],cwd=root,text=True).strip(), functions=rows)
    (Path(__file__).parent/"default_raw_hooks_ast.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result, indent=2))
    assert all(row["exact"] for row in rows)


if __name__ == "__main__":
    main()
