"""t-calleraudit: a nested-AOT lower error prints its full traceback (FINAL c18-c24 lesson).

Both lowering sites fail open to the plain jitted path; that path usually re-raises the same error and
the CLI prints only ``type: message`` -- v-prep's c18 log had no pointer to the failing trace site.
"""
from __future__ import annotations

import ast
from pathlib import Path

from gpuwrf.runtime import domain_tree

ROOT = Path(__file__).resolve().parents[1]


def _failing_trace_site():
    raise TypeError("float() argument must be a string or a real number, not 'dict'")


def test_lower_error_text_prints_the_traceback(capsys):
    try:
        _failing_trace_site()
    except TypeError as exc:
        text = domain_tree._lower_error_text(exc)
    assert text == "TypeError: float() argument must be a string or a real number, not 'dict'"
    err = capsys.readouterr().err
    assert "[gpuwrf:nested-aot] lower-error traceback:" in err
    assert "Traceback (most recent call last)" in err and "_failing_trace_site" in err


def test_both_lowering_sites_report_through_the_helper():
    tree = ast.parse((ROOT / "src/gpuwrf/runtime/domain_tree.py").read_text())
    sites = {node.name: node for node in ast.walk(tree)
             if isinstance(node, ast.FunctionDef) and node.name in ("_lower_advance_variant", "_lower_fused")}
    assert set(sites) == {"_lower_advance_variant", "_lower_fused"}
    for name, node in sites.items():
        handlers = [h for h in ast.walk(node) if isinstance(h, ast.ExceptHandler)]
        assert handlers, name
        for handler in handlers:
            calls = {c.func.id for c in ast.walk(handler) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
            assert "_lower_error_text" in calls, name
