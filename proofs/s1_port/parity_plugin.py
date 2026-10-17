"""Test-only per-family dispatch receipt; no production instrumentation.

Load with pytest -p proofs.s1_port.parity_plugin and PORT_PARITY_COVERAGE=/...json.
The receipt records calls and test outcomes, not numerical qualification by itself.
"""
import json
import os
from pathlib import Path
import sys

_current = None
_calls = {}
_reports = {}
_original = None


def pytest_configure(config):
    global _original
    from jax.experimental import pallas

    _original = pallas.pallas_call

    def record(*args, **kwargs):
        frame = sys._getframe(1)
        module = "unknown"
        while frame is not None:
            name = frame.f_globals.get("__name__", "")
            if (name.startswith("gpuwrf.kernels.") or
                    name in {"gpuwrf.nesting.sint_kernel", "gpuwrf.coupling.boundary_apply"}):
                module = name
                break
            frame = frame.f_back
        key = module + ":" + str(kwargs.get("name", "unnamed"))
        _calls.setdefault(key, set()).add(_current)
        return _original(*args, **kwargs)

    pallas.pallas_call = record


def pytest_runtest_setup(item):
    global _current
    _current = item.nodeid


def pytest_runtest_logreport(report):
    if report.when == "call" or report.failed:
        _reports[report.nodeid] = {"phase": report.when, "outcome": report.outcome}


def pytest_sessionfinish(session, exitstatus):
    output = os.environ.get("PORT_PARITY_COVERAGE")
    if output:
        Path(output).write_text(json.dumps({
            "platform": "cpu", "gpu_executed": False, "exitstatus": int(exitstatus),
            "calls": {k: sorted(v, key=str) for k, v in sorted(_calls.items())},
            "tests": _reports,
            "scope": "dispatch observation; numerical gate is each named test's assertions",
        }, indent=2) + "\n")


def pytest_unconfigure(config):
    if _original is not None:
        from jax.experimental import pallas
        pallas.pallas_call = _original
