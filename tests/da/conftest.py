"""ALISIOS data rule for the gpuwrf da product tests (same rule as tests/a1_da, wrf_gpu2 manager 2026-10-17 01:25Z).

No ALISIOS data may be read outside ``alisios.guard_call`` + read spy. Every test here runs under a refusing spy:
any open of a path below the ALISIOS data roots (Python ``open`` via audit hook, netCDF via ``netCDF4.Dataset``)
raises ``AlisiosReadBlocked`` unless the test itself entered ``alisios_guard.guarded_session`` (which installs its
own listed-path spy). Such a blocked real-file test is reported as SKIPPED, never PASS, with the path in the reason.
Synthetic tests are unaffected.
"""
from __future__ import annotations

import importlib
import os
import sys

import pytest


_AG = "gpuwrf.assimilation.letkf.alisios_guard"


def _ag():
    """The CURRENT alisios_guard module (looked up per call: the mutant overlay replaces it after this conftest)."""
    module = sys.modules.get(_AG)
    return module if module is not None else importlib.import_module(_AG)


class AlisiosReadBlocked(PermissionError):
    pass


_BLOCKING = {"on": False}


def _audit(event, args):
    if event != "open" or not _BLOCKING["on"]:
        return
    alisios_guard = sys.modules.get(_AG)       # no import from inside an audit hook
    if (alisios_guard is None or not hasattr(alisios_guard, "is_alisios_data")
            or alisios_guard._ACTIVE is not None or alisios_guard.in_guard_setup()):
        return
    file, mode, _ = args
    if isinstance(file, (str, bytes, os.PathLike)) and (not isinstance(mode, str) or "r" in mode or "+" in mode):
        if alisios_guard.is_alisios_data(file):
            raise AlisiosReadBlocked(f"unguarded ALISIOS read blocked: {os.fsdecode(file)}")


sys.addaudithook(_audit)


@pytest.fixture(autouse=True)
def _block_unguarded_alisios_reads(monkeypatch):
    import netCDF4

    original = netCDF4.Dataset

    def blocking_dataset(filename, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        reading = isinstance(mode, str) and (mode.startswith("r") or mode.startswith("a"))
        alisios_guard = _ag()
        if (reading and alisios_guard._ACTIVE is None and not alisios_guard.in_guard_setup()
                and alisios_guard.is_alisios_data(filename)):
            raise AlisiosReadBlocked(f"unguarded ALISIOS read blocked: {filename}")
        return original(filename, *args, **kwargs)

    monkeypatch.setattr(netCDF4, "Dataset", blocking_dataset)
    _BLOCKING["on"] = True
    try:
        yield
    finally:
        _BLOCKING["on"] = False


def pytest_configure(config):
    config.addinivalue_line("markers", "alisios_facade: the test may use the sealed ALISIOS reader facade")


@pytest.fixture(autouse=True)
def _alisios_facade_opt_in(request, monkeypatch):
    """The ``alisios`` plug-in prefers the sealed ALISIOS reader facade, which runs the REAL guard; tests use a
    fake guard, so the facade is off unless a test is marked ``alisios_facade`` (and fakes the sealed guard)."""
    if request.node.get_closest_marker("alisios_facade") is None:
        monkeypatch.setattr(_ag(), "facade", lambda sealed_root=None: (
            None, None, {"unavailable": "tests/da: facade disabled unless the test is marked alisios_facade"}))
    yield


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if call.excinfo is not None and call.excinfo.errisinstance(AlisiosReadBlocked):
        report.outcome = "skipped"
        report.longrepr = (str(item.fspath), item.location[1] or 0, f"SKIPPED: {call.excinfo.value}")
