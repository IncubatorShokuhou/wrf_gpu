"""ALISIOS data rule for the a1 test suite (wrf_gpu2 manager 2026-10-17 01:25Z).

No ALISIOS data may be read outside ``alisios.guard_call`` + read spy. Every test here runs under a refusing spy:
any open of a path below the ALISIOS data roots (Python ``open`` via audit hook, netCDF via ``netCDF4.Dataset``)
raises ``AlisiosReadBlocked`` unless the test itself entered ``alisios_guard.guarded_session`` (which installs its
own listed-path spy). Such a blocked real-file test is reported as SKIPPED, never PASS, with the path in the reason.
Synthetic tests are unaffected.
"""
from __future__ import annotations

import os
import sys

import pytest

from gpuwrf.assimilation.letkf import alisios_guard


class AlisiosReadBlocked(PermissionError):
    pass


_BLOCKING = {"on": False}


def _audit(event, args):
    if (event != "open" or not _BLOCKING["on"] or alisios_guard._ACTIVE is not None
            or alisios_guard.in_guard_setup()):
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


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if call.excinfo is not None and call.excinfo.errisinstance(AlisiosReadBlocked):
        report.outcome = "skipped"
        report.longrepr = (str(item.fspath), item.location[1] or 0, f"SKIPPED: {call.excinfo.value}")
