"""Runtime Nt_c bakes bundled freezing/gamma tables: their content must split the AOT key (E80/E45)."""
from dataclasses import replace

import numpy as np
import pytest

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.physics import thompson_nt_c
from gpuwrf.physics.thompson_parameters import validate_parameters
from gpuwrf.runtime.aot_baked_tables import baked_table_hash
from gpuwrf.runtime.operational_mode import OperationalNamelist


@pytest.fixture(autouse=True)
def native(monkeypatch):
    for suffix in ("NATIVE_REAL", "FULL_COLUMN", "MIXED_PHASE_WRF", "COLUMN_SED", "SED_FP32", "WRF_CONSTANTS"):
        monkeypatch.setenv("GPUWRF_THOMPSON_" + suffix, "1")
    monkeypatch.setenv("GPUWRF_THOMPSON_IMPLICIT_SED", "0")
    monkeypatch.setenv("GPUWRF_MP_COLUMN_TILING", "0")


def _fn(x, nl):
    return x


def test_nt_c_table_content_splits_only_the_nt_c_key(monkeypatch):
    off = OperationalNamelist(GridSpec.canary_3km_template(), None, None, 18., run_physics=True, mp_physics=8)
    tno = replace(off, thompson_parameters=validate_parameters({"tno": 5.}))
    nt_a = replace(off, thompson_parameters=validate_parameters({"nt_c": 50e6}))
    nt_b = replace(off, thompson_parameters=validate_parameters({"nt_c": 150e6}))
    before = {name: baked_table_hash(_fn, nl) for name, nl in (("off", off), ("tno", tno), ("nt_a", nt_a), ("nt_b", nt_b))}
    # Only the table selection is keyed here; the Nt value is a traced child (same key for 50/150).
    assert before["off"] == before["tno"] != before["nt_a"] == before["nt_b"]
    changed = {key: np.array(value, copy=True) for key, value in thompson_nt_c.host_tables().items()}
    changed["tpi_qcfz"].flat[0] = np.nextafter(changed["tpi_qcfz"].flat[0], np.inf)
    monkeypatch.setattr(thompson_nt_c, "host_tables", lambda: changed)
    assert baked_table_hash(_fn, nt_a) != before["nt_a"]
    assert baked_table_hash(_fn, off) == before["off"]
    assert baked_table_hash(_fn, tno) == before["tno"]
