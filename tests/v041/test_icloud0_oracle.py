"""w42-icloud0 (ALISIOS F7): icloud=0 against the actual pristine WRF V4.7.1 radiation_driver.

Z01 (proofs/w42_icloud0/driver_oracle.py, receipt fixtures/z01_radiation_driver_icloud.json): the unchanged pristine
radiation_driver + RRTMG on 16 real WN3 0227 RE01 columns (liquid thin/thick/land, MYNN-SGS-only, clear, ice, mixed,
snow), icloud = 0 and the icloud = 1 control.  Z02 (port_vs_driver.py): the release RRTMG path (GPUWRF_FAST_DEFAULTS=1,
REAL entry + band kernels in CPU interpret) fed the port's icloud views, at the frozen RE01 tier-1 bounds (flux
1 W/m2 + 5 %, heating 1e-4 K/s + 5 %), plus the w41-swddir mutant (SWDDIR fix reverted) for the SWDDIR interplay.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
PROOFS = ROOT / "proofs/w42_icloud0"
Z01 = PROOFS / "fixtures/z01_radiation_driver_icloud.json"
RE01 = ROOT / "tests/v025/fid_q2/fixtures/rrtmg_mp_re_0227_v1.npz"


@pytest.fixture(scope="module")
def z01():
    data = json.loads(Z01.read_text())
    rows = {(r["column"], r["icloud"]): r for r in data["rows"]}
    return data, rows


def test_z01_receipt_is_the_pristine_driver_on_the_registered_columns(z01):
    data, rows = z01
    assert data["real_operand_fixture_sha256"] == hashlib.sha256(RE01.read_bytes()).hexdigest()
    assert len(data["input_columns"]) == 16 and len(rows) == 32
    assert {"clear", "ice", "mixed", "snow", "liq_thin_ocean", "liq_thick", "liq_land", "sgs_only"} == set(data["categories"])
    assert data["caller_controls"]["mp_physics"] == 8 and data["caller_controls"]["cldovrlp"] == 2
    from gpuwrf.config.paths import wrf_root
    source = wrf_root() / "phys/module_radiation_driver.F"
    if not source.exists():
        pytest.skip("pristine WRF tree not present (data/wrf_pristine)")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == data["source_sha256"]


def test_pristine_icloud0_is_cloud_free_radiation_and_clouds_matter(z01):
    data, rows = z01
    cloudy = 0
    for col, cat in zip(data["input_columns"], data["categories"]):
        off, on = rows[(col, 0)], rows[(col, 1)]
        s0, s1 = off["scalars"], on["scalars"]
        assert not np.any(off["profiles"]["cldfra"]) and s0["cldt"] == 0.0  # radiation_driver.F:1309-1318
        assert s0["lwcf"] == 0.0 and s0["swcf"] == 0.0
        assert s0["glw"] == s0["lwdnbc"] and abs(s0["swdown"] - s0["swdnbc"]) <= 1e-4  # all-sky == clear-sky
        for key in ("lwdnbc", "lwuptc", "swdnbc", "swuptc", "swdownc"):
            assert s0[key] == s1[key], key  # the clear-sky stream never sees the icloud choice
        np.testing.assert_array_equal(off["profiles"]["rthratenlwc"], on["profiles"]["rthratenlwc"])
        if cat != "clear":
            cloudy += max(abs(s1["glw"] - s0["glw"]) > 1.0 + .05 * abs(s0["glw"]),
                          abs(s1["swdown"] - s0["swdown"]) > 1.0 + .05 * abs(s0["swdown"]))
        else:
            assert s0 == s1
    assert cloudy >= 12  # the switch changes radiation far beyond the frozen bounds on cloudy columns


@pytest.fixture(scope="module")
def z02(tmp_path_factory):
    out = tmp_path_factory.mktemp("z02") / "z02.json"
    env = dict(os.environ, JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0", GPUWRF_FAST_DEFAULTS="1",
               PYTHONPATH=str(ROOT / "src"))
    proc = subprocess.run([sys.executable, str(PROOFS / "port_vs_driver.py"), str(Z01), str(out), "port",
                           "scaled_direct"], env=env, capture_output=True, text=True, timeout=5400)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(out.read_text())


def test_release_port_matches_pristine_icloud0_and_control(z02):
    assert z02["platform"] == "cpu" and z02["real_entry"] and z02["maxrand"] and z02["band_sums"] > 0
    assert z02["gpuwrf"].startswith(str(ROOT / "src"))
    port = z02["variants"]["port"]
    assert sum(port["band_kernel_calls"].values()) > 0  # E114: the release band kernel ran
    for icloud in ("0", "1"):
        assert max(port["arms"][icloud]["bound"].values()) <= 1.0, (icloud, port["arms"][icloud]["bound"])
    assert port["arms"]["0"]["port_clear_minus_all"] == {"lw_sfc_down": 0.0, "sw_sfc_down": 0.0}
    assert port["arms"]["1"]["port_clear_minus_all"]["sw_sfc_down"] > 100.0  # control arm is cloudy


def test_swddir_fix_is_inert_under_icloud0_and_caught_with_clouds(z02):
    """ALISIOS F7 + #146: with icloud=0 the beam is cloud-free, so the v0.4.1 SWDDIR fix cannot change SWDDIR."""
    port, mutant = z02["variants"]["port"]["arms"], z02["variants"]["scaled_direct"]["arms"]
    np.testing.assert_array_equal(mutant["0"]["port"]["swddir"], port["0"]["port"]["swddir"])
    assert max(mutant["0"]["bound"].values()) <= 1.0
    assert mutant["1"]["bound"]["swddir"] > 1.0 and mutant["1"]["bound"]["swddif"] > 1.0
