"""w41-swddir: RRTMG SW SWDDIR/SWDDIF vs pristine WRF V4.7.1 RRTMG_SWRAD (WRF's ztdbt_nodel direct output).

Pre-registered gates (frozen tier-1 RRTMG bounds: flux 1 W/m2 + 5 %, heating 1e-4 K/s + 5 %):
  G1  release path (GPUWRF_FAST_DEFAULTS=1: REAL entry + fused Pallas band kernel, CPU interpret) on every aer_opt=0
      arm of the AE01 oracle (186 real RE01 WN3 0227 columns, own call + 4 other julians) and on the S01 Swiss oracle
      (real Swiss CPU-WRF 2023-01-15 d01 day columns): SWDDIR, SWDDIF and every other SW output inside the bounds on
      ALL columns (v0.4.0 missed SWDDIR by up to 54x on the cloudy WN3 columns);
  G2  the genuine mutant ``kernel_scaled_direct`` (default kernels emit the released delta-scaled beam) fails G1 on
      SWDDIR, and only there (E202: the same solve with one product reverted);
  G3  the XLA band-scan branch (GPUWRF_RRTMG_SW_BAND_SUMS=0) passes G1 on two arms and its mutant
      ``xla_scaled_direct`` fails it.
The legacy entry (GPUWRF_FAST_DEFAULTS=0, arm D wiring) is gated in tests/v025/fid_q2/test_rrtmg_mp_re_port_today.py
(SWDDIR vs pristine arm D).  The default band kernel must actually run (E114 spy).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNNER = HERE / "swddir_oracle_runner.py"
AE01_OFF_ARMS = "0,2,4,6,8"


def _run(tmp: Path, args, *, extra_env=None) -> dict:
    out = tmp / "result.json"
    env = dict(os.environ, JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0", GPUWRF_FAST_DEFAULTS="1",
               PYTHONPATH=str(ROOT / "src"), **(extra_env or {}))
    proc = subprocess.run([sys.executable, str(RUNNER), str(out), *args], env=env, capture_output=True, text=True,
                          timeout=5400)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(out.read_text())


def _check_release(res, oracle, *, band_sums=True):
    assert res["platform"] == "cpu" and res["real_entry"] and res["oracle"] == oracle
    assert res["gpuwrf"].startswith(str(ROOT / "src"))
    assert (res["band_sums"] > 0) == band_sums
    port = res["variants"]["port"]
    assert (sum(port["band_kernel_calls"].values()) > 0) == band_sums  # E114: the path under test ran
    return port


def _assert_inside(port):
    for arm, row in port["arms"].items():
        assert max(row["bound"].values()) <= 1.0, (arm, row["bound"])  # G1, every output incl. SWDDIR/SWDDIF


def _assert_mutant_caught(mutant):
    worst = {arm: row["bound"] for arm, row in mutant["arms"].items()}
    assert max(b["swddir"] for b in worst.values()) > 1.0, worst  # G2/G3
    assert max(b["swddif"] for b in worst.values()) > 1.0, worst
    for b in worst.values():  # one product reverted: nothing else moves
        assert max(v for k, v in b.items() if k not in ("swddir", "swddif")) <= 1.0, b


@pytest.fixture(scope="module")
def ae01_release(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("ae01"), [f"port:{AE01_OFF_ARMS}", "kernel_scaled_direct:0,6"])


def test_release_path_swddir_matches_wrf_on_wn3_columns(ae01_release):
    port = _check_release(ae01_release, "AE01")
    assert sorted(port["arms"]) == sorted(AE01_OFF_ARMS.split(","))
    _assert_inside(port)
    assert all(row["n_cloudy_day"] >= 60 for a, row in port["arms"].items() if a in ("0", "6"))


def test_release_kernel_mutant_is_caught_on_wn3_columns(ae01_release):
    mutant = ae01_release["variants"]["kernel_scaled_direct"]
    assert sum(mutant["band_kernel_calls"].values()) > 0
    _assert_mutant_caught(mutant)


def test_release_path_swddir_matches_wrf_on_swiss_columns(tmp_path):
    res = _run(tmp_path, ["--swiss", "port:all", "kernel_scaled_direct:all"])
    port = _check_release(res, "S01")
    _assert_inside(port)
    _assert_mutant_caught(res["variants"]["kernel_scaled_direct"])


def test_xla_band_scan_branch_swddir_and_mutant(tmp_path):
    res = _run(tmp_path, ["port:0,6", "xla_scaled_direct:0"], extra_env={"GPUWRF_RRTMG_SW_BAND_SUMS": "0"})
    port = _check_release(res, "AE01", band_sums=False)
    _assert_inside(port)
    _assert_mutant_caught(res["variants"]["xla_scaled_direct"])
