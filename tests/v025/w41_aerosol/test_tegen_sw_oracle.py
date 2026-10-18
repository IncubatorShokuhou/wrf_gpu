"""w41-aerosol AE04/AE09: the port's RRTMG SW with aerosol vs pristine WRF V4.7.1 RRTMG_SWRAD (CPU).

Oracles: AE01 (aer_opt=1 Tegen, tegen_sw_0227_v1) and AE07 (aer_opt=3 Thompson, thompson_aer_sw_0227_v1) on the
186 real RE01 WN3 0227 columns (8 cloud regimes, day + night, several dates).  Gates (pre-registered):
  G1  every aerosol arm inside the frozen tier-1 RRTMG bounds (flux 1 W/m2 + 5 %, heating 1e-4 K/s + 5 %) for SWDOWN,
      TOA up, direct, clear-sky surface down / TOA up, surface..model-top interface profiles and heating.
      AE09 (aer_opt=3 only): in the cells where pristine WRF's own result moves outside that bound under 1-ulp input
      perturbations (32-member T/QNWFA ensemble of every column and arm: 5 cells, all column 13 / arm 3, the
      two-stream of a thin top layer carrying gt_aod's 1e-3 AOD floor at coszen .97) the port is gated against WRF's
      ensemble envelope widened by the same bound; the plain member-0 miss there is pinned;
  G2  the aerosol EFFECT (aerosol arm minus its aer_opt=0 arm, both port) within the same bound of WRF's effect.
      Every SW path emits WRF's ztdbt_nodel direct beam since w41-swddir (the released v0.4.0 aer_opt=0 solver used
      the delta-scaled one, SWDDIR off by up to ~130 W/m2 on cloudy columns), so SWDDIR and its effect are gated on
      all columns like every other output;
  G3  each genuine mutant of the aerosol code fails G1 or G2 on some arm (E202: assertion kills, not crashes).
The release path (REAL entry + fused Pallas band kernel, CPU interpret) runs in a subprocess with
GPUWRF_FAST_DEFAULTS=1 and proves the aerosol kernel was dispatched; the legacy XLA branch runs in-process.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNNER = HERE / "sw_oracle_runner.py"
AER1_MUTANTS = ("no_aer", "ssa_one", "asym_zero", "kernel_scaled_direct", "kernel_no_delta")
AER3_MUTANTS = ("no_aer", "aer3_no_floor", "aer3_surface_rh", "kernel_scaled_direct")


def _run(tmp: Path, args, *, fast: bool, extra_env=None) -> dict:
    out = tmp / ("release.json" if fast else "legacy.json")
    env = dict(os.environ, JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0", GPUWRF_FAST_DEFAULTS="1" if fast else "0",
               PYTHONPATH=str(ROOT / "src"), **(extra_env or {}))
    proc = subprocess.run([sys.executable, str(RUNNER), str(out), *args], env=env, capture_output=True, text=True,
                          timeout=5400)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(out.read_text())


def _worst(variant: dict, key: str, *, aerosol_arms_only: bool = True) -> float:
    """Worst ratio of ``key`` over the aerosol arms."""

    rows = [r for r in variant["arms"].values()
            if not aerosol_arms_only or r.get("effect") is not None or r.get("optics") is not None]
    vals = []
    for row in rows:
        for name, v in (row.get(key) or {}).items():
            vals.append(v["max_ratio"] if isinstance(v, dict) else v)
    return max(vals) if vals else 0.0


@pytest.fixture(scope="module")
def aer1_release(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("aer1"), ["port:all", *(f"{m}:0,10" for m in AER1_MUTANTS)], fast=True)


@pytest.fixture(scope="module")
def aer3_release(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("aer3"), ["--aer3", "port:all", *(f"{m}:0,2" for m in AER3_MUTANTS)],
                fast=True)


def _check_arms(port, *, legacy=False):
    """G1 + G2 on the port variant (legacy: the 27 clear columns only, see the legacy test)."""

    bound, effect = ("bound_clear", "effect_clear") if legacy else ("bound", "effect")
    n_effect = 0
    for arm, row in port["arms"].items():
        aerosol = row.get("effect") is not None or row.get("optics") is not None or row.get("aer_arm")
        b = dict(row[bound])
        assert max(b.values()) <= 1.0, (arm, b, aerosol)  # G1
        for key, eff in (row.get(effect) or {}).items():
            assert eff["max_ratio"] <= 1.0, (arm, key, eff)  # G2
        n_effect += bool(row.get(effect))
    return n_effect


def _check_release(res, oracle):
    assert res["platform"] == "cpu" and res["real_entry"] and res["band_sums"] > 0 and res["oracle"] == oracle
    assert res["gpuwrf"].startswith(str(ROOT / "src"))
    port = res["variants"]["port"]
    assert sum(port["aer_kernel_calls"].values()) > 0  # the aerosol band kernel ran (not the released one)
    assert sum(res["variants"]["no_aer"]["aer_kernel_calls"].values()) == 0
    return port


def test_aer1_release_path_inside_frozen_bounds_and_effect(aer1_release):
    port = _check_release(aer1_release, "AE01")
    assert _check_arms(port) == 6
    # w41-swddir: the aer_opt=0 arms emit WRF's ztdbt_nodel beam too (v0.4.0 missed it by up to 54x here)
    assert max(port["arms"][a]["bound"]["swddir"] for a in ("0", "4", "6")) <= 1.0


@pytest.mark.parametrize("mutant", AER1_MUTANTS)
def test_aer1_mutants_are_caught(aer1_release, mutant):
    res = aer1_release["variants"][mutant]
    assert max(_worst(res, "bound"), _worst(res, "effect")) > 1.0, res  # G3


def test_ae09_envelope_cells_are_wrf_unreproducible_and_few():
    """The AE09 fixture is self-justifying: every kept cell's WRF ensemble range exceeds the frozen bound, only
    aer_opt=3 arms are affected, and the cells are a vanishing fraction of the compared AE07 cells."""

    sys.path.insert(0, str(HERE))
    from tegen_fixture import AER3, AER3_ENS, FLUX_BOUND, HEATING_BOUND, load

    ens, ae = load(AER3_ENS), load(AER3)
    assert int(ens["members"]) == 32
    total = compared = 0
    for key in ("swdnb", "swupt", "swddir", "swdnbc", "swuptc", "swdn", "swup", "hsw"):
        cells, lo, hi = ens[f"{key}_cells"], ens[f"{key}_lo"], ens[f"{key}_hi"]
        bound = HEATING_BOUND if key == "hsw" else FLUX_BOUND
        member0 = ae[key][tuple(cells.T)]
        assert np.all((lo <= member0) & (member0 <= hi))  # WRF member 0 (= AE07) lies inside its own envelope
        assert np.all(np.maximum(hi - member0, member0 - lo) > bound[0] + bound[1] * np.abs(member0))
        assert np.all(ae["arm_aer_opt"][cells[:, 1]] == 3)
        total += len(cells)
        compared += ae[key][..., :45].size if key in ("swdn", "swup") else ae[key].size
    assert total == 5 and total < 1e-4 * compared


def test_aer3_release_path_inside_frozen_bounds_and_effect(aer3_release):
    port = _check_release(aer3_release, "AE07")
    assert _check_arms(port) == 3
    # AE09: the envelope replaced member 0 only in the 5 WRF-unreproducible cells; there the plain comparison misses
    assert {a: r["envelope_cells"] for a, r in port["arms"].items() if r["envelope_cells"]} == {"3": 5}
    assert port["arms"]["3"]["bound_raw"]["hsw"] > 1.0
    assert all(r["bound_raw"] == r["bound"] for a, r in port["arms"].items() if a != "3")
    for arm, row in port["arms"].items():
        optics = row.get("optics")
        if optics is not None:  # gt_aod BYTE; spectral Lagrange optics to REAL rounding
            assert optics["finite"] and optics["taod5503d_max_rel"] == 0.0 and optics["taod5502d_max_rel"] < 1e-6
            assert optics["tauaer_max_rel"] < 1e-5 and optics["ssaaer_max_abs"] < 1e-5 and optics["asyaer_max_abs"] < 1e-5


@pytest.mark.parametrize("mutant", AER3_MUTANTS)
def test_aer3_mutants_are_caught(aer3_release, mutant):
    res = aer3_release["variants"][mutant]
    optics_bad = any(r.get("optics") and (r["optics"]["taod5503d_max_rel"] > 1e-3 or r["optics"]["ssaaer_max_abs"] > 1e-3
                                          or r["optics"]["tauaer_max_rel"] > 1e-3) for r in res["arms"].values())
    assert max(_worst(res, "bound"), _worst(res, "effect")) > 1.0 or (mutant != "no_aer" and optics_bad), res


def test_xla_band_scan_branch_matches_wrf_and_its_direct_beam_mutant_is_caught(tmp_path):
    """The XLA band-scan aerosol branch (release REAL entry/reftra/overlap, fused band kernel OFF:
    GPUWRF_RRTMG_SW_BAND_SUMS=0) passes G1/G2 for both options; dropping its nodel direct beam is caught."""

    env = {"GPUWRF_RRTMG_SW_BAND_SUMS": "0"}
    aer1 = _run(tmp_path, ["port:0,1,10", "xla_scaled_direct:0,10"], fast=True, extra_env=env)
    assert aer1["real_entry"] and aer1["band_sums"] == 0 and aer1["oracle"] == "AE01"
    assert sum(aer1["variants"]["port"]["aer_kernel_calls"].values()) == 0  # not the Pallas kernel
    assert _check_arms(aer1["variants"]["port"]) == 2
    assert _worst(aer1["variants"]["xla_scaled_direct"], "bound") > 1.0
    (tmp_path / "a3").mkdir()
    aer3 = _run(tmp_path / "a3", ["--aer3", "port:0,1,2"], fast=True, extra_env=env)
    assert _check_arms(aer3["variants"]["port"]) == 2


def test_legacy_entry_carries_the_aerosol_on_clear_sky(tmp_path):
    """FAST_DEFAULTS=0 (f64 entry, legacy reftra, random McICA overlap): its cloudy columns miss WRF with or without
    aerosol (RE01 arm D), and its two-stream differs from WRF's by ~2 W/m2 on clear days, which no longer cancels in
    the x10 aerosol effect (TOA up 2.9 W/m2 of ~50).  Gate: clear-column G1 for every arm, clear-column G2 for the
    climatological (scale 1) arms."""

    aer1 = _run(tmp_path, ["port:0,1,10"], fast=False)
    assert aer1["band_sums"] == 0 and not aer1["real_entry"]
    port = aer1["variants"]["port"]["arms"]
    for arm, row in port.items():
        assert max(row["bound_clear"].values()) <= 1.0, (arm, row["bound_clear"])
    for key, eff in port["1"]["effect_clear"].items():
        assert eff["max_ratio"] <= 1.0, (key, eff)
    assert max(e["max_err"] for e in port["10"]["effect_clear"].values()) < 4.0  # disclosed legacy-solver residual


def test_bound_helper_rejects_non_finite():
    from tegen_fixture import bound_ratio

    assert np.isinf(bound_ratio(np.ones(3), np.array([1.0, np.nan, np.inf]), (1.0, 0.05))[1:]).all()
