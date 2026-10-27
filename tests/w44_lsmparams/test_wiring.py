"""W2 physics wiring through the actual noah_mp_step (critic F6, E39/E202).

Retained real columns: four warm day/night columns, and a cold cohort with a
frozen surface, a structured snowpack, two real frozen-soil columns and a
declared warm-rain-on-frozen-ground stress column. One compiled program per
cohort and route; values ride the production clock seam.

* Every control, changed alone, must change Noah-MP outputs (no-op mutant).
* Every documented consumer bundle (ROUTES: static.parameters for WATER/
  PHENOLOGY/Jarvis, EnergyParams for ENERGY/PHASECHANGE) is observed: the full
  apply must differ from the variant that drops that bundle's new value, so
  removing either route from lsmparams.apply fails here (static-only and
  energy-only mutants).
* ON at the prior values equals OFF bytes, or a constant-folding association
  class: capturing the same values as compile-time constants (same apply path,
  same equations) restores every OFF byte (the LS10/LS43 intervention), and each
  remaining difference is <= 6 ULP of a REAL field (ERRENG: of its flux scale).
  That class belongs to the shipped Noah-MP REAL path (release defaults). The CPU
  suite keeps the legacy defaults (tests/conftest.py), whose f64 Noah path reads
  f64 table gathers when OFF; there the comparison is skipped and the whole module
  is re-run under the release defaults in a clean child process.
"""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "proofs/w44_lsmparams"))
from columns import build, frozen_soil_columns, source_columns
from gpuwrf.io.noahmp_options import NoahMPOptions
from gpuwrf.physics.noahmp.noahmp_driver import noah_mp_step
from gpuwrf.runtime.operational_mode import OperationalNamelist, build_clock_base
from gpuwrf.surface_params import controls as sc

NAMES = sorted(sc.LSM_CONTROLS)
FACTOR = {"hvb": .01, "snow_emis": .99}  # as the pristine oracle arms: low HVB activates snow burial
# Consumer bundles under the admitted options (opt_run=3, opt_soil=1, opt_irr=0,
# opt_tdrn=0). PSISAT also rides static.parameters, but only the refused
# groundwater/MMF runoff options read it there.
ROUTES = {"smcmax": {"static", "energy"}, "bexp": {"static", "energy"}, "dksat": {"static"},
          "psisat": {"energy"}, "hvt": {"static", "energy"}, "hvb": {"static"}, "rsmin": {"static"},
          "z0sno": {"energy"}, "snow_emis": {"energy"}}
# HVT's PHENOLOGY consumer is snow burial, FB=(SNOWH-HVB)/(HVT-HVB). With the table HVB
# above the 0.3 m snowpack FB=0, so the HVT arms use HVB x.01 in reference AND changed.
CONTEXT = {"hvt": {"hvb": .01}}
ROUNDOFF_ULPS = 6  # LS16/LS42 cold maximum (QFX); warm LS08/LS09 1-2 ULP


def _ulps(a, b):
    def ordered(v):
        bits = v.view(np.int32).astype(np.int64)
        return np.where(bits < 0, -(bits & 0x7fffffff), bits)
    return int(np.abs(ordered(a) - ordered(b)).max())


def _named(out):
    """Field-named host leaves of (land, fluxes, diagnostics)."""
    names = {}
    for part, obj in zip(("land", "flux", "diag"), out, strict=True):
        if isinstance(obj, dict):
            items = obj.items()
        elif hasattr(obj, "_asdict"):
            items = obj._asdict().items()
        else:
            items = ((key, getattr(obj, key, None)) for key in obj.__slots__)
        for key, value in items:
            for sub, leaf in jax.tree_util.tree_flatten_with_path(value)[0]:
                names[f"{part}.{key}{jax.tree_util.keystr(sub)}"] = np.asarray(leaf)
    return names


def _cohort(rows):
    options = NoahMPOptions(opt_crs=2)  # Jarvis, so RSMIN has its WRF consumer
    land, forcing, static, ep, rp, fixture, rows = build(rows=rows, options=options)
    dt = float(rows[0]["dt"])
    nl = OperationalNamelist(grid=None, tendencies=None, metrics=None, dt_s=dt, use_noahmp=True,
                             noahmp_static=static, noahmp_land=land, noahmp_energy_params=ep,
                             noahmp_rad_params=rp, noahmp_options=options)
    prior = sc.prior_controls(fixture, NAMES, device=False)

    def program(land, forcing, nl, cb, other, route):
        s, e, r = nl.noahmp_static, nl.noahmp_energy_params, nl.noahmp_rad_params
        values = sc.traced_controls(nl, cb)
        if values is not None:
            if route == "const":  # causal control: identical values as compile-time constants
                values = {n: jnp.asarray(v) for n, v in prior.items()}
            s1, e1, r1 = sc.apply_surface_controls(s, e, r, values)
            if route in ("static", "energy"):
                # Keep one bundle's new value, take the other bundle from ``other``.
                s0, e0, _ = sc.apply_surface_controls(s, e, r, other)
                s1, e1 = (s1, e0) if route == "static" else (s0, e1)
            s, e, r = s1, e1, r1
        return noah_mp_step(land, forcing, s, dt, energy_params=e, rad_params=r,
                            options=nl.noahmp_options, return_diag=True)

    fn = jax.jit(program, static_argnames="route")

    def run(controls, route="full", other=None):
        arm = nl if controls is None else replace(nl, surface_controls=controls)
        out = fn(land, forcing, arm, build_clock_base(arm), prior if other is None else other, route=route)
        out = _named(jax.device_get(out))
        assert all(np.isfinite(x).all() for x in out.values() if x.dtype.kind == "f"), "non-finite output"
        return out

    def reference(name):
        values = {n: v.copy() for n, v in prior.items()}
        for other, factor in CONTEXT.get(name, {}).items():
            values[other] = sc.expand_controls(fixture, {other: factor}, device=False)[other]
        return values

    def changed(name):
        values = reference(name)
        values[name] = sc.expand_controls(fixture, {name: FACTOR.get(name, 1.1)}, device=False)[name]
        assert values[name].tobytes() != prior[name].tobytes()
        return values

    return run, reference, changed, prior, ep, fn


@pytest.fixture(scope="module")
def cohorts():
    # Retained column research only. Restore public refusal BEFORE tests receive
    # the arrays; there is no production/runtime/API bypass.
    with pytest.MonkeyPatch.context() as policy:
        policy.setattr(sc, "UNAVAILABLE_SURFACE_CONTROLS", frozenset())
        results = {}
        for label, rows in (("warm", source_columns()[:4]), ("cold", source_columns()[4:] + frozen_soil_columns())):
            run, reference, changed, prior, ep, fn = _cohort(rows)
            record = {"off": run(None), "prior": run(prior), "const": run(prior, "const"), "ep": ep,
                      "rows": [r["name"] for r in rows]}
            for name in NAMES:
                base, values = reference(name), changed(name)
                record[name] = {"ref": record["prior"] if name not in CONTEXT else run(base), "full": run(values)}
                for route in ("static", "energy"):
                    record[name][route] = run(values, route, other=base)
            # OFF, full ON, constant ON, static-only and energy-only programs; values never retrace.
            record["cache_size"] = fn._cache_size()
            results[label] = record
        return results


def _differs(a, b):
    assert a.keys() == b.keys()
    return sorted(key for key in a if a[key].tobytes() != b[key].tobytes())


def test_values_ride_operands_without_retrace(cohorts):
    assert all(r["cache_size"] == 5 for r in cohorts.values()), {k: r["cache_size"] for k, r in cohorts.items()}


@pytest.mark.parametrize("name", NAMES)
def test_changed_control_reaches_noahmp_outputs(cohorts, name):
    hits = {label: _differs(r[name]["ref"], r[name]["full"]) for label, r in cohorts.items()}
    assert any(hits.values()), f"{name}: changed value never reaches any Noah-MP output (no-op wiring)"


@pytest.mark.parametrize("name", NAMES)
def test_every_documented_consumer_bundle_is_reached(cohorts, name):
    ep = cohorts["warm"]["ep"]
    assert ("energy" in ROUTES[name]) <= hasattr(ep, name), f"{name}: EnergyParams has no such field"
    # The "static" variant keeps only static.parameters' new value: it differs from the
    # full apply exactly when an EnergyParams consumer sees the value, and vice versa.
    for route, dropped in (("energy", "static"), ("static", "energy")):
        if route not in ROUTES[name]:
            continue
        assert any(_differs(r[name][dropped], r[name]["full"]) for r in cohorts.values()), \
            f"{name}: the {route} bundle's consumers never see the value (full == {dropped}-only)"


def test_soil_ice_is_active_in_the_cold_cohort(cohorts):
    record = cohorts["cold"]
    ice = record["prior"]["land.smois"] - record["prior"]["land.sh2o"]
    frozen = [i for i, name in enumerate(record["rows"]) if "barren16" in name or "grass10" in name]
    assert np.any(ice[:, 0, frozen] > 1e-6), "frozen-soil columns carry no soil ice: PHASECHANGE consumer vacuous"


@pytest.mark.parametrize("cohort", ["warm", "cold"])
def test_on_at_prior_is_off_bytes_or_registered_roundoff(cohorts, cohort):
    record = cohorts[cohort]
    if record["off"]["flux.lh"].dtype != np.float32:
        pytest.skip("legacy f64 Noah-MP path (CPU-suite defaults): OFF reads f64 table gathers, controls are "
                    "WRF REAL; the release-defaults child run checks the shipped REAL path")
    stats = {}
    for key in _differs(record["off"], record["prior"]):
        off, on = record["off"][key], record["prior"][key]
        delta = np.abs(off.astype(np.float64) - on.astype(np.float64))
        stats[key] = {"dtype": str(off.dtype), "cells": int(np.count_nonzero(off != on)),
                      "max_abs": float(delta.max()), "field_max_abs": float(np.abs(off.astype(np.float64)).max()),
                      "max_ulps": _ulps(off, on) if off.dtype == np.float32 else None}
    receipt = os.environ.get("W2_WIRING_RECEIPT")
    if receipt:
        Path(f"{receipt}.{cohort}.json").write_text(json.dumps(
            {"cohort": cohort, "rows": record["rows"], "off_vs_prior": stats,
             "off_vs_constant_prior": _differs(record["off"], record["const"])}, indent=2) + "\n")
    assert not _differs(record["off"], record["const"]), "constant operands do not restore OFF bytes: not association"
    flux = np.abs(record["off"]["flux.hfx"]) + np.abs(record["off"]["flux.lh"]) + np.abs(record["off"]["flux.grdflx"])
    for key, row in stats.items():
        assert row["dtype"] == "float32", f"{key}: non-REAL output differs {row}"
        if key == "diag.erreng":  # a closure residual near 0: rounding scale is that of its flux terms
            limit = ROUNDOFF_ULPS * float(np.spacing(np.float32(flux.max())))
            assert row["max_abs"] <= limit, f"{key}: {row} > {limit}"
        else:
            assert row["max_ulps"] <= ROUNDOFF_ULPS, f"{key}: {row}"


def test_release_defaults_rerun_in_a_clean_process(tmp_path):
    """Release defaults are import-time constants (E148): re-run this module in a child process."""
    if os.environ.get("W2_WIRING_CHILD") == "1" or os.environ.get("GPUWRF_FAST_DEFAULTS") == "1":
        pytest.skip("already running under the release defaults")
    env = dict(os.environ, GPUWRF_FAST_DEFAULTS="1", W2_WIRING_CHILD="1", JAX_PLATFORMS="cpu")
    report = tmp_path / "child.xml"
    result = subprocess.run([sys.executable, "-m", "pytest", __file__, "-q", "-p", "no:cacheprovider",
                             "--basetemp", str(tmp_path / "child"), "--junitxml", str(report)],
                            env=env, capture_output=True, text=True, timeout=2400)
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-2000:]
    import xml.etree.ElementTree as ET
    cases = ET.parse(report).getroot().iter("testcase")
    skipped = {case.get("name") for case in cases if case.find("skipped") is not None}
    # The REAL path must actually run the round-off comparison (no legacy-path skip).
    assert skipped == {"test_release_defaults_rerun_in_a_clean_process"}, skipped
