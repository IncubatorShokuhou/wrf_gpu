"""Frozen ORIGINAL-WRF witnesses for the cycling initialization (w44-warmstart F3a, E39/E202).

Reference values come only from compiled, unedited WRF Fortran (never from JAX):
- rebalance_cycl (dyn_em/start_em.F:2259-2544) via proofs/w44_warmstart/rebalance_oracle.py
  on real cloudy columns (cycled May d01 input; Swiss CPU-WRF 12Z with QC/QR/QI/QS/QG),
  use_theta_m = 0 and 1. Comparison = that oracle's frozen rule + the measured BYTE identity.
- MYNN cycling init (module_bl_mynnedmf.F driver :618-747 + mym_initialize) via
  proofs/w44_warmstart/mynn_init_oracle.py: the WS11 40-case witness plus 24 real Swiss
  ice+snow cases (qi>0 and qs>0 in every column). Comparison = the frozen WS11 gate LIMITS
  of proofs/w44_warmstart/compare_mynn_init.py under its AVX/no-FMA XLA:CPU condition.
Provenance: tests/data/w44_warmstart/f3a_wrf_witnesses.json (generator alongside).
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

DATA = Path(__file__).resolve().parent / "data" / "w44_warmstart"
NPZ = DATA / "f3a_wrf_witnesses.npz"
META = json.loads((DATA / "f3a_wrf_witnesses.json").read_text())
MOIST = tuple(META["rebalance"]["moist"])
REBAL_CASES = tuple(META["rebalance"]["cases"])
MYNN_SETS = tuple(META["mynn_init"]["sets"])
# proofs/w44_warmstart/compare_mynn_init.py LIMITS (frozen WS11 gate; not a new tolerance).
LIMITS = {"QKE_relative_median": 1e-4, "EL_atol": 1e-3, "EL_rtol": 2e-3, "QSQ_atol": 2e-12, "QSQ_rtol": 2e-3}
# The lane gate condition: XLA:CPU without FMA (gfortran build has no FMA contraction).
XLA_CPU_CONDITION = "--xla_cpu_max_isa=AVX"


@pytest.fixture(scope="module")
def witness():
    assert hashlib.sha256(NPZ.read_bytes()).hexdigest() == META["npz_sha256"]
    with np.load(NPZ) as data:
        return {key: data[key] for key in data.files}


def _rebalance_fields(witness, case):
    prefix = f"rebal__{case}__in__"
    return {key[len(prefix):]: value for key, value in witness.items() if key.startswith(prefix)}


def test_rebalance_witness_is_cloudy(witness):
    """E154 census: every hydrometeor the M1 sum must carry is nonzero in the frozen crops."""
    swiss = _rebalance_fields(witness, "swiss_d01_12z")
    for name in MOIST:
        assert np.count_nonzero(swiss[name]) > 100 and swiss[name].max() > 5e-6, name
    may = _rebalance_fields(witness, "may_d01")
    for name in ("QCLOUD", "QRAIN", "QICE", "QSNOW"):
        assert np.count_nonzero(may[name]) > 0, name
    assert may["QCLOUD"].max() > 1e-4


@pytest.mark.parametrize("use_theta_m", (0, 1))
@pytest.mark.parametrize("case", REBAL_CASES)
def test_rebalance_cycling_matches_original_wrf(witness, case, use_theta_m):
    from gpuwrf.io.cycling_init import rebalance_cycling

    fields = _rebalance_fields(witness, case)
    candidate = rebalance_cycling(fields, moist_fields=MOIST, use_theta_m=use_theta_m)
    tolerances = META["rebalance"]["tolerances"]
    assert set(candidate) == set(tolerances)
    failures = []
    for name, limit in tolerances.items():
        reference = witness[f"rebal__{case}__wrf{use_theta_m}__{name}"]
        assert candidate[name].shape == reference.shape and candidate[name].dtype == np.float32, name
        error = np.abs(candidate[name].astype(np.float64) - reference)
        bound = limit["atol"] + limit["rtol"] * np.abs(reference)
        if not np.all(error <= bound):
            failures.append(f"{name}: max_abs {error.max():.6g} exceeds the frozen oracle rule "
                            f"(atol {limit['atol']}, rtol {limit['rtol']})")
        elif candidate[name].tobytes() != reference.tobytes():
            failures.append(f"{name}: not BYTE-identical to original WRF (max_abs {error.max():.6g})")
    assert not failures, f"rebalance {case} use_theta_m={use_theta_m}: " + "; ".join(failures)


@pytest.fixture(scope="module")
def mynn_candidates(witness, tmp_path_factory):
    import gpuwrf

    source = Path(gpuwrf.__file__).resolve().parents[1]
    env = {k: v for k, v in os.environ.items() if not k.startswith("GPUWRF_") and k != "XLA_FLAGS"}
    # Release-default resolution (FAST_DEFAULTS unset) like the lane/critic gate runs;
    # flags must precede the JAX import, hence a separate process (E152: explicit source).
    env.update(JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0", XLA_FLAGS=XLA_CPU_CONDITION, PYTHONPATH=str(source),
               PYTHONDONTWRITEBYTECODE="1")
    out = tmp_path_factory.mktemp("w44_mynn_witness") / "candidate.npz"
    run = subprocess.run([sys.executable, str(DATA / "mynn_init_runner.py"), str(NPZ), str(out)],
                         cwd=str(source.parent), env=env, capture_output=True, text=True, timeout=900)
    if run.returncode:
        raise RuntimeError(f"MYNN witness runner failed rc={run.returncode}\n{run.stdout[-4000:]}\n{run.stderr[-4000:]}")
    with np.load(out) as data:
        result = {key: data[key] for key in data.files}
    meta = json.loads(str(result.pop("meta")))
    if meta["backend"] != "cpu" or meta["xla_flags"] != XLA_CPU_CONDITION:
        raise RuntimeError(f"runner condition differs: {meta}")
    if Path(meta["gpuwrf_file"]).resolve().parents[1] != source:
        raise RuntimeError(f"runner imported a different gpuwrf tree: {meta['gpuwrf_file']} vs {source}")
    return result


def test_mynn_ice_snow_witness_is_not_vacuous(witness, mynn_candidates):
    """E154/E88: the extension really carries ice AND snow; the branch masks equal WRF's."""
    arrays = witness["mynn__swiss_ice_snow__arrays"]
    qi, qs = arrays[:, :, 10], arrays[:, :, 11]
    assert np.all((qi > 1e-6).any(axis=1) & (qs > 1e-6).any(axis=1))
    assert qs.max() > 1e-4 and qi.max() > 1e-5
    ws11 = witness["mynn__ws11__arrays"]
    assert ws11[:, :, 10].max() == 0 and ws11[:, :, 11].max() == 0  # why the extension exists
    for tag in MYNN_SETS:
        cold = witness[f"mynn__{tag}__cold"]
        assert cold.any() and (~cold).any(), tag
        np.testing.assert_array_equal(mynn_candidates[f"{tag}__mask"], cold, err_msg=tag)


@pytest.mark.parametrize("field", ("QKE", "QSQ", "EL"))
@pytest.mark.parametrize("mode", ("cold", "background"))
@pytest.mark.parametrize("tag", MYNN_SETS)
def test_mynn_cycling_init_matches_original_wrf(witness, mynn_candidates, tag, mode, field):
    slot = ("QKE", "QSQ", "EL").index(field)
    cold = witness[f"mynn__{tag}__cold"]
    rows = cold if mode == "cold" else ~cold
    candidate = mynn_candidates[f"{tag}__candidate"][rows, :, slot]
    truth = witness[f"mynn__{tag}__truth"][rows, :, slot]
    assert np.isfinite(candidate).all(), f"{tag} {mode} {field}: non-finite candidate"
    error = np.abs(candidate.astype(np.float64) - truth)
    if field == "QKE" and mode == "background":
        assert np.array_equal(candidate, truth), (
            f"{tag} background QKE must equal WRF's zero (cycling high-QKE branch); max_abs {error.max():.6g}")
    elif field == "QKE":
        metric = float(np.median(error / np.maximum(np.abs(truth), 1e-8)))
        assert metric <= LIMITS["QKE_relative_median"], f"{tag} cold QKE median relative error {metric:.6g}"
    else:
        metric = float(np.max(error / (LIMITS[field + "_atol"] + LIMITS[field + "_rtol"] * np.abs(truth))))
        assert metric <= 1, f"{tag} {mode} {field}: bound ratio {metric:.6g} > 1 (max_abs {error.max():.6g})"
