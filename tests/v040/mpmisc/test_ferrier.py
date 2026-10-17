"""Original-WRF REAL columns; includes fresh dt/GID/multistep/rime holdouts."""
import hashlib
import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics.ferrier.column import reference_column
from gpuwrf.physics.ferrier.native import native_column


HERE = Path(__file__).parent
BOUND = 128 * np.finfo(np.float32).eps
INPUTS = ("TH_IN", "QV_IN", "QC_IN", "QR_IN", "QI_IN", "P", "PII", "RHO", "DZ")


def _load(stem="ferrier_columns"):
    meta = json.loads((HERE / (stem + ".json")).read_text())
    path = HERE / (stem + ".npz")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["sha256"]
    return meta, np.load(path, allow_pickle=False)


def _check(name, actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape, name
    assert np.isfinite(actual).all() and np.isfinite(expected).all(), name
    # Fixed REAL-rounding budget, per column. Fractions use their natural
    # unit scale; temperatures use K. Other fields use their reference's
    # column maximum, so near-zero precipitation cannot borrow a wet case's
    # tolerance. A 128-eps budget covers the many elementary/iterative REAL
    # operations, not model drift. Frozen before fresh holdout measurement.
    floor = 1.0 if name.startswith("F_") or name == "SR" else 1e-12
    scale = np.maximum(np.max(np.abs(expected), axis=0) if expected.ndim > 1 else np.abs(expected), floor)
    tol = BOUND * (scale + np.abs(expected))
    assert np.all(np.abs(actual - expected) <= tol), (name, float(np.max(np.abs(actual - expected) / tol)))


GROUPS = ["caller_mp5", "caller_mp15", "rimed_mp15", "holdout_mp5", "holdout_mp15", "holdout_rimed_mp15"]


def _run(group, fn, stem="ferrier_columns", native=False):
    meta, data = _load(stem)
    cfg = meta["groups"][group]
    get = lambda name: np.asarray(data[group + "__" + name])
    args = [get(name) for name in INPUTS]
    qrim = get("QRIMEF_IN") if cfg["option"] == 15 else None
    rime = (get("F_RIMEF_PHY_IN") if group + "__F_RIMEF_PHY_IN" in data else np.ones_like(args[0])) if cfg["option"] == 5 else None
    if native:
        args = [jnp.asarray(a[:, None, :]) for a in args]
        qrim = None if qrim is None else jnp.asarray(qrim[:, None, :])
        rime = None if rime is None else jnp.asarray(rime[:, None, :])
    rainnc = np.zeros_like(args[0][0])
    for _ in range(cfg["steps"]):
        extra = {"rimef": rime} if cfg["option"] == 5 else {"qrimef": qrim}
        result = fn(*args, cfg["dt_seconds"], gid=cfg["gid"], **extra,
                    **({"option": cfg["option"], "interpret": True} if native else {}))
        if not native:
            result, rates = result
            assert np.isfinite(rates).all()
        rainnc = rainnc + result["RAINNCV"]
        args[:5] = [result[k] for k in ("TH_OUT", "QV_OUT", "QC_OUT", "QR_OUT", "QI_OUT")]
        rime = result["F_RIMEF_PHY_OUT"] if cfg["option"] == 5 else None
        qrim = result["QRIMEF_OUT"] if cfg["option"] == 15 else None
    result["RAINNC"] = rainnc
    actual = {name: np.asarray(result[name]) for name in cfg["fields"]}
    if native:
        actual = {name: value[:, 0, :] if value.ndim == 3 else value[0] for name, value in actual.items()}
    return actual, {name: data[group + "__" + name] for name in cfg["fields"]}


def _literal():
    path = HERE.parents[2] / "proofs/v040/mpmisc/literal_reference.py"
    spec = importlib.util.spec_from_file_location("ferrier_literal_proof", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.literal_column


def _byte_check(actual, expected):
    for name, value in actual.items():
        assert value.dtype == np.float32 and np.isfinite(value).all(), name
        assert value.shape == expected[name].shape, name
        assert value.tobytes() == expected[name].tobytes(), name


@pytest.mark.parametrize("stem,group", [("ferrier_columns", g) for g in GROUPS] +
                         [("ferrier_stress", g) for g in ("stress_mp5", "stress_mp15", "stress_rimed_mp15")])
def test_literal_real_original_wrf_byte(stem, group):
    # Shared equation code objects, scalar REAL libm, no FMA; original WRF is
    # the independent reference. This is the primary arithmetic fidelity gate.
    _byte_check(*_run(group, _literal(), stem))


@pytest.mark.parametrize("group", GROUPS + ["stress_mp5", "stress_mp15", "stress_rimed_mp15"])
def test_native_cpu_compiler_disclosed_deviations(group):
    stress = group.startswith("stress")
    actual, expected = _run(group, native_column, stem="ferrier_stress" if stress else "ferrier_columns", native=True)
    if not group.startswith("holdout") and not stress:
        for name in actual:
            _check(name, actual[name], expected[name])
        return
    meta, bounds = _load("ferrier_stress_compiler_bounds" if stress else "ferrier_compiler_bounds")
    for name, value in actual.items():
        assert value.dtype == np.float32 and np.isfinite(value).all(), name
        key = group + "__" + name
        lower, upper = bounds[key + "__lower"], bounds[key + "__upper"]
        excess = np.maximum(np.maximum(lower.astype(np.float64) - value, value.astype(np.float64) - upper), 0.)
        # Independently derived original-WRF interval plus the sealed MP22
        # small disclosed excess (MP35 for the fresh stress set). This is compiled regression evidence, not
        # a strict interval pass or a forecast D6 qualification.
        assert float(excess.max()) <= meta["groups"][group][name], (group, name, float(excess.max()))


def test_original_wrf_byte_gate_rejects_latent_heat_mutant(monkeypatch):
    from gpuwrf.physics.ferrier import column
    monkeypatch.setattr(column, "XLV1", np.float32(column.XLV1 * np.float32(1.01)))
    with pytest.raises(AssertionError):
        _byte_check(*_run("caller_mp5", _literal()))


def test_reference_gate_rejects_nonfinite():
    for invalid in (np.nan, np.inf, -np.inf):
        with pytest.raises(AssertionError):
            _check("QC_OUT", [invalid], [0.0])
        with pytest.raises(AssertionError):
            _check("QC_OUT", [0.0], [invalid])
