"""Compare the short spectral CPU smoke to ORIGINAL CPU-WRF, using frozen D6.

Usage: python compare_coupled.py CONTROL_JSON SPECTRAL_JSON CPU_ROOT OUT_JSON
The short core-field envelope is not the full-field/24h release gate. Response
correlations and held-rate differences are diagnostics with no invented caps.
"""
from pathlib import Path
import json
import sys

import netCDF4
import numpy as np

control_json, spectral_json, cpu_root, output = map(Path, sys.argv[1:])
control_receipt = json.loads(control_json.read_text())["arms"]["release"]
arms = json.loads(spectral_json.read_text())["arms"]
assert len(arms) == 1
spectral_receipt = next(iter(arms.values()))
for receipt in (control_receipt, spectral_receipt):
    assert receipt.get("nonfinite_values") == 0 and receipt.get("guard_total") == 0, receipt
    assert "final_npz" in receipt
control = dict(np.load(control_receipt["final_npz"]))
spectral = dict(np.load(spectral_receipt["final_npz"]))
manifest_path = Path(__file__).resolve().parents[1] / "v014/grid_delta_atlas/tolerance_manifest_candidate.json"
tolerances = json.loads(manifest_path.read_text())["fields"]
rvrd = 461.6 / 287.0


def port_fields(data):
    return {"U": data["u"], "V": data["v"], "W": data["w"], "QVAPOR": data["qv"],
            "T": data["theta"] / (1 + rvrd * data["qv"]) - 300,
            "PH": data["ph"], "P": data["p"]}


def stats(a, b):
    assert a.shape == b.shape, (a.shape, b.shape)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    error = a.astype(np.float64) - b.astype(np.float64)
    return {"shape": list(a.shape), "rmse": float(np.sqrt(np.mean(error * error))),
            "max_abs": float(np.abs(error).max()), "bias": float(error.mean())}


cpu = {}
for arm in ("control", "spectral"):
    with netCDF4.Dataset(cpu_root / arm / "wrfout_d01_2023-01-15_00:00:54") as ds:
        cpu[arm] = {name: np.asarray(ds[name][0]) for name in ("U", "V", "W", "QVAPOR", "T", "PH", "P")}
mapped = {"control": port_fields(control), "spectral": port_fields(spectral)}
report = {"source": "original CPU-WRF RD11 executable, verified auxinput10 ingestion",
          "scope": "3steps54s, core fields only; NOT the 24h/all-variable release gate",
          "tolerance_manifest": str(manifest_path), "port_vs_cpu": {}, "response_diagnostic": {}}
passed = True
for arm in mapped:
    report["port_vs_cpu"][arm] = {}
    for name, value in mapped[arm].items():
        result = stats(value, cpu[arm][name])
        cap = tolerances[name].get("rmse")
        result["frozen_rmse_limit"] = cap
        result["within_frozen_limit"] = None if cap is None else result["rmse"] <= cap
        if arm == "spectral" and cap is not None:
            passed &= result["within_frozen_limit"]
        report["port_vs_cpu"][arm][name] = result
for name in mapped["spectral"]:
    port_delta = mapped["spectral"][name].astype(np.float64) - mapped["control"][name]
    cpu_delta = cpu["spectral"][name].astype(np.float64) - cpu["control"][name]
    result = stats(port_delta, cpu_delta)
    norm = float(np.linalg.norm(port_delta) * np.linalg.norm(cpu_delta))
    result["cosine"] = None if norm == 0 else float(np.vdot(port_delta.ravel(), cpu_delta.ravel()) / norm)
    result["port_delta_rms"] = float(np.sqrt(np.mean(port_delta * port_delta)))
    result["cpu_delta_rms"] = float(np.sqrt(np.mean(cpu_delta * cpu_delta)))
    report["response_diagnostic"][name] = result
rates_path = output.parent / "spectral_final_rates.npz"
if rates_path.exists():
    rates = dict(np.load(rates_path))
    report["held_rates_vs_original_cpu_diagnostic"] = {}
    with netCDF4.Dataset(cpu_root / "audit/wrfrst_d01_2023-01-15_00:00:54") as ds:
        for name, value in rates.items():
            reference = np.asarray(ds[name.upper()][0])
            if name == "rphndgdten":
                assert value.shape[0] == reference.shape[0] + 1
                assert np.count_nonzero(value[-1]) == 0  # WRF caller leaves top PH zero
                value = value[:-1]
            report["held_rates_vs_original_cpu_diagnostic"][name] = stats(value, reference)
report["short_spectral_core_envelope_pass"] = bool(passed)
Path(output).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
assert passed, report["port_vs_cpu"]["spectral"]
