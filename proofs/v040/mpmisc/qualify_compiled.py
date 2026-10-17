"""Measure Ferrier native CPU output against independent original-WRF spreads.

Inputs: immutable source tree, original fixture, original FMA receipt,
original input-ULP receipt, output directory. No derived interval uses the
candidate. Residual excess is reported, never silently widened or clamped.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time

root, fixture, fma_path, ulp_path, out = map(lambda p: Path(p).resolve(), sys.argv[1:6])
sys.path.insert(0, str(root / "src"))
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.physics.ferrier.native import native_column

assert jax.devices()[0].platform == "cpu"
out.mkdir(exist_ok=False)
meta = json.loads(fixture.with_suffix(".json").read_text())
assert hashlib.sha256(fixture.read_bytes()).hexdigest() == meta["sha256"]
data = np.load(fixture, allow_pickle=False)
fma, ulp = (json.loads(path.read_text()) for path in (fma_path, ulp_path))
rows, intervals = [], {}


def fields(cases, name):
    return (np.asarray([c["columns"][name] for c in cases], np.float32).T
            if name in cases[0]["columns"] else np.asarray([c["scalars"][name] for c in cases], np.float32))


def read_cases(items):
    cases = []
    for item in items:
        path = Path(item["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        cases.append(json.loads(path.read_text()))
    return cases


for group, cfg in meta["groups"].items():
    forcing = ("TH_IN", "QV_IN", "QC_IN", "QR_IN", "QI_IN", "P", "PII", "RHO", "DZ")
    fma_cases = read_cases(fma["arms"][group]["cases"])
    assert all(np.array_equal(fields(fma_cases, name), data[group + "__" + name]) for name in forcing)
    variants = [read_cases(v["cases"]) for v in ulp["arms"][group]["ulp_variants"]]
    args = [jnp.asarray(data[group + "__" + name][:, None, :]) for name in forcing]
    if cfg["option"] == 15:
        extra = jnp.asarray(data[group + "__QRIMEF_IN"][:, None, :])
    else:
        extra = jnp.asarray(data[group + "__F_RIMEF_PHY_IN"][:, None, :])
    rainnc = jnp.zeros_like(args[0][0])
    start = time.monotonic()
    for _ in range(cfg["steps"]):
        actual = native_column(*args, cfg["dt_seconds"], option=cfg["option"], gid=cfg["gid"],
            interpret=True, **({"rimef": extra} if cfg["option"] == 5 else {"qrimef": extra}))
        rainnc = rainnc + actual["RAINNCV"]
        args[:5] = [actual[name] for name in ("TH_OUT", "QV_OUT", "QC_OUT", "QR_OUT", "QI_OUT")]
        extra = actual["F_RIMEF_PHY_OUT"] if cfg["option"] == 5 else actual["QRIMEF_OUT"]
    actual["RAINNC"] = rainnc
    jax.block_until_ready(actual)
    row = {"group": group, "compile_and_steps_s": time.monotonic() - start, "fields": {}}
    arrays = {name: np.asarray(actual[name])[:, 0, :] if actual[name].ndim == 3 else np.asarray(actual[name])[0]
              for name in cfg["fields"]}
    np.savez_compressed(out / (group + ".npz"), **arrays)
    for name in cfg["fields"]:
        expected = data[group + "__" + name]
        values = [expected, fields(fma_cases, name), *(fields(v, name) for v in variants)]
        assert all(v.shape == expected.shape and np.isfinite(v).all() for v in values)
        lower, upper = np.min(values, axis=0), np.max(values, axis=0)
        key = group + "__" + name
        intervals[key + "__lower"], intervals[key + "__upper"] = lower, upper
        candidate = arrays[name]
        assert candidate.dtype == np.float32 and np.isfinite(candidate).all()
        excess = np.maximum(np.maximum(lower.astype(np.float64) - candidate, candidate.astype(np.float64) - upper), 0.)
        row["fields"][name] = {"max_abs": float(np.max(np.abs(candidate - expected))),
            "outside_independent_interval": int(np.count_nonzero(excess)), "max_excess": float(excess.max()),
            "max_fma_spread": float(np.max(np.abs(fields(fma_cases, name) - expected)))}
    rows.append(row)
np.savez_compressed(out / "original_intervals.npz", **intervals)
receipt = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
    "platform": "cpu", "GPU_execution": False, "program": "Pallas interpreter",
    "fixture_sha256": meta["sha256"], "fma_receipt_sha256": hashlib.sha256(fma_path.read_bytes()).hexdigest(),
    "ulp_receipt_sha256": hashlib.sha256(ulp_path.read_bytes()).hexdigest(),
    "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "rows": rows}
(out / "bounds.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
