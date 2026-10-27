"""Deletion and nonfinite controls for the isolated operator's real-data gate."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("GPUWRF_JAX_CACHE", "0")
import jax
import jax.numpy as jnp
import numpy as np

from oracle import compare
from operator_probe import topographic_shadow


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    assert jax.devices()[0].platform == "cpu"
    report = {"deletions": [], "nonfinite_controls": []}
    with np.load(args.fixture) as fixture:
        for hour in (7.5, 18.0):
            get = lambda key: fixture[f"h{hour}_{key}"]
            declin = np.float32(np.arcsin(np.sin(np.float32(23.5) * np.float32(np.pi / 180)) *
                              np.sin(np.float32(360 / 365) * np.float32(58 - 80) * np.float32(np.pi / 180))))
            kwargs = dict(latitude_deg=get("latitude"), longitude_deg=get("longitude"),
                          sina=get("sina"), cosa=get("cosa"), declination_rad=declin,
                          xtime_minutes=hour * 60, gmt_hours=0., radt_minutes=30., dx_m=1000.,
                          dy_m=1000., shadow_length_m=25000.)
            # Genuine deletion mutant: the candidate ignores its actual parent input.
            fn = jax.jit(lambda terrain, parent: topographic_shadow(terrain, **kwargs))
            got, _ = fn(jnp.asarray(get("terrain")), jnp.asarray(get("parent_height")))
            try:
                compare("deletion mutant mask", got, get("mask"), atol=0)
            except AssertionError as exc:
                assert "Not equal to tolerance" in str(exc), str(exc)
                report["deletions"].append({"hour": hour, "assertion_kill": True,
                                            "mask_misses": int(np.count_nonzero(np.asarray(got) != get("mask")))})
            else:
                raise AssertionError("parent-deletion mutant survived the real-data oracle")
            fn.clear_cache()
    for field in ("mask", "height", "flux"):
        for side in ("candidate", "reference"):
            for value in (np.nan, np.inf, -np.inf):
                a, b = np.zeros(3), np.zeros(3)
                (a if side == "candidate" else b)[1] = value
                try:
                    compare(field, a, b, atol=0)
                except AssertionError as exc:
                    assert "nonfinite candidate or reference" in str(exc)
                    report["nonfinite_controls"].append({"field": field, "side": side,
                                                           "value": str(value), "rejected": True})
                else:
                    raise AssertionError("nonfinite control survived")
    report["status"] = "PASS"
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
