"""Pack source-bound raw oracle values into a small, tracked regression fixture."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    blob = (args.oracle / "oracle.bin").read_bytes()
    offset = 0
    values = {}

    def read(name, shape, dtype="<f4"):
        nonlocal offset
        count = int(np.prod(shape))
        value = np.frombuffer(blob, dtype=dtype, count=count, offset=offset).reshape(shape, order="F")
        offset += value.nbytes
        # WRF (i,k,j) -> JAX (k,j,i); WRF (i,j) -> JAX (j,i).
        values[name] = (value.transpose(1, 2, 0) if len(shape) == 3 else value.T if len(shape) == 2 else value).copy()

    plane, cube = (18, 16), (18, 5, 16)
    read("seed_initial", (8,), "<i4")
    read("amplitude", plane)
    read("alpha", ())
    read("vertical_cosine", cube)
    read("vertical_sine", cube)
    read("vertical_amplitude", (5,))
    read("sine_initial", plane)
    read("cosine_initial", plane)
    for step in range(4):
        for name in ("sine_noise", "cosine_noise", "sine", "cosine"):
            read(f"{name}_{step}", plane)
        read(f"seed_{step}", (8,), "<i4")
        for vert in (0, 1):
            for component in "TUV":
                read(f"field_{step}_{vert}_{component}", cube)
    for variable in "WTPRSQO":
        read(f"setup_{variable}_amplitude", plane)
        read(f"setup_{variable}_alpha", ())
        read(f"setup_{variable}_vertical_cosine", cube)
        read(f"setup_{variable}_vertical_sine", cube)
        read(f"setup_{variable}_vertical_amplitude", (5,))
    for name in ("u", "v", "theta", "qv", "u_pattern", "v_pattern", "theta_pattern", "sppt_pattern"):
        read(name, cube)
    read("mu", plane)
    read("mub", plane)
    read("c1h", (5,))
    read("c2h", (5,))
    for name in ("u", "v", "theta"):
        read(f"skebs_{name}", cube)
    for name in ("u", "v", "theta", "qv", "pattern"):
        read(f"sppt_{name}", cube)
    assert offset == len(blob), (offset, len(blob))
    if any(not np.isfinite(v).all() for v in values.values()):
        raise ValueError("oracle contains NaN/Inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **values)
    manifest = json.loads((args.oracle / "manifest.json").read_text())
    manifest["fixture_id"] = "u2-stoch-pristine-real4-v1"
    manifest["scenario"] = "fixed compiler RNG seeds; four AR steps; scalar/U/V FFT; vertical0/1; nonzero tendencies; signed cutoff tails"
    manifest["created_utc"] = "2026-10-13"
    manifest["license_notes"] = "Numerical output from original WRF/FFTPACK5; original source not redistributed in fixture"
    manifest["source_commit"] = "source SHA256 per file (staged source tree)"
    manifest["fixture_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    def units(name):
        if name == "setup_W_amplitude":
            return "m2 s-2"
        if name == "setup_T_amplitude":
            return "K s-1"
        if name in {"mu", "mub", "c2h"}:
            return "Pa"
        if name in {"u_pattern", "v_pattern"}:
            return "m s-2"
        if name == "theta_pattern":
            return "K s-1"
        if name in {"u", "v", "skebs_u", "skebs_v", "sppt_u", "sppt_v"}:
            return "Pa m s-2"
        if name in {"theta", "skebs_theta", "sppt_theta"}:
            return "Pa K s-1"
        if name in {"qv", "sppt_qv"}:
            return "Pa kg kg-1 s-1"
        if name.startswith("field_") and name.endswith(("_U", "_V")):
            return "m-1"
        return "1"
    manifest["variables"] = {name: {"shape": list(v.shape), "dtype": str(v.dtype),
                                    "units": units(name),
                                    "absolute_tolerance": 1e-6 if name.startswith(("skebs_", "sppt_")) else 2e-7,
                                    "relative_tolerance": 3e-6} for name, v in values.items()}
    manifest["tolerances"]["fft_absolute_policy"] = "64*REAL_epsilon*max(abs(reference)); reject nonfinite first"
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Small YAML index for the repository's standard manifest validator.
    summary = {key: manifest[key] for key in ("fixture_id", "source", "source_commit", "scenario", "created_utc", "license_notes")}
    summary["variables"] = {"details": args.output.with_suffix(".json").name,
                            "layout": "WRF(i,k,j) transposed to (k,j,i), includes top/staggered edge slots",
                            "units": "dimensionless spectra and patterns; physical tendency/mass units per JSON variable"}
    summary["files"] = [{"path": args.output.name, "sha256": manifest["fixture_sha256"]}]
    args.output.with_suffix(".yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
    print(json.dumps({"fixture": str(args.output), "bytes": args.output.stat().st_size,
                      "sha256": manifest["fixture_sha256"], "arrays": len(values)}))


if __name__ == "__main__":
    main()
