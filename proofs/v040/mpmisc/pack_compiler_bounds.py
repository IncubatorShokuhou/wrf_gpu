"""Seal independent original-WRF intervals and disclosed predecessor excesses.

This is a compiled regression allowance under the standing compiler policy,
not a replacement for the literal BYTE fidelity test. No candidate result
enters the lower/upper WRF intervals.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def pack(root, fixture, output):
    meta = json.loads(fixture.with_suffix(".json").read_text())
    measured_path = root / "MP22/bounds.json"
    measured = {row["group"]: row for row in json.loads(measured_path.read_text())["rows"]}
    ulp = json.loads((root / "MP17/receipt.json").read_text())
    arrays, groups, sources = {}, {}, {}
    with np.load(fixture, allow_pickle=False) as original:
        for group, cfg in meta["groups"].items():
            if not group.startswith("holdout"):
                continue
            paths = [root / "MP16" / group / f"case_{i}.json" for i in range(1, 5)]
            variants = [paths]
            for variant in ulp["arms"][group]["ulp_variants"]:
                # Those older two input recorders were incomplete (MP17).
                if variant["field"] not in ("qrimef", "f_rimef_phy"):
                    variants.append([Path(c["path"]) for c in variant["cases"]])
            cases = []
            for paths in variants:
                cases.append([json.loads(p.read_text()) for p in paths])
                sources.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
            groups[group] = {}
            for name in cfg["fields"]:
                values = [original[group + "__" + name]]
                for variant in cases:
                    values.append(np.asarray([c["columns"][name] for c in variant], np.float32).T
                                  if name in variant[0]["columns"] else
                                  np.asarray([c["scalars"][name] for c in variant], np.float32))
                assert all(np.isfinite(v).all() for v in values)
                key = group + "__" + name
                arrays[key + "__lower"] = np.min(values, axis=0)
                arrays[key + "__upper"] = np.max(values, axis=0)
                groups[group][name] = measured[group]["fields"][name]["max_excess"]
    np.savez_compressed(output, **arrays)
    output.with_suffix(".json").write_text(json.dumps({
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "groups": groups, "interval_sources": sources,
        "disclosed_excess_source_sha256": hashlib.sha256(measured_path.read_bytes()).hexdigest(),
        "scope": "WRF independent nine-input-ULP/native-FMA intervals; MP22 small excesses are disclosed regression ceilings; D6 pending"
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    pack(args.root, args.fixture, args.output)
