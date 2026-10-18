"""Build the machine-readable v0.4.0 optimize release list from the FINAL release checkpoint.

Usage (CPU, no JAX device work):
  JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 python proofs/w41_optimize/build_released.py \
      [--check]  # --check: rebuild in memory and compare with the shipped JSON byte for byte

Rule (lane w41-optimize contract): a physics configuration is released for `gpuwrf optimize`
iff its exact seven-option tuple (per domain, same domain count) passed frozen bounded D6 against
original CPU-WRF in the v0.4.0 FINAL set: the 66 registered composites of
proofs/r_docs040/RELEASE_CHECKPOINT.json, plus the default-program F20 cases (WN3 3-nest, PROD
2-nest, Swiss) whose E41 was requalified by C5'. Each released tuple lists every qualifying run
with its E41 status, gate hours, geography and non-physics namelist context, so a user sees
exactly which combination was measured. Nothing else is released.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/"src"))

from gpuwrf.io.namelist_parser import parse_namelist  # noqa: E402
from gpuwrf.optimize.released import PHYSICS, RELEASED_PATH, SCHEMA, settings, tuple_id  # noqa: E402

CHECKPOINT = REPO/"proofs/r_docs040/RELEASE_CHECKPOINT.json"
NAMELISTS = REPO/"proofs/v040/final_validation/namelists"
F20_ROOT = Path("<DATA_ROOT>/wrf_gpu2/archive/v040/v-prep/F20_20261016")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def per_domain(config, max_dom):
    physics = config.get("physics", {})
    out = {}
    for key in PHYSICS:
        value = physics.get(key, 0)
        values = value if isinstance(value, list) else [value]*max_dom
        if len(values) < max_dom:
            values = list(values)+[values[-1]]*(max_dom-len(values))
        out[key] = [int(v) for v in values[:max_dom]]
    return out


def context(config, max_dom, base):
    """Normalised non-physics settings that differ from the base Swiss composite c00 (None = absent)."""
    a = settings(config, max_dom)
    return {k: a.get(k) for k in sorted(set(a) | set(base)) if a.get(k) != base.get(k)}


def build():
    checkpoint = json.loads(CHECKPOINT.read_text())
    base = settings(parse_namelist(NAMELISTS/"c00"/"namelist.input"), 1)
    runs = []
    for row in checkpoint["rows"]:
        receipt = row["final_receipt"]
        work = row["registered_work"]
        namelist = NAMELISTS/row["case"]/"namelist.input"
        config = parse_namelist(namelist)
        max_dom = len(work["domains"])
        physics = per_domain(config, max_dom)
        options = {k: physics[k][0] for k in PHYSICS}
        if options != {k: row["options"][k] for k in PHYSICS}:
            raise SystemExit(f"{row['case']}: checkpoint options disagree with the registered namelist")
        if receipt["D6"] != "PASS" or receipt.get("optin_scientific_release_gate") != "PASS":
            raise SystemExit(f"{row['case']}: not a FINAL D6 PASS row")
        runs.append({"run": row["case"], "kind": "FINAL composite", "max_dom": max_dom, "physics": physics,
                     "geography": None, "gate_hours": work["gate_hours"], "D6": receipt["D6"], "E41": receipt["E41"],
                     "release_class": receipt["release_class"], "source_commit": receipt["source"],
                     "sampled_peak_vram_MiB": receipt.get("sampled_peak_vram_MiB"),
                     "gpu_wall_s": round(float(receipt["wall_s"]), 1) if receipt.get("wall_s") else None,
                     "namelist_sha256": sha256(namelist),
                     "registered_namelist_sha256": work["input_sha256"].get("namelist.input"),
                     "context_vs_c00": context(config, max_dom, base)})
    matrix = {}
    for line in (REPO/"proofs/v040/final_validation/MATRIX.md").read_text().splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) >= 2 and cells[0].startswith("c") and cells[0][1:].isdigit():
            matrix[cells[0]] = cells[1]
    for run in runs:
        run["geography"] = matrix.get(run["run"])
    f20, c5 = checkpoint["default_F20"], checkpoint["default_C5prime"]
    if f20["D6"] != "PASS" or c5["verdict"] != "PASS":
        raise SystemExit("default F20/C5' not PASS")
    for row in f20["rows"]:
        namelist = F20_ROOT/row["case"]/"input"/"namelist.input"
        config = parse_namelist(namelist)
        max_dom = len(row["D6"])
        if set(row["D6"].values()) != {"PASS"}:
            raise SystemExit(f"{row['case']}: default F20 D6 not PASS")
        runs.append({"run": f"F20-{row['case']}", "kind": "default program F20 (E41 requalified by C5')",
                     "max_dom": max_dom, "physics": per_domain(config, max_dom), "geography": row["case"],
                     "gate_hours": 24, "D6": "PASS", "E41": "PASS" if c5["verdict"] == "PASS" else row["E41"],
                     "release_class": "release default", "source_commit": row["source"],
                     "sampled_peak_vram_MiB": row.get("sampled_peak_vram_MiB"),
                     "gpu_wall_s": round(float(row["wall_s"]), 1), "namelist_sha256": sha256(namelist),
                     "registered_namelist_sha256": None, "context_vs_c00": context(config, max_dom, base)})
    configurations = {}
    for run in runs:
        key = tuple_id(run["physics"])
        entry = configurations.setdefault(key, {"id": key, "max_dom": run["max_dom"], "physics": run["physics"],
                                                "qualified_by": []})
        entry["qualified_by"].append({k: v for k, v in run.items() if k not in ("physics", "max_dom")})
    order = {"PASS": 0, "FAIL": 1, "UNKNOWN": 2}
    for entry in configurations.values():
        # Preferred evidence first: E41 PASS, no extra context, full 24 h gate, then run name.
        entry["qualified_by"].sort(key=lambda r: (order.get(r["E41"], 3), len(r["context_vs_c00"]) > 0,
                                                 -r["gate_hours"], r["run"]))
        entry["best_E41"] = entry["qualified_by"][0]["E41"]
    options = {key: sorted({v for e in configurations.values() for v in e["physics"][key]}) for key in PHYSICS}
    return {"schema": SCHEMA, "release": "v0.4.0",
            "rule": ("released = exact seven-option tuple per domain (same domain count) with frozen bounded D6 PASS "
                     "vs original CPU-WRF in the v0.4.0 FINAL set (66 registered composites + default F20 cases); "
                     "E41 shown per qualifying run; any other combination is refused by gpuwrf optimize"),
            "authority": checkpoint["authority"], "final_validation_utc": checkpoint["final_validation_utc"],
            "checkpoint": {"path": str(CHECKPOINT.relative_to(REPO)), "sha256": sha256(CHECKPOINT)},
            "release_program_identities": checkpoint["release_program_identities"],
            "physics_keys": list(PHYSICS), "base_settings": base,
            "per_option_union": options,
            "configurations": sorted(configurations.values(), key=lambda e: (e["max_dom"], e["id"]))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = json.dumps(build(), indent=1, sort_keys=True)+"\n"
    if args.check:
        shipped = RELEASED_PATH.read_text()
        print("RELEASED_LIST", "EXACT" if shipped == text else "DIFFERS", sha256(RELEASED_PATH))
        return 0 if shipped == text else 1
    RELEASED_PATH.write_text(text)
    data = json.loads(text)
    print(len(data["configurations"]), "released configurations ->", RELEASED_PATH, sha256(RELEASED_PATH))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
