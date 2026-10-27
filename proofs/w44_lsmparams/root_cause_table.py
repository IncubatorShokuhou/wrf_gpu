"""Per-field ON-default table of retained root_cause.py arms (manager 05:12Z ruling), CPU, no model run.

Same table code as the GPU prime (field_table.py); slot and packed-family names come from the live classes of
the measured pin (imported on CPU) because the CPU arms saved arrays only.
Usage: root_cause_table.py --dir <arms dir> --src <pin>/src/gpuwrf --case <input dir> --out table.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

import field_table as ft

p = argparse.ArgumentParser()
p.add_argument("--dir", type=Path, required=True)
p.add_argument("--src", type=Path, required=True, help="model src/gpuwrf of the measured pin (slot names)")
p.add_argument("--case", type=Path, required=True, help="case input dir (wrfinput_d01: IVGTYP, ISICE)")
p.add_argument("--ref", default="off")
p.add_argument("--cand", action="append", default=None)
p.add_argument("--bound", type=float, default=None, help="optional: flag rows above this max abs / field max")
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()


# Live classes of the measured pin (CPU import, no model work): AST parsing of State.__slots__ is not
# reliable (generated members past index 88).
import sys  # noqa: E402
sys.path.insert(0, str(a.src.parent))
import dataclasses  # noqa: E402
from gpuwrf.contracts.noahmp_state import NoahMPLandState  # noqa: E402
from gpuwrf.contracts.state import State  # noqa: E402
from gpuwrf.runtime.history_accumulators import ENERGY_ACCUMULATORS, LAND_FLUX_FIELDS  # noqa: E402
from gpuwrf.runtime.operational_state import OperationalCarry  # noqa: E402
assert Path(sys.modules["gpuwrf"].__file__).resolve().is_relative_to(a.src.resolve())
SLOTS = ([f.name for f in dataclasses.fields(OperationalCarry)], list(State.__slots__), list(NoahMPLandState.__slots__),
         {"land_history": tuple(LAND_FLUX_FIELDS), "energy_accumulators": tuple(ENERGY_ACCUMULATORS)})


def load(tag):
    meta = json.loads((a.dir / f"{tag}.json").read_text())
    with np.load(a.dir / f"{tag}.npz") as data:
        host = {n: data[f"L{i:04d}"] for i, n in enumerate(meta["leaf_names"]) if not n.startswith(("NAMED.", "CENSUS."))}
    return meta, ft.named_fields(host, "d01", *SLOTS)


ref_meta, ref = load(a.ref)
mask = ft.scale_mask(ref, ft.ice_mask(a.case / "wrfinput_d01"))
report = {"dir": str(a.dir), "src": str(a.src), "head": ref_meta["git_head"], "bound": a.bound,
          "land_columns": int((np.asarray(ref["State.xland"]) == 1).sum()), "scale_columns": int(mask.sum()),
          "scale": "max |field| over land columns without land ice", "tables": {}}
for tag in a.cand or ["on", "const"]:
    meta, cand = load(tag)
    assert meta["git_head"] == ref_meta["git_head"]
    rows = ft.table(ref, cand, mask)
    report["tables"][f"{tag}_vs_{a.ref}"] = rows
    report[f"{tag}_max_ratio"] = max((r["max_abs_over_field_max"] for r in rows), default=0.0)
    if a.bound is not None:
        report[f"{tag}_above_bound"] = [r["field"] for r in rows if r["max_abs_over_field_max"] > a.bound]
a.out.write_text(json.dumps(report, indent=2) + "\n")
for name, rows in report["tables"].items():
    print(f"## {name}" + (f"  (bound {a.bound:g})" if a.bound is not None else ""))
    print("| field | cells | ULP median/max | max abs | field max | max abs / field max |")
    print("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        flag = " **above**" if a.bound is not None and r["max_abs_over_field_max"] > a.bound else ""
        print(f"| {r['field']} | {r['cells']} | {r['ulp_median']:g}/{r['ulp_max']} | {r['max_abs']:.3g} | "
              f"{r['field_abs_max']:.3g} | {r['max_abs_over_field_max']:.2g}{flag} |")
