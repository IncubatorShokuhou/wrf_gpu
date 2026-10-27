"""Real PROD namelist + actual callback scheduler; no model or GPU execution."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess

from product_schedule import product_schedule


@dataclass(frozen=True)
class Carry:
    step: int


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("JAX_PLATFORMS") == "cpu"
    from gpuwrf.contracts.grid import DomainHierarchy, DomainNest
    from gpuwrf.io.gen2_accessor import Gen2Run
    from gpuwrf.integration.nested_pipeline import _domain_list_value
    from gpuwrf.runtime import domain_tree as dt

    names = ("d01", "d02")
    raw = Gen2Run(args.input_dir).namelist["domains"]
    value = lambda key, i: _domain_list_value(raw[key], i, None)
    edge = DomainNest(f"d{int(value('parent_id', 1)):02d}", "d02",
                      int(value("parent_grid_ratio", 1)),
                      int(value("i_parent_start", 1)), int(value("j_parent_start", 1)))
    hierarchy = DomainHierarchy.from_edges(names, (edge,))
    steps = {"d01": float(value("time_step", 0))}
    steps["d02"] = steps["d01"] / edge.parent_grid_ratio
    kwargs, schedule = product_schedule(args.input_dir, names, steps, hours=3)
    assert steps == {"d01": 54., "d02": 18.}
    assert schedule["output_cadence_steps"] == {"d01": 67, "d02": 200}
    assert schedule["output_alarm_steps"] == {"d01": (67, 134, 200)}
    assert (schedule["block_between"], schedule["root_sync_cadence"],
            schedule["event_aware_fusion_k"]) == (False, 1, 0)
    rows = {}

    def run(label, start, count=1, *, legacy=False, omit_observer=False):
        calls, fused_calls, outputs = [], [], []
        def advance(name, carry, own, n):
            assert carry.step + 1 == own  # Scheduler callbacks use WRF 1-based starts.
            calls.append([name, own, n])
            return Carry(own + n - 1)
        def lookup(name):
            if name != "d01":
                return None
            def fused(parent, children, own, child_owns):
                fused_calls.append([name, own, list(child_owns)])
                return Carry(own), (Carry(child_owns[0] + edge.parent_grid_ratio - 1),)
            return fused
        controls = {} if legacy else dict(kwargs)
        if not legacy:
            controls["output"] = None if omit_observer else (
                lambda name, own, carry: outputs.append([name, own]))
        initial = {"d01": start, "d02": start * edge.parent_grid_ratio}
        result = dt.run_domain_tree_callbacks(
            hierarchy, {name: Carry(own) for name, own in initial.items()},
            root_steps=count, advance=advance, force=lambda edge, parent, child: child,
            fused_cascade=lookup, initial_own_steps=initial, **controls)
        assert result.own_steps == {"d01": start + count, "d02": (start + count) * 3}
        assert {name: carry.step for name, carry in result.carries.items()} == result.own_steps
        rows[label] = dict(advance=calls, fused=fused_calls, outputs=outputs,
                           own_steps=result.own_steps)
        return rows[label]

    for start, label in ((0, "first"), (1, "ordinary")):
        old = run("legacy_" + label, start, legacy=True)
        assert len(old["fused"]) == 1 and not old["advance"]
        new = run("product_" + label, start)
        assert not new["fused"]
        assert new["advance"] == [["d01", start + 1, 1], ["d02", 3 * start + 1, 3]]
    tail = run("product_first_alarm", 66)
    assert tail["advance"] == [["d01", 67, 1], ["d02", 199, 2], ["d02", 201, 1]]
    assert tail["outputs"] == [["d02", 200], ["d01", 67]]
    tail2 = run("product_second_alarm", 133)
    assert tail2["advance"] == [["d01", 134, 1], ["d02", 400, 1], ["d02", 401, 2]]
    missing = run("negative_omitted_observer", 0, omit_observer=True)
    assert missing["fused"] and not missing["advance"]
    whole = run("product_3h", 0, count=200)
    assert not whole["fused"]
    assert sorted(whole["outputs"]) == [
        ["d01", 67], ["d01", 134], ["d01", 200],
        ["d02", 200], ["d02", 400], ["d02", 600]]
    whole["advance_chunk_census"] = dict(Counter(f"{name}:{n}" for name, own, n in whole.pop("advance")))
    source = Path(dt.__file__).resolve().parents[3]
    receipt = dict(status="PASS", schedule=schedule, cases=rows,
                   source=str(source), source_pin=subprocess.check_output(
                       ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
                   tools_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (Path(__file__), Path(__file__).with_name("product_schedule.py"))},
                   scheduler_sha256=hashlib.sha256(Path(dt.__file__).read_bytes()).hexdigest(),
                   scope="Scheduling proof only; opaque integer carries and recorder callbacks. No model arithmetic, numerical fidelity, native compile, or GPU claim.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "cases": list(rows), "schedule": schedule,
                      "full_advance_chunks": whole["advance_chunk_census"]}, indent=2))


if __name__ == "__main__":
    main()
