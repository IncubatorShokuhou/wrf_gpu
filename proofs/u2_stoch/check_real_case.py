"""Offline real-input arithmetic + existing ensemble preparation, never a forecast."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np

from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.perturb.inputs import prepare
from gpuwrf.ens.workflows import prepare as prepare_ensemble


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    result = prepare(args.case, args.output / "members", members=2, seed=451, wind_std=.05)
    prep_s = time.perf_counter() - start
    counts = {"scaled": 0, "unchanged": 0, "sources_unchanged": 0}
    for member in result["members"]:
        for filename in member["prepared"]:
            with Dataset(args.case / filename) as source, Dataset(Path(member["case"]) / filename) as got:
                assert source.dimensions.keys() == got.dimensions.keys()
                assert source.ncattrs() == got.ncattrs()
                for name in source.ncattrs():
                    np.testing.assert_array_equal(source.getncattr(name), got.getncattr(name))
                for name in source.variables:
                    selected = name in member["changes"][filename]
                    expected = source[name][:]
                    if selected:
                        expected = expected * np.float32(member["factor_real32"])
                    np.testing.assert_array_equal(got[name][:], expected)
                    counts["scaled" if selected else "unchanged"] += 1
    for item in result["source"].values():
        digest = hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()
        assert digest == item["sha256"]
        counts["sources_unchanged"] += 1
    plan = prepare_ensemble("batch", args.output / "plan", hours=24,
                            cases=[member["case"] for member in result["members"]])
    assert len(plan["members"]) == 2 and plan["mode"] == "batch"
    receipt = {"source_case": str(args.case), "counts": counts, "member_manifest": result,
               "existing_ensemble_plan": plan, "prepare_wall_s": prep_s,
               "whole_probe_wall_s": time.perf_counter() - start,
               "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               "backend": "NumPy/NetCDF only; no forecast", "gpu": False,
               "qualification": "REAL INPUT ARITHMETIC AND WORKFLOW; NOT FORECAST FIDELITY"}
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k not in {"member_manifest", "existing_ensemble_plan"}}))


if __name__ == "__main__":
    main()
