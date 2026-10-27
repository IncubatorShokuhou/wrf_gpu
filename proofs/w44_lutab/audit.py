"""CPU-only land-use admission reproduction and initialized-leaf byte receipt.

Use a committed detached tree; output is under the lane artifact directory.
The USGS variant changes only metadata on the shipped real Swiss input. It
proves admission/table selection, and is NOT a USGS scientific forecast.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import jax
import numpy as np

from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io.noahmp_land_init import build_noahmp_land_state
from gpuwrf.physics.noahmp.tables import load_noahmp_parameters


def main():
    root = Path(__file__).resolve().parents[2]
    output = Path(sys.argv[1]).resolve()
    assert jax.devices()[0].platform == "cpu"
    source = root / "examples/switzerland_d01"
    receipt = {"commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "platform": "cpu", "cache": "disabled", "gpu_executed": False}
    paths = [source / "wrfinput_d01"]
    paths += sorted(Path('<DATA_ROOT>/wrf_gpu2/scratch/w43-release/cases').glob('*/wrfinput_d0*'))
    receipt["input_attributes"] = []
    for path in paths:
        with Dataset(path) as ds:
            receipt["input_attributes"].append({"path": str(path),
                "MMINLU": str(ds.MMINLU), "NUM_LAND_CAT": int(ds.NUM_LAND_CAT)})
    t0 = time.perf_counter()
    land, static, metadata = build_noahmp_land_state(source, "d01")
    leaves = jax.tree_util.tree_leaves(jax.device_get((land, static)))
    receipt["modis_init_seconds"] = time.perf_counter() - t0
    receipt["modis_leaves"] = []
    for leaf in leaves:
        a = np.asarray(leaf)
        assert np.isfinite(a).all()
        receipt["modis_leaves"].append({"dtype": str(a.dtype), "shape": list(a.shape),
            "sha256": hashlib.sha256(a.tobytes()).hexdigest()})
    receipt["modis_metadata"] = metadata
    receipt["table_sentinels"] = {}
    for dataset in ("USGS", "MODIFIED_IGBP_MODIS_NOAH"):
        p = load_noahmp_parameters(dataset=dataset)
        receipt["table_sentinels"][dataset] = {name: int(getattr(p, name))
            for name in ("isurban", "iswater", "isbarren", "isice")}
    with tempfile.TemporaryDirectory(prefix="usgs_metadata_", dir=output.parent) as tmp:
        case = Path(tmp)
        (case / "namelist.input").symlink_to(source / "namelist.input")
        shutil.copyfile(source / "wrfinput_d01", case / "wrfinput_d01")
        with Dataset(case / "wrfinput_d01", "r+") as ds:
            ds.MMINLU = "USGS"
            ds.NUM_LAND_CAT = np.int32(24)
        t0 = time.perf_counter()
        try:
            _, usgs_static, _ = build_noahmp_land_state(case, "d01")
        except ValueError as error:
            receipt["usgs_metadata_variant"] = {"admitted": False, "error": str(error)}
        else:
            receipt["usgs_metadata_variant"] = {"admitted": True,
                "actual_sentinels": {name: int(getattr(usgs_static.parameters, name))
                    for name in ("isurban", "iswater", "isbarren", "isice")}}
        receipt["usgs_metadata_variant"]["elapsed_seconds"] = time.perf_counter() - t0
    output.write_text(json.dumps(receipt, indent=2, default=str) + "\n")
    print(json.dumps({"output": str(output), "leaves": len(leaves),
        "modis_init_seconds": receipt["modis_init_seconds"],
        "usgs": receipt["usgs_metadata_variant"]}))


if __name__ == "__main__":
    main()
