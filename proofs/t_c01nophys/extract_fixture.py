"""Extract a small independent original-WRF history fixture, with provenance."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np
from gpuwrf.io.netcdf_lock import Dataset

case = Path("<USER_HOME>/wrf_gpu2_lanes/v-prep/c01")
output = Path(__file__).parent / "held_surface.npz"
fields = ("PSFC", "T2", "Q2", "U10", "V10", "TH2")
arrays = {}
sources = {}
paths = {"input": case / "input/wrfinput_d01"}
frames = sorted((case / "cpu").glob("wrfout_d01_*"))
paths.update({f"h{h}": frames[h] for h in (0, 1, 2, 24)})
for label, path in paths.items():
    sources[label] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    with Dataset(path) as ds:
        for name in fields:
            # Fixed real terrain crop, same spatial cells in input and history.
            arrays[f"{label}_{name}"] = np.asarray(ds[name][0, 19:23, 18:23])
np.savez_compressed(output, **arrays)
manifest = {
    "source": "original serial WRF v4.7.1, c01 all physics off; never GPU reference",
    "case_inputs": json.loads((case / "inputs.json").read_text()),
    "cpu_executable": json.loads((case / "cpu_run.json").read_text())["executable"],
    "crop": {"south_north": [19, 23], "west_east": [18, 23]},
    "source_files": sources,
    "fixture_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    "all_fields_all_hours_byte_equal_input": all(
        arrays[f"h{h}_{name}"].tobytes() == arrays[f"input_{name}"].tobytes()
        for h in (0, 1, 2, 24) for name in fields),
}
output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"fixture": str(output), "sha256": manifest["fixture_sha256"], "byte_equal": manifest["all_fields_all_hours_byte_equal_input"]}))
