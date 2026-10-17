"""Full-frame original-WRF byte audit for the six held surface fields."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from gpuwrf.io.netcdf_lock import Dataset

parser = argparse.ArgumentParser()
parser.add_argument("case", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
fields = ("PSFC", "T2", "Q2", "U10", "V10", "TH2")
rows = []
cpu_paths = sorted((args.case / "cpu").glob("wrfout_d01_*"))
gpu_paths = sorted((args.case / "gpu").glob("wrfout_d01_*"))
assert len(cpu_paths) == len(gpu_paths) == 25
for cpu_path, gpu_path in zip(cpu_paths, gpu_paths, strict=True):
    assert cpu_path.name == gpu_path.name
    with Dataset(cpu_path) as cpu, Dataset(gpu_path) as gpu:
        for name in fields:
            a, b = cpu[name][0], gpu[name][0]
            assert not np.ma.getmaskarray(a).any() and not np.ma.getmaskarray(b).any()
            a, b = np.asarray(a), np.asarray(b)
            assert np.isfinite(a).all() and np.isfinite(b).all()
            rows.append({"frame": cpu_path.name, "field": name,
                         "dtype_cpu": str(a.dtype), "dtype_gpu": str(b.dtype),
                         "shape_cpu": list(a.shape), "shape_gpu": list(b.shape),
                         "cpu_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
                         "gpu_sha256": hashlib.sha256(b.tobytes()).hexdigest(),
                         "byte_equal": a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()})
result = {"reference": "original serial CPU-WRF c01, no GPU self-compare",
          "rows": rows, "frame_count": 25, "field_count": 6,
          "byte_equal_count": sum(r["byte_equal"] for r in rows),
          "all_byte_equal": all(r["byte_equal"] for r in rows),
          "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "rows"}))
assert result["all_byte_equal"]
