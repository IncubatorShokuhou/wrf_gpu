"""Read-only, all-frame original-CPU proof for manager-authorized class B.

No model import or GPU. Missing frames/fields, masks, nonfinite values and any
nonconstant bit pattern refuse admission. The receipt binds the complete bytes
of every original file, its sealed inventory and its actual namelist.
"""
import hashlib
import json
from pathlib import Path
import sys
import time

import netCDF4
import numpy as np

FIELDS = {
    **{f"c{i:02d}": ("REFD_MAX",) for i in range(12, 17)},
    "c20": (*[f"CCN{i}_GS" for i in range(1, 8)], "EFCG", "EFIG", "EFSG", "WACT"),
    "c46": ("FIELD_CONV", "FIELD_PBL", "FIELD_SF"),
    "c87": ("ICE_TYPE", "ICE_TYPE2", "ICE_TYPE3"),
}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(4 << 20), b""):
            h.update(data)
    return h.hexdigest()


def audit(root, output):
    started = time.monotonic()
    result = {"schema": "t-writer-inactive-all-frame-v1", "source": "original CPU-WRF",
              "script_sha256": sha(Path(__file__)), "cases": {}}
    for case, names in FIELDS.items():
        folder = root / case
        manifest = folder / "CPU_OUTPUTS.json"
        sealed = json.loads(manifest.read_text())
        expected = sorted(x["file"] for x in sealed if x["file"].startswith("wrfout_d01_"))
        actual = sorted(p.name for p in (folder / "cpu").glob("wrfout_d01_*"))
        # These are ordinary 24 h cases, including lead zero. Both the sealed
        # manifest and the independent case duration must establish completeness.
        if expected != actual or len(expected) != 25:
            raise ValueError(f"{case}: incomplete original frames: sealed={len(expected)} actual={len(actual)}")
        record = {"cpu_manifest_sha256": sha(manifest),
                  "namelist_sha256": sha(folder / "input/namelist.input"),
                  "frames": [], "fields": {name: {"constant": True} for name in names}}
        for name in expected:
            path = folder / "cpu" / name
            frame = {"file": name, "sha256": sha(path), "fields": {}}
            with netCDF4.Dataset(path, "r") as ds:
                ds.set_auto_mask(False)
                for field in names:
                    var = ds.variables[field]
                    a = np.asarray(var[:])
                    first = a.reshape(-1)[0]
                    bits = np.ascontiguousarray(a).view(f"u{a.dtype.itemsize}")
                    bit = bits.reshape(-1)[0]
                    constant = bool(np.isfinite(a).all() and np.all(bits == bit))
                    info = record["fields"][field]
                    value = {"dtype": a.dtype.str, "shape": list(a.shape),
                             "value": float(first), "bits": int(bit),
                             "all_equal_bits": constant,
                             "data_sha256": hashlib.sha256(a.tobytes()).hexdigest()}
                    frame["fields"][field] = value
                    if "bits" not in info:
                        info.update(dtype=a.dtype.str, value=float(first), bits=int(bit),
                                    dimensions=list(var.dimensions))
                    info["constant"] &= constant and info["bits"] == int(bit) and info["dtype"] == a.dtype.str
            record["frames"].append(frame)
        result["cases"][case] = record
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(case, len(record["frames"]), {n: (r["constant"], r["value"]) for n, r in record["fields"].items()}, flush=True)
    result["wall_s"] = time.monotonic() - started
    result["admitted_pairs"] = sum(r["constant"] for c in result["cases"].values() for r in c["fields"].values())
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    if "--fasdas" in sys.argv:
        # This arm checks the discovered Noah-MP inactive branch. It does not
        # authorize a new constant: the manager decides its disposition.
        FIELDS = {"c78": ("HFX_BOTH", "QFX_BOTH")}
    audit(Path(sys.argv[1]), Path(sys.argv[2]))
