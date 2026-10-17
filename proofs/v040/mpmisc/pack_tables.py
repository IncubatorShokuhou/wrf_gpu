"""Package the WRF binary tables and original-Fortran saturation export."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


def pack(binary: Path, saturation: Path, receipt: Path, target: Path):
    raw = binary.read_bytes()
    arrays = {}
    offset = 0
    names = "ventr1 ventr2 accrr massr vrain rrate venti1 venti2 accri massi vsnowi vel_rf".split()
    # The file has a final MY_GROWTH(35) record. FER_HIRES_INIT deliberately
    # reads only the first 12 records and computes its own timestep growth.
    inventory = []
    for name in names + ["unused_growth"]:
        count = struct.unpack_from(">i", raw, offset)[0]
        offset += 4
        assert struct.unpack_from(">i", raw, offset + count)[0] == count
        values = np.frombuffer(raw[offset:offset + count], ">f4").astype(np.float32)
        expected = 328 if name == "vel_rf" else 35 if name == "unused_growth" else 951
        assert values.size == expected, (name, values.size)
        inventory.append({"name": name, "bytes": count})
        if name != "unused_growth":
            arrays[name] = values.reshape((8, 41), order="F") if name == "vel_rf" else values
        offset += count + 4
    assert offset == len(raw)
    tables = np.loadtxt(saturation, dtype=np.float32)
    assert tables.shape == (7501, 2) and np.isfinite(tables).all()
    arrays["tbpvs"], arrays["tbpvs0"] = tables[:, 0], tables[:, 1]
    np.savez_compressed(target, **arrays)
    source_receipt = json.loads(receipt.read_text())
    meta = {
        "source": "pristine WRF module_mp_fer_hires.F:GPVS/FPVSX/FPVSX0 + ETAMPNEW_DATA.expanded_rain",
        "source_commit": source_receipt["source_commit"],
        "source_hashes": source_receipt["sources"],
        "asset_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "table_sha256": hashlib.sha256(raw).hexdigest(),
        "saturation_export_sha256": hashlib.sha256(saturation.read_bytes()).hexdigest(),
        "precision": "WRF REAL32",
        "indexing": "size tables 50:1000; vel_rf(2:9,0:40); saturation 1:7501",
        "record_inventory": inventory,
        "fixture_boundary": "static tables; timestep-dependent MY_GROWTH excluded",
        "units": {"tbpvs": "kPa", "tbpvs0": "kPa", "massi": "kg", "massr": "kg m", "vsnowi": "m s-1", "vrain": "m s-1"},
        "tolerance": "stored lookup values exact REAL32; downstream arithmetic independently oracle-tested",
    }
    target.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--saturation", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--target", type=Path, required=True)
    a = p.parse_args()
    pack(a.binary, a.saturation, a.receipt, a.target)
