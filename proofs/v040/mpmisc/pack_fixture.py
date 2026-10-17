"""Package a fresh original-WRF Ferrier receipt without changing its values."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def pack(receipt_path, output):
    receipt = json.loads(receipt_path.read_text())
    arrays, groups = {}, {}
    for group, arm in receipt["arms"].items():
        if group.startswith("legacy"):
            continue
        cases = [json.loads(Path(case["path"]).read_text()) for case in arm["cases"]]
        option = 15 if "15" in group else 5
        fields = ["TH_OUT", "QV_OUT", "QC_OUT", "QI_OUT", "QR_OUT", "RAINNCV", "SR"]
        fields += (["QT_OUT", "F_ICE_PHY_OUT", "F_RAIN_PHY_OUT", "F_RIMEF_PHY_OUT"]
                   if option == 5 else ["QRIMEF_OUT"])
        if arm["steps"] > 1:
            fields += ["RAINNC"]
        for name in cases[0]["columns"]:
            arrays[group + "__" + name] = np.asarray([c["columns"][name] for c in cases], np.float32).T
        for name in cases[0]["scalars"]:
            arrays[group + "__" + name] = np.asarray([c["scalars"][name] for c in cases], np.float32)
        groups[group] = {"option": option, "dt_seconds": arm["dt_seconds"], "gid": arm["gid"],
                         "steps": arm["steps"], "fields": fields,
                         "oracle_source_commit": receipt["source_commit"],
                         "driver_sha256": arm["driver_sha256"],
                         "reference_sha256": [c["sha256"] for c in arm["cases"]]}
    np.savez_compressed(output, **arrays)
    metadata = {"sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "groups": groups,
                "wrf_sources": receipt["sources"], "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest()}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    pack(args.receipt, args.output)
