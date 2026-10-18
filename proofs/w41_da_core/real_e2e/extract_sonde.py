"""Guarded extract of ONE IGRA2 launch (Guimar SPM00060018, 2026-02-28 12Z, A1 development case 0227) for the
w41-da-core real-data CPU end-to-end. ALISIOS data rule: alisios_guard.guarded_session (guard scan of issue day
20260227 + valid day 20260228 -> admit CALIBRATION_READ -> read spy). The raw file is scanned from a byte offset near
the target (headers only); only the target launch block is parsed and copied. Values are never logged.

  JAX_PLATFORMS=cpu PYTHONPATH=src taskset -c 8,9 nice -n 19 python proofs/w41_da_core/real_e2e/extract_sonde.py OUT_DIR
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

from gpuwrf.assimilation.letkf import alisios_guard

IGRA = Path("<DATA_ROOT>/alisios/state/manager/runs/core_interp_research_20260930/shared_truth/radiosonde_60018/"
            "raw_igra2/SPM00060018-data_before20260701.txt")
STATION, TARGET = "SPM00060018", ("2026", "02", "28", "12")
ISSUE, VALID = ["20260227"], ["20260228"]


def _header_key(line):
    return (line[13:17], line[18:20], line[21:23], line[24:26])


def main(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    extract = out_dir / "igra2_SPM00060018_20260228T12_admitted.txt"
    with alisios_guard.guarded_session({str(IGRA): VALID}, ISSUE, out_dir / "guard",
                                       purpose="w41-da-core real-data CPU e2e (dev case 0227): one sonde launch") as s:
        size = IGRA.stat().st_size
        start = max(size - 8_000_000, 0)                 # the file is time-ordered; 2026 is near its end
        block, scanned_from = None, None
        while block is None:
            with open(IGRA, "rb") as f:
                f.seek(start)
                f.readline()                             # resynchronise on a line boundary
                first = None
                for raw in f:
                    line = raw.decode("ascii", "replace")
                    if not line.startswith("#"):
                        continue
                    key = _header_key(line)
                    first = first or key
                    if line[1:12].strip() == STATION and key == TARGET:
                        n = int(line[32:36])
                        block = [line] + [f.readline().decode("ascii") for _ in range(n)]
                        break
                    if key > TARGET:
                        break
            scanned_from = start
            if block is None:
                if start == 0 or (first is not None and first <= TARGET):
                    raise LookupError("target launch not in the raw file")
                start = max(start - 8_000_000, 0)
        extract.write_text("".join(block))
    receipt = {"schema": "w41_da_core.guarded_extract.v1", "source": str(IGRA), "station": STATION,
               "launch": "-".join(TARGET[:3]) + "T" + TARGET[3] + "Z", "levels": len(block) - 1,
               "scan": {"byte_offset_start": scanned_from, "method": "headers only from the offset; target block copied"},
               "extract": str(extract), "extract_sha256": hashlib.sha256(extract.read_bytes()).hexdigest(),
               "guard": s.receipt, "values_logged": False}
    (out_dir / "EXTRACT_RECEIPT.json").write_text(json.dumps(receipt, indent=1, default=str) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "guard"}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
