"""Extract the original caller's cached DOUBLE tables without recomputation."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time

import numpy as np


def extract(source, output, receipt):
    start = time.monotonic()
    names = ("tpi_qrfz", "tni_qrfz", "tpg_qrfz", "tnr_qrfz", "tpi_qcfz", "tni_qcfz")
    rows = []
    output.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as stream:
        for name in names:
            length = struct.unpack(">i", stream.read(4))[0]
            shape = (37, 37, 45, 55) if name.endswith("qrfz") else (37, 100, 45, 55)
            assert length == int(np.prod(shape)) * 8
            raw = stream.read(length)
            assert len(raw) == length and struct.unpack(">i", stream.read(4))[0] == length
            array = np.frombuffer(raw, dtype=">f8").reshape(shape, order="F").astype(np.float64)
            # Conversion changes endian/layout only; certify every original
            # DOUBLE bit, including signed zero, before packing.
            assert array.astype(">f8").tobytes(order="F") == raw
            target = output / ("freezing_tables_" + name + ".npz")
            np.savez_compressed(target, values=array)
            with np.load(target, allow_pickle=False) as packed:
                assert packed["values"].tobytes() == array.tobytes()
            rows.append(dict(name=name, shape=shape, bytes=length, sha256=hashlib.sha256(raw).hexdigest(),
                asset=target.name, asset_sha256=hashlib.sha256(target.read_bytes()).hexdigest(), compressed_bytes=target.stat().st_size))
            del raw, array
        assert stream.read(1) == b""
    receipt.write_text(json.dumps(dict(scope="Original cached tables read by unmodified WRF caller; no quadrature rebuild",
        source=str(source.resolve()), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        compressed_bytes=sum(row["compressed_bytes"] for row in rows),
        raw_bytes=sum(row["bytes"] for row in rows), records=rows, wall_s=time.monotonic()-start,
        every_original_double_bit_preserved=True), indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    extract(args.source, args.output, args.receipt)
