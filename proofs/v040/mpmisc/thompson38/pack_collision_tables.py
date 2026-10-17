"""Lossless 9-part packaging of WRF's five DOUBLE hail collision channels.

Each density part fits GitHub's per-blob limit. Only one 15 MB channel piece
is materialized at a time; runtime reconstructs the original 5-D tensors.
"""
from pathlib import Path
import argparse
import hashlib
import json
import struct
import zipfile
import numpy as np

NAMES = ("tcg_racg", "tmr_racg", "tcr_gacr", "tnr_racg", "tnr_gacr")
SHAPE = (37, 37, 9, 37, 37)
SIZE = int(np.prod(SHAPE)) * 8


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pack(source, target, receipt):
    target.parent.mkdir(parents=True, exist_ok=True)
    offsets = {}
    with source.open("rb") as raw:
        for name in NAMES:
            length = struct.unpack(">i", raw.read(4))[0]
            assert length == SIZE, (name, length, SIZE)
            offsets[name] = raw.tell()
            raw.seek(length, 1)
            assert struct.unpack(">i", raw.read(4))[0] == length
        assert not raw.read(1)
    data = {"source": str(source), "source_sha256": digest_file(source), "shape": SHAPE,
            "dtype": "float64", "raw_payload_bytes": SIZE*5, "parts": []}
    for density in range(9):
        part = target.with_name(f"{target.stem}_{density}.npz")
        meta = {"density_index0": density, "path": str(part), "channels": {}}
        with zipfile.ZipFile(part, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
            for name in NAMES:
                mapped = np.memmap(source, mode="r", dtype=">f8", offset=offsets[name], shape=SHAPE, order="F")
                array = np.asarray(mapped[:,:,density,:,:], dtype=np.float64, order="C")
                assert np.isfinite(array).all(), (density,name)
                with archive.open(name+".npy", "w", force_zip64=True) as entry:
                    np.lib.format.write_array(entry, array, allow_pickle=False)
                meta["channels"][name] = {"sha256_C_native_f8": hashlib.sha256(memoryview(array).cast("B")).hexdigest(), "nonzero": int(np.count_nonzero(array))}
                del array,mapped
        assert part.stat().st_size < 95_000_000, "keep every blob under the hosting limit"
        with np.load(part, allow_pickle=False) as archive:
            for name in NAMES:
                mapped = np.memmap(source, mode="r", dtype=">f8", offset=offsets[name], shape=SHAPE, order="F")
                array = archive[name]
                assert array.shape == (37,37,37,37) and array.dtype == np.float64
                assert np.array_equal(array,mapped[:,:,density,:,:]), (density,name)
                del array,mapped
        meta.update(bytes=part.stat().st_size,sha256=digest_file(part))
        data["parts"].append(meta)
        print(density,meta["bytes"],flush=True)
    data["compressed_bytes"] = sum(p["bytes"] for p in data["parts"])
    data["roundtrip"] = "all 45 channel/density pieces byte-exact FLOAT64 values against original Fortran records"
    receipt.write_text(json.dumps(data,indent=2)+"\n")


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("source",type=Path)
    parser.add_argument("target",type=Path)
    parser.add_argument("receipt",type=Path)
    args=parser.parse_args()
    pack(args.source,args.target,args.receipt)
