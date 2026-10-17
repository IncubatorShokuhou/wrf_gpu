"""Check lane assets from a clean immutable git archive without extracting it.

Usage: check_archive.py SOURCE_TREE OUTPUT_JSON
Only NumPy and standard-library imports; no JAX, build, or persistent cache.
"""
from pathlib import Path
import fnmatch
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import time
import tomllib
import zipfile

import numpy as np


tree, output = map(Path, sys.argv[1:3])
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tree, text=True).strip()
paths = ("src/gpuwrf/physics/ferrier", "src/gpuwrf/physics/goddard4ice",
         "src/gpuwrf/physics/thompson_hail", "tests/v040/mpmisc", "proofs/v040/mpmisc")
start = time.monotonic()
proc = subprocess.Popen(["git", "archive", commit, "--", *paths], cwd=tree, stdout=subprocess.PIPE)
assets = {}
with tarfile.open(fileobj=proc.stdout, mode="r|") as archive:
    for member in archive:
        if not member.isfile() or not member.name.endswith((".npz", ".json")):
            continue
        raw = archive.extractfile(member).read()
        entry = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        if member.name.endswith(".npz"):
            assert len(raw) < 100_000_000, member.name
            count = values = 0
            # ZipFile.read checks each CRC. Decode each NPY only once, keeping
            # peak memory bounded to one compressed asset and one array.
            with zipfile.ZipFile(io.BytesIO(raw)) as packed:
                for name in packed.namelist():
                    assert name.endswith(".npy"), (member.name, name)
                    array = np.load(io.BytesIO(packed.read(name)), allow_pickle=False)
                    assert np.issubdtype(array.dtype, np.number) or array.dtype == np.bool_, (member.name, name, array.dtype)
                    assert np.isfinite(array).all(), (member.name, name)
                    count += 1
                    values += array.size
                    del array
            entry.update(arrays=count, values=values)
        else:
            def nonfinite(value):
                raise ValueError("nonfinite JSON number: " + value)
            json.loads(raw, parse_constant=nonfinite)
        assets[member.name] = entry
        del raw
assert proc.wait() == 0
root = "src/gpuwrf/physics/"
for pattern, expected in (("thompson_hail/collision_tables_*.npz", 9),
                          ("thompson_hail/freezing_tables_*.npz", 6),
                          ("thompson_hail/cloud_coefficients.npz", 1),
                          ("ferrier/tables.npz", 1), ("ferrier/tables.json", 1)):
    assert sum(fnmatch.fnmatch(name, root + pattern) for name in assets) == expected, pattern
packaging = tomllib.loads(subprocess.check_output(
    ["git", "show", commit + ":pyproject.toml"], cwd=tree, text=True))["tool"]["setuptools"]["package-data"]
for name in assets:
    if name.startswith(root):
        package = str(Path(name).parent).replace("/", ".").removeprefix("src.")
        assert any(fnmatch.fnmatch(Path(name).name, pattern) for pattern in packaging.get(package, ())), name
receipt = {"commit": commit, "platform": "CPU/no JAX", "assets": assets,
           "asset_count": len(assets), "compressed_bytes": sum(x["bytes"] for x in assets.values()),
           "wall_s": time.monotonic() - start, "cache": "NONE", "archive_extracted": False,
           "gate": "PASS: original assets present and packaged, NPZ CRC/dtype/finite and strict JSON checks"}
output.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({key: value for key, value in receipt.items() if key != "assets"}))
