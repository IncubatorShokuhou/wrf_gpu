#!/usr/bin/env python3
"""Download UCAR's small educational geography separately from the image.

Uses only stdlib; no model/JAX imports. Refuses existing output directories,
oversized downloads/extraction, symlinks, hardlinks and path traversal.
"""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from urllib.request import urlopen

URL = "https://www2.mmm.ucar.edu/wrf/src/wps_files/geog_low_res_mandatory.tar.gz"
LIMIT = 2_000_000_000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    root = args.destination.resolve()
    root.mkdir(parents=True, exist_ok=False)
    archive = root / "download.tar.gz.part"
    digest, size = hashlib.sha256(), 0
    try:
        with urlopen(URL, timeout=90) as response, archive.open("xb") as out:
            while block := response.read(1024 * 1024):
                size += len(block)
                if size > LIMIT:
                    raise ValueError("geography download exceeds 2 GB cap")
                digest.update(block)
                out.write(block)
        with tarfile.open(archive) as tar:
            members = tar.getmembers()
            if sum(m.size for m in members) > LIMIT:
                raise ValueError("geography extraction exceeds 2 GB cap")
            for member in members:
                path = (root / member.name).resolve()
                if not path.is_relative_to(root) or not (member.isdir() or member.isfile()):
                    raise ValueError(f"unsafe geography archive entry: {member.name}")
            tar.extractall(root, members=members)
        (root / "download.json").write_text(json.dumps(dict(url=URL, bytes=size,
                                                         sha256=digest.hexdigest()), indent=2) + "\n")
        print(root)
    finally:
        archive.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
