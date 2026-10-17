"""Small explicit member inventories. Paths are relative to the JSON file."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def identity(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def read_members(path):
    path = Path(path).resolve(strict=True)
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("member inventory must be a JSON object")
    members = data.get("members")
    if not isinstance(members, list) or len(members) < 2:
        raise ValueError("member inventory requires at least two members")
    out, names, inventories = [], set(), set()
    for member in members:
        if not isinstance(member, dict):
            raise ValueError("each member must be a JSON object")
        name, files = member.get("id"), member.get("files")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("member IDs must be unique nonempty strings")
        if not isinstance(files, list) or not files or not all(isinstance(f, str) and f for f in files):
            raise ValueError(f"{name}: files must be a nonempty list of paths")
        files = [(path.parent/f).resolve(strict=True) for f in files]
        if not all(f.is_file() for f in files) or len(set(files)) != len(files):
            raise ValueError(f"{name}: duplicate paths or non-file inputs")
        signature = tuple(sorted(str(f) for f in files))
        if signature in inventories:
            raise ValueError("duplicate member file inventory; one run cannot count twice")
        names.add(name)
        inventories.add(signature)
        out.append({"id": name, "files": files})
    return out
