#!/usr/bin/env python3
"""Regenerate or check ``src/gpuwrf/assets_manifest.json`` (pinned runtime tables, see ``gpuwrf/assets.py``).

usage: update_assets_manifest.py [--check]

Registered files = OUT_OF_PACKAGE (runtime files outside src/gpuwrf) + every tracked src/gpuwrf file larger
than 10 MiB. Kind: > 10 MiB -> ``remote`` (release asset, fetched on first use); otherwise ``bundle``
(copied into the wheel by setup.py). ``--check`` exits 1 when the committed manifest is stale (edited table
without a new pin, new big table, drifted size).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "src" / "gpuwrf" / "assets_manifest.json"
MAX_BUNDLE_BYTES = 10 * 1024 * 1024  # = the public curation MAX_NEW_BLOB gate
URL_TEMPLATE = "https://github.com/wrf-gpu/wrf_gpu/releases/download/v{version}/{asset}"
# Runtime files the package reads from outside src/gpuwrf (repo-root relative). Grep for
# `parents[3] / "data"` / `scripts/` readers in src before adding a new one.
OUT_OF_PACKAGE = (
    "data/fixtures/rrtmg-tables-v1.npz",          # physics/rrtmg_tables.py (import time, ra=4)
    "data/fixtures/wrf-cam-ozone-v1.npz",         # physics/wrf_cam_ozone.py (ra=3/4)
    "data/fixtures/thompson-tables-v1.npz",       # physics/thompson_tables.py (import time)
    "data/fixtures/thompson-cold-collection-v1.npz",  # physics/thompson_tables.py (import time, default ON)
    "data/fixtures/thompson-mixed-phase-v1.npz",  # physics/thompson_tables.py (mixed-phase default ON)
    "data/fixtures/thompson-aero-tables-v1.npz",  # physics/thompson_aero_tables.py (mp=28/38)
    "scripts/extract_rrtmg_tables.py",            # physics/rrtmg_lw.py RRTMG source parser (ra=4)
)


def asset_name(repo_path: str) -> str:
    parts = repo_path.split("/")
    if parts[:3] == ["src", "gpuwrf", "physics"] and len(parts) > 4:
        return parts[3] + "-" + parts[-1]  # physics/<scheme>/[data/]<file> -> <scheme>-<file>
    return parts[-1]


def tracked_big_src() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z", "src/gpuwrf"], cwd=ROOT, check=True, capture_output=True).stdout
    paths = [p for p in out.decode().split("\0") if p]
    return sorted(p for p in paths if (ROOT / p).is_file() and (ROOT / p).stat().st_size > MAX_BUNDLE_BYTES)


def build() -> dict:
    assets = {}
    # A curated public tree has no copy of the release-asset tables: keep their committed pins.
    committed = json.loads(MANIFEST.read_text(encoding="utf-8"))["assets"] if MANIFEST.is_file() else {}
    for repo_path, info in committed.items():
        if info["kind"] == "remote" and not (ROOT / repo_path).exists():
            assets[repo_path] = info
    for repo_path in sorted(set(OUT_OF_PACKAGE) | set(tracked_big_src())):
        path = ROOT / repo_path
        if not path.exists() and committed.get(repo_path, {}).get("kind") == "remote":
            continue  # release-asset table absent from this tree: committed pin kept above
        data = path.read_bytes()
        assets[repo_path] = dict(kind="remote" if len(data) > MAX_BUNDLE_BYTES else "bundle", asset=asset_name(repo_path),
                                 bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    names = [info["asset"] for info in assets.values() if info["kind"] == "remote"]
    assert len(names) == len(set(names)), f"release asset names collide: {names}"
    return dict(schema="gpuwrf-runtime-assets-v1", url_template=URL_TEMPLATE, max_bundle_bytes=MAX_BUNDLE_BYTES,
                assets=dict(sorted(assets.items())))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    text = json.dumps(build(), indent=1, sort_keys=True) + "\n"
    if ap.parse_args().check:
        if MANIFEST.read_text(encoding="utf-8") != text:
            print(f"STALE {MANIFEST.relative_to(ROOT)}: run scripts/update_assets_manifest.py", file=sys.stderr)
            return 1
        print("assets manifest up to date")
        return 0
    MANIFEST.write_text(text, encoding="utf-8")
    print(f"wrote {MANIFEST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
