#!/usr/bin/env python3
"""SHA-pinned release attachments: repository files > 10 MiB that are kept outside the git tree (v0.4.1, B17).

Each attachment leaves a small provenance index ``<path>.attachment.json`` where the file used to be. The index pins
the bytes (size, SHA-256, git blob id), the release asset that carries them (GitHub release ``release_tag``), the
public commits whose history holds the same blob, and where the file came from. Whoever needs the file stages it:

    python3 tools/release_attachments.py stage --all        # or: stage <index> ...
    python3 tools/release_attachments.py list

``stage`` returns a read-only copy in the content-addressed cache ``<GPUWRF_TABLES_DIR or
$XDG_CACHE_HOME/gpuwrf/tables or ~/.cache/gpuwrf/tables>/<sha256>/<asset>``, the same cache as the gpuwrf runtime
tables. It fills the cache from, in this order:
  1. the cache itself (re-verified);
  2. the git object store of this checkout (``git cat-file blob <git_blob>``; every clone with the v0.3.x/v0.4.0
     history has it);
  3. the release URL (``GPUWRF_ATTACHMENTS_URL`` template with {tag} {asset} {sha256} for mirrors, and
     ``GPUWRF_TABLES_OFFLINE=1`` never downloads).
Every source is checked against the pinned size and SHA-256 before the cache file appears (atomic rename).
Nothing found raises :class:`AttachmentUnavailable`, whose message says what was tried and how to fix it (tests turn
it into a skip). Standard library only; never imports gpuwrf or touches a GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SCHEMA = "gpuwrf-release-attachment-v1"
SUFFIX = ".attachment.json"
ROOT = Path(__file__).resolve().parents[1]
URL = "https://github.com/wrf-gpu/wrf_gpu/releases/download/{tag}/{asset}"
ASSET_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_CHUNK = 1 << 20


class AttachmentUnavailable(FileNotFoundError):
    """A pinned attachment is neither cached, in this checkout's git history, nor downloadable."""


def load(index) -> dict:
    """Read and validate one provenance index."""
    index = Path(index)
    return check(json.loads(index.read_text(encoding="utf-8")), str(index))


def check(info: dict, index: str) -> dict:
    """Validate a provenance index (``index`` = its path; it must end in ``<path of the attached file>.attachment.json``)."""
    problems = []
    if info.get("schema") != SCHEMA:
        problems.append(f"schema {info.get('schema')!r}")
    attached = str(info.get("path", ""))
    if (not attached or attached.startswith("/") or ".." in attached.split("/")
            or not index.replace("\\", "/").endswith(attached + SUFFIX)):
        problems.append("index must sit at <repo path of the attached file>" + SUFFIX)
    if not ASSET_RE.fullmatch(str(info.get("asset", ""))):
        problems.append(f"asset {info.get('asset')!r}")
    if not re.fullmatch(r"[0-9a-f]{64}", str(info.get("sha256", ""))):
        problems.append("sha256")
    if not re.fullmatch(r"[0-9a-f]{40}", str(info.get("git_blob", ""))):
        problems.append("git_blob")
    if not isinstance(info.get("bytes"), int) or info["bytes"] <= 0:
        problems.append("bytes")
    if not re.fullmatch(r"v\d+\.\d+\.\d+", str(info.get("release_tag", ""))):
        problems.append("release_tag")
    if problems:
        raise ValueError(f"{index}: invalid release-attachment index ({', '.join(problems)})")
    return info


def indexes(root: Path = ROOT) -> list[Path]:
    """Every provenance index of a checkout (tracked files when git is available)."""
    r = subprocess.run(["git", "ls-files", "-z", f"*{SUFFIX}"], cwd=root, capture_output=True)
    if r.returncode == 0:
        return sorted(root / p for p in r.stdout.decode().split("\0") if p)
    return sorted(p for p in root.rglob(f"*{SUFFIX}") if ".git" not in p.parts)


def cache_root() -> Path:
    env = os.environ.get("GPUWRF_TABLES_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    return (Path(xdg).expanduser() if xdg else Path.home() / ".cache") / "gpuwrf" / "tables"


def url(info: dict) -> str:
    template = os.environ.get("GPUWRF_ATTACHMENTS_URL", "").strip() or URL
    return template.format(tag=info["release_tag"], asset=info["asset"], sha256=info["sha256"])


def _digest(path: Path) -> tuple[int, str]:
    h, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(_CHUNK), b""):
            h.update(block)
            size += len(block)
    return size, h.hexdigest()


def _matches(path: Path, info: dict) -> bool:
    return path.is_file() and not path.is_symlink() and _digest(path) == (info["bytes"], info["sha256"])


def _fill(target: Path, info: dict, writer) -> str | None:
    """Write via ``writer(stream)`` into a temp file beside ``target``; keep it only with the pinned bytes."""
    created = not target.parent.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{info['asset']}.", suffix=".part", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            writer(out)
        got = _digest(Path(tmp))
        if got != (info["bytes"], info["sha256"]):
            return f"got {got[0]} B sha256 {got[1]}"
        os.chmod(tmp, 0o444)
        os.replace(tmp, target)  # atomic; concurrent stagers write identical bytes
        return None
    except Exception as exc:  # noqa: BLE001 - reported by the caller
        return f"{type(exc).__name__}: {exc}"
    finally:
        Path(tmp).unlink(missing_ok=True)
        if created and not target.exists():
            try:
                target.parent.rmdir()  # a failed source leaves no empty <sha256>/ directory behind
            except OSError:
                pass


def stage(index, root: Path = ROOT) -> Path:
    """Path of a verified copy of the attached file (see the module docstring for the sources)."""
    info = load(index)
    target = cache_root() / info["sha256"] / info["asset"]
    if _matches(target, info):
        return target
    tried = []
    if target.exists():
        tried.append(f"cache {target}: does not match the pin (left in place; delete it)")
    else:
        def from_git(out):
            with subprocess.Popen(["git", "cat-file", "blob", info["git_blob"]], cwd=root, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL) as proc:
                for block in iter(lambda: proc.stdout.read(_CHUNK), b""):
                    out.write(block)
            if proc.returncode:
                raise FileNotFoundError(f"blob {info['git_blob']} not in this checkout's object store")

        problem = _fill(target, info, from_git)
        if problem is None:
            return target
        tried.append(f"git object store of {root}: {problem}")
        if os.environ.get("GPUWRF_TABLES_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}:
            tried.append("download: GPUWRF_TABLES_OFFLINE is set")
        else:
            import urllib.request

            def from_url(out):
                with urllib.request.urlopen(url(info), timeout=60) as response:
                    for block in iter(lambda: response.read(_CHUNK), b""):
                        out.write(block)

            problem = _fill(target, info, from_url)
            if problem is None:
                return target
            tried.append(f"download {url(info)}: {problem}")
    raise AttachmentUnavailable(
        f"release attachment {info['path']} ({info['bytes'] / 1e6:.1f} MB, sha256 {info['sha256']}) is not available: "
        + "; ".join(tried) + f". Fix: download {url(info)} to {target}, or run in a clone with the published history "
        f"(public commits {', '.join(info.get('public_commits', [])) or 'see the index'}), or set GPUWRF_TABLES_DIR to "
        "a cache that holds it.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("stage", help="stage attachments into the cache and print their paths")
    st.add_argument("--all", action="store_true")
    st.add_argument("index", nargs="*", type=Path)
    sub.add_parser("list", help="list the provenance indexes of this checkout")
    a = ap.parse_args(argv)
    found = indexes()
    if a.cmd == "list":
        for index in found:
            info = load(index)
            print(f"{info['path']}  {info['bytes']} B  sha256 {info['sha256']}  {info['release_tag']}/{info['asset']}")
        return 0
    todo = found if a.all else a.index
    if not todo:
        ap.error("stage needs --all or index paths")
    rc = 0
    for index in todo:
        try:
            print(f"{load(index)['path']} -> {stage(index)}")
        except AttachmentUnavailable as exc:
            print(f"UNAVAILABLE: {exc}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
