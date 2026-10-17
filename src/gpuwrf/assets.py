"""Runtime table files that do not live inside the installed ``gpuwrf`` package.

``assets_manifest.json`` (next to this file) declares two kinds, each pinned by SHA-256:

* ``bundle`` -- small (<= 10 MB) runtime files kept outside ``src/gpuwrf`` in the
  repository (``data/fixtures`` tables, the RRTMG source parser). A wheel build copies
  them into ``gpuwrf/_bundled/<repo path>`` (``setup.py``); a checkout reads the repo copy.
* ``remote`` -- tables larger than 10 MB. They are not in the wheel, and the new ones are
  not in the curated public git tree either. A checkout that has the file reads it in
  place; otherwise it is downloaded once from the GitHub release assets of the matching
  version, checked against the pinned SHA-256 and stored content-addressed in the user
  cache ``<GPUWRF_TABLES_DIR or ~/.cache/gpuwrf/tables>/<sha256>/<asset>``. When it
  cannot be downloaded the run stops with :class:`AssetUnavailableError` (fail-closed:
  a missing table never switches physics off).

Environment: ``GPUWRF_TABLES_DIR`` (cache root), ``GPUWRF_TABLES_URL`` (URL template
with ``{version}``, ``{asset}``, ``{sha256}``; for mirrors) and ``GPUWRF_TABLES_OFFLINE=1``
(never download). None of them changes a traced program: the AOT key hashes table
CONTENT, and a registered table that is not downloaded yet is keyed by its pinned SHA
(:func:`pinned_sha256`), which equals the content digest once it is present.

Import never downloads: a module-level table bundle whose file is not installed yet is bound
to a :class:`LazyTable` stand-in that downloads and loads on first use. ``gpuwrf run`` fetches
the tables of the selected schemes before the model is imported (:func:`prefetch_for_schemes`),
so a run loads them exactly as a checkout does. A remote table found in the user cache is
checked against its pin once per process before use (prefilled/copied caches included).

Prefetch for offline hosts or containers: ``python -m gpuwrf.assets fetch --all``.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import threading
import time
from functools import lru_cache
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
# src/gpuwrf/assets.py -> repository root of a checkout; inside a wheel install this is
# just a site-packages ancestor where no repo copy exists.
REPO_ROOT = PACKAGE_DIR.parents[1]
BUNDLED_DIR = PACKAGE_DIR / "_bundled"
MANIFEST_PATH = PACKAGE_DIR / "assets_manifest.json"
DEFAULT_URL = "https://github.com/wrf-gpu/wrf_gpu/releases/download/v{version}/{asset}"
_CHUNK = 1 << 20


class AssetUnavailableError(FileNotFoundError):
    """A pinned runtime table is neither installed nor downloadable (fail-closed)."""


@lru_cache(maxsize=1)
def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def entry(repo_path: str) -> dict:
    try:
        return manifest()["assets"][repo_path]
    except KeyError:
        raise KeyError(f"{repo_path!r} is not a registered gpuwrf runtime asset ({MANIFEST_PATH})") from None


def cache_root() -> Path:
    env = os.environ.get("GPUWRF_TABLES_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    return (Path(xdg).expanduser() if xdg else Path.home() / ".cache") / "gpuwrf" / "tables"


def _cache_path(info: dict) -> Path:
    return cache_root() / info["sha256"] / info["asset"]


def _candidates(repo_path: str) -> list[Path]:
    info = entry(repo_path)
    found = [REPO_ROOT / repo_path]
    if info["kind"] == "bundle":
        found.append(BUNDLED_DIR / repo_path)
    else:
        found.append(_cache_path(info))
    return found


def asset_path(repo_path: str) -> Path:
    """Where ``repo_path`` is (repo copy, wheel bundle or user cache), never downloading.

    A remote table that is not downloaded yet resolves to its cache location, which
    :func:`ensure_asset` fills. A checkout therefore keeps today's repo path.
    """
    candidates = _candidates(repo_path)
    for path in candidates:
        if path.is_file():
            return path
    return candidates[-1] if entry(repo_path)["kind"] == "remote" else candidates[0]


@lru_cache(maxsize=None)
def _registered_locations() -> dict[str, tuple[str, dict]]:
    out = {}
    for repo_path, info in manifest()["assets"].items():
        for path in (REPO_ROOT / repo_path, BUNDLED_DIR / repo_path):
            out[str(path)] = (repo_path, info)
    return out


def _lookup(path: Path) -> tuple[str, dict] | None:
    key = str(path)
    hit = _registered_locations().get(key)
    if hit is not None:
        return hit
    # Cache locations: <any cache root>/<sha256>/<asset>, so a GPUWRF_TABLES_DIR change after import
    # still maps a module constant to its table.
    for repo_path, info in manifest()["assets"].items():
        if info["kind"] == "remote" and path.name == info["asset"] and path.parent.name == info["sha256"]:
            return repo_path, info
    return None


def pinned_sha256(path: Path) -> str | None:
    """Pinned SHA-256 of a registered asset location (repo, bundle or cache), else ``None``.

    The AOT keys use it for a registered table that is not present yet, so a run keys
    the same program before and after the first download (content, never location).
    """
    hit = _lookup(Path(path))
    return hit[1]["sha256"] if hit else None


def _version() -> str:
    from gpuwrf import __version__

    return __version__


def asset_url(info: dict) -> str:
    template = os.environ.get("GPUWRF_TABLES_URL", "").strip() or manifest().get("url_template", DEFAULT_URL)
    return template.format(version=_version(), asset=info["asset"], sha256=info["sha256"])


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _unavailable(repo_path: str, info: dict, reason: str) -> AssetUnavailableError:
    target = _cache_path(info)
    return AssetUnavailableError(
        f"gpuwrf runtime table {info['asset']} ({info['bytes'] / 1e6:.1f} MB, sha256 {info['sha256']}) is not "
        f"installed: {reason}. It is larger than 10 MB, so it is not shipped in the wheel or in the public git tree. "
        f"Fix one of: (1) rerun with network access (downloaded once from {asset_url(info)}); (2) on a connected host "
        f"run `python -m gpuwrf.assets fetch --all`, copy {cache_root()} here and set GPUWRF_TABLES_DIR to it; "
        f"(3) download that URL to {target}. Repository path: {repo_path}."
    )


def _download(repo_path: str, info: dict) -> Path:
    import urllib.request

    target = _cache_path(info)
    if os.environ.get("GPUWRF_TABLES_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}:
        raise _unavailable(repo_path, info, "GPUWRF_TABLES_OFFLINE is set")
    url = asset_url(info)
    target.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for attempt in range(3):
        fd, tmp = tempfile.mkstemp(prefix=f".{info['asset']}.", suffix=".part", dir=target.parent)
        try:
            h, size = hashlib.sha256(), 0
            with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=60) as response:
                for block in iter(lambda: response.read(_CHUNK), b""):
                    out.write(block)
                    h.update(block)
                    size += len(block)
            if size != info["bytes"] or h.hexdigest() != info["sha256"]:
                raise ValueError(f"downloaded {size} B sha256 {h.hexdigest()}, expected {info['bytes']} B {info['sha256']}")
            os.replace(tmp, target)  # atomic; concurrent fetchers write identical bytes
            return target
        except Exception as exc:  # noqa: BLE001 - reported in the fail-closed error below
            last = exc
            Path(tmp).unlink(missing_ok=True)
            if isinstance(exc, ValueError):
                break  # wrong bytes: retrying the same URL does not help
            time.sleep(1.0 + attempt)
    raise _unavailable(repo_path, info, f"download from {url} failed ({type(last).__name__}: {last})")


_VERIFIED: dict[str, tuple[int, int]] = {}  # remote-cache file -> (size, mtime_ns) checked in this process


def _verified(path: Path, repo_path: str, info: dict) -> Path:
    """A remote table from the user cache must match its pin (checked once per process, not per step).

    Checkout copies stay content-keyed (development edits are hashed by the AOT key) and wheel
    bundles were checked by setup.py; cache files may be prefilled or copied by hand.
    """
    if path in (REPO_ROOT / repo_path, BUNDLED_DIR / repo_path):
        return path
    st = path.stat()
    stamp = (st.st_size, st.st_mtime_ns)
    if _VERIFIED.get(str(path)) != stamp:
        if st.st_size != info["bytes"] or _digest(path) != info["sha256"]:
            raise AssetUnavailableError(
                f"gpuwrf runtime table {path} does not match its pin (expected {info['bytes']} B, sha256 "
                f"{info['sha256']}). Delete it and rerun (or `python -m gpuwrf.assets fetch {info['asset']}`).")
        _VERIFIED[str(path)] = stamp
    return path


def ensure_asset(path) -> Path:
    """Return a readable file for ``path``; fetch a registered remote table on first use.

    ``path`` is a repo-relative asset name or a location returned by :func:`asset_path`.
    Unregistered paths (explicit user files) are returned unchanged.
    """
    if isinstance(path, str) and path in manifest()["assets"]:
        repo_path, info = path, entry(path)
        path = asset_path(path)
    else:
        path = Path(path)
        hit = _lookup(path)
        if hit is None:
            return path
        repo_path, info = hit
        if not path.is_file():
            path = asset_path(repo_path)  # e.g. captured under another cache root
    if path.is_file():
        return _verified(path, repo_path, info)
    if info["kind"] != "remote":
        raise AssetUnavailableError(f"gpuwrf runtime file {repo_path} is missing from this installation "
                                    f"(expected {path}); reinstall the package")
    target = _download(repo_path, info)
    st = target.stat()
    _VERIFIED[str(target)] = (st.st_size, st.st_mtime_ns)
    return target


def is_present(path) -> bool:
    """True when ``path`` (asset name or location) is readable now -- never downloads."""
    if isinstance(path, str) and path in manifest()["assets"]:
        return asset_path(path).is_file()
    path = Path(path)
    if path.is_file():
        return True
    hit = _lookup(path)
    return hit is not None and asset_path(hit[0]).is_file()


_LAZY_PYTREE_REGISTERED = False


class LazyTable:
    """Stand-in for a module-level table bundle whose pinned file is not installed yet.

    Bound at import instead of loading (import never downloads). The first attribute, item or
    iteration access -- or an AOT key walk -- downloads the table if needed and runs ``loader``
    exactly as the eager import would: concrete arrays even when first touched inside a jit trace
    (``jax.ensure_compile_time_eval``) and with the x64 setting captured at module import, so a
    REAL32 scope cannot truncate float64 tables (E75/E206). Identity checks (``is``) see one
    object; as a pytree it is its loaded value.
    """

    __slots__ = ("_name", "_loader", "_x64", "_value", "_lock")

    def __init__(self, name: str, loader):
        import jax

        global _LAZY_PYTREE_REGISTERED
        if not _LAZY_PYTREE_REGISTERED:
            jax.tree_util.register_pytree_node(LazyTable, lambda lazy: ((lazy.resolve(),), None),
                                               lambda _aux, children: children[0])
            _LAZY_PYTREE_REGISTERED = True
        self._name, self._loader = name, loader
        self._x64 = bool(jax.config.jax_enable_x64)
        self._value, self._lock = None, threading.Lock()

    def resolve(self):
        if self._value is None:
            with self._lock:
                if self._value is None:
                    import jax

                    with jax.ensure_compile_time_eval(), jax.enable_x64(self._x64):
                        self._value = self._loader()
        return self._value

    def __getattr__(self, name):
        # Dunder probes (pytest's __test__, copy/inspect hooks) must never load (or download) the table.
        if name in LazyTable.__slots__ or (name.startswith("__") and name.endswith("__")):
            raise AttributeError(name)
        return getattr(self.resolve(), name)

    def __reduce__(self):
        return _loaded_value, (self.resolve(),)

    def __iter__(self):
        return iter(self.resolve())

    def __len__(self):
        return len(self.resolve())

    def __getitem__(self, index):
        return self.resolve()[index]

    def __repr__(self):
        return f"<gpuwrf table {self._name}: {'loaded' if self._value is not None else 'not loaded yet'}>"


def _loaded_value(value):
    return value


def load_or_lazy(name: str, paths, loader):
    """``loader()`` now when every table file is readable, else a :class:`LazyTable` (no download at import)."""
    return loader() if all(is_present(p) for p in paths) else LazyTable(name, loader)


# Remote tables each microphysics scheme reads (mp_physics -> manifest repo paths).
_THOMPSON = ("data/fixtures/thompson-tables-v1.npz", "data/fixtures/thompson-cold-collection-v1.npz",
             "data/fixtures/thompson-mixed-phase-v1.npz")
_SCHEME_PREFIXES = {
    8: _THOMPSON,
    28: _THOMPSON + ("data/fixtures/thompson-aero-tables-v1.npz",),
    38: _THOMPSON + ("data/fixtures/thompson-aero-tables-v1.npz", "src/gpuwrf/physics/thompson_hail/"),
    55: ("src/gpuwrf/physics/jensen_ishmael/",),
}


def tables_for_schemes(mp_physics) -> list[str]:
    """Remote tables (repo paths) the given mp_physics options read."""
    remote = [p for p, i in manifest()["assets"].items() if i["kind"] == "remote"]
    wanted = set()
    for mp in mp_physics:
        for prefix in _SCHEME_PREFIXES.get(int(mp), ()):
            wanted.update(p for p in remote if p == prefix or (prefix.endswith("/") and p.startswith(prefix)))
    return sorted(wanted)


def prefetch_for_schemes(mp_physics) -> list[Path]:
    """Download (if needed) and pin-check the remote tables of these schemes, before the model is imported."""
    return [ensure_asset(p) for p in tables_for_schemes(mp_physics)]


def status() -> list[dict]:
    rows = []
    for repo_path, info in manifest()["assets"].items():
        path = asset_path(repo_path)
        where = "missing"
        if path.is_file():
            where = ("repo" if path == REPO_ROOT / repo_path else
                     "bundled" if path == BUNDLED_DIR / repo_path else "cache")
        rows.append(dict(repo_path=repo_path, kind=info["kind"], bytes=info["bytes"], sha256=info["sha256"],
                         where=where, path=str(path)))
    return rows


def _main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m gpuwrf.assets", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="show every pinned runtime table and where it resolves")
    fetch = sub.add_parser("fetch", help="download missing remote tables into the cache")
    fetch.add_argument("names", nargs="*", help="repo paths or asset names (default with --all: every remote table)")
    fetch.add_argument("--all", action="store_true")
    verify = sub.add_parser("verify", help="check the SHA-256 of every present table")
    verify.add_argument("--strict", action="store_true", help="also fail on missing remote tables")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        for row in status():
            print(f"{row['kind']:6} {row['where']:7} {row['bytes'] / 1e6:8.1f} MB  {row['repo_path']}  -> {row['path']}")
        print(f"cache root: {cache_root()}")
        return 0
    if a.cmd == "fetch":
        assets = manifest()["assets"]
        by_asset = {info["asset"]: p for p, info in assets.items()}
        names = [by_asset.get(n, n) for n in a.names] or (
            [p for p, info in assets.items() if info["kind"] == "remote"] if a.all else [])
        if not names:
            ap.error("name tables to fetch or pass --all")
        for repo_path in names:
            print(ensure_asset(repo_path))
        return 0
    bad = 0
    for row in status():
        if row["where"] == "missing":
            bad += bool(a.strict or row["kind"] == "bundle")
            print(f"MISSING {row['repo_path']}")
        elif _digest(Path(row["path"])) != row["sha256"]:
            bad += 1
            print(f"SHA MISMATCH {row['repo_path']} ({row['path']})")
        else:
            print(f"ok {row['repo_path']}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(_main())
