"""Pinned runtime tables (gpuwrf.assets) and wheel package-data completeness. CPU only, no network."""
from __future__ import annotations

import fnmatch
import hashlib
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import gpuwrf.assets as assets

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 10 * 1024 * 1024
# Tracked src/gpuwrf files that are documentation or optional build sources, never read at run time.
NOT_RUNTIME = {
    "src/gpuwrf/README.md", "src/gpuwrf/contracts/README.md", "src/gpuwrf/profiling/README.md",
    "src/gpuwrf/validation/README.md", "src/gpuwrf/physics/nssl2mom/PORTING.md",
    "src/gpuwrf/validation/operational_relaxed_bands.json",  # no reader in src (tests/scripts only)
}
NOT_RUNTIME_PREFIXES = ("src/gpuwrf/backends/kokkos/", "src/gpuwrf/backends/cuda_tile/")  # optional native builds


def _tracked_src_data():
    out = subprocess.run(["git", "ls-files", "-z", "src/gpuwrf"], cwd=ROOT, check=True, capture_output=True).stdout
    return sorted(p for p in out.decode().split("\0") if p and not p.endswith(".py"))


def _package_data():
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["setuptools"]["package-data"]


def _shipped(repo_path, package_data):
    for package, patterns in package_data.items():
        base = "src/" + package.replace(".", "/") + "/"
        if repo_path.startswith(base) and any(fnmatch.fnmatch(repo_path[len(base):], pat) for pat in patterns):
            return True
    return False


def test_manifest_matches_the_tracked_tables():
    rc = subprocess.run([sys.executable, "scripts/update_assets_manifest.py", "--check"], cwd=ROOT,
                        capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr


def test_every_src_data_file_reaches_a_non_editable_install():
    """Each tracked non-Python src file is wheel package-data, a pinned remote table, or not runtime."""
    package_data = _package_data()
    registered = assets.manifest()["assets"]
    unshipped = [p for p in _tracked_src_data() if not _shipped(p, package_data)
                 and registered.get(p, {}).get("kind") != "remote"
                 and p not in NOT_RUNTIME and not p.startswith(NOT_RUNTIME_PREFIXES)]
    assert not unshipped, f"add to [tool.setuptools.package-data] or the assets manifest: {unshipped}"


def test_package_data_never_ships_a_table_over_10_mib():
    package_data = _package_data()
    big = [p for p in _tracked_src_data() if _shipped(p, package_data) and (ROOT / p).stat().st_size > LIMIT]
    assert not big, big
    for repo_path, info in assets.manifest()["assets"].items():
        assert (info["kind"] == "remote") == (info["bytes"] > LIMIT), repo_path


def test_checkout_resolves_every_table_to_its_repo_copy():
    """Dev/editable installs keep today's paths (and therefore today's content keys)."""
    for repo_path, info in assets.manifest()["assets"].items():
        if info["kind"] == "remote" and not (ROOT / repo_path).exists():
            # curated public tree: the table is a release asset, resolved to the cache (no download here)
            assert assets.asset_path(repo_path).parent.name == info["sha256"]
            continue
        assert assets.asset_path(repo_path) == ROOT / repo_path
        assert assets.ensure_asset(repo_path) == ROOT / repo_path
    from gpuwrf.physics import rrtmg_tables, thompson_tables, wrf_cam_ozone

    assert rrtmg_tables.TABLE_ASSET == ROOT / "data/fixtures/rrtmg-tables-v1.npz"
    assert thompson_tables.COLD_TABLE_ASSET == ROOT / "data/fixtures/thompson-cold-collection-v1.npz"
    assert wrf_cam_ozone.OZONE_ASSET == ROOT / "data/fixtures/wrf-cam-ozone-v1.npz"
    assert thompson_tables.COLD_COLLECTION_TABLES is not None


@pytest.fixture
def wheel_like(tmp_path, monkeypatch):
    """A synthetic 'installed package' without repo copies: one bundle + one remote table, served by file://."""
    payload = b"pinned-table-bytes" * 1000
    served = tmp_path / "release"
    served.mkdir()
    (served / "tiny-v1.npz").write_bytes(payload)
    manifest = dict(schema="gpuwrf-runtime-assets-v1", url_template="unused", max_bundle_bytes=LIMIT, assets={
        "data/fixtures/tiny-v1.npz": dict(kind="remote", asset="tiny-v1.npz", bytes=len(payload),
                                          sha256=hashlib.sha256(payload).hexdigest()),
        "scripts/tiny.py": dict(kind="bundle", asset="tiny.py", bytes=1, sha256="0" * 64)})
    monkeypatch.setattr(assets, "manifest", lambda: manifest)
    monkeypatch.setattr(assets, "REPO_ROOT", tmp_path / "site-packages-parent")
    monkeypatch.setattr(assets, "BUNDLED_DIR", tmp_path / "pkg" / "_bundled")
    assets._registered_locations.cache_clear()
    monkeypatch.setenv("GPUWRF_TABLES_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("GPUWRF_TABLES_URL", served.as_uri() + "/{asset}")
    monkeypatch.delenv("GPUWRF_TABLES_OFFLINE", raising=False)
    yield dict(payload=payload, served=served, cache=tmp_path / "cache", sha=hashlib.sha256(payload).hexdigest())
    assets._registered_locations.cache_clear()


def test_remote_table_is_fetched_once_verified_and_content_addressed(wheel_like):
    target = assets.asset_path("data/fixtures/tiny-v1.npz")
    assert target == wheel_like["cache"] / wheel_like["sha"] / "tiny-v1.npz" and not target.exists()
    assert assets.ensure_asset(target) == target and target.read_bytes() == wheel_like["payload"]
    (wheel_like["served"] / "tiny-v1.npz").unlink()  # second use must not touch the network
    assert assets.ensure_asset("data/fixtures/tiny-v1.npz") == target
    assert not list(target.parent.glob("*.part"))


def test_wrong_bytes_fail_closed_and_leave_no_file(wheel_like):
    (wheel_like["served"] / "tiny-v1.npz").write_bytes(b"tampered")
    with pytest.raises(assets.AssetUnavailableError, match="expected"):
        assets.ensure_asset("data/fixtures/tiny-v1.npz")
    assert not any(p.is_file() for p in wheel_like["cache"].rglob("*"))


def test_offline_fails_closed_with_instructions(wheel_like, monkeypatch):
    monkeypatch.setenv("GPUWRF_TABLES_OFFLINE", "1")
    with pytest.raises(FileNotFoundError) as info:  # AssetUnavailableError is a FileNotFoundError
        assets.ensure_asset("data/fixtures/tiny-v1.npz")
    message = str(info.value)
    for needle in (wheel_like["sha"], "GPUWRF_TABLES_DIR", "python -m gpuwrf.assets fetch --all", "tiny-v1.npz"):
        assert needle in message
    with pytest.raises(assets.AssetUnavailableError, match="missing from this installation"):
        assets.ensure_asset("scripts/tiny.py")  # bundle files never download


def test_aot_key_hashes_table_content_before_and_after_download(wheel_like, monkeypatch, tmp_path):
    """Never by path: an undownloaded table keys its pin == the downloaded content, for any cache root."""
    from gpuwrf.runtime.aot_cheap_key import canonical_digest

    before = canonical_digest((assets.asset_path("data/fixtures/tiny-v1.npz"),))
    monkeypatch.setenv("GPUWRF_TABLES_DIR", str(tmp_path / "elsewhere"))
    elsewhere = canonical_digest((assets.asset_path("data/fixtures/tiny-v1.npz"),))
    fetched = assets.ensure_asset("data/fixtures/tiny-v1.npz")
    after = canonical_digest((fetched,))
    plain_copy = tmp_path / "copy.npz"
    plain_copy.write_bytes(wheel_like["payload"])
    assert before == elsewhere == after == canonical_digest((plain_copy,))
    assert canonical_digest((tmp_path / "not-registered.npz",)) != before  # positive control


def test_table_env_does_not_split_any_key():
    from gpuwrf.runtime import aot_cheap_key as ck

    for name in ("GPUWRF_TABLES_DIR", "GPUWRF_TABLES_URL", "GPUWRF_TABLES_OFFLINE"):
        assert ck.is_process_infra_env(name) and ck.trace_env_is_inert(name), name


def test_assets_cli_lists_every_table():
    rc = subprocess.run([sys.executable, "-m", "gpuwrf.assets", "list"], cwd=ROOT, capture_output=True, text=True,
                        env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
    assert rc.returncode == 0, rc.stderr
    assert sum(" repo " in line for line in rc.stdout.splitlines()) == len(assets.manifest()["assets"])
    assert json.loads((ROOT / "src/gpuwrf/assets_manifest.json").read_text())["url_template"].startswith(
        "https://github.com/wrf-gpu/wrf_gpu/releases/download/v{version}/")


def test_onboarding_container_builds_the_wheel_with_every_bundle_file():
    dockerfile = (ROOT / "containers/onboarding/Dockerfile").read_text()
    allow = (ROOT / "containers/onboarding/Dockerfile.dockerignore").read_text().split()
    assert "setup.py" in dockerfile and "!setup.py" in allow and "GPUWRF_TABLES_DIR=/work/cache/tables" in dockerfile
    for repo_path, info in assets.manifest()["assets"].items():
        if info["kind"] == "bundle":
            assert repo_path in dockerfile and "!" + repo_path in allow, repo_path


def test_manifest_check_passes_in_a_tree_without_release_tables(tmp_path):
    """The curated public tree lacks the remote src tables; their committed pins must survive --check."""
    tree = tmp_path / "pub"
    files = subprocess.run(["git", "ls-files", "-z", "--", "scripts/update_assets_manifest.py", "src/gpuwrf/assets_manifest.json",
                            *[p for p in assets.manifest()["assets"]]], cwd=ROOT, check=True, capture_output=True).stdout
    for rel in (p for p in files.decode().split("\0") if p):
        if assets.manifest()["assets"].get(rel, {}).get("kind") == "remote" and rel.startswith("src/"):
            continue  # curated out of the public tree
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith((".py", ".json")):
            (tree / rel).write_bytes((ROOT / rel).read_bytes())  # real copy: the script resolves its own root
        else:
            (tree / rel).symlink_to(ROOT / rel)
    subprocess.run(["git", "init", "-q", str(tree)], check=True)
    subprocess.run(["git", "add", "-A"], cwd=tree, check=True)
    rc = subprocess.run([sys.executable, str(tree / "scripts/update_assets_manifest.py"), "--check"], cwd=tree,
                        capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr


def test_cache_paths_map_to_tables_under_any_cache_root(tmp_path):
    info = next(i for i in assets.manifest()["assets"].values() if i["kind"] == "remote")
    for root in (tmp_path / "a", tmp_path / "b"):
        assert assets.pinned_sha256(root / info["sha256"] / info["asset"]) == info["sha256"]
    assert assets.pinned_sha256(tmp_path / ("0" * 64) / info["asset"]) is None


def test_baked_table_hash_is_the_same_before_and_after_a_table_download(monkeypatch, tmp_path):
    """A module-level table Path that is not downloaded yet must give the same table key as the present file."""
    import copy

    from gpuwrf.physics import rrtmg_tables
    from gpuwrf.runtime.aot_baked_tables import baked_table_hash

    def fn(x):
        return x
    fn.__module__ = "gpuwrf.physics.rrtmg_sw"  # selects the RRTMG table modules
    present = baked_table_hash(fn, None)
    patched = copy.deepcopy(assets.manifest())
    info = patched["assets"]["data/fixtures/rrtmg-tables-v1.npz"]
    info["kind"] = "remote"  # pretend: a wheel install that has not fetched it yet
    monkeypatch.setattr(assets, "manifest", lambda: patched)
    assets._registered_locations.cache_clear()
    monkeypatch.setattr(rrtmg_tables, "TABLE_ASSET", tmp_path / info["sha256"] / info["asset"])
    assert not rrtmg_tables.TABLE_ASSET.exists()
    assert baked_table_hash(fn, None) == present
    assets._registered_locations.cache_clear()


def test_missing_cold_collection_table_fails_closed_instead_of_switching_physics_off(wheel_like, monkeypatch):
    """Cold collection is default ON: an absent table must stop the run on use, never switch the process off."""
    import urllib.request

    from gpuwrf.physics import thompson_column, thompson_tables

    tiny = assets.asset_path("data/fixtures/tiny-v1.npz")
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("presence check downloaded"))
    assert thompson_tables.cold_tables_available(tiny) is False  # presence only: test collection never downloads
    monkeypatch.undo()
    monkeypatch.setenv("GPUWRF_TABLES_DIR", str(wheel_like["cache"]))
    monkeypatch.setenv("GPUWRF_TABLES_URL", wheel_like["served"].as_uri() + "/{asset}")
    monkeypatch.setattr(assets, "manifest", lambda: {"assets": {"data/fixtures/tiny-v1.npz": dict(
        kind="remote", asset="tiny-v1.npz", bytes=len(wheel_like["payload"]), sha256=wheel_like["sha"])}})
    assets._registered_locations.cache_clear()
    monkeypatch.setenv("GPUWRF_TABLES_OFFLINE", "1")
    standin = assets.load_or_lazy("COLD_COLLECTION_TABLES", (tiny,), lambda: assets.ensure_asset(tiny))
    monkeypatch.setattr(thompson_column, "COLD_COLLECTION_TABLES", standin)
    assert thompson_column._cold_collection_enabled()  # not switched off by the missing file
    with pytest.raises(assets.AssetUnavailableError):
        standin.resolve()  # first use offline: fail closed


def test_compile_cache_lock_dependency_is_declared():
    """runtime/compile_cache.py fails open to an UNLOCKED shared cache without filelock (jax does not install it)."""
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    assert any(d.split(">")[0].split("=")[0].strip().lower() == "filelock" for d in deps)


_NO_NETWORK_IMPORT = r'''
import importlib, json, os, pkgutil, sys, urllib.request
import gpuwrf.assets as assets
calls = []
def refuse(*args, **kwargs):
    calls.append(str(args[0])[:120])
    raise OSError("network refused by test")
urllib.request.urlopen = refuse
real = assets._candidates
def wheel_like(repo_path):  # remote tables: no checkout copy, empty cache
    found = real(repo_path)
    return found[1:] if assets.entry(repo_path)["kind"] == "remote" else found
assets._candidates = wheel_like
import gpuwrf.physics
errors = {}
for info in pkgutil.walk_packages(gpuwrf.physics.__path__, "gpuwrf.physics."):
    try:
        importlib.import_module(info.name)
    except Exception as exc:  # optional extras are allowed to be absent; network is not
        errors[info.name] = type(exc).__name__ + ": " + str(exc)[:160]
import gpuwrf.integration.nested_pipeline
from gpuwrf.physics import thompson_tables, thompson_aero_tables
print(json.dumps(dict(calls=calls, errors=errors, lazy=[type(v).__name__ for v in (
    thompson_tables.THOMPSON_TABLES, thompson_tables.COLD_COLLECTION_TABLES, thompson_aero_tables.THOMPSON_AERO_TABLES)])))
'''


def test_importing_every_physics_module_never_downloads(tmp_path):
    """Fresh install, no table installed: imports bind stand-ins and make zero network calls."""
    env = {**__import__("os").environ, "PYTHONPATH": str(ROOT / "src"), "JAX_PLATFORMS": "cpu",
           "GPUWRF_JAX_CACHE": "0", "GPUWRF_TABLES_DIR": str(tmp_path / "empty-cache")}
    env.pop("GPUWRF_TABLES_OFFLINE", None)
    rc = subprocess.run([sys.executable, "-c", _NO_NETWORK_IMPORT], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr[-2000:]
    report = json.loads(rc.stdout.strip().splitlines()[-1])
    assert report["calls"] == [], report["calls"]
    assert report["lazy"] == ["LazyTable"] * 3
    assert not any("AssetUnavailable" in e or "network" in e for e in report["errors"].values()), report["errors"]


def _synthetic_bundle():
    import jax.numpy as jnp
    import numpy as np

    from gpuwrf.physics.thompson_tables import ThompsonTableBundle
    rng = np.random.default_rng(7)
    return ThompsonTableBundle(*(jnp.asarray(rng.random(4), dtype=jnp.float64) for _ in ThompsonTableBundle._fields))


def test_lazy_table_first_use_inside_a_real32_trace_matches_the_eager_bundle():
    import jax
    import jax.numpy as jnp
    import numpy as np

    from gpuwrf.physics import thompson_tables  # noqa: F401  (x64 configured as at import)
    eager = _synthetic_bundle()
    lazy = assets.LazyTable("SYNTH", _synthetic_bundle)

    @jax.jit
    def step(x):
        with jax.enable_x64(False):  # REAL32 scope: must not truncate the float64 table
            return x + lazy.t_Efrw[0].astype(x.dtype)

    step(jnp.float32(1.0))
    loaded = lazy.resolve()
    assert type(loaded.t_Efrw).__name__ == "ArrayImpl" and loaded.t_Efrw.dtype == np.float64  # concrete, not a tracer
    for a, b in zip(loaded, eager):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert list(lazy)[0] is loaded.t_Efrw and lazy._replace(cse=lazy.cse).t_Efrw is loaded.t_Efrw


def test_lazy_table_keys_like_the_eager_bundle(monkeypatch):
    from gpuwrf.physics import thompson_tables
    from gpuwrf.runtime.aot_baked_tables import baked_table_hash
    from gpuwrf.runtime.aot_cheap_key import canonical_digest

    real = thompson_tables.THOMPSON_TABLES
    lazy = assets.LazyTable("THOMPSON_TABLES", lambda: real)
    assert canonical_digest((lazy,)) == canonical_digest((real,))

    class Namelist:  # minimal mp=8 namelist for the selector
        pass
    from gpuwrf.runtime import aot_baked_tables
    monkeypatch.setattr(aot_baked_tables, "_namelists", lambda fn, nl: [nl])
    nl = Namelist()
    for name, value in dict(mp_physics=8, ra_sw_physics=0, ra_lw_physics=0, sf_sfclay_physics=0, bl_pbl_physics=0,
                            cu_physics=0, noahmp_options=None).items():
        setattr(nl, name, value)
    eager_key = baked_table_hash(lambda x: x, nl)
    monkeypatch.setattr(thompson_tables, "THOMPSON_TABLES", lazy)
    assert baked_table_hash(lambda x: x, nl) == eager_key
    monkeypatch.setattr(thompson_tables, "THOMPSON_TABLES", real._replace(snow_sa=real.snow_sa + 1.0))
    assert baked_table_hash(lambda x: x, nl) != eager_key  # positive control: content is keyed


def test_load_or_lazy_loads_now_only_when_every_file_is_installed(wheel_like):
    calls = []
    present = assets.load_or_lazy("X", (), lambda: calls.append(1) or "loaded")
    assert present == "loaded" and calls == [1]
    lazy = assets.load_or_lazy("Y", (assets.asset_path("data/fixtures/tiny-v1.npz"),), lambda: calls.append(2) or ("v",))
    assert isinstance(lazy, assets.LazyTable) and calls == [1]  # nothing loaded (or downloaded) yet
    assert getattr(lazy, "__test__", False) is False and not hasattr(lazy, "__wrapped__")  # pytest/inspect probes
    assert calls == [1]
    assert lazy[0] == "v" and calls == [1, 2]


def test_prefilled_cache_files_are_checked_against_their_pin_once(wheel_like, monkeypatch):
    target = assets.asset_path("data/fixtures/tiny-v1.npz")
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x" * len(wheel_like["payload"]))  # right size, wrong bytes (hand-copied cache)
    monkeypatch.setenv("GPUWRF_TABLES_OFFLINE", "1")
    with pytest.raises(assets.AssetUnavailableError, match="does not match its pin"):
        assets.ensure_asset("data/fixtures/tiny-v1.npz")
    target.write_bytes(wheel_like["payload"])
    hashed = []
    real_digest = assets._digest
    monkeypatch.setattr(assets, "_digest", lambda p: hashed.append(p) or real_digest(p))
    assert assets.ensure_asset(target) == target and assets.ensure_asset("data/fixtures/tiny-v1.npz") == target
    assert len(hashed) == 1  # verified once per process, not per use


def test_run_prefetches_exactly_the_tables_of_the_selected_schemes():
    thompson = {"data/fixtures/thompson-tables-v1.npz", "data/fixtures/thompson-cold-collection-v1.npz",
                "data/fixtures/thompson-mixed-phase-v1.npz"}
    assert set(assets.tables_for_schemes([8])) == thompson
    assert set(assets.tables_for_schemes([28])) == thompson | {"data/fixtures/thompson-aero-tables-v1.npz"}
    hail = set(assets.tables_for_schemes([38])) - thompson - {"data/fixtures/thompson-aero-tables-v1.npz"}
    assert len(hail) == 15 and all(p.startswith("src/gpuwrf/physics/thompson_hail/") for p in hail)
    assert assets.tables_for_schemes([55]) == ["src/gpuwrf/physics/jensen_ishmael/data/lookup_tables.npz"]
    assert assets.tables_for_schemes([6, 10]) == []
    remote = {p for p, i in assets.manifest()["assets"].items() if i["kind"] == "remote"}
    assert set(assets.tables_for_schemes([8, 28, 38, 55])) == remote  # every remote table has a scheme
    cli = (ROOT / "src/gpuwrf/cli.py").read_text()
    assert cli.index("prefetch_for_schemes(") < cli.index("from gpuwrf.integration.nested_pipeline import (")
