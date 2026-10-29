"""Content binding and coverage guards; physical LANDUSE repro is separate."""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_restart_store import initialized

from gpuwrf.runtime.checkpoint_branch import branch_identity
from gpuwrf.runtime.runtime_data_identity import check_reader_coverage, runtime_data_identity


@pytest.fixture
def data_tree(tmp_path, monkeypatch):
    from gpuwrf import assets
    from gpuwrf.config import paths
    from gpuwrf.physics import rrtmg_lw
    root, package, store = (tmp_path / name for name in ("wrf", "package", "asset_store"))
    (root / "run").mkdir(parents=True)
    (root / "phys").mkdir()
    (package / "physics").mkdir(parents=True)
    store.mkdir()
    (package / "assets.py").write_text("# source bytes are bound by source_sha256\n")
    manifest = {"assets": {}}
    monkeypatch.setattr(paths, "wrf_root", lambda: root)
    monkeypatch.setattr(assets, "__file__", str(package / "assets.py"))
    monkeypatch.setattr(assets, "manifest", lambda: manifest)
    monkeypatch.setattr(assets, "asset_path", lambda name: store / Path(name).name)
    monkeypatch.setattr(assets, "ensure_asset", lambda *args: pytest.fail("identity attempted to download/read a loader"))
    monkeypatch.setattr(rrtmg_lw, "_extract_rrtmg_tables_module", lambda: SimpleNamespace(
        LW_SOURCE=root / "phys/module_ra_rrtmg_lw.F", SW_SOURCE=root / "phys/module_ra_rrtmg_sw.F",
        LW_DATA_CANDIDATES=(root / "run/RRTMG_LW_DATA",), SW_DATA_CANDIDATES=(root / "run/RRTMG_SW_DATA",)))
    bundles = {"d01": SimpleNamespace(namelist=SimpleNamespace(mp_physics=8))}
    return root, package, store, manifest, bundles


def fork(tmp_path, parent, current):
    generation = tmp_path / "generation"
    generation.mkdir()
    manifest = {"branch_schema_version": 1, "identity": {"output_dir": "parent", "runtime_data": parent},
                "domains": ["d01"], "own_steps": {"d01": 67}, "sha256": "payload"}
    (generation / "manifest.json").write_text(json.dumps(manifest))
    return branch_identity(manifest, {"output_dir": "child", "runtime_data": current}, {"schema_version": 1}, generation)


@pytest.mark.parametrize("relative", [
    "run/LANDUSE.TBL", "run/MPTABLE.TBL", "run/SOILPARM.TBL", "run/VEGPARM.TBL", "run/GENPARM.TBL",
    "run/URBPARM.TBL", "run/URBPARM_LCZ.TBL", "run/LAKEPARM.TBL", "run/RRTM_DATA_DBL",
    "run/RRTMG_LW_DATA", "run/ETAMPNEW_DATA", "run/thompson_lookup.dat", "run/ishmael-qi-qr.bin",
    "run/CAMtr_volume_mixing_ratio", "phys/module_ra_rrtm.F", "phys/module_ra_rrtmg_lw.F",
])
def test_same_path_same_size_restored_mtime_content_change_refuses_empty_fork(data_tree, tmp_path, relative):
    root, _, _, _, bundles = data_tree
    path = root / relative
    path.write_bytes(b"first-table")
    parent = runtime_data_identity(bundles)
    stat = path.stat()
    path.write_bytes(b"other-table")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    current = runtime_data_identity(bundles)
    assert parent != current
    assert parent["files"]["wrf/" + relative]["sha256"] == hashlib.sha256(b"first-table").hexdigest()
    with pytest.raises(ValueError, match="undeclared.*runtime_data"):
        fork(tmp_path, parent, current)


@pytest.mark.parametrize("name", ["lake/parameters.table", "surface.npz", "closure.json", "coefficients.dat",
                                  "gfdl/tr49t85", "gfdl/co2_host.f90", "_bundled/lake/extra.table"])
def test_all_package_data_kinds_are_content_bound(data_tree, tmp_path, name):
    _, package, _, _, bundles = data_tree
    path = package / name if name.startswith("_bundled/") else package / "physics" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"original")
    parent = runtime_data_identity(bundles)
    path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="runtime_data"):
        fork(tmp_path, parent, runtime_data_identity(bundles))


def test_active_assets_use_loader_scheme_map_and_content_not_manifest_pin(data_tree, tmp_path):
    _, package, store, manifest, bundles = data_tree
    name = "data/fixtures/thompson-tables-v1.npz"
    data = b"original"
    manifest["assets"][name] = {"kind": "remote", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    path = store / Path(name).name
    path.write_bytes(data)
    parent = runtime_data_identity(bundles)
    assert parent["files"]["asset/" + name]["sha256"] == manifest["assets"][name]["sha256"]
    path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="runtime_data"):
        fork(tmp_path, parent, runtime_data_identity(bundles))
    hail = "src/gpuwrf/physics/thompson_hail/collision_tables_0.npz"
    manifest["assets"][hail] = {"kind": "remote", "bytes": 8, "sha256": hashlib.sha256(data).hexdigest()}
    (store / Path(hail).name).write_bytes(data)
    assert "asset/" + hail not in runtime_data_identity(bundles)["files"]
    bundles["d01"].namelist.mp_physics = 38
    assert "asset/" + hail in runtime_data_identity(bundles)["files"]


def test_pinned_unmaterialized_asset_identity_stays_equal_after_verified_materialization(data_tree):
    _, _, store, manifest, bundles = data_tree
    name = "data/fixtures/thompson-tables-v1.npz"
    data = b"original"
    manifest["assets"][name] = {"kind": "remote", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    before = runtime_data_identity(bundles)
    (store / Path(name).name).write_bytes(data)
    assert runtime_data_identity(bundles) == before


def test_added_or_deleted_runtime_table_changes_identity(data_tree):
    root, _, _, _, bundles = data_tree
    empty = runtime_data_identity(bundles)
    path = root / "run/LAKEPARM.TBL"
    path.write_text("parameters")
    assert runtime_data_identity(bundles) != empty
    path.unlink()
    assert runtime_data_identity(bundles) == empty


def test_reader_registry_covers_actual_package_and_runtime_extractor():
    assert check_reader_coverage()["unbound"] == []


@pytest.mark.parametrize("site", [
    "integration/nested_pipeline.py:<module>.execute_nested_pipeline:call:d6c6632b6a4857586b02bd9b7527b345e275600a61ca930c5ff76b75c1d4aff5:1",
    "runtime/restart_store.py:<module>.HistoryJournal.__init__:call:48613a14dbfaf17e3a472ff21312acc767d31b12135caa82a79964b44c1538f6:1",
    "runtime/restart_store.py:<module>.require_resume_history:path:history-*/owner.json",
])
def test_resume_journal_readers_have_reviewed_binding_and_cannot_be_deleted(tmp_path, site):
    import shutil
    from gpuwrf.runtime import runtime_data_identity as binding
    source = Path(binding.__file__).resolve().parents[1]
    registry = source / "runtime/runtime_file_readers.json"
    recorded = json.loads(registry.read_text())
    assert site in recorded["reader_sites"]
    categories = [value for value in recorded["reader_categories"].values() if site in value["reader_sites"]]
    assert len(categories) == 1 and categories[0]["binding"] == "bindings.case_and_restart"
    # Audit the actual two changed modules, without masking unrelated global
    # registry debt; the global coverage test above remains unchanged and active.
    scoped = tmp_path / "source"
    for relative in ("integration/nested_pipeline.py", "runtime/restart_store.py"):
        target = scoped / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
    assert check_reader_coverage(scoped, registry)["unbound"] == []
    recorded["reader_sites"].remove(site)
    deleted = tmp_path / "registry-with-one-deleted-site.json"
    deleted.write_text(json.dumps(recorded))
    with pytest.raises(ValueError, match="lacks a reviewed content binding") as error:
        check_reader_coverage(scoped, deleted)
    assert site in str(error.value)


def test_every_remote_asset_has_a_loader_scheme_selection():
    from gpuwrf import assets
    remote = {name for name, item in assets.manifest()["assets"].items() if item["kind"] == "remote"}
    # New remote data cannot silently be omitted by the active-asset selector.
    assert set(assets.tables_for_schemes(range(256))) == remote


def test_symlinked_package_data_directory_is_followed_and_cycles_are_bounded(data_tree, tmp_path):
    _, package, _, _, bundles = data_tree
    external = tmp_path / "external_data"
    external.mkdir()
    table = external / "ice.table"
    table.write_bytes(b"original")
    (package / "physics/data").symlink_to(external, target_is_directory=True)
    (external / "cycle").symlink_to(package, target_is_directory=True)
    parent = runtime_data_identity(bundles)
    assert "package/physics/data/ice.table" in parent["files"]
    table.write_bytes(b"modified")
    with pytest.raises(ValueError, match="runtime_data"):
        fork(tmp_path, parent, runtime_data_identity(bundles))


def test_new_loader_read_of_unbound_file_fails_coverage(tmp_path):
    from gpuwrf.runtime.runtime_data_identity import _readers
    source = tmp_path / "source"
    source.mkdir()
    (source / "known.py").write_text("from pathlib import Path\ndef load():\n return Path('bound.dat').read_bytes()\n")
    registry = tmp_path / "readers.json"
    registry.write_text(json.dumps({"reader_sites": _readers(source)}))
    assert check_reader_coverage(source, registry)["unbound"] == []
    (source / "new_loader.py").write_text("from pathlib import Path\ndef load_extra():\n return Path('/outside/unbound.dat').read_bytes()\n")
    with pytest.raises(ValueError, match="lacks a reviewed content binding.*new_loader"):
        check_reader_coverage(source, registry)


def test_new_read_inside_existing_loader_is_not_hidden_by_module_whitelist(tmp_path):
    from gpuwrf.runtime.runtime_data_identity import _readers
    source = tmp_path / "source"
    source.mkdir()
    path = source / "known.py"
    path.write_text("from pathlib import Path\ndef load():\n return Path('bound.dat').read_bytes()\n")
    registry = tmp_path / "readers.json"
    registry.write_text(json.dumps({"reader_sites": _readers(source)}))
    path.write_text("from pathlib import Path\ndef load():\n return Path('unbound.dat').read_bytes()\n")
    with pytest.raises(ValueError, match="lacks a reviewed content binding"):
        check_reader_coverage(source, registry)


def test_native_identity_binds_runtime_data_and_refuses_table_changes(initialized, data_tree, tmp_path):
    from dataclasses import replace
    from datetime import datetime
    from gpuwrf.integration.nested_pipeline import NestedPipelineConfig
    from gpuwrf.runtime.restart_store import nested_identity
    root, _, _, _, _ = data_tree
    (root / "run/LANDUSE.TBL").write_bytes(b"first-table")
    _, nl, _ = initialized
    bundles = {"d01": SimpleNamespace(namelist=nl)}
    config = NestedPipelineConfig(tmp_path / "input", tmp_path / "parent", tmp_path / "proof", 3, 1)
    parent = nested_identity(config, bundles, datetime(2026, 2, 27))
    assert parent.get("runtime_data") == runtime_data_identity(bundles)
    (root / "run/LANDUSE.TBL").write_bytes(b"other-table")
    current = nested_identity(replace(config, output_dir=tmp_path / "child"), bundles, datetime(2026, 2, 27))
    generation = tmp_path / "native-generation"
    generation.mkdir()
    manifest = {"branch_schema_version": 1, "identity": parent, "domains": ["d01"],
                "own_steps": {"d01": 67}, "sha256": "payload"}
    (generation / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="undeclared.*runtime_data"):
        branch_identity(manifest, current, {"schema_version": 1}, generation)


@pytest.mark.parametrize("addition", [
    "\ndef extra_loader(path):\n return path.read_bytes()\n",
    "\ndef load(path):\n value = path.read_bytes()\n return path.read_bytes()\n",
])
def test_identical_reader_expression_cannot_hide_new_loader_or_extra_read(tmp_path, addition):
    from gpuwrf.runtime.runtime_data_identity import _readers
    source = tmp_path / "source"
    source.mkdir()
    path = source / "known.py"
    path.write_text("def load(path):\n return path.read_bytes()\n")
    registry = tmp_path / "readers.json"
    registry.write_text(json.dumps({"reader_sites": _readers(source)}))
    path.write_text(path.read_text() + addition)
    with pytest.raises(ValueError, match="lacks a reviewed content binding"):
        check_reader_coverage(source, registry)
