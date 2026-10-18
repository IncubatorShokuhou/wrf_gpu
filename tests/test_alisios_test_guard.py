"""Suite entry-point refusal, using only private files and a fake admitting guard."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
from types import SimpleNamespace

import netCDF4
import pytest

from gpuwrf.assimilation.letkf import alisios_guard as ag


@pytest.fixture
def protected(tmp_path, monkeypatch, request):
    # Locate the already installed root conftest, without installing a second hook.
    root_conftest = Path(__file__).with_name("conftest.py").resolve()
    suite = next(m for m in request.config.pluginmanager.get_plugins()
                 if getattr(m, "__file__", None) == str(root_conftest))
    root = tmp_path / "protected"
    root.mkdir()
    text = root / "payload.txt"
    text.write_text("private fixture")
    nc = root / "payload.nc"
    with netCDF4.Dataset(nc, "w") as ds:
        ds.createDimension("x", 1)
        ds.createVariable("x", "f4", ("x",))[:] = [7]
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    roots = (root.resolve(),)
    monkeypatch.setattr(suite, "_ALISIOS_ROOTS", roots)
    monkeypatch.setattr(ag, "DATA_ROOTS", roots)
    return root, alias, text, nc, suite


@pytest.mark.parametrize("reader", ["python", "os", "netcdf", "netcdf_extension", "listdir", "scandir"])
@pytest.mark.parametrize("aliased", [False, True])
def test_unadmitted_open_skips_before_reader(protected, reader, aliased):
    root, alias, text, nc, suite = protected
    base = alias if aliased else root
    # A missing path still skips before FileNotFoundError/the C reader. No real
    # ALISIOS data is needed even if the guard is accidentally deleted.
    path = base / "missing"
    with pytest.raises(pytest.skip.Exception, match=suite._ALISIOS_SKIP):
        if reader == "python":
            path.read_bytes()
        elif reader == "os":
            os.close(os.open(os.fsencode(path), os.O_RDONLY))
        elif reader == "netcdf":
            with netCDF4.Dataset(path):
                pass
        elif reader == "netcdf_extension":
            with netCDF4._netCDF4.Dataset(path):
                pass
        elif reader == "listdir":
            os.listdir(base)
        else:
            with os.scandir(base):
                pass


def test_real_roots_and_proc_fd_alias_resolve(protected, monkeypatch):
    root, alias, text, nc, suite = protected
    for base in ("<DATA_ROOT>/server/work/src/alisios", "<DATA_ROOT>/server/work/alisios", "<DATA_ROOT>/alisios"):
        assert Path(base) in suite._ALISIOS_REFERENCE_ROOTS
    # Supply a private already-open descriptor before treating its root as
    # protected, then check that integer-fd opens cannot bypass realpath.
    with monkeypatch.context() as setup:
        setup.setattr(suite, "_ALISIOS_ROOTS", ())
        fd = os.open(text, os.O_RDONLY)
    try:
        with pytest.raises(pytest.skip.Exception, match=suite._ALISIOS_SKIP):
            open(fd, "rb", closefd=False)
    finally:
        os.close(fd)
    with suite._NetCDFDataset(nc) as ds:
        assert isinstance(ds, netCDF4.Dataset)
    with pytest.raises(pytest.skip.Exception, match=suite._ALISIOS_SKIP):
        text.write_text("must not write")


def test_unadmitted_existing_file_is_not_read(protected):
    root, alias, text, nc, suite = protected
    refused = False
    try:
        text.read_bytes()
    except pytest.skip.Exception:
        refused = True
    assert refused, "central guard allowed an unadmitted payload read"


def test_unadmitted_existing_netcdf_is_not_read(protected):
    root, alias, text, nc, suite = protected
    refused = False
    try:
        with netCDF4.Dataset(nc):
            pass
    except pytest.skip.Exception:
        refused = True
    assert refused, "central guard allowed an unadmitted C-level netCDF read"


def test_admission_context_and_receipts(protected, tmp_path, monkeypatch):
    root, alias, text, nc, suite = protected
    calls = []
    class Refused(PermissionError):
        pass

    def scan(days, **kw):
        calls.append("scan")
        assert ag.admitted_session() is None
        return {day: "ALLOW" for day in days}

    def admit(*args, **kw):
        calls.append("admit")
        assert ag.admitted_session() is None
        return {"result": "ALLOW"}

    monkeypatch.setattr(ag, "_guard_call", lambda: SimpleNamespace(
        scan=scan, admit=admit, GuardRefused=Refused))
    cached_dataset = netCDF4.Dataset
    allowed = {text: "20260503", nc: "20260503"}
    with ag.guarded_session(allowed, ["20260502"], tmp_path / "receipts") as session:
        assert calls == ["scan", "admit"]
        assert ag.admitted_session() is session
        assert (alias / text.name).read_text() == "private fixture"
        with netCDF4.Dataset(nc) as ds:
            assert ds["x"][0] == 7
        with cached_dataset(nc):
            pass
        with netCDF4._netCDF4.Dataset(nc):
            pass
        import xarray as xr
        with xr.open_dataset(nc, engine="netcdf4") as ds:
            assert ds.x.values[0] == 7
        with pytest.raises(Refused):
            (root / "unlisted").read_bytes()
        # Another thread has no admission merely because the global spy is active.
        with ThreadPoolExecutor(1) as pool:
            with pytest.raises(pytest.skip.Exception, match=suite._ALISIOS_SKIP):
                pool.submit(text.read_bytes).result()
    assert ag.admitted_session() is None
    assert session.receipt["guard_result"] == "ALLOW"
    assert set(session.receipt["spy"]["observed"]) == {str(p) for p in allowed}
    assert session.receipt["spy"]["observed"][str(nc)]["open_count"] == 4
    assert (tmp_path / "receipts" / "guarded_session.json").is_file()
    with pytest.raises(pytest.skip.Exception, match=suite._ALISIOS_SKIP):
        text.read_bytes()


def test_nonprotected_netcdf_keeps_xarray_class_semantics(tmp_path):
    import xarray as xr
    assert issubclass(netCDF4.Dataset, netCDF4.Dataset)
    path = tmp_path / "ours.nc"
    xr.Dataset({"v": ("x", [1., 2.])}).to_netcdf(path, engine="netcdf4")
    with xr.open_dataset(path, engine="netcdf4") as ds:
        assert ds.v.values.tolist() == [1., 2.]


@pytest.fixture
def protected_fds(protected, monkeypatch):
    root, alias, text, nc, suite = protected
    with monkeypatch.context() as setup:
        setup.setattr(suite, "_ALISIOS_ROOTS", ())
        directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        file = os.open(text, os.O_RDONLY)
    try:
        yield protected, directory, file
    finally:
        os.close(directory)
        os.close(file)


@pytest.mark.parametrize("kind", ["stat", "exists", "lstat", "fstat", "dirfd_open",
                                  "dirfd_stat", "h5py", "h5netcdf", "xarray_h5netcdf", "h5py_native"])
def test_metadata_and_native_bypasses_refused(protected_fds, kind):
    (root, alias, text, nc, suite), directory, file = protected_fds
    refused = False
    try:
        if kind == "stat":
            text.stat()
        elif kind == "exists":
            (alias / text.name).exists()
        elif kind == "lstat":
            text.lstat()
        elif kind == "fstat":
            os.fstat(file)
        elif kind == "dirfd_open":
            os.close(os.open(text.name, os.O_RDONLY, dir_fd=directory))
        elif kind == "dirfd_stat":
            os.stat(text.name, dir_fd=directory)
        elif kind == "h5py":
            import h5py
            with h5py.File(nc, "r"):
                pass
        elif kind == "h5netcdf":
            import h5netcdf
            with h5netcdf.File(nc, "r"):
                pass
        elif kind == "xarray_h5netcdf":
            import xarray as xr
            with xr.open_dataset(nc, engine="h5netcdf"):
                pass
        else:
            import h5py
            h5py.h5f.open(os.fsencode(nc)).close()
    except pytest.skip.Exception:
        refused = True
    assert refused, f"central guard allowed unadmitted {kind}"


@pytest.mark.parametrize("kind", ["fd", "dirfd", "h5py", "h5netcdf"])
def test_admitted_native_and_fd_reads_have_spy(protected_fds, kind, tmp_path, monkeypatch):
    (root, alias, text, nc, suite), directory, file = protected_fds
    monkeypatch.setattr(ag, "_guard_call", lambda: SimpleNamespace(
        scan=lambda days, **kw: {d: "ALLOW" for d in days},
        admit=lambda *args, **kw: {"result": "ALLOW"}, GuardRefused=PermissionError))
    with ag.guarded_session({text: "20260503", nc: "20260503"}, ["20260502"],
                            tmp_path / "receipts") as session:
        if kind == "fd":
            with open(file, "rb", closefd=False) as src:
                assert src.read() == b"private fixture"
        elif kind == "dirfd":
            opened = os.open(text.name, os.O_RDONLY, dir_fd=directory)
            os.close(opened)
        elif kind == "h5py":
            import h5py
            with h5py.File(nc, "r") as ds:
                assert ds["x"][0] == 7
        else:
            import h5netcdf
            with h5netcdf.File(nc, "r") as ds:
                assert ds["x"][0] == 7
        if kind.startswith("h5"):
            reader = h5py.File if kind == "h5py" else h5netcdf.File
            with pytest.raises(PermissionError):
                reader(root / "unlisted.nc", "r")
    path = nc if kind.startswith("h5") else text
    assert session.receipt["spy"]["observed"][str(path)]["open_count"] == 1
    if kind.startswith("h5"):
        assert session.receipt["spy"]["refused_paths"] == [str(root / "unlisted.nc")]


def test_collection_inventory_includes_helpers_and_json(request):
    root_conftest = Path(__file__).with_name("conftest.py").resolve()
    suite = next(p for p in request.config.pluginmanager.get_plugins()
                 if getattr(p, "__file__", None) == str(root_conftest))
    rows = suite.pytest_report_collectionfinish(request.config, Path.cwd(), request.session.items)
    assert any("tests/v025/b_core/score_cpu_fidelity.py" in row for row in rows)
    assert any("tests/v025/b_noahmp/fixtures/albedo_hold.json" in row for row in rows)
