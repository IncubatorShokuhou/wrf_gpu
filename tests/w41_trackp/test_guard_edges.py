"""ALISIOS read guard at every Track P entry point (critic cr-w41-trackp FIX 17:19Z; the first two tests are the
critic's sentinels). Nothing here opens ALISIOS payload: protected paths are fakes, or local synthetic files under a
fake protected root installed by monkeypatch, and readers are wrapped to record any read outside admission."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwrf.surface_params import cli
from gpuwrf.surface_params import stations as st
from test_eki_collect import _inputs


class PayloadSentinel(Exception):
    pass


def _fake_guard(monkeypatch, admission):
    from gpuwrf.assimilation.letkf import alisios_guard

    @contextmanager
    def fake_session(allowed, *a, **k):
        admission["allowed"] = {str(Path(p).resolve()) for p in allowed}
        admission["active"] = True
        admission["sessions"] = admission.get("sessions", 0) + 1
        try:
            yield SimpleNamespace(receipt={"synthetic_guard": True, "allowed": sorted(admission["allowed"])})
        finally:
            admission["active"] = False
    monkeypatch.setattr(alisios_guard, "guarded_session", fake_session)


@pytest.mark.parametrize("kind,with_guard", [("member", False), ("member", True), ("stations", False)])
def test_all_collector_alisios_inputs_admitted_before_reader(tmp_path, monkeypatch, kind, with_guard):
    from gpuwrf.verify import observations, wrfout
    dirs, stations, obs = _inputs(tmp_path)
    members = [sorted(d.glob("wrfout_d01_*")) for d in dirs]
    touched = []
    admission = {"active": False, "allowed": set()}
    if with_guard:
        _fake_guard(monkeypatch, admission)
    fake = Path("<DATA_ROOT>/alisios/__cr_trackp_no_payload__/wrfout_d01_2026-01-01_00:00:00")
    if kind == "member":
        members[0] = [fake]

        def reader(paths, *a, **k):
            for path in paths:
                if not admission["active"] or str(Path(path).resolve()) not in admission["allowed"]:
                    touched.append(str(path))
            raise PayloadSentinel()
        monkeypatch.setattr(wrfout, "extract_wrfout", reader)
    else:
        stations = fake.with_name("stations.csv")

        def reader(path, *a, **k):
            touched.append(str(path))
            raise PayloadSentinel()
        monkeypatch.setattr(observations, "read_stations", reader)
    guard = {"issue_days": ["20260101"], "valid_days": ["20260101"],
             "receipt_dir": str(tmp_path / "guard")} if with_guard else None
    try:
        st.collect(members, stations, obs, sigma={"t2": 1., "wind": 1.5}, guard=guard)
    except (PayloadSentinel, ValueError):
        pass
    assert not touched, f"Protected {kind} reached payload reader outside admission: {touched}"


def test_basis_protected_case_refused_before_loader(tmp_path, monkeypatch):
    from gpuwrf.integration import nested_pipeline as pipeline
    touched = []

    def loader(config, *a, **k):
        touched.append(str(config.input_dir))
        raise PayloadSentinel()
    monkeypatch.setattr(pipeline, "_load_domains", loader)
    try:
        cli.main_args(["basis", "--input-dir", "<DATA_ROOT>/alisios/__cr_trackp_no_payload__",
                       "--max-dom", "1", "--output", str(tmp_path / "basis.npz")])
    except (PayloadSentinel, ValueError):
        pass
    assert not touched, f"Protected case reached _load_domains before admission: {touched}"


@pytest.fixture
def fake_root(tmp_path, monkeypatch):
    """A local directory treated as an ALISIOS data root (synthetic files only)."""
    from gpuwrf.assimilation.letkf import alisios_guard
    root = tmp_path / "fake_alisios_root"
    root.mkdir()
    monkeypatch.setattr(st, "ALISIOS_DATA_ROOTS", st.ALISIOS_DATA_ROOTS + (str(root.resolve()),))
    monkeypatch.setattr(alisios_guard, "DATA_ROOTS", alisios_guard.DATA_ROOTS + (root.resolve(),))
    return root


def test_guarded_route_admits_whole_read_set_for_whole_lifetime(tmp_path, monkeypatch, fake_root):
    """Protected observations, station metadata AND one member (via a symlink): every read happens inside ONE session
    that admitted exactly the protected set; local members are read in the same session; the receipt is kept."""
    from gpuwrf.verify import observations, wrfout
    dirs, stations, obs = _inputs(tmp_path)
    for f in (stations, obs):
        f.rename(fake_root / f.name)
    stations, obs = fake_root / "stations.csv", fake_root / "obs.csv"
    member0 = sorted(dirs[0].glob("wrfout_d01_*"))[0]
    target = fake_root / "m0" / member0.name
    target.parent.mkdir()
    member0.rename(target)
    member0.symlink_to(target)  # local path, protected real path
    admission = {"active": False, "allowed": set()}
    _fake_guard(monkeypatch, admission)
    reads = []
    for module, name in ((observations, "read_stations"), (observations, "read_observations"),
                         (wrfout, "extract_wrfout")):
        real = getattr(module, name)

        def wrapped(*a, _real=real, _name=name, **k):
            reads.append((_name, admission["active"]))
            return _real(*a, **k)
        monkeypatch.setattr(module, name, wrapped)
    guard = {"issue_days": ["20260101"], "valid_days": ["20260101"], "receipt_dir": str(tmp_path / "guard")}
    members = [sorted(d.glob("wrfout_d01_*")) for d in dirs]
    pred, y, _sig, keys = st.collect(members, stations, obs, sigma={"t2": 1., "wind": 1.5}, guard=guard)
    assert pred.shape == (3, len(keys)) and y.size == len(keys) > 0
    assert admission["sessions"] == 1
    assert admission["allowed"] == {str(stations.resolve()), str(obs.resolve()), str(target.resolve())}
    assert [r[0] for r in reads] == ["read_stations", "read_observations"] + ["extract_wrfout"] * 3
    assert all(active for _name, active in reads), reads
    assert guard["receipt"]["synthetic_guard"] is True


def test_symlinked_member_refused_without_guard(tmp_path, monkeypatch, fake_root):
    from gpuwrf.verify import wrfout
    dirs, stations, obs = _inputs(tmp_path)
    member0 = sorted(dirs[0].glob("wrfout_d01_*"))[0]
    target = fake_root / member0.name
    member0.rename(target)
    member0.symlink_to(target)
    monkeypatch.setattr(wrfout, "extract_wrfout", lambda *a, **k: pytest.fail("member read before refusal"))
    with pytest.raises(ValueError, match="alisios_guard"):
        st.collect([sorted(d.glob("wrfout_d01_*")) for d in dirs], stations, obs, sigma={"t2": 1., "wind": 1.5})


def test_basis_local_case_linking_into_protected_root_refused(tmp_path, monkeypatch, capsys, fake_root):
    from gpuwrf.integration import nested_pipeline as pipeline
    case = tmp_path / "case"
    case.mkdir()
    (fake_root / "wrfinput_d01").write_bytes(b"synthetic")
    (case / "wrfinput_d01").symlink_to(fake_root / "wrfinput_d01")
    (case / "namelist.input").write_text("&domains\n max_dom = 1,\n/\n")
    touched = []

    def loader(config, *a, **k):  # PayloadSentinel is not caught by the CLI handler: reaching it fails the test
        touched.append(str(config.input_dir))
        raise PayloadSentinel()
    monkeypatch.setattr(pipeline, "_load_domains", loader)
    rc = cli.main_args(["basis", "--input-dir", str(case), "--output", str(tmp_path / "basis.npz")])
    assert rc != 0 and not touched
    assert "ALISIOS data is read only through" in capsys.readouterr().err
    assert not (tmp_path / "basis.npz").exists()
    assert st.protected_paths([case]) == [str((fake_root / "wrfinput_d01").resolve())]


def test_local_inputs_need_no_guard(tmp_path):
    dirs, stations, obs = _inputs(tmp_path)
    assert st.protected_paths([stations, obs, *dirs]) == []


def test_roots_match_the_guard():
    from gpuwrf.assimilation.letkf import alisios_guard
    assert {Path(r).resolve() for r in st.ALISIOS_DATA_ROOTS} == set(alisios_guard.DATA_ROOTS)


# --- critic delta repros (cr-w41-trackp VERDICT_e7fa8e2ac8.md, test_delta_derived.py): derived basis inputs ---------


def test_directory_alias_chain_cannot_hide_protected_descendant(tmp_path, monkeypatch):
    from gpuwrf.integration import nested_pipeline as pipeline
    case = tmp_path / "case"
    case.mkdir()
    outside = tmp_path / "ordinary_table_directory"
    outside.mkdir()
    target = Path("<DATA_ROOT>/alisios/__cr_trackp_no_payload__/MPTABLE.TBL")
    (outside / "MPTABLE.TBL").symlink_to(target)
    (case / "tables").symlink_to(outside, target_is_directory=True)
    touched = []

    def loader(*a, **k):
        touched.append(str(case / "tables/MPTABLE.TBL"))
        raise PayloadSentinel()
    monkeypatch.setattr(pipeline, "_load_domains", loader)
    rc = cli.main_args(["basis", "--input-dir", str(case), "--max-dom", "1", "--output", str(tmp_path / "basis.npz")])
    assert rc != 0 and not touched, "local directory alias hides a protected descendant from basis preflight"


def test_basis_runtime_table_root_checked_before_real_table_reader(tmp_path, monkeypatch):
    from gpuwrf.config import paths
    from gpuwrf.integration import nested_pipeline as pipeline
    from gpuwrf.physics.noahmp import tables
    case = tmp_path / "case"
    case.mkdir()
    (case / "namelist.input").write_text("&domains\n max_dom=1,\n/\n")
    runtime = tmp_path / "ordinary_wrf_root"
    (runtime / "run").mkdir(parents=True)
    target = Path("<DATA_ROOT>/alisios/__cr_trackp_no_payload__/MPTABLE.TBL")
    (runtime / "run/MPTABLE.TBL").symlink_to(target)
    monkeypatch.setenv("GPUWRF_WRF_ROOT", str(runtime))
    touched = []
    real_read = Path.read_text

    def sentinel(path, *a, **k):
        if path.resolve() == target:
            touched.append(str(path))
            raise PayloadSentinel()
        return real_read(path, *a, **k)
    monkeypatch.setattr(Path, "read_text", sentinel)
    monkeypatch.setattr(pipeline, "_load_domains", lambda *a, **k: tables.load_noahmp_parameters(paths.wrf_run_dir()))
    rc = cli.main_args(["basis", "--input-dir", str(case), "--max-dom", "1", "--output", str(tmp_path / "basis.npz")])
    assert rc != 0 and not touched, "basis reached protected MPTABLE read_text through runtime resolver before refusal"


def test_basis_open_backstop_refuses_paths_the_preflight_cannot_see(tmp_path, monkeypatch, capsys):
    """A path built only while loading (invisible to the static preflight) is refused at the open layer."""
    from gpuwrf.assimilation.letkf import member_driver
    from gpuwrf.integration import nested_pipeline as pipeline
    case = tmp_path / "case"
    case.mkdir()
    (case / "namelist.input").write_text("&domains\n max_dom=1,\n/\n")
    monkeypatch.setattr(member_driver, "refuse_protected_inputs", lambda *a, **k: None)  # blind preflight
    hidden = "<DATA_ROOT>/alisios/" + "__cr_trackp_no_payload__" + "/late_table.TBL"
    opened = []

    def loader(*a, **k):
        with open(hidden) as fh:  # the audit hook raises BEFORE the open happens
            opened.append(fh.read())
        raise PayloadSentinel()
    monkeypatch.setattr(pipeline, "_load_domains", loader)
    rc = cli.main_args(["basis", "--input-dir", str(case), "--max-dom", "1", "--output", str(tmp_path / "basis.npz")])
    assert rc != 0 and not opened
    assert "refused open of ALISIOS path" in capsys.readouterr().err
