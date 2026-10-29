"""Known-answer consumer/QC checks; these are not forecast fidelity tests."""
import hashlib
import json

from netCDF4 import Dataset
import numpy as np
import pytest

from gpuwrf.optimize import diagnostics as D


POINTS = [dict(id="north", region="north_windward", domain="d03", j=0, i=0),
          dict(id="ocean", region="sea", domain="d03", j=0, i=1)]


@pytest.fixture
def history(tmp_path):
    path = tmp_path / "wrfout_d03"
    with Dataset(path, "w") as ds:
        for name, size in (("Time", 2), ("DateStrLen", 19), ("bottom_top", 4),
                           ("bottom_top_stag", 5), ("south_north", 1), ("west_east", 2)):
            ds.createDimension(name, size)
        ds.GRID_ID = 3
        ds.SIMULATION_START_DATE = "2026-02-27_00:00:00"
        v = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        v[:] = np.array([list(b"2026-02-27_00:00:00".decode()), list(b"2026-02-27_01:00:00".decode())], "S1")

        def put(name, values, *, surface=False, stagger=False):
            dims = ("Time",) + (() if surface else ("bottom_top_stag" if stagger else "bottom_top",)) + ("south_north", "west_east")
            var = ds.createVariable(name, "f4", dims)
            var.units = D._units(name)
            if surface:
                var[:] = np.broadcast_to(values, (2, 1, 2))
            else:
                var[:] = np.broadcast_to(np.array(values)[None, :, None, None], (2, 5 if stagger else 4, 1, 2))

        for name, values in {"T": [0, 1, 9, 10], "P": [0]*4, "PB": [100000]*4,
                             "QVAPOR": [.01]*4, "QCLOUD": [0, .001, .002, 0], "QICE": [0, 0, .0005, 0],
                             "PH": np.array([0, 200, 600, 1200, 2000])*9.81,
                             "PHB": [0]*5}.items():
            put(name, values, stagger=name in ("PH", "PHB"))
        for name in D.RAD_FIELDS:
            put(name, [-.001, -.002, -.003, -.004] if name.startswith("RTH") else [0]*4)
        for name in D.METADATA:
            values = {"qSHEAR": [1, 2, 3, 0, 0], "qBUOY": [2, 3, 4, 0, 0],
                      "qDISS": [4, 5, 6, 0, 0], "qWT": [5, 6, 7, 0, 0],
                      "Dtke": [4.25, 6.5, 8.75, 0]}.get(name, [1.e-5]*4)
            put(name, values, stagger=name in ("qSHEAR", "qBUOY", "qDISS", "qWT"))
        for name, value in {"HGT": 0, "XLAT": 28.4, "XLONG": -16.5,
                            "HFX": 20, "LH": 40, "GRDFLX": -5, "QFX": 1.e-5,
                            "LANDMASK": [1, 0]}.items():
            put(name, value, surface=True)
    return path


def first(report):
    return next(r["metrics"] for r in report["rows"] if r["point"] == "north")


def test_known_answer_profiles_closure_and_read_only(history, tmp_path):
    before = hashlib.sha256(history.read_bytes()).hexdigest()
    output = tmp_path / "profiles.json"
    report = D.write_profile_report([history], POINTS, output)
    assert hashlib.sha256(history.read_bytes()).hexdigest() == before
    assert report == json.loads(output.read_text())
    assert len(report["rows"]) == 4
    m = first(report)
    np.testing.assert_allclose(m["theta"]["value"], [300, 301, 309, 310])
    np.testing.assert_allclose(m["height_agl"]["value"], [100, 400, 900, 1600], atol=1.e-4)
    assert m["inversion_height_agl"]["value"] == pytest.approx(650, abs=1.e-4)
    assert m["cloud_top_agl"]["value"] == pytest.approx(900, abs=1.e-4)
    assert m["cloud_top_lw_cooling"]["value"] == pytest.approx(.003)
    rho = 100000 / (287 * np.array([300, 301, 309, 310]) * (1 + .01/(287/461.6)))
    lwp = 1000 * (rho[1]*400*.001 + rho[2]*600*.002)
    assert m["LWP"]["value"] == pytest.approx(lwp, rel=2.e-7)
    assert m["CWP"]["value"] == pytest.approx(lwp + 1000*rho[2]*600*.0005, rel=2.e-7)
    np.testing.assert_array_equal(m["tke_budget_residual"]["value"], [.25, .5, .75])
    assert m["tke_budget_residual"]["max_abs"] == .75
    assert m["tke_budget_residual"]["rms"] == pytest.approx(np.sqrt((.25**2+.5**2+.75**2)/3))
    np.testing.assert_array_equal(m["qSHEAR"]["value"], [1, 2, 3, 0])  # no second destagger
    assert m["pbl_water_tendency"]["value"] == pytest.approx(np.sum(rho*np.array([200,400,600,800]))*3.e-5)
    assert m["HFX"]["value"] == 20
    assert m["QFX"]["units"] == "kg m-2 s-1"
    for name in D.LATER_TAPS:
        assert m[name]["status"] == "UNAVAILABLE" and m[name]["value"] is None and m[name]["reason"]
    entry = D.wave_report_entry(output)
    assert entry["profile_rows"] == 4 and entry["max_abs_tke_residual_m2_s3"] == .75
    assert entry["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert 0 < report["sample_bytes"] < 16000
    with pytest.raises(ValueError, match="new destination"):
        D.write_profile_report([history], POINTS, output)


def test_missing_diagnostics_are_unavailable_not_zero(history, tmp_path):
    with Dataset(history, "r+") as ds:
        for name in ("RTHRATLW", "qWT", "QCLOUD"):
            ds.renameVariable(name, "omitted_" + name)
    report = D.write_profile_report([history], POINTS, tmp_path / "missing.json")
    m = first(report)
    for name in ("RTHRATLW", "qWT", "CWP", "LWP", "cloud_top_lw_cooling", "tke_budget_residual"):
        assert m[name]["status"] == "UNAVAILABLE" and m[name]["value"] is None
    assert m["theta"]["status"] == "AVAILABLE"
    assert "qWT" in m["tke_budget_residual"]["reason"]
    entry = D.wave_report_entry(tmp_path / "missing.json")
    assert entry["status"] == report["status"]
    assert entry["max_abs_tke_residual_m2_s3"] is None


@pytest.mark.parametrize("damage", ["nan", "fill", "units", "top", "mass_top", "negative_q", "height", "land", "duplicate_time", "moving_grid"])
def test_bad_history_fails_before_publication(history, tmp_path, damage):
    with Dataset(history, "r+") as ds:
        if damage == "nan": ds["RTHRATLW"][1, 0, 0, 1] = np.nan
        if damage == "fill": ds["RTHRATLW"][1, 0, 0, 1] = np.ma.masked
        if damage == "units": ds["RTHRATLW"].units = "K h-1"
        if damage == "top": ds["qWT"][:, -1] = 1
        if damage == "mass_top": ds["Dtke"][:, -1] = 1
        if damage == "negative_q": ds["QCLOUD"][1, 0, 0, 1] = -.1
        if damage == "height": ds["PH"][:, 1] = -1
        if damage == "land": ds["LANDMASK"][:, 0, 1] = 1
        if damage == "duplicate_time": ds["Times"][1] = ds["Times"][0]
        if damage == "moving_grid": ds["XLAT"][1] = 29
    output = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        D.write_profile_report([history], POINTS, output)
    assert not output.exists()


@pytest.mark.parametrize("points", [[], [dict(POINTS[0], j=-1)], [dict(POINTS[0], i=True)],
                                    [dict(POINTS[0], i=10)], [dict(POINTS[0], domain="d3")], POINTS*2])
def test_bad_point_contract_refuses(history, tmp_path, points):
    with pytest.raises(ValueError):
        D.write_profile_report([history], points, tmp_path / "badpoint.json")


def test_absent_domain_and_no_cloud_are_explicit(history, tmp_path):
    with Dataset(history, "r+") as ds:
        ds["QCLOUD"][:] = 0
        ds["QICE"][:] = 0
    report = D.write_profile_report([history], [*POINTS, dict(POINTS[0], id="other", domain="d02")], tmp_path / "clear.json")
    assert report["status"] == "UNAVAILABLE" and report["missing_points"] == ["other"]
    m = first(report)
    assert m["CWP"]["value"] == 0
    assert m["cloud_top_lw_cooling"]["status"] == "UNAVAILABLE"
    entry = D.wave_report_entry(tmp_path / "clear.json")
    assert entry["status"] == "UNAVAILABLE" and entry["missing_points"] == ["other"]


def _numeric_paths(value, path=()):
    """Every numeric leaf, including metadata, later rows and profile cells."""
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _numeric_paths(child, (*path, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _numeric_paths(child, (*path, index))
    elif type(value) in (int, float):
        yield path


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")], ids=["nan", "inf", "neginf"])
def test_wave_entry_rejects_one_nonfinite_per_numeric_leaf(history, tmp_path, monkeypatch, nonfinite):
    output = tmp_path / "profiles.json"
    report = D.write_profile_report([history], POINTS, output)
    paths = list(_numeric_paths(report))
    assert len(paths) > 100
    assert ("rows", 0, "metrics", "tke_budget_residual", "max_abs") in paths
    assert ("rows", 3, "metrics", "tke_budget_residual", "max_abs") in paths
    original_read = type(output).read_bytes
    raw = output.read_bytes()
    def read_bytes(path):
        return raw if path == output else original_read(path)
    monkeypatch.setattr(type(output), "read_bytes", read_bytes)
    for path in paths:
        parent = report
        for key in path[:-1]:
            parent = parent[key]
        original = parent[path[-1]]
        parent[path[-1]] = nonfinite
        raw = json.dumps(report, allow_nan=True).encode()
        with pytest.raises(ValueError, match="nonfinite.*profile report"):
            D.wave_report_entry(output)
        parent[path[-1]] = original


@pytest.mark.parametrize("literal", ["1e309", "-1e309"])
def test_wave_entry_rejects_json_exponent_overflow(history, tmp_path, literal):
    output = tmp_path / "overflow.json"
    report = D.write_profile_report([history], POINTS, output)
    report["extract_wall_s"] = "OVERFLOW_SENTINEL"
    output.write_text(json.dumps(report).replace('"OVERFLOW_SENTINEL"', literal))
    with pytest.raises(ValueError, match="nonfinite.*profile report"):
        D.wave_report_entry(output)


def test_cli_consumer_and_domain_issue_qc(history, tmp_path):
    points = tmp_path / "points.json"
    points.write_text(json.dumps(POINTS))
    assert D.main(["--files", str(history), "--points", str(points), "--output", str(tmp_path / "cli.json")]) == 0
    copy = tmp_path / "other.nc"
    copy.write_bytes(history.read_bytes())
    with Dataset(copy, "r+") as ds:
        ds.SIMULATION_START_DATE = "2026-02-28_00:00:00"
    with pytest.raises(ValueError, match="one forecast issue"):
        D.write_profile_report([history, copy], POINTS, tmp_path / "mix.json")


def test_changed_history_refuses_and_never_opens_for_write(history, tmp_path, monkeypatch):
    original = D._stat
    calls = []
    def changed(path):
        calls.append(path)
        return {**original(path), "mtime_ns": len(calls)}
    monkeypatch.setattr(D, "_stat", changed)
    with pytest.raises(ValueError, match="changed during"):
        D.write_profile_report([history], POINTS, tmp_path / "changed.json")


def test_alisios_paths_refused_before_open(tmp_path, monkeypatch):
    from gpuwrf.assimilation.letkf import alisios_guard
    protected = tmp_path / "private"
    monkeypatch.setattr(alisios_guard, "DATA_ROOTS", (protected,))
    def opened(*args, **kwargs):
        raise AssertionError("protected payload opened")
    monkeypatch.setattr(D, "Dataset", opened)
    with pytest.raises(ValueError, match="ALISIOS"):
        D.write_profile_report([protected / "wrfout"], POINTS, tmp_path / "blocked.json")
