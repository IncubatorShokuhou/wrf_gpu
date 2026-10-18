"""ensemble.time_expanded_members: (lagged) TES ensembles (lane a1-io). CPU/NumPy only.

Synthetic run directories hold tiny wrfout-like files with the dt = 54 s hourly naming
(hh:00:00/18/36); the real test uses the CPU-WRF WN3 0502 + 0503 runs read-only (skip if absent).
Checkers must FAIL on deletion mutants (E39).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from gpuwrf.assimilation.letkf import ensemble as ens
from gpuwrf.assimilation.letkf.ensemble import hourly_wrfout, perturbation_spectrum, time_expanded_members
from gpuwrf.assimilation.letkf.state_io import EnsembleFrames, read_frame
from gpuwrf.io.netcdf_lock import Dataset

T_A = "2026-05-04_12:00:00"
NAMES = ("T", "QVAPOR", "MU", "U", "T2", "PB")
OFFSETS = (-3, -2, -1, 1, 2, 3)


def _write_frame(path: Path, stamp: str, rng, nz=4, ny=6, nx=7, grid_shift=0.0, qv_shift=0.0):
    with Dataset(str(path), "w") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("DateStrLen", 19)
        for dim, n in (("bottom_top", nz), ("bottom_top_stag", nz + 1), ("south_north", ny), ("west_east", nx),
                       ("west_east_stag", nx + 1)):
            ds.createDimension(dim, n)
        ds.DX = np.float32(9000.0)
        ds.DY = np.float32(9000.0)
        t = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        t[0] = np.frombuffer(stamp.encode(), dtype="S1")
        jj, ii = np.mgrid[0:ny, 0:nx]
        static = {"XLAT": (27.0 + 0.1 * jj + grid_shift, ("south_north", "west_east")),
                  "XLONG": (-18.0 + 0.1 * ii, ("south_north", "west_east")),
                  "HGT": (100.0 * ii, ("south_north", "west_east")),
                  "ZNU": (np.linspace(0.95, 0.05, nz), ("bottom_top",)),
                  "ZNW": (np.linspace(1.0, 0.0, nz + 1), ("bottom_top_stag",)),
                  "PB": (np.full((nz, ny, nx), 5.0e4), ("bottom_top", "south_north", "west_east"))}
        fields = {"T": (rng.normal(5, 2, (nz, ny, nx)), ("bottom_top", "south_north", "west_east")),
                  "QVAPOR": (0.01 + rng.normal(0.0, 0.0005, (nz, ny, nx)) + qv_shift,
                             ("bottom_top", "south_north", "west_east")),
                  "MU": (rng.normal(0, 300, (ny, nx)), ("south_north", "west_east")),
                  "U": (rng.normal(3, 4, (nz, ny, nx + 1)), ("bottom_top", "south_north", "west_east_stag")),
                  "T2": (rng.normal(290, 3, (ny, nx)), ("south_north", "west_east"))}
        for name, (value, dims) in {**static, **fields}.items():
            var = ds.createVariable(name, "f4", ("Time",) + dims)
            var[0] = np.asarray(value, np.float32)


def make_run(root: Path, seed: int, *, hours=range(-4, 5), grid_shift=0.0, qv_shift=0.0, skip=()):
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    t_a = datetime(2026, 5, 4, 12)
    for h in hours:
        if h in skip:
            continue
        when = t_a + timedelta(hours=h)
        sec = 0 if h == 0 else (18 * (h % 3))
        stamp = (when + timedelta(seconds=sec)).strftime("%Y-%m-%d_%H:%M:%S")
        _write_frame(root / f"wrfout_d01_{stamp}", stamp, rng, grid_shift=grid_shift, qv_shift=qv_shift)
    return root


# --------------------------------------------------------------------------- checkers

def _frame(run, t=T_A):
    return {n: v.astype(np.float64) for n, v in read_frame(Path(run) / f"wrfout_d01_{t}", t, NAMES).items()}


def check_tes(fn, runs, scale=1.0, control=None):
    control = control if control is not None else runs[0]
    e = fn(runs, "d01", T_A, OFFSETS, scale, names=NAMES, control_run=control)
    n_off = len(OFFSETS)
    assert e.n_members == len(runs) * n_off and all(v.shape[0] == e.n_members for v in e.values())
    ctrl = _frame(control)
    for name in NAMES:  # (4) TES mean = control frame x(t_a)
        dev = np.abs(np.asarray(e[name], np.float64).mean(axis=0) - ctrl[name]).max()
        assert dev <= 1e-12 * max(1.0, np.abs(ctrl[name]).max()), f"{name} mean deviates {dev}"
    for r, run in enumerate(runs):  # member = control + scale·(x_r(t_a+k) − mean_k x_r(t_a+k))
        offs = []
        for k in OFFSETS:
            path = hourly_wrfout(run, "d01", datetime(2026, 5, 4, 12) + timedelta(hours=k))
            with Dataset(str(path)) as ds:
                ds.set_auto_mask(False)
                offs.append({n: ds[n][0].astype(np.float64) for n in NAMES})
        mean_k = {n: np.mean([o[n] for o in offs], axis=0) for n in NAMES}
        for i, k in enumerate(OFFSETS):
            m = r * n_off + i
            assert e.members[m]["offset_h"] == k and e.members[m]["run_index"] == r
            assert len(e.members[m]["sha256"]) == 64 and e.members[m]["frame_index"] == 0
            for n in NAMES:
                expect = ctrl[n] + scale * (offs[i][n] - mean_k[n])
                np.testing.assert_allclose(e[n][m], expect, rtol=0, atol=1e-9 * max(1.0, np.abs(expect).max()))
    assert np.all(np.asarray(e["PB"]) == np.asarray(e["PB"])[0]), "static fields must not be perturbed"
    assert e.control_file.endswith(f"wrfout_d01_{T_A}") and e.times == (T_A,)
    return e


def _mut_no_recentre(runs, domain, t_a, offsets_h, scale, **kw):
    e = time_expanded_members(runs, domain, t_a, offsets_h, 0.0, **kw)  # members = raw offset frames
    for m, rec in enumerate(e.members):
        with Dataset(rec["file"]) as ds:
            ds.set_auto_mask(False)
            for n in e:
                e[n][m] = ds[n][0]
    return e


def _mut_scale_ignored(runs, domain, t_a, offsets_h, scale, **kw):
    return time_expanded_members(runs, domain, t_a, offsets_h, 1.0, **kw)


def _mut_own_centre(runs, domain, t_a, offsets_h, scale, **kw):  # each run around its own t_a state
    kw.pop("control_run", None)
    parts = [time_expanded_members([r], domain, t_a, offsets_h, scale, control_run=r, **kw) for r in runs]
    fields = {n: np.concatenate([p[n] for p in parts]) for n in parts[0]}
    members = [dict(rec, run_index=i) for i, p in enumerate(parts) for rec in p.members]
    return EnsembleFrames(fields, member_ids=[x for p in parts for x in p.member_ids], valid_time=t_a,
                          members=members, control_file=parts[0].control_file)


# --------------------------------------------------------------------------- synthetic tests

@pytest.fixture
def runs(tmp_path):
    return [make_run(tmp_path / "runA", 1), make_run(tmp_path / "runB", 2)]


def test_single_run_tes(runs):
    e = check_tes(time_expanded_members, runs[:1], scale=0.7)
    assert e.member_ids[0] == "r0-3h" and e.valid_time == T_A
    secs = {rec["offset_h"]: rec["time_offset_s"] for rec in e.members}
    assert secs[1] == 18.0 and secs[2] == 36.0 and secs[3] == 0.0 and secs[-1] == 36.0
    assert e.meta["negative"]["QVAPOR"]["total"] == 0 and "noncausal" in e.meta["label"]


@pytest.mark.parametrize("control_index", [0, 1])
def test_lagged_tes_centred_on_control(runs, control_index):
    e = check_tes(time_expanded_members, runs, scale=1.0, control=runs[control_index])
    assert e.n_members == 12 and e.meta["control_run"] == str(runs[control_index])


def test_control_outside_runs(tmp_path, runs):
    ctrl = make_run(tmp_path / "runCtrl", 9)
    check_tes(time_expanded_members, runs, scale=1.0, control=ctrl)


@pytest.mark.parametrize("mutant", [_mut_no_recentre, _mut_scale_ignored, _mut_own_centre])
def test_tes_mutants_caught(runs, mutant):
    with pytest.raises(AssertionError):
        check_tes(mutant, runs, scale=0.5)


def test_refusals(tmp_path, runs):
    with pytest.raises(ValueError):
        time_expanded_members(runs, "d01", T_A, (0, 1), names=NAMES)
    shifted = make_run(tmp_path / "runC", 3, grid_shift=0.01)
    with pytest.raises(ValueError, match="grid differs"):
        time_expanded_members([runs[0], shifted], "d01", T_A, OFFSETS, names=NAMES)
    holed = make_run(tmp_path / "runD", 4, skip=(2,))
    with pytest.raises(FileNotFoundError):
        time_expanded_members(holed, "d01", T_A, OFFSETS, names=NAMES)
    with pytest.raises(FileNotFoundError):  # t_a itself must be an exact-time frame
        time_expanded_members(runs[0], "d01", "2026-05-04_13:00:00", (-1, 1), names=NAMES)


def test_negative_qv_counted_never_silent(tmp_path):
    run = make_run(tmp_path / "runN", 5)
    e = time_expanded_members(run, "d01", T_A, OFFSETS, 25.0, names=NAMES)
    neg = e.meta["negative"]["QVAPOR"]
    assert neg["total"] > 0 and not neg["clipped"] and (np.asarray(e["QVAPOR"]) < 0).sum() == neg["total"]
    c = time_expanded_members(run, "d01", T_A, OFFSETS, 25.0, names=NAMES, clip_negative=True)
    assert c.meta["negative"]["QVAPOR"]["total"] == neg["total"] and (np.asarray(c["QVAPOR"]) < 0).sum() == 0


def test_perturbation_spectrum():
    rng = np.random.default_rng(0)
    basis = rng.standard_normal((3, 500))
    x = rng.standard_normal((10, 3)) @ basis  # rank-3 ensemble of 10 members
    s = perturbation_spectrum({"A": x.reshape(10, 5, 100)})
    sv = np.asarray(s["singular_values"])
    assert int((sv > 1e-6 * sv.max()).sum()) == 3
    assert 1.0 < s["effective_rank_entropy"] <= 3.0 + 1e-9 and 1.0 < s["participation_ratio"] <= 3.0 + 1e-9
    iso = perturbation_spectrum({"A": np.eye(6)})  # 6 orthogonal members: centred rank 5, equal energy
    assert abs(iso["effective_rank_entropy"] - 5.0) < 1e-9


# --------------------------------------------------------------------------- real CPU-WRF lagged pair

RUN_A = Path("<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260502_18z_a1/run/run")
RUN_B = Path("<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260503_18z_a1/run/run")


def _alisios_reads():
    """Every ALISIOS file the real TES test opens (t_a and ±1…±5 h frames of both runs; ALISIOS rule)."""
    reads = {}
    for run in (RUN_A, RUN_B):
        if not (run / f"wrfout_d01_{T_A}").is_file():
            return {}
        reads[(run / f"wrfout_d01_{T_A}").resolve()] = ["20260504"]
        for k in ens.DEFAULT_OFFSETS_H:
            path = hourly_wrfout(run, "d01", datetime(2026, 5, 4, 12) + timedelta(hours=k))
            reads[path.resolve()] = [path.name.split("_")[2].replace("-", "")]
    return reads


from _a1io_guard import module_guard  # noqa: E402

_alisios_guard = module_guard(_alisios_reads, autouse=False)


@pytest.mark.usefixtures("_alisios_guard")
def test_real_lagged_tes_d01():
    if not (RUN_A / f"wrfout_d01_{T_A}").is_file() or not (RUN_B / f"wrfout_d01_{T_A}").is_file():
        pytest.skip("CPU-WRF 0502/0503 runs absent")
    names = ("T", "QVAPOR", "MU", "T2")
    e = time_expanded_members(control_run=RUN_B, runs=[RUN_B, RUN_A], domain="d01",
                              t_a=datetime(2026, 5, 4, 12, tzinfo=timezone.utc), offsets_h=list(ens.DEFAULT_OFFSETS_H),
                              scale=1.0, names=names)  # the lead's CLI call shape
    assert e.n_members == 20 and e["T"].shape == (20, 44, 70, 120)
    ctrl = read_frame(RUN_B / f"wrfout_d01_{T_A}", T_A, names)
    for n in names:
        dev = np.abs(np.asarray(e[n]).mean(axis=0) - ctrl[n]).max()
        assert dev <= 1e-12 * np.abs(ctrl[n]).max(), (n, dev)
    assert sorted({rec["time_offset_s"] for rec in e.members}) == [0.0, 18.0, 36.0]
    spread = np.asarray(e["T"]).std(axis=0, ddof=1).mean(axis=(1, 2))
    assert np.all(spread > 0.01) and np.all(spread < 10.0)


# --------------------------------------------------------------------------- perturbed-IC members

def _mk_archive(root: Path, issue: str, valids, seed):
    rng = np.random.default_rng(seed)
    run = root / f"wg_{issue}_18z_a1" / "run" / "run"
    run.mkdir(parents=True)
    for v in valids:
        _write_frame(run / f"wrfout_d01_{v}_00:00:00", f"{v}_00:00:00", rng)
    return run


def test_select_archive_states_dedupe_exclusion(tmp_path):
    a = _mk_archive(tmp_path, "20260420", ["2026-04-21", "2026-04-22", "2026-04-23"], 1)
    b = _mk_archive(tmp_path, "20260421", ["2026-04-22", "2026-04-23", "2026-04-24"], 2)
    ctrl = _mk_archive(tmp_path, "20260502", ["2026-05-03"], 3)
    late = _mk_archive(tmp_path, "20260503", ["2026-05-04"], 4)  # issued after t_a: never causal
    used, dropped = ens.select_archive_states([a, b, ctrl, late], "d01", "2026-05-03_00:00:00", 3,
                                              exclude_runs=[ctrl])
    assert [u["valid"] for u in used] == ["2026-04-22_00:00:00", "2026-04-23_00:00:00", "2026-04-24_00:00:00"]
    assert all(u["run"] == str(b) for u in used)  # duplicates → shortest lead (later issue)
    reasons = sorted(d["reason"].split(" ")[0] for d in dropped)
    assert reasons == ["control", "duplicate", "duplicate", "older"]
    with pytest.raises(ValueError):
        ens.select_archive_states([a, b], "d01", "2026-05-03_00:00:00", 9)


def check_ic_fields(fn, control, states, scale):
    out = fn(control, states, scale, names=("T", "QVAPOR", "U"), spec_bdy_width=1, ramp=1)
    for n in ("T", "QVAPOR", "U"):
        x = out[n]
        assert x.shape == (len(states),) + control[n].shape
        np.testing.assert_allclose(x.mean(axis=0), control[n], rtol=0, atol=1e-12 * max(1.0, np.abs(control[n]).max()))
    assert (out["QVAPOR"] > 0).all(), "log-space moisture must stay positive"
    assert np.all(out["T"][:, :, 0, :] == control["T"][:, 0, :]) and np.all(out["U"][:, :, :, 0] == control["U"][:, :, 0])
    dev = out["T"][:, :, 2:-2, 2:-2] - control["T"][:, 2:-2, 2:-2]
    stack = np.stack([s["T"] for s in states])[:, :, 2:-2, 2:-2]
    np.testing.assert_allclose(dev, scale * (stack - stack.mean(axis=0)), rtol=0, atol=1e-9)
    return out


def test_ic_member_fields():
    rng = np.random.default_rng(0)
    nz, ny, nx = 3, 8, 9
    control = {"T": rng.normal(5, 1, (nz, ny, nx)), "QVAPOR": np.abs(rng.normal(0.006, 0.002, (nz, ny, nx))) + 1e-4,
               "U": rng.normal(3, 2, (nz, ny, nx + 1))}
    states = [{"T": rng.normal(5, 2, (nz, ny, nx)), "QVAPOR": np.abs(rng.normal(0.006, 0.004, (nz, ny, nx))) + 1e-5,
               "U": rng.normal(3, 3, (nz, ny, nx + 1))} for _ in range(6)]
    check_ic_fields(ens.ic_member_fields, control, states, 0.3)
    check_ic_fields(ens.ic_member_fields, control, states, 3.0)  # log-space moisture stays positive at large scale

    def additive_q(c, s, k, **kw):  # mutant: additive moisture deviations (archive_ensemble pre-log behaviour)
        out = ens.ic_member_fields(c, s, k, **kw)
        stack = np.stack([x["QVAPOR"] for x in s])
        out["QVAPOR"] = c["QVAPOR"][None] + k * (stack - stack.mean(axis=0))
        return out

    with pytest.raises(AssertionError):
        check_ic_fields(additive_q, control, states, 3.0)
    with pytest.raises(AssertionError):  # mutant: no taper
        check_ic_fields(lambda c, s, k, **kw: ens.ic_member_fields(c, s, k, **{**kw, "spec_bdy_width": 0, "ramp": 0}),
                        control, states, 0.3)
    with pytest.raises(AssertionError):  # mutant: scale ignored
        check_ic_fields(lambda c, s, k, **kw: ens.ic_member_fields(c, s, 1.0, **kw), control, states, 0.3)


def test_select_archive_states_drops_guard_deny_before_open(tmp_path, monkeypatch):
    """ALISIOS rule: DENY days are dropped by a guard scan before anything is opened (fake guard, synthetic tree)."""
    import types
    import netCDF4
    from gpuwrf.assimilation.letkf import alisios_guard as ag

    root = tmp_path / "alisios"
    a = _mk_archive(root, "20260420", ["2026-04-21", "2026-04-22", "2026-04-23"], 1)
    monkeypatch.setattr(ag, "DATA_ROOTS", (root.resolve(),))
    scanned = []
    fake = types.SimpleNamespace(scan=lambda days, mode=None: scanned.extend(days) or
                                 {d: ("DENY" if d == "20260422" else "ALLOW") for d in days})
    monkeypatch.setattr(ag, "_guard_call", lambda: fake)
    opened = []
    original = netCDF4.Dataset
    monkeypatch.setattr(netCDF4, "Dataset", lambda *a, **k: opened.append(a[0]) or original(*a, **k))
    used, dropped = ens.select_archive_states([a], "d01", "2026-05-03_00:00:00", 2)
    assert [u["valid"][:10] for u in used] == ["2026-04-21", "2026-04-23"] and not opened
    assert sorted(scanned) == ["20260421", "20260422", "20260423"]
    assert [d["valid"][:10] for d in dropped if d["reason"].startswith("ALISIOS guard DENY")] == ["2026-04-22"]
    monkeypatch.setattr(ag, "DATA_ROOTS", ())  # mutant: data not recognised as ALISIOS → the DENY day is used
    used2, _ = ens.select_archive_states([a], "d01", "2026-05-03_00:00:00", 2)
    assert "2026-04-22" in [u["valid"][:10] for u in used2]


# --------------------------------------------------------------------------- w41-da-ens: combinations + balanced MU

def test_ic_member_fields_combination_weights_untapered():
    from gpuwrf.assimilation.letkf.spread import gaussian_combinations
    rng = np.random.default_rng(1)
    nz, ny, nx = 3, 12, 13
    control = {"T": rng.normal(5, 1, (nz, ny, nx)), "QVAPOR": np.abs(rng.normal(0.006, 0.002, (nz, ny, nx))) + 1e-4,
               "MU": rng.normal(0, 50, (ny, nx))}
    states = [{"T": rng.normal(5, 2, (nz, ny, nx)), "QVAPOR": np.abs(rng.normal(0.006, 0.004, (nz, ny, nx))) + 1e-5,
               "MU": rng.normal(0, 80, (ny, nx))} for _ in range(4)]
    z = gaussian_combinations(9, 4, 5)                              # N = 9 members from M = 4 states
    out = ens.ic_member_fields(control, states, 0.4, names=("T", "QVAPOR", "MU"), spec_bdy_width=2, ramp=2,
                               weights=z, untapered=("T",))
    assert out["T"].shape == (9, nz, ny, nx) and out["MU"].shape == (9, ny, nx)
    for n in ("T", "QVAPOR", "MU"):
        np.testing.assert_allclose(out[n].mean(0), control[n], rtol=1e-12, atol=1e-9)
    assert (out["QVAPOR"] > 0).all()
    assert np.all(out["MU"][:, 0, :] == control["MU"][0, :])        # tapered
    assert np.abs(out["T"][:, :, 0, :] - control["T"][:, 0, :]).max() > 0.01   # untapered
    pool = np.stack([s["T"] for s in states])
    expect = 0.4 * np.einsum("im,m...->i...", z, pool - pool.mean(0))
    np.testing.assert_allclose(out["T"] - control["T"], expect - expect.mean(0), atol=1e-12)


SWISS_REF = Path("<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu")
SWISS_CASE = Path(__file__).resolve().parents[2] / "examples" / "switzerland_d01"


@pytest.mark.skipif(not (SWISS_REF / "wrfout_d01_2023-01-15_08:00:00").is_file() or
                    not (SWISS_CASE / "wrfinput_d01").is_file(), reason="Swiss CPU-WRF reference absent")
def test_perturbed_ic_members_balanced_combination_on_real_swiss_frames(tmp_path):
    """Non-ALISIOS real frames: control = CPU-WRF 03Z, pool = 04–08Z states; N = 6 > M = 5, MU balanced."""
    from gpuwrf.assimilation.letkf import perturb
    from gpuwrf.assimilation.letkf.state_io import read_geometry
    t_a = "2023-01-15_03:00:00"
    used = [{"file": str(SWISS_REF / f"wrfout_d01_2023-01-15_{h:02d}:00:00"), "valid": f"2023-01-15_{h:02d}:00:00"}
            for h in range(4, 9)]
    man = ens.perturbed_ic_members(SWISS_REF / f"wrfout_d01_{t_a}", [], tmp_path / "members",
                                   template_wrfinput=SWISS_CASE / "wrfinput_d01", scale=0.4, n_members=5,
                                   archive_states=(used, []), combination={"n": 6, "seed": 2}, balanced=True)
    assert man["n_members"] == 6 and man["combination"]["pool"] == 5 and len(man["members"]) == 7
    assert man["mu"].startswith("perturbed")
    geo = read_geometry(SWISS_REF / f"wrfout_d01_{t_a}", base_wrfinput=SWISS_CASE / "wrfinput_d01")
    names = ("U", "V", "T", "QVAPOR", "MU", "P", "PH")
    ctrl = read_frame(tmp_path / "members" / "ctrl" / "wrfinput_d01", t_a, names)
    mus = []
    for k in range(6):
        m = read_frame(tmp_path / "members" / f"m{k:02d}" / "wrfinput_d01", t_a, names)
        mus.append(np.asarray(m["MU"], np.float64))
        assert perturb.hydrostatic_residual_change(ctrl, m, m, geo) < 1e-5
        assert np.all(m["MU"][:5, :] == ctrl["MU"][:5, :])            # specified zone: MU frozen like wrfbdy
    assert np.std(mus, axis=0)[15:-15, 15:-15].mean() > 1.0           # MU genuinely perturbed in the interior


def test_perturbed_ic_members_protect_the_case_boundary_width(tmp_path):
    """Sibling of cr-w41-da-core (2): with the control's namelist at spec_bdy_width = 9 the tapered IC increment
    must leave rows/faces 0..8 from every edge byte-identical to the control (the shared wrfbdy owns them), not
    only 0..4 (the old default 5); the manifest records the case width."""
    import re
    import shutil
    if not (SWISS_REF / "wrfout_d01_2023-01-15_03:00:00").is_file():
        pytest.skip("Swiss reference run absent")
    t_a = "2023-01-15_03:00:00"
    run = tmp_path / "ctrl_run"
    run.mkdir()
    shutil.copy(SWISS_REF / f"wrfout_d01_{t_a}", run / f"wrfout_d01_{t_a}")
    nml = (SWISS_CASE / "namelist.input").read_text()
    nml = re.sub(r"spec_bdy_width\s*=\s*\d+", "spec_bdy_width = 9", nml)
    nml = re.sub(r"relax_zone\s*=\s*\d+", "relax_zone = 8", nml)
    (run / "namelist.input").write_text(nml)
    used = [{"file": str(SWISS_REF / f"wrfout_d01_2023-01-15_{h:02d}:00:00"), "valid": f"2023-01-15_{h:02d}:00:00"}
            for h in range(4, 9)]
    man = ens.perturbed_ic_members(run / f"wrfout_d01_{t_a}", [], tmp_path / "members",
                                   template_wrfinput=SWISS_CASE / "wrfinput_d01", scale=0.4, n_members=5,
                                   archive_states=(used, []))
    assert man["taper"]["spec_bdy_width"] == 9 and man["taper"]["spec_bdy_width_source"].startswith("case namelist")
    names = ("U", "V", "T", "QVAPOR")
    ctrl = read_frame(tmp_path / "members" / "ctrl" / "wrfinput_d01", t_a, names)
    mem = read_frame(tmp_path / "members" / "m00" / "wrfinput_d01", t_a, names)
    for n in names:
        a, b = np.asarray(mem[n]), np.asarray(ctrl[n])
        for sl in ((Ellipsis, slice(0, 9), slice(None)), (Ellipsis, slice(-9, None), slice(None)),
                   (Ellipsis, slice(None), slice(0, 9)), (Ellipsis, slice(None), slice(-9, None))):
            assert np.array_equal(a[sl], b[sl]), n
        assert not np.array_equal(a, b), n
