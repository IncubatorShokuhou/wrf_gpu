"""`gpuwrf optimize` continuous mode (lane w41-optcont): EKI over traced Track P controls through the launcher.

The fake scheduler below is an ANALYTIC forward model, not the WRF model: each member reads ITS OWN control file
(the per-member ``surface_controls.npz`` the optimizer prepared) and writes hourly wrfouts whose T2 rises and 10 m
wind falls with ln(z0) at every cell (log-linear in the log multipliers). It also writes the run proof (AOT cheap key
of the namelist, never of the control values) and launcher receipts (first member of a program cold, the rest warm),
so the orchestration, EKI recovery, program proof, resume and refusals are exercised end to end on the CPU. Real
model members: proofs/w41_optcont (GPU OSSE smoke); no real-world skill is claimed here.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
from functools import partial

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/"tests"/"w41_trackp"))
from w41_trackp_fixtures import SHAPE, VEG, pytestmark, _fixture, _namelist  # noqa: E402,F401

from gpuwrf.optimize import continuous  # noqa: E402
from gpuwrf.optimize.data import identity, write_json  # noqa: E402
from gpuwrf.surface_params import cli as spc  # noqa: E402
from gpuwrf.surface_params import controls as sc  # noqa: E402

TRUTH = {"z0_forest": 1.6, "z0_low": 0.7}


@pytest.fixture(autouse=True)
def legacy_chunk_reference(monkeypatch):
    """Uninterrupted/chunk EKI regression reference; production wave defaults tested in v45_waveops."""
    from gpuwrf.optimize import study
    monkeypatch.setattr(study, "run", partial(study.run, wave_mode=False, wall_seconds=None, max_retries=None))
LAT = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
LON = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])


def fake_case(root, day):
    from gpuwrf.io.netcdf_lock import Dataset
    case = root/f"case-{day}"
    case.mkdir()
    stamp = f"2026-01-{day:02d}"
    for name, times in (("wrfinput_d01", [f"{stamp}_00:00:00"]),
                        ("wrfbdy_d01", [f"{stamp}_00:00:00", f"{stamp}_03:00:00"])):
        with Dataset(case/name, "w") as ds:
            ds.createDimension("Time", len(times))
            ds.createDimension("DateStrLen", 19)
            ds.createVariable("Times", "S1", ("Time", "DateStrLen"))[:] = np.array([list(t) for t in times], dtype="S1")
            if name == "wrfinput_d01":
                for dim, size in (("west_east", 3), ("south_north", 2), ("bottom_top", 3)):
                    ds.createDimension(dim, size)
                ds.createVariable("IVGTYP", "i4", ("Time", "south_north", "west_east"))[:] = VEG[None]
    (case/"namelist.input").write_text(f"""&time_control
 start_year=2026, start_month=1, start_day={day}, start_hour=0,
 /
 &domains
 max_dom=1, time_step=3600,
 /
 &physics
 mp_physics=8, bl_pbl_physics=5, sf_sfclay_physics=5, sf_surface_physics=4,
 ra_lw_physics=4, ra_sw_physics=4, cu_physics=0, radt=30, cudt=0,
 /
 &dynamics
 diff_opt=1, km_opt=4,
 /
""")
    return case


def forward(z0, day, hour):
    """Analytic surface response: T2 up and wind down with ln z0 (exactly log-linear in the multipliers)."""
    lz = np.log(z0/0.1)
    t2 = 273.15+5.0+0.8*lz+0.1*day+0.2*hour+LON*0.3
    wind = 4.0+0.3*hour-0.9*lz+LAT*0.5
    return t2, wind


def make_fake_launch(calls, *, clock=None, interrupt=None, programs=None):
    from gpuwrf.io.namelist_parser import parse_namelist
    from gpuwrf.io.netcdf_lock import Dataset

    compiled = set() if programs is None else programs

    def fake_launch(plan_path, **kwargs):
        plan = json.loads(Path(plan_path).read_text())
        calls.append(plan)
        assert "--case-surface-controls" in plan["launcher_options"]
        for member in plan["members"]:
            case = Path(member["case"])
            cfg = parse_namelist(case/"namelist.input")
            day = cfg["time_control"]["start_day"]
            fields = sc.load_controls(case/"surface_controls.npz")[0]["d01"]
            if "ctopo" in fields:  # topo drag: 10 m wind falls with CTOPO (linear), T2 barely moves
                assert np.all(fields["ctopo"] >= 0), "a member got a negative drag factor"
                ctopo = fields["ctopo"].astype(np.float64)
                response = lambda day, hour: (273.15+5.0+0.1*ctopo+0.1*day+LON*0.3,  # noqa: E731
                                              6.0+0.3*hour-1.5*ctopo+LAT*0.5)
            else:
                z0 = fields["z0mvt"].astype(np.float64)
                response = lambda day, hour: forward(z0, day, hour)  # noqa: E731
            out = Path(plan["out_root"])/member["id"]/"wrfout"
            (out/"proofs").mkdir(parents=True)
            for hour in (1, 2, 3):
                t2, wind = response(day, hour)
                stamp = f"2026-01-{day:02d}_{hour:02d}:00:00"
                with Dataset(out/f"wrfout_d01_{stamp}", "w") as ds:
                    for dim, count in (("Time", 1), ("DateStrLen", 19), ("south_north", 2), ("west_east", 3)):
                        ds.createDimension(dim, count)
                    ds.createVariable("Times", "S1", ("Time", "DateStrLen"))[:] = np.array([list(stamp)], dtype="S1")
                    ds.setncatts({"GRID_ID": 1, "SIMULATION_START_DATE": f"2026-01-{day:02d}_00:00:00", "MAP_PROJ": 3})
                    for name, values, units in (("XLAT", LAT, "degree_north"), ("XLONG", LON, "degree_east"),
                                                ("LANDMASK", np.ones(SHAPE), ""), ("T2", t2, "K"),
                                                ("U10", wind, "m s-1"), ("V10", np.zeros(SHAPE), "m s-1")):
                        v = ds.createVariable(name, "f8", ("Time", "south_north", "west_east"))
                        v[:] = values
                        v.units = units
            # The program is the configuration (physics + geometry), never the control VALUES.
            key = hashlib.sha256(json.dumps(cfg["physics"], sort_keys=True).encode()).hexdigest()[:16]
            warm = key in compiled
            compiled.add(key)
            (out/"proofs"/"nested_pipeline_run.json").write_text(json.dumps({"metadata": {"nested_aot": {"domains": {
                "d01": {"cheap_key": key, "loaded": True, "source": "aot_blob" if warm else "compiled_memory_cache"}}}}}))
            (out.parent/"receipt.json").write_text(json.dumps({"rc": 0, "wall_s": 10.0, "aot_cache_warm": warm,
                                                               "aot_cold_executables": [] if warm else ["d01"],
                                                               "vmhwm_kb": 1_000_000, "vram_peak_mib": 1000}))
        if clock is not None:
            clock[0] += 100.0
        if interrupt is not None and len(calls)-1 == interrupt:
            raise KeyboardInterrupt("GPU window ended")
        return 0
    return fake_launch


@pytest.fixture
def study_files(tmp_path):
    _land, _forcing, static = _fixture()
    cases = [fake_case(tmp_path, d) for d in (1, 2, 5)]
    basis = tmp_path/"basis.npz"
    spc.write_basis({"d01": _namelist(static)}, basis, input_dir=cases[0])
    stations = tmp_path/"stations.csv"
    stations.write_text("station_uid,lat,lon,elevation_m\n" + "".join(
        f"s{j}{i},{float(LAT[j, i])!r},{float(LON[j, i])!r},0\n" for j in range(2) for i in range(3)))
    raw = {"schema": continuous.SCHEMA, "region": "synthetic-twin", "hours": 3, "budget_case_hours": 195,
           "test_start": "2026-01-05", "configuration": "c00", "basis": "basis.npz", "stations": "stations.csv",
           "cases": [{"id": f"c{d}", "date": f"2026-01-{d:02d}", "input_dir": str(c)} for d, c in zip((1, 2, 5), cases)],
           "parameters": [{"name": "z0_forest", "control": "z0mvt", "classes": [2, 5]},
                          {"name": "z0_low", "control": "z0mvt", "classes": [7, 10, 12]}],
           "eki": {"members": 8, "iterations": 3, "seed": 11},
           "scorer": {"kind": "twin", "variables": ["t2", "wind"], "sigma": {"t2": 0.05, "wind": 0.05},
                      "truth": TRUTH, "noise_seed": 3, "tolerance_log": 0.05},
           "sampling": {"method": "nearest", "land_aware": False, "wind_frame": "earth"}}
    path = tmp_path/"study.json"
    write_json(path, raw)
    return path, raw


def rewrite(path, raw, **changes):
    data = copy.deepcopy(raw)
    data.update(changes)
    path.unlink()
    write_json(path, data)
    return path


def run_study(study, output, monkeypatch, **kwargs):
    from gpuwrf.ens import workflows
    calls = []
    monkeypatch.setattr(workflows, "launch", make_fake_launch(calls, **{k: kwargs.pop(k) for k in
                                                                         ("clock", "interrupt", "programs") if k in kwargs}))
    return continuous.run(study, output, **kwargs), calls


def test_twin_recovers_truth_with_one_program_and_cold_first(study_files, tmp_path, monkeypatch):
    path, _ = study_files
    study = continuous.load_continuous(path)
    assert study["planned_case_hours"] == 195 and study["configuration"]["status"] == "RELEASED_v0.4.0"
    assert study["configuration"]["physics"]["cu_physics"] == [0]  # released c00
    report, calls = run_study(study, tmp_path/"run", monkeypatch)
    twin = report["twin"]
    assert twin["verdict"] == "RECOVERED", twin
    for name, truth in TRUTH.items():
        assert abs(np.log(report["estimate_multipliers"][name]/truth)) <= 0.05
    # misfit falls from the prior ensemble to the last iteration; the posterior beats the prior on held-out dates
    misfits = [h["normalised_misfit"] for h in report["iterations"]]
    assert misfits[-1] < misfits[0]
    held = report["evaluation"]["test"]
    assert held["posterior"]["normalised_misfit"] < 0.25*held["prior"]["normalised_misfit"]
    # ONE program: every member's proof names the same cheap key; only the very first member compiled
    program = report["program"]
    assert program["verdict"] == "ONE_PROGRAM" and len(program["cheap_keys"]["d01"]) == 1
    assert program["cold_or_unknown_members"] == ["truth|c1"] and program["members"] == 65
    assert [len(c["members"]) for c in calls][:2] == [1, 2]  # cold wave = one member, then the rest of the truth
    assert report["scheduled_case_hours"] == 195
    # every member got its own control file, bound by content in its member manifest
    digests = set()
    for plan in calls:
        for member in plan["members"]:
            manifest = json.loads((Path(member["case"])/"member_manifest.json").read_text())
            (item,) = [i for i in manifest["inputs"] if i["name"] == "surface_controls.npz"]
            digests.add(item["sha256"])
    # 31 parameter sets: truth, four +/- probes, 3 x 8 members, prior, posterior.
    assert len(digests) == 31
    assert (tmp_path/"run"/"result.json").is_file()


def test_resume_after_pause_and_interrupt_reproduces_the_study(study_files, tmp_path, monkeypatch):
    from gpuwrf.ens import workflows
    path, _ = study_files
    study = continuous.load_continuous(path)
    whole, _ = run_study(study, tmp_path/"whole", monkeypatch)
    clock, calls, programs = [0.0], [], set()
    monkeypatch.setattr(workflows, "launch", make_fake_launch(calls, clock=clock, programs=programs))
    paused = continuous.run(study, tmp_path/"chunked", chunk_minutes=4, member_wall_s=60, clock=lambda: clock[0])
    assert paused["status"] == "PAUSED" and 0 < paused["completed_members"] < 65
    # the next window is cut by an interrupt after its second batch wrote its members (salvaged on resume)
    monkeypatch.setattr(workflows, "launch", make_fake_launch(calls, interrupt=len(calls)+1, programs=programs))
    with pytest.raises(KeyboardInterrupt):
        continuous.run(study, tmp_path/"chunked", resume=True)
    monkeypatch.setattr(workflows, "launch", make_fake_launch(calls, programs=programs))
    done = continuous.run(study, tmp_path/"chunked", resume=True)
    assert done["estimate_multipliers"] == whole["estimate_multipliers"]
    assert done["evaluation"] == whole["evaluation"] and done["iterations"] == whole["iterations"]
    ledger = json.loads((tmp_path/"chunked"/"ledger.json").read_text())
    assert any(m.get("salvaged") for m in ledger["members"].values()) and len(ledger["members"]) == 65
    assert done["program"]["verdict"] == "ONE_PROGRAM" and done["program"]["cold_or_unknown_members"] == ["truth|c1"]
    with pytest.raises(ValueError, match="same study"):  # a changed study never resumes this directory
        continuous.run({**study, "eki": {**study["eki"], "seed": 12}}, tmp_path/"chunked", resume=True)


def test_member_control_files_are_bound_and_refused_when_changed(study_files, tmp_path, monkeypatch):
    path, _ = study_files
    study = continuous.load_continuous(path)
    from gpuwrf.ens import workflows
    calls = []
    monkeypatch.setattr(workflows, "launch", make_fake_launch(calls, interrupt=2))
    with pytest.raises(KeyboardInterrupt):
        continuous.run(study, tmp_path/"run")
    target = tmp_path/"run"/"controls"/"truth.npz"
    fields, _ = sc.load_controls(target)
    fields["d01"]["z0mvt"] = fields["d01"]["z0mvt"]*np.float32(1.01)
    target.unlink()
    sc.save_controls(target, fields)
    monkeypatch.setattr(workflows, "launch", make_fake_launch(calls))
    with pytest.raises(ValueError, match="other values"):
        continuous.run(study, tmp_path/"run", resume=True)


def test_stations_scorer_runs_from_an_observation_file(study_files, tmp_path, monkeypatch):
    path, raw = study_files
    # observations = the analytic truth at the station cells (no noise): EKI recovers it from a plain CSV
    table = sc.expand_controls(_namelist(_fixture()[2]), {"z0mvt": {2: 1.6, 5: 1.6, 7: 0.7, 10: 0.7, 12: 0.7}},
                               device=False)["z0mvt"].astype(np.float64)
    lines = ["station_uid,valid_time,variable,value"]
    for day in (1, 2, 5):
        for hour in (1, 2, 3):
            t2, wind = forward(table, day, hour)
            for j in range(2):
                for i in range(3):
                    stamp = f"2026-01-{day:02d}T{hour:02d}:00:00Z"
                    lines += [f"s{j}{i},{stamp},t2,{float(t2[j, i])-273.15!r}", f"s{j}{i},{stamp},wind,{float(wind[j, i])!r}"]
    (tmp_path/"obs.csv").write_text("\n".join(lines)+"\n")
    scorer = {"kind": "stations", "variables": ["t2", "wind"], "sigma": {"t2": 0.05, "wind": 0.05},
              "observations": "obs.csv"}
    study = continuous.load_continuous(rewrite(path, raw, scorer=scorer, budget_case_hours=186))
    assert study["planned_case_hours"] == 186
    report, calls = run_study(study, tmp_path/"run", monkeypatch)
    assert "twin" not in report
    # Without truth members the prior and +/- probes precede EKI: one cold member, then nine warm members.
    assert [len(c["members"]) for c in calls][:2] == [1, 9]
    batches = json.loads((tmp_path/"run"/"ledger.json").read_text())["batches"]
    assert [b["warm"] for b in batches] == [False]+[True]*(len(batches)-1)
    assert report["program"]["cold_or_unknown_members"] == ["prior|c1"]
    for name, truth in TRUTH.items():
        assert abs(np.log(report["estimate_multipliers"][name]/truth)) <= 0.05


@pytest.mark.parametrize("change, error", [
    ({"schema": "gpuwrf.optimize.study.v1"}, "requires schema"),
    ({"budget_case_hours": 194}, "budget needs at least 195"),
    ({"parameters": [{"name": "a", "control": "nope"}]}, "unknown control"),
    ({"parameters": [{"name": "a", "control": "ctopo"}]}, "not an active control"),
    ({"parameters": [{"name": "a", "control": "z0mvt"}, {"name": "b", "control": "z0mvt", "classes": [5]}]},
     "same cells"),
    ({"parameters": [{"name": "a", "control": "z0mvt", "bounds": [1.2, 5]}]}, "bounds"),
    ({"parameters": [{"name": "a", "control": "z0mvt", "domains": ["d02"]}]}, "not an active control"),
    ({"eki": {"members": 1, "iterations": 2, "seed": 1}}, "at least 2"),
    ({"eki": {"members": 4, "iterations": 2}}, "seed"),
    ({"configuration": "c99"}, "unknown released"),
    ({"extra": 1}, "unknown study keys"),
    ({"test_start": "2026-01-01"}, "training date"),
])
def test_study_fail_closed(study_files, change, error):
    path, raw = study_files
    with pytest.raises(ValueError, match=error):
        continuous.load_continuous(rewrite(path, raw, **change))


@pytest.mark.parametrize("change, error", [
    ({"truth": {"z0_forest": 1.6}}, "one multiplier per parameter"),
    ({"truth": {"z0_forest": 9.0, "z0_low": 0.7}}, "outside its bounds"),
    ({"sigma": {"t2": 0.05}}, "one observation error per variable"),
    ({"variables": ["wind_direction"], "sigma": {"wind_direction": 5}}, "subset"),
    ({"tolerance_log": None}, "tolerance_log"),
])
def test_twin_scorer_fail_closed(study_files, change, error):
    path, raw = study_files
    scorer = {**raw["scorer"], **change}
    with pytest.raises(ValueError, match=error):
        continuous.load_continuous(rewrite(path, raw, scorer=scorer))


@pytest.mark.parametrize("kind", ["twin", "stations"])
def test_accumulated_precip_is_refused_until_interval_aware(study_files, kind):
    """Critic cr-w41-optcont 00:10Z: the reused collector keys omit interval_start (an [01,02] forecast would meet an
    [00,02] observation) and the twin CSV has no interval column -> continuous studies refuse precip_rate explicitly."""
    path, raw = study_files
    scorer = {**raw["scorer"], "variables": ["t2", "precip_rate"], "sigma": {"t2": 0.5, "precip_rate": 0.2}}
    if kind == "stations":
        scorer = {"kind": "stations", "variables": scorer["variables"], "sigma": scorer["sigma"],
                  "observations": "stations.csv"}
    with pytest.raises(ValueError, match=r"precip_rate.*not supported by continuous studies.*interval_start"):
        continuous.load_continuous(rewrite(path, raw, scorer=scorer))
    assert "precip_rate" not in continuous.VARIABLES


def test_basis_must_match_case_statics_and_runtime_tables(study_files, tmp_path):
    from gpuwrf.io.netcdf_lock import Dataset
    path, raw = study_files
    with Dataset(Path(raw["cases"][2]["input_dir"])/"wrfinput_d01", "r+") as ds:
        ds.variables["IVGTYP"][0, 0, 0] = 12
    with pytest.raises(ValueError, match="static land/terrain fields differ"):
        continuous.load_continuous(path)
    with Dataset(Path(raw["cases"][2]["input_dir"])/"wrfinput_d01", "r+") as ds:
        ds.variables["IVGTYP"][0, 0, 0] = VEG[0, 0]
    continuous.load_continuous(path)
    data = dict(np.load(tmp_path/"basis.npz"))
    # the release path holds the tables in REAL: a basis of REAL-rounded table values (consumer dtype float32) is current
    manifest = json.loads(bytes(data["manifest"]).decode())
    manifest["domains"]["d01"]["controls"]["z0mvt"]["dtype"] = "float32"
    real = dict(data, **{"d01/z0mvt": data["d01/z0mvt"].astype(np.float32).astype(np.float64),
                         "manifest": np.frombuffer(json.dumps(manifest, sort_keys=True).encode(), dtype=np.uint8)})
    assert not np.array_equal(real["d01/z0mvt"], data["d01/z0mvt"])  # really REAL-rounded (0.06 -> 0.0599999986)
    (tmp_path/"basis.npz").unlink()
    with open(tmp_path/"basis.npz", "wb") as handle:
        np.savez(handle, **real)
    continuous.load_continuous(path)
    data["d01/z0mvt"] = data["d01/z0mvt"]*1.01  # a basis built with other MPTABLE values
    (tmp_path/"basis.npz").unlink()
    with open(tmp_path/"basis.npz", "wb") as handle:
        np.savez(handle, **data)
    with pytest.raises(ValueError, match="differs from the runtime Noah-MP tables"):
        continuous.load_continuous(path)


def test_configuration_must_run_the_basis_case_physics(study_files):
    path, raw = study_files
    with pytest.raises(ValueError, match="other physics than the basis case"):
        continuous.load_continuous(rewrite(path, raw, configuration="c36"))  # released KF tuple, basis case cu=0


def test_alisios_inputs_refused_before_any_read(study_files, tmp_path, monkeypatch):
    from gpuwrf.assimilation.letkf import alisios_guard
    path, raw = study_files
    root = tmp_path/"protected"
    root.mkdir()
    monkeypatch.setattr(alisios_guard, "DATA_ROOTS", alisios_guard.DATA_ROOTS+(root.resolve(),))
    (root/"stations.csv").write_bytes((tmp_path/"stations.csv").read_bytes())
    (tmp_path/"linked.csv").symlink_to(root/"stations.csv")
    import gpuwrf.ens.workflows as workflows
    monkeypatch.setattr(workflows, "_case", lambda *a, **k: pytest.fail("case read before the refusal"))
    with pytest.raises(ValueError, match="ALISIOS|protected"):
        continuous.load_continuous(rewrite(path, raw, stations="linked.csv"))


def test_cli_refuses_a_protected_study_before_the_schema_dispatch_reads_it(study_files, tmp_path, monkeypatch, capsys):
    """Critic cr-w41-optcont 00:24Z: the CLI used to read the study (schema dispatch) before any refusal. A study
    linked into a (fake) protected root is refused by its real path before one byte of it is read."""
    import pathlib
    from gpuwrf.assimilation.letkf import alisios_guard
    from gpuwrf.cli import main
    path, _ = study_files
    root = tmp_path/"protected"
    root.mkdir()
    (root/"study.json").write_bytes(path.read_bytes())
    (tmp_path/"study_link.json").symlink_to(root/"study.json")
    monkeypatch.setattr(alisios_guard, "DATA_ROOTS", alisios_guard.DATA_ROOTS+(root.resolve(),))
    protected = str(root.resolve())
    for name in ("read_text", "read_bytes", "open"):
        original = getattr(pathlib.Path, name)

        def spy(self, *a, _original=original, **k):
            if os.path.realpath(self).startswith(protected):
                pytest.fail(f"protected study read before the refusal: {self}")
            return _original(self, *a, **k)
        monkeypatch.setattr(pathlib.Path, name, spy)
    assert main(["optimize", "plan", "--study", str(tmp_path/"study_link.json"), "--output", str(tmp_path/"p.json")]) == 2
    assert "ALISIOS" in capsys.readouterr().err and not (tmp_path/"p.json").exists()
    assert main(["optimize", "plan", "--study", str(path), "--output", str(tmp_path/"ok.json")]) == 0  # admitted path


def test_cli_plan_dispatch_and_score_refusal(study_files, tmp_path, capsys):
    from gpuwrf.cli import main
    path, _ = study_files
    assert main(["optimize", "plan", "--study", str(path), "--output", str(tmp_path/"plan.json")]) == 0
    plan = json.loads((tmp_path/"plan.json").read_text())
    assert plan["schema"] == continuous.SCHEMA and plan["planned_case_hours"] == 195
    assert main(["optimize", "score", "--study", str(path), "--output", str(tmp_path/"s.json"),
                 "--forecasts", str(path)]) == 2
    assert "continuous studies run their members" in capsys.readouterr().err


# ------------------------------------------------------------------------------------------- launcher plumbing
def test_prepared_members_get_their_own_control_file_through_the_real_scheduler(tmp_path, monkeypatch):
    """workflows.prepare links one control file per member; the actual parallel_cases launcher passes it as that
    member's --surface-controls (fake product CLI records argv); a changed control file is refused at launch."""
    import shlex
    from gpuwrf.ens.workflows import launch, prepare

    sources = []
    for day in (1, 2):
        case = fake_case(tmp_path, day)
        controls = tmp_path/f"ctl-{day}"/"surface_controls.npz"
        controls.parent.mkdir()
        sc.save_controls(controls, {"d01": {"z0mvt": np.full(SHAPE, float(day), np.float32)}})
        sources.append((case, controls))
    plan = prepare("batch", tmp_path/"prepared", cases=[c for c, _ in sources], hours=3,
                   surface_controls=[s for _, s in sources])
    assert plan["launcher_options"][-2:] == ["--case-surface-controls", "surface_controls.npz"]
    fake_cli = tmp_path/"fake_cli.py"
    fake_cli.write_text('''import json,pathlib,sys
a=sys.argv
out=pathlib.Path(a[a.index('--output-dir')+1]);out.mkdir(parents=True,exist_ok=True)
(out/'argv.json').write_text(json.dumps(a))
print(json.dumps({'all_domains_finite':True,'all_outputs_present':True}))
''')
    runner = tmp_path/"scheduler.py"
    runner.write_text(f'''import importlib.util,json,os,sys
spec=importlib.util.spec_from_file_location('parallel_cases',{str(REPO/'scripts/parallel_cases.py')!r})
pc=importlib.util.module_from_spec(spec);spec.loader.exec_module(pc)
os.environ['GPUWRF_GPU_LOCK_HELD']='1'
pc.vram_by_pid=lambda:{{}}
argv=sys.argv[1:]
argv[0:0]=['--gpu-log-interval','0','--host-gb-per-case','0.001','--min-mem-available-gb','0','--cli-json',json.dumps([sys.executable,{str(fake_cli)!r}])]
sys.exit(pc.main(argv,sizer=lambda ds,a,e:[dict(need_bytes=1024) for d in ds],gpu=lambda:(8*1024**3,8*1024**3),vram=lambda:{{}}))
''')
    wrapper = tmp_path/"run_parallel_cases.sh"
    wrapper.write_text(f"#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(runner))} \"$@\"\n")
    wrapper.chmod(0o755)
    monkeypatch.setenv("PYTHONPATH", str(REPO/"src"))
    monkeypatch.delenv("GPUWRF_GPU_LOCK_FD", raising=False)
    for key in ("XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_MEM_FRACTION",
                "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"):
        monkeypatch.delenv(key, raising=False)
    assert launch(tmp_path/"prepared"/"plan.json", launcher=wrapper) == 0
    seen = {}
    for member in plan["members"]:
        argv = json.loads((tmp_path/"prepared"/"runs"/member["id"]/"wrfout"/"argv.json").read_text())
        assert argv.count("--surface-controls") == 1, argv
        given = Path(argv[argv.index("--surface-controls")+1])
        assert given == Path(member["case"])/"surface_controls.npz"
        seen[member["id"]] = float(sc.load_controls(given)[0]["d01"]["z0mvt"][0, 0])
    assert sorted(seen.values()) == [1.0, 2.0]
    # a control file changed after preparation is refused before the scheduler starts
    sc.save_controls(tmp_path/"ctl-1"/"surface_controls.npz", {"d01": {"z0mvt": np.full(SHAPE, 9.0, np.float32)}})
    with pytest.raises(ValueError, match="shared member input changed"):
        launch(tmp_path/"prepared"/"plan.json", launcher=wrapper)


def test_launcher_refuses_missing_or_shared_control_files(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("parallel_cases", REPO/"scripts"/"parallel_cases.py")
    pc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pc)
    case = tmp_path/"case"
    case.mkdir()
    for argv in (["--out-root", "o", "--case-surface-controls", "surface_controls.npz", str(case)],
                 ["--out-root", "o", "--case-surface-controls", "../x.npz", str(case)],
                 ["--out-root", "o", str(case), "--", "--surface-controls", "x.npz"]):
        with pytest.raises(SystemExit):
            pc.parse(argv)
    (case/"surface_controls.npz").write_bytes(b"x")
    a, _ = pc.parse(["--out-root", "o", "--case-surface-controls", "surface_controls.npz", str(case)])
    assert a.case_surface_controls == "surface_controls.npz"


def test_prepare_refuses_misnamed_or_partial_control_files(tmp_path):
    from gpuwrf.ens.workflows import prepare
    cases = [fake_case(tmp_path, 1), fake_case(tmp_path, 2)]
    bad = tmp_path/"other.npz"
    bad.write_bytes(b"x")
    with pytest.raises(ValueError, match="named surface_controls.npz"):
        prepare("batch", tmp_path/"p1", cases=cases, hours=3, surface_controls=[bad, bad])
    with pytest.raises(ValueError, match="one control file per case"):
        prepare("batch", tmp_path/"p2", cases=cases, hours=3, surface_controls=[bad])


def test_program_report_flags_a_second_program_or_missing_proof(tmp_path):
    def member(name, key, warm):
        out = tmp_path/name/"wrfout"
        (out/"proofs").mkdir(parents=True)
        (out/"wrfout_d01_x").write_bytes(b"x")
        if key is not None:
            (out/"proofs"/"nested_pipeline_run.json").write_text(json.dumps(
                {"metadata": {"nested_aot": {"domains": {"d01": {"cheap_key": key}}}}}))
        return {"files": [{"path": str(out/"wrfout_d01_x")}], "aot_cache_warm": warm,
                "aot_cold_executables": [] if warm else ["d01"]}
    same = {"members": {"a|c1": member("a", "k1", False), "b|c1": member("b", "k1", True)}}
    report = continuous.program_report(same)
    assert report["verdict"] == "ONE_PROGRAM" and report["cold_or_unknown_members"] == ["a|c1"]
    two = {"members": {**same["members"], "c|c1": member("c", "k2", True)}}
    assert continuous.program_report(two)["verdict"] == "NOT_PROVEN"
    blind = {"members": {**same["members"], "d|c1": member("d", None, True)}}
    report = continuous.program_report(blind)
    assert report["verdict"] == "NOT_PROVEN" and report["members_without_proof"] == ["d|c1"]


# ------------------------------------------------------------------------------------- feasible prior support
def test_bounds_must_stay_physically_feasible_and_defaults_are_tightened(study_files):
    """z0mvt <= 5 m: forest classes (EBF 2 z0 1.1 m, mixed forest 5) cap the factor near 4.5; user bounds that reach
    beyond are refused at LOAD (never discovered when a member is expanded mid-study); default bounds are intersected."""
    path, raw = study_files
    params = [{"name": "z0_forest", "control": "z0mvt", "classes": [2, 5], "bounds": [0.2, 5.0]},
              {"name": "z0_low", "control": "z0mvt", "classes": [7, 10, 12]}]
    with pytest.raises(ValueError, match="physically invalid z0mvt"):
        continuous.load_continuous(rewrite(path, raw, parameters=params))
    params[0].pop("bounds")
    study = continuous.load_continuous(rewrite(path, raw, parameters=params))
    forest, low = study["parameters"]
    assert forest["bounds_source"] == "default" and forest["bounds"][0] == 0.2 and 4.0 < forest["bounds"][1] < 5.0
    # analytic edge 5/1.1; REAL rounding of 1.1 m x 5/1.1 exceeds 5 m, so the edge steps 1e-6 inside
    assert forest["bounds"][1] == forest["feasible_factors"][1] and 5/1.1*(1-2e-6) < forest["bounds"][1] <= 5/1.1
    assert low["bounds"] == [0.2, 5.0]
    params[0]["bounds"] = [0.2, forest["feasible_factors"][1]]  # the feasible endpoint itself is a valid user bound
    continuous.load_continuous(rewrite(path, raw, parameters=params))
    eg = [{"name": "emis", "control": "eg"}]  # EG in [0.5, 1]: both ends constrained
    (p,) = continuous.load_continuous(rewrite(path, raw, parameters=eg, scorer={**raw["scorer"], "truth": {"emis": 1.0}}))["parameters"]
    assert 0.2 < p["bounds"][0] < 1.0 < p["bounds"][1] < 1.1


def _ctopo_study(tmp_path, raw, path, **parameter):
    """Fake cases (2x3) + a ctopo basis with a valley ramp cell (CTOPO 0), a blend cell (.5) and ridge cells."""
    import dataclasses
    import jax.numpy as jnp
    from gpuwrf.io.topo_wind_static import TopoWindStatic
    sys.path.insert(0, str(REPO/"tests"))
    from test_v013_operational_smoke import _grid, _namelist as ops_namelist
    nl = ops_namelist(_grid(nz=12, ny=2, nx=3), mp_physics=0, cu_physics=0, bl_pbl_physics=1, sf_sfclay_physics=1,
                      sf_surface_physics=0, use_noahmp=False, ra_lw_physics=0, ra_sw_physics=0)
    ctopo = np.array([[0.0, 0.5, 1.0], [1.0, 2.0, 3.0]], np.float32)
    nl = dataclasses.replace(nl, topo_wind_static=TopoWindStatic(jnp.asarray(ctopo), None))
    (tmp_path/"ctopo_basis.npz").unlink(missing_ok=True)
    spc.write_basis({"d01": nl}, tmp_path/"ctopo_basis.npz", input_dir=raw["cases"][0]["input_dir"])
    return rewrite(path, raw, basis="ctopo_basis.npz", parameters=[{"name": "drag", "control": "ctopo", **parameter}],
                   eki={"members": 10, "iterations": 3, "seed": 20261017},
                   # One-sided sensitivity probe: +2 training dates x 3 h beyond the old 207 h budget.
                   scorer={**raw["scorer"], "truth": {"drag": 0.6}, "tolerance_log": 0.1}, budget_case_hours=213)


def test_ctopo_valley_cells_feasible_bounds_and_driver_regression(study_files, tmp_path, monkeypatch):
    """Critic cr-w41-optcont 00:10Z: CTOPO = 0 ramp cells allow only factors <= 1. [0.2, 5] is refused at load,
    [0.2, 1] is a valid study bound, the default prior is tightened to it, and the whole continuous driver runs the
    seed-20261017, 10-member ensemble without one invalid member (each member asserts CTOPO >= 0)."""
    path, raw = study_files
    with pytest.raises(ValueError, match=r"feasible factors are \[0, 1\]"):
        continuous.load_continuous(_ctopo_study(tmp_path, raw, path, bounds=[0.2, 5.0]))
    continuous.load_continuous(_ctopo_study(tmp_path, raw, path, bounds=[0.2, 1.0]))
    study = continuous.load_continuous(_ctopo_study(tmp_path, raw, path))
    (drag,) = study["parameters"]
    assert drag["bounds_source"] == "default" and drag["bounds"][0] == 0.2 and 1.0-1e-5 < drag["bounds"][1] <= 1.0
    report, _ = run_study(study, tmp_path/"run", monkeypatch)
    factors = np.asarray(report["ensemble_final_multipliers"])[:, 0]
    assert np.all((factors >= 0.2) & (factors <= drag["bounds"][1]))
    assert report["twin"]["verdict"] == "RECOVERED", report["twin"]
    assert report["program"]["verdict"] == "ONE_PROGRAM"
