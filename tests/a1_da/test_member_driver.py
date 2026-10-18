"""member_driver / cycle_loop orchestration (lane w41-da-cycle): no model run here.

The forecast itself is the unchanged ``gpuwrf run`` (cli.main), replaced by a recording fake; the real
in-process == separate-process bitwise proof is proofs/w41_da_cycle (Swiss d01, 2 members x 2 windows).
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwrf.assimilation.letkf import cycle_loop, member_driver
from gpuwrf.runtime import domain_tree

SWISS = Path(__file__).resolve().parents[2] / "examples" / "switzerland_d01"


@pytest.fixture
def case(tmp_path):
    out = tmp_path / "case"
    out.mkdir()
    shutil.copyfile(SWISS / "namelist.input", out / "namelist.input")
    return out


@pytest.fixture
def fake_cli(monkeypatch):
    """Record argv; write one fake history file + the run payload like the real CLI does."""
    calls = []

    def fake_main(argv):
        calls.append(list(argv))
        arg = dict(zip(argv[1::2], argv[2::2]))
        out, proofs = Path(arg["--output-dir"]), Path(arg["--proof-dir"])
        out.mkdir(parents=True, exist_ok=True)
        proofs.mkdir(parents=True, exist_ok=True)
        (out / "wrfout_d01_2023-01-15_03:00:00").write_bytes(b"history " + arg["--input-dir"].encode())
        verdict = "PIPELINE_GREEN" if "fail" not in arg["--input-dir"] else "PIPELINE_RED"
        (proofs / "nested_pipeline_run.json").write_text(json.dumps({"verdict": verdict}))
        print(json.dumps({"verdict": verdict}))
        return 0 if verdict == "PIPELINE_GREEN" else 1

    import gpuwrf.cli

    monkeypatch.setattr(gpuwrf.cli, "main", fake_main)
    yield calls
    domain_tree.set_process_aot_reuse(False)


def test_namelist_signature_ignores_time_control_only(case, tmp_path):
    other = tmp_path / "other"
    shutil.copytree(case, other)
    text = (other / "namelist.input").read_text()
    (other / "namelist.input").write_text(text.replace("start_hour                          = 0",
                                                        "start_hour                          = 3"))
    assert member_driver.namelist_signature(case) == member_driver.namelist_signature(other)
    (other / "namelist.input").write_text(text.replace("radt                                = 10",
                                                        "radt                                = 20"))
    assert member_driver.namelist_signature(case) != member_driver.namelist_signature(other)


def test_driver_runs_each_member_as_the_cli(case, tmp_path, fake_cli):
    driver = member_driver.MemberDriver(tmp_path / "scratch")
    first = driver.run("m00", case, tmp_path / "fc" / "m00", 3)
    second = driver.run("m01", case, tmp_path / "fc" / "m01", 3)
    assert [c[0] for c in fake_cli] == ["run", "run"]
    for argv, label in zip(fake_cli, ("m00", "m01")):
        arg = dict(zip(argv[1::2], argv[2::2]))
        assert arg["--input-dir"] == str(case)
        assert arg["--output-dir"] == str(tmp_path / "fc" / label)
        assert arg["--domain"] == "d01" and arg["--max-dom"] == "1" and arg["--hours"] == "3"
    # The first member may prefetch; later members reuse the resident executables.
    assert "--no-aot-prefetch" not in fake_cli[0] and "--no-aot-prefetch" in fake_cli[1]
    assert first["verdict"] == second["verdict"] == "PIPELINE_GREEN"
    assert list(second["wrfout"]) == ["wrfout_d01_2023-01-15_03:00:00"]
    assert (tmp_path / "fc" / "m01" / "member_receipt_0001.json").is_file()
    assert json.loads((tmp_path / "fc" / "m00" / "cli_stdout_0000.json").read_text())["verdict"] == "PIPELINE_GREEN"
    assert not any((tmp_path / "scratch").glob("*")), "per-member scratch must be removed"
    receipt = driver.receipt()
    assert receipt["n_runs"] == 2 and receipt["share_executables"] is True
    assert domain_tree._PROCESS_AOT_MEMO is not None


def test_driver_refuses_a_different_namelist(case, tmp_path, fake_cli):
    driver = member_driver.MemberDriver(tmp_path / "scratch")
    driver.run("m00", case, tmp_path / "fc" / "m00", 3)
    other = tmp_path / "other"
    shutil.copytree(case, other)
    text = (other / "namelist.input").read_text()
    (other / "namelist.input").write_text(text.replace("mp_physics                          = 8",
                                                        "mp_physics                          = 6"))
    with pytest.raises(ValueError, match="import-time switches are pinned"):
        driver.run("m01", other, tmp_path / "fc" / "m01", 3)
    assert len(fake_cli) == 1


def test_driver_refuses_an_environment_change(case, tmp_path, fake_cli, monkeypatch):
    driver = member_driver.MemberDriver(tmp_path / "scratch")
    driver.run("m00", case, tmp_path / "fc" / "m00", 3)
    original = os.environ.get("GPUWRF_DYN_RK_FP32")
    monkeypatch.setenv("GPUWRF_DYN_RK_FP32", "0" if original == "1" else "1")
    with pytest.raises(ValueError, match="GPUWRF_DYN_RK_FP32"):
        driver.run("m01", case, tmp_path / "fc" / "m01", 3)
    if original is None:
        monkeypatch.delenv("GPUWRF_DYN_RK_FP32")
    else:
        monkeypatch.setenv("GPUWRF_DYN_RK_FP32", original)
    # Scratch locations are rewritten by the CLI per run and are not pinned.
    monkeypatch.setenv("GPUWRF_TMPDIR", str(tmp_path / "elsewhere"))
    driver.run("m01", case, tmp_path / "fc" / "m01", 3)


def test_failed_member_is_raised_and_recorded(case, tmp_path, fake_cli):
    bad = tmp_path / "fail_case"
    shutil.copytree(case, bad)
    driver = member_driver.MemberDriver(tmp_path / "scratch")
    with pytest.raises(RuntimeError, match="verdict=PIPELINE_RED"):
        driver.run("m00", bad, tmp_path / "fc" / "m00", 3)
    assert driver.runs[0]["rc"] == 1 and driver.runs[0]["verdict"] == "PIPELINE_RED"
    with pytest.raises(ValueError, match="new member directory"):
        driver.run("m00", case, tmp_path / "fc" / "m00", 3)


def test_driver_refuses_a_process_without_the_package_bootstrap(tmp_path, monkeypatch):
    # `gpuwrf da check|analyse` skip x64/command buffers/compile cache; members there would be another program
    import gpuwrf

    monkeypatch.setattr(gpuwrf, "_OFFLINE_COMMAND", True)
    with pytest.raises(RuntimeError, match="package bootstrap"):
        member_driver.MemberDriver(tmp_path / "scratch")


def _memo_of(advance):
    cells = dict(zip(advance.__code__.co_freevars, advance.__closure__))
    return tuple(cells[k].cell_contents for k in ("_aot_calls", "_aot_attempted", "_aot_loaded_keys"))


def test_process_aot_reuse_shares_only_blob_loaded_executables():
    import jax

    tree = SimpleNamespace(domains={})
    domain_tree.set_process_aot_reuse(False)
    a, b = (_memo_of(domain_tree._operational_advance_factory(tree)) for _ in range(2))
    assert a[0] is not b[0], "default: one memo per tree"
    try:
        domain_tree.set_process_aot_reuse(True)
        calls, attempted, loaded = _memo_of(domain_tree._operational_advance_factory(tree))

        def blob_call(*args, **kwargs):  # stands for a loaded aot_call (leaf contract, treedef-tolerant)
            return args

        compiled = jax.jit(lambda x: x + 1.0).lower(1.0).compile()  # strict on its own input treedef
        calls[("d01", "blob")] = blob_call
        loaded.add(("d01", "blob"))
        calls[("d01", "compiled")] = compiled
        loaded.add(("d01", "compiled"))
        later_calls, later_attempted, later_loaded = _memo_of(domain_tree._operational_advance_factory(tree))
        assert later_calls[("d01", "blob")] is blob_call
        assert ("d01", "blob") in later_attempted and ("d01", "blob") in later_loaded
        assert ("d01", "compiled") not in later_calls and ("d01", "compiled") not in later_attempted
    finally:
        domain_tree.set_process_aot_reuse(False)
    assert domain_tree._PROCESS_AOT_MEMO is None


def test_second_tree_executes_the_first_trees_loaded_executable(monkeypatch):
    """Real advance-factory Step-C path (fake blob): a later member's tree reuses the load, a default tree does not."""
    from tests.test_aot_executable import _aot_carry, _aot_namelist, _fake_tree, _make_aot_advance_like
    from gpuwrf.runtime import aot_precompile as aot

    monkeypatch.setenv("GPUWRF_NESTED_AOT", "1")

    def no_jit(*args, **kwargs):
        raise AssertionError("jitted path used despite a loaded executable")

    monkeypatch.setattr(domain_tree, "_advance_chunk", no_jit)
    monkeypatch.setattr(domain_tree, "_advance_chunk_fori", _make_aot_advance_like())
    loads, calls = [], []

    def aot_call(carry, namelist, start, clock_base, *, n_steps, cadence):
        calls.append(n_steps)
        return ("AOT", carry)

    def fake_load(name, *args, **kwargs):
        loads.append(kwargs.get("cheap_key"))
        return aot_call

    monkeypatch.setattr(aot, "load_domain_blob", fake_load)
    try:
        domain_tree.set_process_aot_reuse(True)
        for _ in range(3):  # three members, each with its own tree and namelist object
            out = domain_tree._operational_advance_factory(_fake_tree(_aot_namelist()))(
                "d01", _aot_carry(1), start_step=2, n_steps=5)
            assert out[0] == "AOT"
        assert len(loads) == 1 and len(calls) == 3
        assert domain_tree.nested_aot_report()["domains"]["d01"]["source"] == "aot_memory_cache"
        domain_tree.set_process_aot_reuse(False)
        domain_tree._operational_advance_factory(_fake_tree(_aot_namelist()))("d01", _aot_carry(1), start_step=2,
                                                                             n_steps=5)
        assert len(loads) == 2, "default: every tree loads its own executable"
    finally:
        domain_tree.set_process_aot_reuse(False)


def _config(tmp_path, **extra):
    """A cycle-only (gpuwrf.da.cycle.v1) file; letkf arms go through _da_config."""
    cfg = {"schema": cycle_loop.CYCLE_SCHEMA, "source_case": str(SWISS), "t0": "2023-01-15T00:00:00Z",
           "window_hours": 3, "cycles": 2, "arms": {"zero_update": {"kind": "zero_update"}},
           "initial_cases": {"m00": str(tmp_path / "a"), "m01": str(tmp_path / "b")}}
    cfg.update(extra)
    path = tmp_path / "cycle.json"
    path.write_text(json.dumps(cfg))
    return path


def _da_config(tmp_path, arms, **cycle_extra):
    """A gpuwrf.da.v1 analysis config with a 'cycle' section (LETKF arms read its analysis settings)."""
    cycle = {"source_case": str(SWISS), "t0": "2023-01-15T00:00:00Z", "window_hours": 3, "cycles": 2,
             "arms": arms, "initial_cases": {"m00": str(tmp_path / "a"), "m01": str(tmp_path / "b")}}
    cycle.update(cycle_extra)
    cfg = {"schema": "gpuwrf.da.v1", "t_a": "2023-01-15T03:00:00Z",
           "ensemble": {"kind": "members", "member_glob": str(tmp_path / "*" / "wrfout_d01_*")},
           "obs": {"sources": [{"format": "igra2", "path": str(tmp_path / "sonde.txt"), "station": "SWZ00000001"}]},
           "experiments": [{"name": "A", "assimilate": ["sonde"]}, {"name": "B", "assimilate": ["sonde"]}],
           "cycle": cycle}
    path = tmp_path / "da.json"
    path.write_text(json.dumps(cfg))
    return path


def _load(path, tmp_path):
    return cycle_loop.load_config(path, out=tmp_path / "loop")


@pytest.mark.parametrize("bad", [{"window_hours": 4}, {"cycles": 0}, {"arms": {"x": {"kind": "nudge"}}},
                                 {"initial_cases": {"m00": "/a"}}, {"schema": 2}, {"t0": "2023-01-15T00:00:00"},
                                 {"source_case": "relative/case"}, {"arms": {"shared": {"kind": "zero_update"}}},
                                 {"arms": {"z": {"kind": "letkf"}}}, {"surprise": 1}])
def test_cycle_config_is_validated(tmp_path, bad):
    with pytest.raises(ValueError):
        _load(_config(tmp_path, **bad), tmp_path)


def test_letkf_arms_take_their_experiment_from_the_da_config(tmp_path):
    cfg = _load(_da_config(tmp_path, {"l": {"kind": "letkf"}, "lb": {"kind": "letkf", "experiment": "B"},
                                      "z": {"kind": "zero_update", "seam": "restart"}}), tmp_path)
    assert cfg["arms"]["l"]["experiment"] == "A" and cfg["arms"]["lb"]["experiment"] == "B"
    assert cfg["da_config"] == str((tmp_path / "da.json").resolve()) and cfg["t0"] == "2023-01-15_00:00:00"
    with pytest.raises(ValueError, match="not in"):
        _load(_da_config(tmp_path, {"l": {"kind": "letkf", "experiment": "C"}}), tmp_path)
    from gpuwrf.assimilation.letkf import config as dacfg
    with pytest.raises(dacfg.ConfigError, match="cycle"):
        _load(_da_config(tmp_path, {"l": {"kind": "letkf", "seam": "glue"}}), tmp_path)


def test_cycle_dry_run_plan(tmp_path, capsys):
    cfg = _da_config(tmp_path, {"zero_update": {"kind": "zero_update"}, "letkf": {"kind": "letkf"}})
    rc = cycle_loop._cycle_command(SimpleNamespace(config=cfg, out=tmp_path / "loop", dry_run=True))
    plan = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert plan["windows"] == [["2023-01-15_00:00:00", "2023-01-15_03:00:00"],
                               ["2023-01-15_03:00:00", "2023-01-15_06:00:00"]]
    assert plan["member_runs"] == 2 * (1 + 1 * 2), "window 0 is shared by all arms"


def test_cycle_loop_streams_and_window_zero_sharing(tmp_path, monkeypatch):
    """Window 0 runs once; every arm prepares its own next cases from the shared backgrounds."""
    runs, prepared = [], []

    class FakeDriver:
        def run(self, label, case, root, hours, **kwargs):
            runs.append((label, str(case), str(root), hours))
            root.mkdir(parents=True)
            return {"wall_seconds": 0.0, "wrfout": {}, "aot": {}}

        def receipt(self):
            return {"n_runs": len(runs)}

    def fake_prepare(cfg, arm, fc_root, labels, t_a, prep, work):
        prepared.append((arm, str(fc_root), t_a))
        return {k: prep / "cases" / k for k in labels}, {"arm": arm}

    monkeypatch.setattr(cycle_loop, "prepare_next", fake_prepare)
    cfg = _load(_da_config(tmp_path, {"zero_update": {"kind": "zero_update"}, "letkf": {"kind": "letkf"}},
                           cycles=3), tmp_path)
    receipt = cycle_loop.run_cycles(cfg, driver=FakeDriver())
    loop = tmp_path / "loop"
    assert [r[2] for r in runs[:2]] == [str(loop / "c00/shared/fc/m00"), str(loop / "c00/shared/fc/m01")]
    assert len(runs) == 2 + 2 * 2 * 2
    assert prepared[:2] == [("zero_update", str(loop / "c00/shared/fc"), "2023-01-15_03:00:00"),
                            ("letkf", str(loop / "c00/shared/fc"), "2023-01-15_03:00:00")]
    assert prepared[2:] == [("zero_update", str(loop / "c01/zero_update/fc"), "2023-01-15_06:00:00"),
                            ("letkf", str(loop / "c01/letkf/fc"), "2023-01-15_06:00:00")]
    assert [w["cycle"] for w in receipt["windows"]] == [0, 1, 2]
    assert json.loads((loop / "CYCLE_RECEIPT.json").read_text())["driver"]["n_runs"] == len(runs)


# ----------------------------------------------------------------------------- restart_inject (R5)

def _frame(path, values, time="2023-01-15_03:00:00"):
    from gpuwrf.io.netcdf_lock import Dataset
    import numpy as np

    with Dataset(str(path), "w") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("DateStrLen", 19)
        for dim, size in (("z", 3), ("zs", 4), ("y", 4), ("ys", 5), ("x", 5), ("xs", 6)):
            ds.createDimension(dim, size)
        t = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        t[0] = np.array(list(time), dtype="S1")
        dims = {"U": ("z", "y", "xs"), "V": ("z", "ys", "x"), "T": ("z", "y", "x"), "QVAPOR": ("z", "y", "x"),
                "P": ("z", "y", "x"), "PH": ("zs", "y", "x"), "MU": ("y", "x")}
        for name, d in dims.items():
            var = ds.createVariable(name, "f4", ("Time",) + d)
            var[0] = values[name]
    return path


def _values(seed=0):
    import numpy as np

    rng = np.random.default_rng(seed)
    shapes = {"U": (3, 4, 6), "V": (3, 5, 5), "T": (3, 4, 5), "QVAPOR": (3, 4, 5), "P": (3, 4, 5),
              "PH": (4, 4, 5), "MU": (4, 5)}
    out = {k: rng.normal(size=s).astype("f4") for k, s in shapes.items()}
    out["QVAPOR"] = np.abs(out["QVAPOR"]) * np.float32(1e-3)
    return out


def test_increments_zero_for_a_zero_update_and_exact_for_t(tmp_path):
    import numpy as np
    from gpuwrf.assimilation.letkf import restart_inject as ri

    bg = _values()
    b = _frame(tmp_path / "bg.nc", bg)
    inc = ri.analysis_increments(_frame(tmp_path / "zero.nc", bg), b, "2023-01-15_03:00:00")
    assert set(inc) == set(ri._LEAVES) and all(not np.any(v) for v in inc.values())
    an = dict(bg, T=bg["T"] + np.float32(1.0))
    inc = ri.analysis_increments(_frame(tmp_path / "an.nc", an), b, "2023-01-15_03:00:00")
    q = 1.0 + ri.RVOVRD * bg["QVAPOR"].astype(float)
    expect = (an["T"].astype(float) + 300.0) * q - (bg["T"].astype(float) + 300.0) * q
    np.testing.assert_allclose(inc["THM"], expect, rtol=0, atol=1e-12)
    assert not np.any(inc["U"]) and not np.any(inc["P"])
    with pytest.raises(ValueError, match="changed MU"):
        ri.analysis_increments(_frame(tmp_path / "mu.nc", dict(bg, MU=bg["MU"] + 1)), b, "2023-01-15_03:00:00")


def test_apply_keeps_stored_bits_where_the_increment_is_zero():
    import numpy as np
    from gpuwrf.assimilation.letkf.restart_inject import _apply

    old = np.array([-0.0, 1.0, 2.5], np.float32)
    new = _apply(old, np.array([0.0, 0.0, 0.25]))
    assert new.dtype == np.float32 and np.signbit(new[0]) and new[1] == 1.0 and new[2] == np.float32(2.75)
    with pytest.raises(ValueError, match="shape"):
        _apply(old, np.zeros(2))


def test_cycle_loop_restart_seam_continues_one_stream_per_member(tmp_path, monkeypatch):
    """Restart arm: whole-horizon cases, growing --hours, end checkpoints, resume from the injected generation."""
    calls, injected = [], []

    class FakeDriver:
        def run(self, label, case, root, hours, **kwargs):
            calls.append({"label": label, "case": str(case), "root": str(root), "hours": hours, **kwargs})
            root.mkdir(parents=True, exist_ok=kwargs.get("resume_checkpoint") is not None)
            return {"wall_seconds": 0.0, "wrfout": {}, "aot": {}}

        def receipt(self):
            return {"n_runs": len(calls)}

    first = []

    def fake_first(cfg, root, *, hours=None):
        first.append((str(root), cycle_loop.forcing_hours(cfg)))
        return {k: root / "cases" / k for k in ("m00", "m01")}, {}

    def fake_prepare(cfg, arm, fc_root, labels, t_a, prep, work):
        return {k: prep / "cases" / k for k in labels}, {"arm": arm, "fc_root": str(fc_root),
                                                         "analysis_wrfinputs": {k: f"an-{k}" for k in labels}}

    def fake_inject(out, arm, fc_root, labels, t_a, arm_root, wrfinputs):
        assert wrfinputs == {"m00": "an-m00", "m01": "an-m01"}
        injected.append((arm, str(fc_root), t_a))
        return {"members": {k: {"generation": str(arm_root / f"gen-{k}")} for k in labels}}

    monkeypatch.setattr(cycle_loop, "first_window_cases", fake_first)
    monkeypatch.setattr(cycle_loop, "prepare_next", fake_prepare)
    monkeypatch.setattr(cycle_loop, "inject_next", fake_inject)
    cfg = _load(_da_config(tmp_path, {"zr": {"kind": "letkf", "seam": "restart"}}), tmp_path)
    cycle_loop.run_cycles(cfg, driver=FakeDriver())
    loop = tmp_path / "loop"
    assert first == [(str(loop / "c00/shared"), 6)], "one whole-horizon case per member"
    assert [(c["root"], c["hours"]) for c in calls] == [(str(loop / "streams/zr/m00"), 3), (str(loop / "streams/zr/m01"), 3),
                                                       (str(loop / "streams/zr/m00"), 6), (str(loop / "streams/zr/m01"), 6)]
    assert all(c["checkpoint_dir"] == loop / "checkpoints/zr" / c["label"] for c in calls)
    assert all(c["checkpoint_interval_steps"] == 600 for c in calls), "Swiss dt 18 s: 3 h = 600 root steps"
    assert [c["resume_checkpoint"] for c in calls[:2]] == [None, None] and not calls[0]["extend_run"]
    assert [str(c["resume_checkpoint"]) for c in calls[2:]] == [str(loop / "c01/zr/gen-m00"), str(loop / "c01/zr/gen-m01")]
    assert all(c["extend_run"] for c in calls[2:])
    assert injected == [("zr", str(loop / "streams/zr"), "2023-01-15_03:00:00")]


def test_forcing_hours_give_every_case_one_record_count(tmp_path):
    cfg = _load(_config(tmp_path, cycles=3), tmp_path)
    assert cycle_loop.forcing_hours(cfg) == 3
    cfg = _load(_config(tmp_path, cycles=3, arms={"z": {"kind": "zero_update"},
                                                  "r": {"kind": "zero_update", "seam": "restart"}}), tmp_path)
    assert cycle_loop.forcing_hours(cfg) == 9, "restart stream keeps its first case for the whole horizon"
    with pytest.raises(ValueError, match="cycles"):
        _load(_config(tmp_path, cycles=3, forcing_hours=6, arms={"r": {"kind": "zero_update", "seam": "restart"}}),
              tmp_path)
    with pytest.raises(ValueError, match="multiple of 3"):
        _load(_config(tmp_path, forcing_hours=4), tmp_path)


def test_window_zero_runs_once_with_a_restart_arm_present(tmp_path, monkeypatch):
    """File arms start window 1 from the backgrounds of the first restart arm's checkpointed window 0."""
    calls, prepared = [], []

    class FakeDriver:
        def run(self, label, case, root, hours, **kwargs):
            calls.append((str(root), hours, kwargs.get("checkpoint_dir") is not None))
            root.mkdir(parents=True, exist_ok=True)
            (root / f"wrfout_d01_2023-01-15_0{hours}:00:00").write_bytes(b"frame")
            return {"wall_seconds": 0.0, "wrfout": {}, "aot": {}}

        def receipt(self):
            return {"n_runs": len(calls)}

    monkeypatch.setattr(cycle_loop, "first_window_cases",
                        lambda cfg, root, hours=None: ({k: root / "cases" / k for k in ("m00", "m01")}, {}))

    def fake_prepare(cfg, arm, fc_root, labels, t_a, prep, work):
        prepared.append((arm, str(fc_root)))
        return {k: prep / "cases" / k for k in labels}, {"analysis_wrfinputs": {}}

    monkeypatch.setattr(cycle_loop, "prepare_next", fake_prepare)
    monkeypatch.setattr(cycle_loop, "inject_next", lambda out, arm, fc, labels, t_a, root, wrfinputs:
                        {"members": {k: {"generation": str(root / k)} for k in labels}})
    arms = {"zf": {"kind": "zero_update"}, "zr": {"kind": "zero_update", "seam": "restart"}}
    cfg = _load(_config(tmp_path, arms=arms), tmp_path)
    cycle_loop.run_cycles(cfg, driver=FakeDriver())
    loop = tmp_path / "loop"
    assert calls[:2] == [(str(loop / "streams/zr/m00"), 3, True), (str(loop / "streams/zr/m01"), 3, True)]
    assert len(calls) == 2 + 2 * 2, "window 0 once, then both arms"
    assert prepared == [("zf", str(loop / "streams/zr"))], "the zero restart arm needs no analysis step"
    assert cycle_loop.plan(cfg)["member_runs"] == 2 * (1 + 2)


def test_zero_update_restart_arm_injects_from_the_background_frame(tmp_path, monkeypatch):
    """No analysis step for a zero-update restart arm: the 'analysis wrfinput' is the background frame."""
    injected = []

    class FakeDriver:
        def run(self, label, case, root, hours, **kwargs):
            root.mkdir(parents=True, exist_ok=True)
            (root / f"wrfout_d01_2023-01-15_0{hours}:00:00").write_bytes(b"frame")
            return {"wall_seconds": 0.0, "wrfout": {}, "aot": {}}

        def receipt(self):
            return {}

    monkeypatch.setattr(cycle_loop, "first_window_cases",
                        lambda cfg, root, hours=None: ({k: root / "cases" / k for k in ("m00", "m01")}, {}))
    monkeypatch.setattr(cycle_loop, "prepare_next", lambda *a, **k: pytest.fail("zero restart arm needs no analysis"))
    monkeypatch.setattr(cycle_loop, "inject_next", lambda out, arm, fc, labels, t_a, root, wrfinputs:
                        injected.append(dict(wrfinputs)) or {"members": {k: {"generation": "g"} for k in labels}})
    cfg = _load(_config(tmp_path, arms={"zr": {"kind": "zero_update", "seam": "restart"}}), tmp_path)
    cycle_loop.run_cycles(cfg, driver=FakeDriver())
    loop = tmp_path / "loop"
    assert injected == [{k: str(loop / "streams/zr" / k / "wrfout_d01_2023-01-15_03:00:00") for k in ("m00", "m01")}]


# --- review cr-w41-da-cycle 15:57Z: canonical member mapping, one-component labels, ALISIOS inputs fail closed ---

# Synthetic, non-date path under an ALISIOS data root: it does not exist, so nothing there is listed or opened.
PROTECTED = "<DATA_ROOT>/alisios/__w41_da_cycle_test__"


@pytest.fixture
def no_alisios_reads(monkeypatch):
    """Fail (never open) on any Python-level open of a protected path, so even a mutated refusal reads nothing."""
    import builtins

    from gpuwrf.assimilation.letkf.alisios_guard import is_alisios_data

    def guarded(real):
        def wrapper(target, *args, **kwargs):
            if isinstance(target, (str, os.PathLike)) and is_alisios_data(target):
                pytest.fail(f"opened protected path {target}")
            return real(target, *args, **kwargs)
        return wrapper

    monkeypatch.setattr(builtins, "open", guarded(builtins.open))
    real_read_text = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: guarded(real_read_text)(self, *a, **k))


def _manifest(path, members):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"members": members}))
    return path


def test_analyses_map_to_loop_members_by_their_background_frame(tmp_path):
    frames = {"m01": str(tmp_path / "fc/m01/frame"), "m00": str(tmp_path / "fc/m00/frame")}   # reversed input order
    api = {"m00": {"case": "/an/c0", "wrfinput": "/an/w0", "background_wrfout": frames["m01"]},   # API: input order
           "m01": {"case": "/an/c1", "wrfinput": "/an/w1", "background_wrfout": frames["m00"]},
           "ctrl": {"case": "/an/ctrl", "wrfinput": "/an/wc"}}
    cases, inputs = cycle_loop.manifest_members(_manifest(tmp_path / "a.json", api), frames)
    assert cases == {"m01": Path("/an/c0"), "m00": Path("/an/c1")}
    assert inputs == {"m01": Path("/an/w0"), "m00": Path("/an/w1")}
    arbitrary = {"zeta": frames["m01"], "alpha": frames["m00"]}
    cases, _ = cycle_loop.manifest_members(_manifest(tmp_path / "b.json", api), arbitrary)
    assert cases == {"zeta": Path("/an/c0"), "alpha": Path("/an/c1")}
    bad = [({k: dict(v, background_wrfout=frames["m01"]) for k, v in api.items() if k != "ctrl"}, "share"),
           ({"m00": api["m00"]}, "no analysed member"),
           ({**api, "m02": {"case": "/x", "wrfinput": "/y", "background_wrfout": "/elsewhere"}}, "no loop member"),
           ({"m00": {"case": "/x", "wrfinput": "/y"}, "m01": api["m01"]}, "background_wrfout")]
    for k, (members, match) in enumerate(bad):
        with pytest.raises(RuntimeError, match=match):
            cycle_loop.manifest_members(_manifest(tmp_path / f"bad{k}.json", members), frames)


@pytest.mark.parametrize("labels", [["m01", "m00"], ["zeta", "alpha"]])
def test_every_member_keeps_its_own_analysis_through_file_and_restart_seams(tmp_path, monkeypatch, labels):
    """Real prepare_next/manifest_members; the CPU child is replaced by the API's naming (m00.. = input order)."""
    runs, injected, api_background = [], [], {}

    class FakeDriver:
        def run(self, label, case, root, hours, **kwargs):
            runs.append((label, str(case), str(root)))
            root.mkdir(parents=True, exist_ok=True)
            (root / "wrfout_d01_2023-01-15_03:00:00").write_bytes(label.encode())
            return {"wall_seconds": 0.0, "wrfout": {}, "aot": {}}

        def receipt(self):
            return {}

    def fake_child(argv, log):
        assert argv[2] == "letkf-step", argv
        spec = json.loads(Path(argv[3]).read_text())
        analysis = Path(spec["out"])
        members = {}
        for i, frame in enumerate(spec["member_files"]):
            api = f"m{i:02d}"
            case, wrfinput = analysis / "members_A/cases" / api, analysis / "members_A/members" / api / "wrfinput_d01"
            api_background[str(case)] = api_background[str(wrfinput)] = frame
            members[api] = {"case": str(case), "wrfinput": str(wrfinput), "background_wrfout": frame}
        manifest = _manifest(analysis / "members_A/ENSEMBLE_MANIFEST.json", members)
        (analysis / "cycle_step.json").write_text(json.dumps({"manifest": str(manifest)}))
        return {"argv": argv, "rc": 0}

    monkeypatch.setattr(cycle_loop, "first_window_cases",
                        lambda cfg, root, hours=None: ({k: root / "cases" / k for k in labels}, {}))
    monkeypatch.setattr(cycle_loop, "_cpu_child", fake_child)
    monkeypatch.setattr(cycle_loop, "inject_next", lambda out, arm, fc, labs, t_a, root, wrfinputs:
                        injected.append(dict(wrfinputs)) or {"members": {k: {"generation": f"g-{k}"} for k in labs}})
    initial = {label: str(tmp_path / label) for label in labels}
    cfg = _load(_da_config(tmp_path, {"lr": {"kind": "letkf", "seam": "restart"}, "lf": {"kind": "letkf"}},
                           initial_cases=initial), tmp_path)
    cycle_loop.run_cycles(cfg, driver=FakeDriver())
    loop = tmp_path / "loop"
    own = lambda label: str(loop / "streams/lr" / label / "wrfout_d01_2023-01-15_03:00:00")  # noqa: E731
    file_seam = [(label, case) for label, case, root in runs if "/c01/lf/fc/" in root]
    assert sorted(label for label, _ in file_seam) == sorted(labels)
    assert all(api_background[case] == own(label) for label, case in file_seam), file_seam
    assert len(injected) == 1 and sorted(injected[0]) == sorted(labels)
    assert all(api_background[str(w)] == own(label) for label, w in injected[0].items()), injected


@pytest.mark.parametrize("label", ["/outside/new-member", "../escape", "..", ".", "", "a/b", "-x", "x" * 65])
def test_member_and_arm_labels_are_one_path_component(tmp_path, fake_cli, case, label):
    for key in ("initial_cases", "initial_members"):
        with pytest.raises(ValueError, match="path component"):
            cycle_loop.validate_config_section({
                "source_case": "/case", "t0": "2023-01-15T00:00:00Z", "window_hours": 3, "cycles": 2,
                "arms": {"zero": {"kind": "zero_update"}}, key: {label: "/member1", "m01": "/member2"}})
    if label:
        with pytest.raises(ValueError, match="path component"):
            cycle_loop.validate_config_section({
                "source_case": "/case", "t0": "2023-01-15T00:00:00Z", "window_hours": 3, "cycles": 2,
                "arms": {label: {"kind": "zero_update"}}, "initial_cases": {"m00": "/a", "m01": "/b"}})
    manifest = _manifest(tmp_path / "members.json", {label: {"case": str(case)}, "m01": {"case": str(case)}})
    with pytest.raises(ValueError, match="path component"):
        member_driver.members_from_manifest(manifest)
    with pytest.raises(ValueError, match="path component"):
        member_driver.MemberDriver(tmp_path / "scratch").run(label, case, tmp_path / "fc" / "m", 3)
    assert fake_cli == [], "refused before any member run"


def test_first_window_refuses_alisios_inputs_before_reading_anything(tmp_path, monkeypatch, no_alisios_reads):
    from gpuwrf.assimilation.letkf import cycle_case

    listed = []
    real_iterdir = Path.iterdir
    monkeypatch.setattr(Path, "iterdir", lambda self: listed.append(str(self)) or real_iterdir(self))
    monkeypatch.setattr(cycle_case, "write_cycle_case", lambda *a, **k: pytest.fail("protected input reached writer"))
    linked = tmp_path / "linked_case"
    linked.mkdir()
    (linked / "wrfbdy_d01").symlink_to(PROTECTED + "/wrfbdy_d01")
    for source, member in ((str(tmp_path), PROTECTED + "/member/wrfinput_d01"),
                           (PROTECTED + "/case", str(tmp_path / "m00/wrfinput_d01")),
                           (str(linked), str(tmp_path / "m00/wrfinput_d01"))):
        spec = tmp_path / "first_window_spec.json"
        spec.write_text(json.dumps({"source_case": source, "members": {"m00": member}, "t_a": "2023-01-15_00:00:00",
                                    "hours": 3, "out": str(tmp_path / "out")}))
        with pytest.raises(member_driver.ProtectedInputError, match="ALISIOS"):
            cycle_loop.first_window(spec)
        assert not (tmp_path / "out").exists()
    assert not any(p.startswith("<DATA_ROOT>/alisios") for p in listed), listed


def test_zero_update_child_refuses_alisios_inputs_before_reading(tmp_path, monkeypatch, no_alisios_reads):
    from gpuwrf.assimilation.letkf import state_io

    monkeypatch.setattr(state_io, "read_member_frames", lambda *a, **k: pytest.fail("protected frames read"))
    spec = tmp_path / "zero.json"
    spec.write_text(json.dumps({"members": {"m00": str(tmp_path / "f0"), "m01": str(tmp_path / "f1")},
                                "t_a": "2023-01-15_03:00:00", "template_wrfinput": PROTECTED + "/wrfinput_d01",
                                "source_case": str(SWISS), "hours": 3, "out": str(tmp_path / "prep")}))
    with pytest.raises(member_driver.ProtectedInputError):
        cycle_loop.zero_update(spec)


def test_cycle_loop_refuses_alisios_inputs_before_writing(tmp_path, monkeypatch, no_alisios_reads):
    monkeypatch.setattr(cycle_loop, "first_window_cases", lambda *a, **k: pytest.fail("first window started"))
    for extra in ({"initial_members": {"m00": PROTECTED + "/m00/wrfinput_d01", "m01": str(tmp_path / "w1")}},
                  {"initial_cases": {"m00": str(tmp_path / "a"), "m01": PROTECTED + "/m01"}},
                  {"source_case": PROTECTED + "/case"}, {"template_wrfinput": PROTECTED + "/wrfinput_d01"}):
        path = _config(tmp_path, **extra)
        if "initial_members" in extra:
            raw = json.loads(path.read_text())
            raw.pop("initial_cases")
            path.write_text(json.dumps(raw))
        cfg = _load(path, tmp_path)
        with pytest.raises(member_driver.ProtectedInputError):
            cycle_loop.run_cycles(cfg, driver=SimpleNamespace(run=lambda *a, **k: pytest.fail("member run")))
        assert not (tmp_path / "loop").exists()
    with pytest.raises(member_driver.ProtectedInputError):
        cycle_loop.load_config(Path(PROTECTED) / "cycle.json")


def test_member_driver_refuses_alisios_cases_and_manifests(tmp_path, fake_cli, case, no_alisios_reads):
    (case / "wrfinput_d01").symlink_to(PROTECTED + "/wrfinput_d01")
    driver = member_driver.MemberDriver(tmp_path / "scratch")
    with pytest.raises(member_driver.ProtectedInputError):
        driver.run("m00", case, tmp_path / "fc" / "m00", 3)
    with pytest.raises(member_driver.ProtectedInputError):
        member_driver.members_from_manifest(Path(PROTECTED) / "ENSEMBLE_MANIFEST.json")
    manifest = _manifest(tmp_path / "members.json", {"m00": {"case": PROTECTED + "/m00"}, "m01": {"case": str(case)}})
    with pytest.raises(member_driver.ProtectedInputError):
        member_driver.members_from_manifest(manifest)
    assert fake_cli == [] and driver.runs == []


# --- review cr-w41-da-cycle delta 16:20Z: every file name the run resolves (namelist keys, links, environment) ---

def _with_time_control(case, *lines):
    text = (case / "namelist.input").read_text()
    assert text.count("&time_control") == 1
    (case / "namelist.input").write_text(text.replace("&time_control", "&time_control\n" + "\n".join(lines), 1))


@pytest.mark.parametrize("lines,link", [
    ((f" auxinput4_inname = '{PROTECTED}/wrflowinp_d<domain>',", " io_form_auxinput4 = 2,"), None),  # critic repro
    ((f" gfdda_inname = '{PROTECTED}/wrffdda_d<domain>',",), None),                                   # another key
    ((f" auxinput4_inname = '{PROTECTED}/wrflowinp_<date>',",), None),                                # date template
    ((f" some_future_file = \"{PROTECTED}/x.nc\",",), None),                                         # any key
    ((" sgfdda_inname = 'sub/sgfdda_d<domain>',",), "sub"),                                         # relative via link
    ((" sgfdda_inname = '../outside_link/sgfdda_d<domain>',",), "../outside_link"),   # leaves the case dir: run_dir/name
])
def test_member_run_refuses_namelist_file_names_into_alisios(tmp_path, fake_cli, case, no_alisios_reads, monkeypatch,
                                                            lines, link):
    from gpuwrf.io import lower_boundary

    monkeypatch.setattr(lower_boundary, "Dataset", lambda *a, **k: pytest.fail("protected lower boundary opened"))
    _with_time_control(case, *lines)
    if link:
        (case / link).symlink_to(PROTECTED)
    driver = member_driver.MemberDriver(tmp_path / "scratch")
    with pytest.raises(member_driver.ProtectedInputError, match="ALISIOS"):
        driver.run("m00", case, tmp_path / "fc" / "m00", 3)
    assert fake_cli == [] and driver.runs == []


def test_member_run_refuses_nested_links_environment_paths_and_output_roots(tmp_path, fake_cli, case,
                                                                            no_alisios_reads, monkeypatch):
    driver = member_driver.MemberDriver(tmp_path / "scratch")
    driver.run("ok", case, tmp_path / "fc" / "ok", 3)                    # the plain Swiss case passes
    nested = case / "tables" / "deeper"
    nested.mkdir(parents=True)
    (nested / "LANDUSE.TBL").symlink_to(PROTECTED + "/LANDUSE.TBL")
    with pytest.raises(member_driver.ProtectedInputError):
        driver.run("m00", case, tmp_path / "fc" / "m00", 3)
    (nested / "LANDUSE.TBL").unlink()
    for key, value in (("GPUWRF_TEST_INPUT_DIR", PROTECTED), ("XLA_FLAGS", f"--xla_dump_to={PROTECTED}/dump")):
        monkeypatch.setenv(key, value)
        with pytest.raises(member_driver.ProtectedInputError):
            driver.run("m01", case, tmp_path / "fc" / "m01", 3)
        monkeypatch.delenv(key)
    with pytest.raises(member_driver.ProtectedInputError):
        driver.run("m02", case, Path(PROTECTED) / "fc" / "m02", 3)
    assert len(fake_cli) == 1


def test_first_window_refuses_a_source_namelist_naming_alisios_files(tmp_path, monkeypatch, case, no_alisios_reads):
    from gpuwrf.assimilation.letkf import cycle_case

    monkeypatch.setattr(cycle_case, "write_cycle_case", lambda *a, **k: pytest.fail("protected input reached writer"))
    _with_time_control(case, f" auxinput4_inname = '{PROTECTED}/wrflowinp_d<domain>',")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"source_case": str(case), "members": {"m00": str(tmp_path / "w0")},
                                "t_a": "2023-01-15_00:00:00", "hours": 3, "out": str(tmp_path / "out")}))
    with pytest.raises(member_driver.ProtectedInputError):
        cycle_loop.first_window(spec)
    member_driver.refuse_protected_inputs("swiss example", [SWISS])     # no false positive on a real case


# --- review cr-w41-da-cycle 16:45Z: files under runtime roots (env roots, resolver defaults, table/cache dirs) ---

@pytest.fixture
def fresh_runtime_scan(monkeypatch):
    monkeypatch.setattr(member_driver, "_RUNTIME_SCANS", {})


def test_member_run_refuses_a_protected_table_under_an_environment_root(tmp_path, fake_cli, case, no_alisios_reads,
                                                                        fresh_runtime_scan, monkeypatch):
    """Critic repro: GPUWRF_WRF_ROOT=<mirror>, <mirror>/run/MPTABLE.TBL -> protected: refused before the forecast."""
    from gpuwrf.config import paths

    mirror = tmp_path / "mirror"
    (mirror / "run").mkdir(parents=True)
    (mirror / "run" / "MPTABLE.TBL").symlink_to(PROTECTED + "/MPTABLE.TBL")
    monkeypatch.setenv("GPUWRF_WRF_ROOT", str(mirror))
    assert paths.wrf_run_dir() == mirror / "run"
    with pytest.raises(member_driver.ProtectedInputError, match="MPTABLE"):
        member_driver.MemberDriver(tmp_path / "scratch").run("m00", case, tmp_path / "fc" / "m00", 3)
    assert fake_cli == []


@pytest.mark.parametrize("where", ["wrf_run_dir", "tables_env", "xla_flags"])
def test_member_run_refuses_links_under_every_runtime_root(tmp_path, fake_cli, case, no_alisios_reads,
                                                           fresh_runtime_scan, monkeypatch, where):
    from gpuwrf.config import paths

    root = tmp_path / "root"
    (root / "a" / "b").mkdir(parents=True)
    (root / "a" / "b" / "VEGPARM.TBL").symlink_to(PROTECTED + "/VEGPARM.TBL")
    if where == "wrf_run_dir":                       # the resolver's default/override, not an environment name
        monkeypatch.setattr(paths, "wrf_run_dir", lambda: root)
    elif where == "tables_env":
        monkeypatch.setenv("GPUWRF_TABLES_DIR", str(root))
    else:
        monkeypatch.setenv("XLA_FLAGS", f"--xla_gpu_per_fusion_autotune_cache_dir={root} --xla_foo=1")
    with pytest.raises(member_driver.ProtectedInputError, match="VEGPARM"):
        member_driver.MemberDriver(tmp_path / "scratch").run("m00", case, tmp_path / "fc" / "m00", 3)
    assert fake_cli == []


@pytest.mark.parametrize("swallow", [False, True])
def test_netcdf_backstop_refuses_what_no_preflight_can_list(tmp_path, monkeypatch, case, swallow):
    """A reader that builds a protected name at run time (no file, link or namelist names it beforehand)."""
    import gpuwrf.cli

    target = PROTECTED + "/built_at_run_time.nc"

    def reader_cli(argv):
        arg = dict(zip(argv[1::2], argv[2::2]))
        Path(arg["--proof-dir"]).mkdir(parents=True, exist_ok=True)
        try:
            from gpuwrf.io.netcdf_lock import Dataset

            with Dataset(target):
                pass
        except member_driver.ProtectedInputError:
            if not swallow:
                raise
        (Path(arg["--proof-dir"]) / "nested_pipeline_run.json").write_text(json.dumps({"verdict": "PIPELINE_GREEN"}))
        return 0

    monkeypatch.setattr(gpuwrf.cli, "main", reader_cli)
    driver = member_driver.MemberDriver(tmp_path / "scratch")
    try:
        with pytest.raises(member_driver.ProtectedInputError):
            driver.run("m00", case, tmp_path / "fc" / "m00", 3)
        assert driver.runs[0]["error"]["refused_opens"] == [target]
        assert member_driver._OPEN_SPY["refused"] is None, "backstop inactive outside a run"
    finally:
        domain_tree.set_process_aot_reuse(False)


def test_python_open_backstop_in_a_process_without_the_test_guard():
    """The suite's own conftest blocks ALISIOS opens first; a clean child shows the driver's hook alone refuses
    builtins.open, os.open and netCDF (through gpuwrf.io.netcdf_lock) before any system call (the synthetic path
    does not exist either way)."""
    import subprocess
    import sys

    code = (
        "import json, os, sys\n"
        "from gpuwrf.assimilation.letkf import member_driver as md\n"
        "from gpuwrf.io.netcdf_lock import Dataset\n"
        "out = {}\n"
        "with md.refuse_protected_opens() as refused:\n"
        "    for name, opener in (('open', lambda p: open(p)), ('os.open', lambda p: os.open(p, os.O_RDONLY)),\n"
        "                         ('netcdf', lambda p: Dataset(p).__enter__())):\n"
        "        try:\n"
        "            opener(sys.argv[1]); out[name] = 'OPENED'\n"
        "        except md.ProtectedInputError:\n"
        "            out[name] = 'refused'\n"
        "        except OSError as exc:\n"
        "            out[name] = type(exc).__name__\n"
        "out['refused'] = refused; out['inactive_after'] = md._OPEN_SPY['refused'] is None\n"
        "print(json.dumps(out))\n")
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0", PYTHONPATH=str(root / "src"))
    proc = subprocess.run([sys.executable, "-c", code, PROTECTED + "/built_at_run_time.nc"], capture_output=True,
                          text=True, env=env, cwd=str(root), timeout=300)
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result == {"open": "refused", "os.open": "refused", "netcdf": "refused",
                      "refused": [PROTECTED + "/built_at_run_time.nc"] * 3,
                      "inactive_after": True}, proc.stderr
