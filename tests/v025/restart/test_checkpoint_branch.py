"""Branch seam/identity tests. These are structural CPU evidence, not WRF fidelity."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from gpuwrf.runtime.checkpoint_branch import (
    apply_overrides, branch_identity, digest, load_manifest, resumed_identity, validate_paths,
)
from gpuwrf.runtime.run_window import aligned_steps, explicit_window


@pytest.fixture
def parent(tmp_path):
    identity = {"source_sha256": "source", "inputs": {"wrfinput_d01": "input"},
                "output_dir": str(tmp_path / "parent"), "namelists": {"d01": "static"},
                "surface_controls": {"d01": "a" * 64, "d02": "a" * 64}}
    manifest = {"branch_schema_version": 1, "identity": identity, "sha256": "payload",
                "domains": ["d01", "d02"], "own_steps": {"d01": 10, "d02": 30}}
    generation = tmp_path / "store" / "generation-1"
    generation.mkdir(parents=True)
    (generation / "manifest.json").write_text(json.dumps(manifest))
    return generation, manifest, {**identity, "output_dir": str(tmp_path / "child")}


def test_empty_manifest_identity_and_parent_chain(parent):
    generation, manifest, current = parent
    changes = {"schema_version": 1}
    child = branch_identity(manifest, current, changes, generation)
    assert child["branch"]["parent_identity"] == manifest["identity"]
    assert child["branch"]["manifest_sha256"] == digest(changes)
    path = generation.parent / "generation-2"
    path.mkdir()
    child_manifest = {**manifest, "identity": child}
    (path / "manifest.json").write_text(json.dumps(child_manifest))
    assert resumed_identity(current, path) == child
    grandchild = branch_identity(child_manifest, {**current, "output_dir": current["output_dir"] + "2"}, changes, path)
    assert grandchild["branch"]["parent_identity"] == child
    child["branch"]["manifest_sha256"] = "bad"
    (path / "manifest.json").write_text(json.dumps({**manifest, "identity": child}))
    with pytest.raises(ValueError, match="provenance"):
        resumed_identity(current, path)


@pytest.mark.parametrize("key", ["source_sha256", "inputs", "namelists", "aot_environment", "run_start"])
def test_undeclared_changes_refuse(parent, key):
    generation, manifest, current = parent
    with pytest.raises(ValueError, match="undeclared"):
        branch_identity(manifest, {**current, key: "changed"}, {"schema_version": 1}, generation)


def test_old_schema_and_same_stream_refuse(parent):
    generation, manifest, current = parent
    with pytest.raises(ValueError, match="E78"):
        branch_identity({k: v for k, v in manifest.items() if k != "branch_schema_version"}, current, {}, generation)
    with pytest.raises(ValueError, match="new output"):
        branch_identity(manifest, manifest["identity"], {}, generation)


def test_parameter_changes_are_exactly_declared(parent):
    generation, manifest, current = parent
    changes = {"schema_version": 1, "parameters": {"surface_controls": {"d01": "b" * 64}}}
    candidate = {**current, "surface_controls": {"d01": "b" * 64, "d02": "a" * 64}}
    assert branch_identity(manifest, candidate, changes, generation)["surface_controls"] == candidate["surface_controls"]
    with pytest.raises(ValueError, match="undeclared"):
        branch_identity(manifest, {**candidate, "surface_controls": dict.fromkeys(("d01", "d02"), "b" * 64)}, changes, generation)
    with pytest.raises(ValueError, match="already be enabled"):
        branch_identity(manifest, candidate, {"parameters": {"mynn_parameters": {"d01": "b" * 64}}}, generation)


def snapshot():
    return {"carries": {name: {"state_fields": {"qv": np.array([.01, .02], np.float32),
                "theta": np.array([300, 301], np.float32), "u": np.array([-0., 3.], np.float32),
                "p_perturbation": np.array([1, 2], np.float32), "p_total": np.array([90001, 90002], np.float32)},
                "noahmp_land_fields": {"smois": np.array([.2, .3], np.float32)},
                "scratch_fields": {"held": np.array([9.], np.float32)}} for name in ("d01", "d02", "d03")},
            "own_steps": {"d01": 10, "d02": 30, "d03": 90},
            "driver_state": {"outputs": {"d01": ["parent-file"]}, "history_stream": "parent", "forecast_hours": 3}}


def arrays(tree, prefix=()):
    if isinstance(tree, dict):
        for key, val in tree.items():
            yield from arrays(val, prefix + (key,))
    elif isinstance(tree, np.ndarray):
        yield prefix, tree


def test_null_branch_payload_is_byte_exact_and_detaches_output():
    old = snapshot()
    validated = []
    new, report = apply_overrides(old, {"schema_version": 1}, validate=lambda *args: validated.append(args[0]))
    assert validated == ["d01", "d02", "d03"]
    for (key, a), (key2, b) in zip(arrays(old["carries"]), arrays(new["carries"]), strict=True):
        assert key == key2 and a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    assert new["driver_state"]["outputs"] == {"d01": [], "d02": [], "d03": []}
    assert "history_stream" not in new["driver_state"]
    assert new["driver_state"]["branch_start_steps"] == old["own_steps"]
    assert old["driver_state"]["history_stream"] == "parent"


def test_all_nests_only_declared_arrays_change():
    old = snapshot()
    changes = {"domains": {name: {"state": {"qv": {"multiply": .5}}, "land": {"smois": {"add": .01}}}
                           for name in old["carries"]}}
    new, report = apply_overrides(old, changes, validate=lambda *args: {"passed": True})
    for (key, a), (_, b) in zip(arrays(old["carries"]), arrays(new["carries"]), strict=True):
        if key[-1] in {"qv", "smois"}:
            assert a.tobytes() != b.tobytes()
        else:
            assert a.tobytes() == b.tobytes(), key
    assert all(counts["state.qv"] == counts["land.smois"] == 2 for counts in report["fields"].values())


def test_pressure_pair_and_signed_zero_retained():
    old = snapshot()
    new, _ = apply_overrides(old, {"domains": {"d02": {"state": {"p_perturbation": {"add": 2.}, "u": {"add": 0.}}}}},
                             validate=lambda *args: {})
    state = new["carries"]["d02"]["state_fields"]
    np.testing.assert_array_equal(state["p_perturbation"], [3, 4])
    np.testing.assert_array_equal(state["p_total"], [90003, 90004])
    assert state["u"].tobytes() == old["carries"]["d02"]["state_fields"]["u"].tobytes()


def test_validator_refusal_keeps_parent_bytes():
    old = snapshot()
    before = copy.deepcopy(old)
    def refuse(*args):
        raise ValueError("consistency refused")
    with pytest.raises(ValueError, match="consistency"):
        apply_overrides(old, {"domains": {"d01": {"state": {"qv": {"add": 5.}}}}}, validate=refuse)
    for (_, a), (_, b) in zip(arrays(old), arrays(before), strict=True):
        assert a.tobytes() == b.tobytes()


@pytest.mark.parametrize("domain", ["d01", "d02", "d03"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_null_branch_rejects_nonfinite_retained_leaf(domain, value):
    from gpuwrf.runtime.checkpoint_consistency import checkpoint_validator
    old = snapshot()
    old["carries"][domain]["scratch_fields"]["held"][0] = value
    with pytest.raises(ValueError, match="NaN/Inf"):
        apply_overrides(old, {"schema_version": 1}, validate=checkpoint_validator(None, {}))


def test_actual_null_validator_keeps_all_bytes():
    from gpuwrf.runtime.checkpoint_consistency import checkpoint_validator
    old = snapshot()
    new, report = apply_overrides(old, {"schema_version": 1}, validate=checkpoint_validator(None, {}))
    assert all(item["passed"] for item in report["consistency"].values())
    for (_, a), (_, b) in zip(arrays(old), arrays(new), strict=True):
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 1e300])
def test_nonfinite_override_refuses(value):
    with pytest.raises(ValueError, match="nonfinite|NaN/Inf"):
        apply_overrides(snapshot(), {"domains": {"d01": {"state": {"qv": {"set": value}}}}}, validate=lambda *args: {})


@pytest.mark.parametrize("value", [
    {}, {"schema_version": 0}, {"schema_version": True}, {"schema_version": 1, "unknown": 1},
    {"schema_version": 1, "domains": {"d01": {"state": {"mu_perturbation": {"set": 1}}}}},
    {"schema_version": 1, "domains": {"d01": {"state": {"qv": {"set": 1, "add": 2}}}}},
    {"schema_version": 1, "domains": {"d01": {"state": {"qv": {"set": True}}}}},
    {"schema_version": 1, "parameters": {"namelists": {}}},
])
def test_manifest_schema_refuses(tmp_path, value):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load_manifest(path)


def test_array_hash_shape_and_race(tmp_path):
    source, path = tmp_path / "a.npy", tmp_path / "manifest.json"
    np.save(source, np.array([.003, .004], np.float32))
    value = {"schema_version": 1, "domains": {"d01": {"state": {"qv": {"set": {
        "path": "a.npy", "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}}}}}}
    path.write_text(json.dumps(value))
    changes = load_manifest(path)
    new, _ = apply_overrides(snapshot(), changes, validate=lambda *args: {})
    np.testing.assert_array_equal(new["carries"]["d01"]["state_fields"]["qv"], np.load(source))
    np.save(source, np.ones((1, 2), np.float32))
    with pytest.raises(ValueError, match="hash"):
        apply_overrides(snapshot(), changes, validate=lambda *args: {})
    value["domains"]["d01"]["state"]["qv"]["set"]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="shape"):
        apply_overrides(snapshot(), load_manifest(path), validate=lambda *args: {})


def test_paths_separate_and_empty(parent, tmp_path):
    generation, manifest, current = parent
    validate_paths(generation, current["output_dir"], tmp_path / "child-store")
    for output in (manifest["identity"]["output_dir"], str(Path(manifest["identity"]["output_dir"]) / "child")):
        with pytest.raises(ValueError, match="separate"):
            validate_paths(generation, output, None)
    with pytest.raises(ValueError, match="parent store"):
        validate_paths(generation, current["output_dir"], generation.parent)


@pytest.mark.parametrize("minutes,steps", [(.9, 1), (9, 10), (18, 20), (180, 200)])
def test_short_windows(minutes, steps):
    nml = {"domains": {"time_step": [54]}}
    hours = explicit_window(nml, minutes=minutes)
    assert aligned_steps(hours, 54) == steps


@pytest.mark.parametrize("minutes", [1, 10, 60, 0, -1, float("nan"), float("inf")])
def test_unaligned_or_invalid_windows_refuse(minutes):
    with pytest.raises(ValueError, match="positive|multiple"):
        explicit_window({"domains": {"time_step": 54}}, minutes=minutes)


def test_absolute_endpoint():
    tc = {"start_year": [2026], "start_month": [2], "start_day": [27], "start_hour": [0]}
    assert explicit_window({"time_control": tc, "domains": {"time_step": 54}}, end_time="2026-02-27_00:09:00") == .15
    from gpuwrf.cli import build_parser
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--input-dir", "x", "--output-dir", "y", "--hours", "3", "--minutes", "18"])


@pytest.mark.parametrize("flag,value,expected", [("--minutes", "9", .15), ("--end-time", "2023-01-15T00:09:00Z", .15)])
def test_public_cli_keeps_fractional_duration(tmp_path, monkeypatch, capsys, flag, value, expected):
    from gpuwrf import cli
    from gpuwrf.integration import nested_pipeline
    source = Path(__file__).resolve().parents[3] / "examples" / "switzerland_d01"
    # __file__ is tests/v025/restart/...; repository root is parents[3].
    captured = []
    monkeypatch.setattr(nested_pipeline, "execute_nested_pipeline", lambda config: captured.append(config) or
                        {"verdict": "PIPELINE_GREEN", "wrfout_files": [], "metadata": {}})
    monkeypatch.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
    assert cli.main(["run", "--input-dir", str(source), "--output-dir", str(tmp_path / "out"),
                     "--scratch-dir", str(tmp_path / "scratch"), flag, value]) == 0
    assert captured[0].hours == expected
    assert json.loads(capsys.readouterr().out)["effective_hours"] == expected


def test_parameter_binding_helper_off_and_on(monkeypatch):
    import sys
    from types import SimpleNamespace
    from gpuwrf.runtime.restart_parameters import parameter_digests, validate_parameters
    assert parameter_digests(SimpleNamespace()) == {}
    assert parameter_digests({}) == {}
    monkeypatch.setitem(sys.modules, "gpuwrf.runtime_parameters", SimpleNamespace(
        restart_parameter_digests=lambda nl: {"diffusion_parameters": nl.diffusion_parameters}))
    nl = SimpleNamespace(diffusion_parameters="a" * 64)
    assert parameter_digests(nl) == {"diffusion_parameters": "a" * 64}
    validate_parameters({"diffusion_parameters": "a" * 64}, nl)
    with pytest.raises(ValueError, match="E78"):
        validate_parameters({}, nl)
    with pytest.raises(ValueError, match="E78"):
        validate_parameters({"diffusion_parameters": "b" * 64}, nl)


def test_static_identity_masks_only_authenticated_value_bytes():
    from collections import namedtuple
    from gpuwrf.runtime.restart_parameters import static_namelist
    from gpuwrf.runtime.aot_cheap_key import canonical_digest
    Topo = namedtuple("Topo", "ctopo untouched")
    base = {"diffusion_parameters": {"factor": np.array(.1, np.float32)},
            "topo_wind_static": Topo(np.ones((2, 3), np.float32), np.array(9., np.float32)),
            "static": 7}
    changed = {**base, "diffusion_parameters": {"factor": np.array(.9, np.float32)},
               "topo_wind_static": base["topo_wind_static"]._replace(ctopo=np.full((2, 3), 2., np.float32))}
    assert canonical_digest(static_namelist(base)) == canonical_digest(static_namelist(changed))
    for delta in (
        {"static": 8},
        {"diffusion_parameters": {"different": np.array(.1, np.float32)}},
        {"diffusion_parameters": {"factor": np.array(.1, np.float64)}},
        {"diffusion_parameters": {"factor": np.array([.1], np.float32)}},
        {"topo_wind_static": base["topo_wind_static"]._replace(untouched=np.array(10., np.float32))},
    ):
        assert canonical_digest(static_namelist(base)) != canonical_digest(static_namelist({**base, **delta}))
    empty = {"static": 7}
    assert static_namelist(empty) is empty


def test_static_operand_identity_is_stable_across_fresh_processes():
    import os
    import subprocess
    import sys
    import gpuwrf.runtime.restart_parameters as binding
    script = """import sys,numpy as np
from gpuwrf.runtime.restart_parameters import static_namelist
from gpuwrf.runtime.aot_cheap_key import canonical_digest
print(canonical_digest(static_namelist({'diffusion_parameters': {'factor':np.array(float(sys.argv[1]),np.float32)},'static':7})))
"""
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "GPUWRF_JAX_CACHE": "0",
           "PYTHONPATH": str(Path(binding.__file__).resolve().parents[2])}
    values = [subprocess.check_output([sys.executable, "-c", script, value], env=env, text=True).strip()
              for value in ("0.1", "0.9")]
    assert len(values[0]) == 64 and values[0] == values[1]
