"""Output identity is separate from the HLO key, and binds resolved choices."""
from dataclasses import replace
from datetime import datetime
import json
from types import SimpleNamespace

import pytest

from test_restart_store import initialized
from gpuwrf.integration import nested_pipeline as pipeline
from gpuwrf.io import wrfout_writer as writer
from gpuwrf.runtime import operational_mode
from gpuwrf.runtime.aot_cheap_key import global_trace_env_hash
from gpuwrf.runtime.checkpoint_branch import branch_identity
from gpuwrf.runtime.restart_store import RestartStore, _nested_output_policy, nested_identity


@pytest.fixture
def case(initialized, tmp_path, monkeypatch):
    for name in ("GPUWRF_TRAINING_OUTPUT_SUBSET", "GPUWRF_FULL_WRFOUT_VARIABLES", "GPUWRF_FULL_WRFOUT",
                 "GPUWRF_COLONFREE_OUTPUT", "GPUWRF_NESTED_ASYNC_OUTPUT", "GPUWRF_NEST_OUTPUT_PIPELINE",
                 "GPUWRF_NEST_OUTPUT_PIPELINE_DEPTH", "GPUWRF_NESTED_M9_RADIATION_FROM_CARRY",
                 "GPUWRF_WRITER_RVOVRD_REAL", "GPUWRF_HISTORY_SWDOWN_HORIZONTAL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(writer, "_WRITER_RVOVRD_REAL", False)
    monkeypatch.setattr(operational_mode, "_HISTORY_SWDOWN_HORIZONTAL", False)
    _, nl, carry = initialized
    config = pipeline.NestedPipelineConfig(tmp_path / "input", tmp_path / "parent", tmp_path / "proof", 3, 1)
    return config, {"d01": SimpleNamespace(namelist=nl)}, carry


@pytest.mark.parametrize("flag,value,target", [
    ("GPUWRF_TRAINING_OUTPUT_SUBSET", "1", None),
    ("GPUWRF_FULL_WRFOUT_VARIABLES", "0", None),
    ("GPUWRF_FULL_WRFOUT", "0", None),
    ("GPUWRF_COLONFREE_OUTPUT", "1", None),
    ("GPUWRF_NESTED_ASYNC_OUTPUT", "0", None),
    ("GPUWRF_NEST_OUTPUT_PIPELINE", "1", None),
    ("GPUWRF_NEST_OUTPUT_PIPELINE_DEPTH", "3", None),
    ("GPUWRF_NESTED_M9_RADIATION_FROM_CARRY", "1", None),
    ("GPUWRF_WRITER_RVOVRD_REAL", "1", "rvovrd"),
    ("GPUWRF_HISTORY_SWDOWN_HORIZONTAL", "1", "swdown"),
    ("GPUWRF_DYN_RK_FP32", "1", None),
])
def test_empty_branch_refuses_each_output_flag(case, tmp_path, monkeypatch, flag, value, target):
    config, bundles, _ = case
    if flag == "GPUWRF_NEST_OUTPUT_PIPELINE_DEPTH":
        monkeypatch.setenv("GPUWRF_NEST_OUTPUT_PIPELINE", "1")
    if flag == "GPUWRF_DYN_RK_FP32":
        monkeypatch.setenv(flag, "0")
    parent = nested_identity(config, bundles, datetime(2026, 2, 27))
    generation = tmp_path / "generation"
    generation.mkdir()
    manifest = {"branch_schema_version": 1, "identity": parent, "domains": ["d01"],
                "own_steps": {"d01": 67}, "sha256": "payload"}
    (generation / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setenv(flag, value)
    # These flags are captured at import. Model the newly configured process's
    # actual constant, rather than pretending a late getenv changes the writer.
    if target == "rvovrd":
        monkeypatch.setattr(writer, "_WRITER_RVOVRD_REAL", True)
    elif target == "swdown":
        monkeypatch.setattr(operational_mode, "_HISTORY_SWDOWN_HORIZONTAL", True)
    current = nested_identity(replace(config, output_dir=tmp_path / "child"), bundles, datetime(2026, 2, 27))
    assert current["output_policy"] != parent["output_policy"], flag
    with pytest.raises(ValueError, match="undeclared.*output_policy"):
        branch_identity(manifest, current, {"schema_version": 1}, generation)


def test_training_subset_binds_real_variables_and_compression_without_hlo_key_change(case, monkeypatch):
    config, bundles, _ = case
    before = _nested_output_policy(config, bundles)
    trace_env = global_trace_env_hash()
    monkeypatch.setenv("GPUWRF_TRAINING_OUTPUT_SUBSET", "yes")
    after = _nested_output_policy(config, bundles)
    assert global_trace_env_hash() == trace_env
    assert before["history"]["variable_subset"] is None
    assert before["history"]["compression"] is None
    assert after["history"]["variable_subset"] == sorted(writer.MINIMAL_TRAINING_SET)
    assert after["history"]["compression"] == {"zlib": True, "complevel": 4}
    assert after["history"]["mandatory_coordinates"] == sorted(writer.MANDATORY_WRFOUT_COORDINATES)


def test_full_alias_precedence_and_equivalent_spellings(case, monkeypatch):
    config, bundles, _ = case
    parent = _nested_output_policy(config, bundles)
    for flag, value in (("GPUWRF_FULL_WRFOUT_VARIABLES", "yes"), ("GPUWRF_FULL_WRFOUT", "off"),
                        ("GPUWRF_TRAINING_OUTPUT_SUBSET", "off"), ("GPUWRF_NESTED_ASYNC_OUTPUT", "true")):
        monkeypatch.setenv(flag, value)
    assert _nested_output_policy(config, bundles) == parent
    monkeypatch.setenv("GPUWRF_NESTED_ASYNC_OUTPUT", "0")
    serial = _nested_output_policy(config, bundles)
    monkeypatch.setenv("GPUWRF_NEST_OUTPUT_PIPELINE", "1")
    monkeypatch.setenv("GPUWRF_NEST_OUTPUT_PIPELINE_DEPTH", "99")
    assert _nested_output_policy(config, bundles) == serial  # inactive scheduler options


def test_import_time_writer_policy_uses_actual_constant(case, monkeypatch):
    config, bundles, _ = case
    original = _nested_output_policy(config, bundles)
    monkeypatch.setenv("GPUWRF_WRITER_RVOVRD_REAL", "1")
    monkeypatch.setenv("GPUWRF_HISTORY_SWDOWN_HORIZONTAL", "1")
    assert _nested_output_policy(config, bundles) == original


def test_history_intervals_follow_the_actual_per_domain_writer_resolver(case, tmp_path):
    config, bundles, _ = case
    config.input_dir.mkdir()
    namelist = config.input_dir / "namelist.input"
    namelist.write_text("&time_control\n history_interval=30,15,5,\n/\n")
    bundles = dict.fromkeys(("d01", "d02", "d03"), bundles["d01"])
    policy = _nested_output_policy(config, bundles)
    assert policy["history"]["interval_minutes"] == {"d01": 30., "d02": 15., "d03": 5.}
    namelist.write_text("&time_control\n history_interval=60,\n/\n")
    assert _nested_output_policy(config, bundles)["history"]["interval_minutes"] == dict.fromkeys(bundles, 60.)


def test_regional_stream_set_and_compression_options_are_bound(case, monkeypatch):
    config, bundles, _ = case
    original = _nested_output_policy(config, bundles)
    config.input_dir.mkdir()
    path = config.input_dir / "gpuwrf.streams.json"
    path.write_text(json.dumps({"version": 1, "streams": [{"name": "surface", "domain": "d01",
        "variables": ["T2"], "region": {"bbox": [0, 0, 1, 1]}, "interval_steps": 3}]}))
    streams = _nested_output_policy(config, bundles)
    assert original["regional_streams"] == []
    assert streams["regional_streams"][0]["name"] == "surface"
    monkeypatch.setenv("GPUWRF_TRAINING_OUTPUT_SUBSET", "1")
    compressed = _nested_output_policy(config, bundles)
    monkeypatch.setitem(writer._SUBSET_STREAM_COMPRESSION, "complevel", 7)
    assert _nested_output_policy(config, bundles)["history"]["compression"] != compressed["history"]["compression"]


def test_ordinary_resume_also_refuses_changed_output_policy(case, tmp_path, monkeypatch):
    config, bundles, carry = case
    identity = nested_identity(config, bundles, datetime(2026, 2, 27))
    store = RestartStore(tmp_path / "checkpoints", 10**8, 2, 0)
    generation, _ = store.save({"d01": carry}, {"d01": 67}, {"d01": 54.}, identity)
    store.read(generation, expected_identity=identity, device=False)
    monkeypatch.setenv("GPUWRF_TRAINING_OUTPUT_SUBSET", "1")
    changed = nested_identity(config, bundles, datetime(2026, 2, 27))
    with pytest.raises(ValueError, match="identity"):
        store.read(generation, expected_identity=changed, device=False)
