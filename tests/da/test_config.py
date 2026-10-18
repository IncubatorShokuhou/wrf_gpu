"""Strict ``gpuwrf.da.v1`` configuration: every unknown key, wrong type and unsafe choice is refused up front."""
from __future__ import annotations

import copy
import json

import pytest

from gpuwrf.assimilation.letkf import config as dacfg


def minimal(**changes):
    cfg = {
        "schema": "gpuwrf.da.v1",
        "t_a": "2023-01-15T00:00:00Z",
        "ensemble": {"kind": "members", "member_glob": "members/*/wrfout_d01_*"},
        "obs": {"sources": [{"format": "csv_surface", "path": "obs.csv"}]},
        "experiments": [{"name": "A", "assimilate": ["surface"]}],
    }
    cfg.update(changes)
    return cfg


def test_minimal_config_normalises_with_defaults(tmp_path):
    cfg = dacfg.validate(minimal(), base_dir=tmp_path)
    data = cfg.data
    assert data["domain"] == "d01" and data["t_a"] == "2023-01-15T00:00:00Z"
    assert data["ensemble"]["member_glob"] == str(tmp_path / "members/*/wrfout_d01_*")
    assert data["obs"]["sources"][0]["path"] == str(tmp_path / "obs.csv") and data["obs"]["sonde_superob"] is True
    assert data["moisture"]["negative"] == "clip" and data["governance"]["plugin"] == "none"
    assert data["experiments"][0] == {"name": "A", "assimilate": ["surface"], "withhold": [], "params": {},
                                      "write_mean": False, "land": "cycled"}
    assert cfg.t_a.tzinfo is not None and len(cfg.sha256) == 64
    assert dacfg.validate(minimal(), base_dir=tmp_path).sha256 == cfg.sha256


def test_offset_times_are_converted_to_utc(tmp_path):
    cfg = dacfg.validate(minimal(t_a="2023-01-15T01:00:00+01:00"), base_dir=tmp_path)
    assert cfg.data["t_a"] == "2023-01-15T00:00:00Z"


@pytest.mark.parametrize("mutate, message", [
    (lambda c: c.update(schema="gpuwrf.da.v0"), "schema"),
    (lambda c: c.update(t_a="2023-01-15T00:00:00"), "UTC offset"),
    (lambda c: c.update(t_a="yesterday"), "ISO 8601"),
    (lambda c: c.update(domain="d02"), "only ['d01']"),
    (lambda c: c.update(extra=1), "unknown key"),
    (lambda c: c["ensemble"].update(kind="tes"), "must be one of"),
    (lambda c: c["ensemble"].update(member_files=["a", "b"]), "exactly one of"),
    (lambda c: c["ensemble"].update(scale="innovation_bands"), "same_phase"),
    (lambda c: c["ensemble"].update(names=["U", "V"]), "must include"),
    (lambda c: c["obs"]["sources"][0].update(format="bufr"), "must be one of"),
    (lambda c: c["obs"]["sources"][0].update(window_minutes=600), "<= 180"),
    (lambda c: c["obs"]["sources"].append({"format": "igra2", "path": "x", "station": "guimar"}), "does not match"),
    (lambda c: c["obs"]["sources"].append({"format": "isd_lite", "path": "x"}), "exactly one of"),
    (lambda c: c["obs"].update(sources=[]), "non-empty"),
    (lambda c: c.update(params={"rtps_alpha": 1.5}), "<= 1.0"),
    (lambda c: c.update(params={"inflation": 0.5}), ">= 1.0"),
    (lambda c: c.update(params={"stride_h": 1.5}), "integer"),
    (lambda c: c.update(params={"negative_qv": "clip"}), "unknown key"),
    (lambda c: c.update(params={"variables": ["T", "W"]}), "must be one of"),
    (lambda c: c.update(moisture={"negative": "drop"}), "moisture.negative"),
    (lambda c: c.update(moisture={"max_added_fraction": float("nan")}), "finite"),
    (lambda c: c.update(governance={"plugin": "rm -rf"}), "plugin"),
    (lambda c: c["experiments"][0].update(assimilate=[]), "at least one"),
    (lambda c: c["experiments"][0].update(withhold=["surface"]), "both assimilated and withheld"),
    (lambda c: c["experiments"][0].update(write=True), "unknown key"),
    (lambda c: c["experiments"][0].update(write_mean=True), "template_wrfinput"),
    (lambda c: c["experiments"].append({"name": "A", "assimilate": ["sonde"]}), "unique"),
    (lambda c: c["experiments"][0].update(name="../escape"), "does not match"),
    (lambda c: c.update(localisation={"variables": {"T": {"radar": 0.5}}}), "bad obs key"),
    (lambda c: c.update(localisation={"variables": {"T": {"*": 1.5}}}), "<= 1.0"),
])
def test_invalid_configs_are_refused(tmp_path, mutate, message):
    cfg = minimal()
    mutate(cfg)
    with pytest.raises(dacfg.ConfigError, match=None) as info:
        dacfg.validate(cfg, base_dir=tmp_path)
    assert message in str(info.value), str(info.value)


def test_write_mean_with_members_needs_control_and_same_phase_needs_control(tmp_path):
    cfg = minimal(template_wrfinput="wrfinput_d01")
    cfg["experiments"][0]["write_mean"] = True
    with pytest.raises(dacfg.ConfigError, match="control"):
        dacfg.validate(cfg, base_dir=tmp_path)
    cfg["control_file"] = "ctrl"
    assert dacfg.validate(cfg, base_dir=tmp_path).data["control_file"] == str(tmp_path / "ctrl")
    sp = minimal(ensemble={"kind": "same_phase", "runs_glob": "runs/*"})
    with pytest.raises(dacfg.ConfigError, match="same_phase"):
        dacfg.validate(sp, base_dir=tmp_path)


def test_delegated_sections_refused_without_a_delegate(tmp_path, monkeypatch):
    from gpuwrf.assimilation.letkf import inflation

    monkeypatch.delattr(inflation, "validate_config_section", raising=False)
    with pytest.raises(dacfg.ConfigError, match="not supported by this build"):
        dacfg.validate(minimal(inflation={"mode": "adaptive"}), base_dir=tmp_path)
    monkeypatch.setattr(inflation, "validate_config_section", lambda section: {"checked": dict(section)},
                        raising=False)
    assert dacfg.validate(minimal(inflation={"mode": "adaptive"}), base_dir=tmp_path).data["inflation"] == {
        "checked": {"mode": "adaptive"}}

    def bad(section):
        raise ValueError("mode must be fixed")
    monkeypatch.setattr(inflation, "validate_config_section", bad, raising=False)
    with pytest.raises(dacfg.ConfigError, match="mode must be fixed"):
        dacfg.validate(minimal(inflation={"mode": "x"}), base_dir=tmp_path)


def test_load_reads_json_relative_to_its_directory(tmp_path):
    (tmp_path / "cfg").mkdir()
    path = tmp_path / "cfg" / "da.json"
    path.write_text(json.dumps(minimal()))
    cfg = dacfg.load(path)
    assert cfg.data["obs"]["sources"][0]["path"] == str(tmp_path / "cfg" / "obs.csv")
    assert cfg.source == str(path.resolve())
    path.write_text("{not json")
    with pytest.raises(dacfg.ConfigError, match="not valid JSON"):
        dacfg.load(path)


def test_with_overrides_revalidates(tmp_path):
    path = tmp_path / "da.json"
    path.write_text(json.dumps(minimal()))
    cfg = dacfg.load(path)
    later = cfg.with_overrides(t_a="2023-01-15T06:00:00Z")
    assert later.data["t_a"] == "2023-01-15T06:00:00Z" and later.sha256 != cfg.sha256
    with pytest.raises(dacfg.ConfigError):
        cfg.with_overrides(t_a="2023-01-15T06:00:00")
    assert cfg.data["t_a"] == "2023-01-15T00:00:00Z"          # original unchanged
    assert copy.deepcopy(cfg.data) == cfg.data
