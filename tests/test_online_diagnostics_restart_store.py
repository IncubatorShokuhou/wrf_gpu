"""Native checkpoint admission, using authenticated public-store generations.

These are structural IO controls; sentinel driver stepping is not physics
validation. Damage is rehashed so the E78 guard, rather than the hash gate,
must refuse a scientifically incomplete carry.
"""
from dataclasses import replace
from datetime import datetime
import hashlib
import json
import pickle
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from gpuwrf.runtime.restart_store import RestartStore, _diagnostic_restart_config, nested_identity
from test_online_diagnostics import case


def identity(nl):
    from gpuwrf.runtime.aot_cheap_key import canonical_digest
    from gpuwrf.diagnostics.solar_config import configuration as solar_configuration
    from gpuwrf.io.pbl_epsilon_schema import epsilon_pbl_config
    from gpuwrf.io.windfarm_schema import configuration as windfarm_configuration
    from gpuwrf.io.urban_schema import configuration as urban_configuration
    return {"namelists": {"d01": canonical_digest(nl)},
            "windfarm_config": {"d01": windfarm_configuration(nl)},
            "temf_config": {"d01": {"bl_pbl_physics": int(nl.bl_pbl_physics),
                                      "sf_sfclay_physics": int(nl.sf_sfclay_physics),
                                      "domain_shape": [nl.grid.nz, nl.grid.ny, nl.grid.nx]}},
            "epsilon_pbl_config": {"d01": {**epsilon_pbl_config(nl),
                                      "sf_sfclay_physics": int(nl.sf_sfclay_physics),
                                      "domain_shape": [nl.grid.nz, nl.grid.ny, nl.grid.nx]}},
            "solar_interpolation_config": {"d01": solar_configuration(nl)},
            "online_diagnostics_config": {"d01": _diagnostic_restart_config(nl)},
            "urban_inputs_config": {"d01": urban_configuration(nl)},
            "urban_bep_config": {"d01": {"sf_urban_physics": int(getattr(nl, "sf_urban_physics", 0) or 0),
                                         "domain_shape": [nl.grid.nz, nl.grid.ny, nl.grid.nx]}},
            "surface_fdda_config": {"d01": None},
            "ssib_active": {"d01": False},
            "sas_config": {"d01": {"cu_physics": int(nl.cu_physics), "osas_domain_id": int(nl.osas_domain_id),
                                     "domain_shape": [nl.grid.nz, nl.grid.ny, nl.grid.nx]}}}


def rewrite(generation, change):
    path = generation / "snapshot.pkl"
    snapshot = pickle.loads(path.read_bytes())
    manifest_path = generation / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    change(snapshot["carries"]["d01"], manifest)
    data = pickle.dumps(snapshot)
    path.write_bytes(data)
    manifest.update(payload_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    manifest_path.write_text(json.dumps(manifest))


def damage(payload, manifest, kind):
    if kind.endswith("_marker"):
        key = "precipitation_buckets_schema_version" if kind == "bucket_marker" else "online_diagnostics_schema_version"
        payload.pop(key)
        return
    if kind == "config_missing":
        manifest["identity"].pop("online_diagnostics_config")
        return
    if kind == "config_domain":
        manifest["identity"]["online_diagnostics_config"] = {}
        return
    if kind in ("bucket_off", "nwp_off", "radar_off"):
        key = {"bucket_off": "prec_acc_dt", "nwp_off": "nwp_diagnostics", "radar_off": "do_radar_ref"}[kind]
        manifest["identity"]["online_diagnostics_config"]["d01"][key] = 0
        return
    group, operation = kind.split("_", 1)
    group = {"bucket": "precipitation_buckets", "nwp": "nwp_diagnostics", "radar": "radar_reflectivity"}[group]
    value = payload["extra_fields"][group]
    if operation == "group":
        payload["extra_fields"][group] = None
        return
    key = "prec_acc_nc" if group == "precipitation_buckets" else "w_mean"
    leaf = value if group == "radar_reflectivity" else getattr(value, key)
    changed = (None if operation == "leaf" else np.asarray(leaf, np.float64)
               if operation == "precision" else np.asarray(leaf)[..., :-1])
    payload["extra_fields"][group] = changed if group == "radar_reflectivity" else value._replace(**{key: changed})


DAMAGES = ("bucket_marker", "online_marker", "bucket_group", "nwp_group", "radar_group",
           "bucket_leaf", "nwp_leaf", "bucket_precision", "nwp_precision", "radar_precision",
           "bucket_shape", "nwp_shape", "radar_shape", "config_missing", "config_domain",
           "bucket_off", "nwp_off", "radar_off")


def test_public_native_store_exact_active_control(case, tmp_path):
    _, nl, carry = case
    store = RestartStore(tmp_path, 10**7, 2, 0)
    generation, _ = store.save({"d01": carry}, {"d01": 2}, {"d01": nl.dt_s}, identity(nl))
    restored = store.read(generation, expected_identity=identity(nl))[0]["carries"]["d01"]
    assert jax.tree.structure(restored) == jax.tree.structure(carry)
    for left, right in zip(jax.tree.leaves(restored), jax.tree.leaves(carry), strict=True):
        left, right = np.asarray(left), np.asarray(right)
        assert (left.dtype, left.shape, left.tobytes()) == (right.dtype, right.shape, right.tobytes())


@pytest.mark.parametrize("kind", DAMAGES)
@pytest.mark.parametrize("device", [False, True])
def test_public_native_store_restore_refuses_authenticated_damage(case, tmp_path, kind, device):
    _, nl, carry = case
    store = RestartStore(tmp_path, 10**7, 2, 0)
    generation, _ = store.save({"d01": carry}, {"d01": 2}, {"d01": nl.dt_s}, identity(nl))
    rewrite(generation, lambda payload, manifest: damage(payload, manifest, kind))
    with pytest.raises(ValueError, match="E78"):
        store.read(generation, device=device)


@pytest.mark.parametrize("kind", ("bucket_group", "nwp_group", "radar_group", "bucket_leaf", "nwp_leaf",
                                  "bucket_precision", "nwp_precision", "radar_precision",
                                  "bucket_shape", "nwp_shape", "radar_shape"))
def test_public_native_store_save_refuses_damage(case, tmp_path, kind):
    from gpuwrf.io.restart import _carry_to_payload
    _, nl, carry = case
    payload = _carry_to_payload(carry)
    damage(payload, {"identity": identity(nl)}, kind)
    groups = ("precipitation_buckets", "nwp_diagnostics", "radar_reflectivity")
    broken = carry.replace(**{name: payload["extra_fields"][name] for name in groups})
    with pytest.raises(ValueError, match="E78"):
        RestartStore(tmp_path, 10**7, 2, 0).save({"d01": broken}, {"d01": 2}, {"d01": nl.dt_s}, identity(nl))
    assert not list(tmp_path.glob("generation-*"))


def test_native_identity_binds_each_domains_diagnostic_cadences(case, tmp_path):
    _, nl, carry = case
    config = SimpleNamespace(input_dir=tmp_path, output_dir=tmp_path / "out", feedback=False,
                             emit_initial_history=True)
    other = replace(nl, prec_acc_dt=60., diagnostic_history_interval=17.)
    bundles = {name: SimpleNamespace(namelist=selected) for name, selected in (("d01", nl), ("d02", other))}
    result = nested_identity(config, bundles, datetime(2023, 1, 15))
    assert result["online_diagnostics_config"] == {
        "d01": {"prec_acc_dt": 15., "nwp_diagnostics": 1, "do_radar_ref": 1, "history_interval": 30.},
        "d02": {"prec_acc_dt": 60., "nwp_diagnostics": 1, "do_radar_ref": 1, "history_interval": 17.}}


@pytest.mark.parametrize("patch", [{"prec_acc_dt": 30.}, {"diagnostic_history_interval": 15.},
                                   {"nwp_diagnostics": 0}, {"do_radar_ref": 0}])
def test_public_native_store_requested_selection_or_cadence_mismatch(case, tmp_path, patch):
    _, nl, carry = case
    store = RestartStore(tmp_path, 10**7, 2, 0)
    generation, _ = store.save({"d01": carry}, {"d01": 2}, {"d01": nl.dt_s}, identity(nl))
    with pytest.raises(ValueError, match="identity mismatch"):
        store.read(generation, expected_identity=identity(replace(nl, **patch)))


@pytest.mark.parametrize("kind", ["valid", "bucket_marker", "online_marker", "bucket_shape", "nwp_precision",
                                  "radar_shape", "nwp_group"])
def test_actual_native_driver_rejects_diagnostic_damage(case, tmp_path, monkeypatch, kind):
    from tests.v025.restart.test_single_domain_restart import _install_driver, DT
    grid, nl, carry = case
    nl = replace(nl, dt_s=DT)
    finals = []
    pipeline, storage, _ = _install_driver(monkeypatch, (grid, nl, carry), finals)
    bound = {**identity(nl), "output_dir": str((tmp_path / "out").resolve())}
    monkeypatch.setattr(storage, "nested_identity", lambda *args: bound)
    admitted = []
    def after_admission(carries, source):
        restored = carries["d01"]
        for group in ("precipitation_buckets", "nwp_diagnostics", "radar_reflectivity"):
            for left, right in zip(jax.tree.leaves(getattr(restored, group)),
                                   jax.tree.leaves(getattr(carry, group)), strict=True):
                np.testing.assert_array_equal(left, right)
        admitted.append(source)
        raise RuntimeError("valid diagnostic carry admitted")
    monkeypatch.setattr(pipeline, "_require_native_base_state", after_admission)
    store = RestartStore(tmp_path / "store", 10**7, 2, 0)
    generation, _ = store.save({"d01": carry}, {"d01": 1}, {"d01": DT}, bound,
                               driver_state={"forecast_hours": 3.})
    if kind != "valid":
        rewrite(generation, lambda payload, manifest: damage(payload, manifest, kind))
    config = pipeline.NestedPipelineConfig(tmp_path / "in", tmp_path / "out", tmp_path / "proof", 3., 1,
                                          resume_checkpoint=generation, checkpoint_reserve_bytes=0,
                                          aot_prefetch=False)
    with pytest.raises(RuntimeError if kind == "valid" else ValueError,
                       match="valid diagnostic carry admitted" if kind == "valid" else "E78"):
        pipeline.execute_nested_pipeline(config)
    assert len(admitted) == (1 if kind == "valid" else 0)
    assert not finals, "admission must fail before any step is advanced"
