"""TEMF E78 admission through the native RestartStore public API."""
from datetime import datetime, timezone
import hashlib
import json
import pickle
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from gpuwrf.integration.nested_pipeline import NestedPipelineConfig
from gpuwrf.runtime import restart_store as storage
from gpuwrf.runtime.operational_mode import _initial_carry_for_run, _physics_step_forcing
from test_v040_temf import _grid_state, _namelist


@pytest.fixture(scope="module")
def temf_case():
    grid, state = _grid_state()
    ns = _namelist(grid)
    carry = _physics_step_forcing(_initial_carry_for_run(state, ns), ns, 0., run_radiation=False).carry
    assert np.max(np.asarray(carry.temf.surface["wm_temf"])) > 0
    return ns, carry


def identity(tmp_path, ns):
    config = NestedPipelineConfig(tmp_path, tmp_path / "out", tmp_path / "proof", hours=1, max_dom=1)
    value = storage.nested_identity(config, {"d01": SimpleNamespace(namelist=ns)},
                                    datetime(2024, 6, 1, 12, tzinfo=timezone.utc))
    assert value["temf_config"]["d01"] == {
        "bl_pbl_physics": int(ns.bl_pbl_physics), "sf_sfclay_physics": int(ns.sf_sfclay_physics),
        "domain_shape": [ns.grid.nz, ns.grid.ny, ns.grid.nx],
    }
    return value


def exact(left, right):
    a, at = jax.tree.flatten(left)
    b, bt = jax.tree.flatten(right)
    assert at == bt
    for x, y in zip(a, b, strict=True):
        x, y = np.asarray(x), np.asarray(y)
        assert x.shape == y.shape and x.dtype == y.dtype and x.tobytes() == y.tobytes()


def save(store, carry, config):
    return store.save({"d01": carry}, {"d01": 1}, {"d01": 18.}, config)[0]


def damage(payload, kind):
    group = payload["extra_fields"]["temf"]
    if kind == "missing_group":
        payload["extra_fields"]["temf"] = None
    elif kind == "wrong_group":
        payload["extra_fields"]["temf"] = {}
    elif kind == "missing_profile":
        payload["extra_fields"]["temf"] = group._replace(
            profiles={k: v for k, v in group.profiles.items() if k != "te_temf"})
    elif kind == "missing_surface":
        payload["extra_fields"]["temf"] = group._replace(
            surface={k: v for k, v in group.surface.items() if k != "wm_temf"})
    elif kind == "coherent_wrong_domain":
        payload["extra_fields"]["temf"] = group._replace(
            profiles={k: v[:1] for k, v in group.profiles.items()},
            surface={k: v[:1] for k, v in group.surface.items()})
    elif kind == "wrong_shape":
        payload["extra_fields"]["temf"] = group._replace(
            profiles=dict(group.profiles, te_temf=np.zeros((1, 1), np.float32)))
    elif kind == "fp64":
        payload["extra_fields"]["temf"] = group._replace(
            profiles=dict(group.profiles, te_temf=np.asarray(group.profiles["te_temf"], np.float64)))
    elif kind == "wrong_integer":
        payload["extra_fields"]["temf"] = group._replace(
            surface=dict(group.surface, kpbl=np.asarray(group.surface["kpbl"], np.float32)))
    elif kind == "nonfinite":
        value = np.asarray(group.profiles["te_temf"]).copy()
        value[0, 0] = np.inf
        payload["extra_fields"]["temf"] = group._replace(profiles=dict(group.profiles, te_temf=value))
    elif kind == "missing_marker":
        payload.pop("temf_schema_version")
    else:
        raise AssertionError(kind)


def rewrite(generation, mutate):
    """Recompute outer checksums so read reaches the actual schema gate."""
    path = generation / "snapshot.pkl"
    snapshot = pickle.loads(path.read_bytes())
    mutate(snapshot["carries"]["d01"])
    data = pickle.dumps(snapshot, protocol=pickle.HIGHEST_PROTOCOL)
    path.write_bytes(data)
    manifest_path = generation / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(payload_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    manifest_path.write_text(json.dumps(manifest))


def test_native_temf_public_roundtrip_and_evolved_continuation(temf_case, tmp_path, monkeypatch):
    from gpuwrf.io import restart_schema
    validate = restart_schema.validate_temf
    host_admissions = []
    def checked_host(group, namelist, version):
        assert all(isinstance(x, np.ndarray) for x in jax.tree.leaves(group))
        host_admissions.append(True)
        return validate(group, namelist, version)
    monkeypatch.setattr(restart_schema, "validate_temf", checked_host)
    ns, carry = temf_case
    config = identity(tmp_path, ns)
    store = storage.RestartStore(tmp_path / "store", 10**7, 2, 0)
    generation = save(store, carry, config)
    snapshot, _ = store.read(generation, expected_identity=config)
    restored = snapshot["carries"]["d01"]
    exact(carry, restored)
    uninterrupted = _physics_step_forcing(carry, ns, 18., run_radiation=False).carry
    resumed = _physics_step_forcing(restored, ns, 18., run_radiation=False).carry
    exact(uninterrupted, resumed)
    # Initial save, temporary/published readback, and this explicit public read.
    assert len(host_admissions) == 4


@pytest.mark.parametrize("operation", ["save", "read"])
@pytest.mark.parametrize("kind", ["missing_group", "wrong_group", "missing_profile", "missing_surface",
                                  "wrong_shape", "coherent_wrong_domain", "fp64", "wrong_integer",
                                  "nonfinite", "missing_marker"])
def test_native_temf_rejects_invalid_carry(temf_case, tmp_path, monkeypatch, operation, kind):
    ns, carry = temf_case
    config = identity(tmp_path, ns)
    store = storage.RestartStore(tmp_path / "store", 10**7, 2, 0)
    generation = save(store, carry, config)
    if operation == "save":
        if kind == "missing_marker":
            original = storage._carry_to_payload
            def malformed(value):
                payload = original(value)
                damage(payload, kind)
                return payload
            monkeypatch.setattr(storage, "_carry_to_payload", malformed)
            invalid = carry
        else:
            payload = {"extra_fields": {"temf": carry.temf}}
            damage(payload, kind)
            invalid = carry.replace(temf=payload["extra_fields"]["temf"])
        with pytest.raises(ValueError, match="TEMF.*E78"):
            save(store, invalid, config)
        assert store.latest(expected_identity=config) == generation
        assert not list(store.root.glob(".partial-*")), "save must refuse before publication/readback"
    else:
        rewrite(generation, lambda payload: damage(payload, kind))
        for device in (False, True):
            with pytest.raises(ValueError, match="TEMF.*E78"):
                store.read(generation, expected_identity=config, device=device)


@pytest.mark.parametrize("operation", ["save", "read"])
@pytest.mark.parametrize("kind", ["missing_binding", "missing_domain", "wrong_domain", "unpaired"])
def test_native_temf_rejects_unbound_selection_and_domain(temf_case, tmp_path, operation, kind):
    ns, carry = temf_case
    config = identity(tmp_path, ns)
    store = storage.RestartStore(tmp_path / "store", 10**7, 2, 0)
    generation = save(store, carry, config)
    if kind == "missing_binding":
        config.pop("temf_config")
    elif kind == "missing_domain":
        config["temf_config"].clear()
    elif kind == "wrong_domain":
        config["temf_config"]["d01"]["domain_shape"] = [ns.grid.nz, 1, 1]
    else:
        config["temf_config"]["d01"]["sf_sfclay_physics"] = 5
    if operation == "save":
        with pytest.raises(ValueError, match="TEMF.*E78"):
            save(store, carry, config)
    else:
        path = generation / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["identity"] = config
        path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="TEMF.*E78"):
            store.read(generation, expected_identity=config)


@pytest.mark.parametrize("configured", [False, True])
def test_native_inactive_default_accepts_legacy_missing_temf_marker(tmp_path, configured):
    grid, state = _grid_state()
    ns = _namelist(grid, bl_pbl_physics=5, sf_sfclay_physics=5)
    carry = _initial_carry_for_run(state, ns)
    assert carry.temf is None
    config = identity(tmp_path, ns) if configured else {}
    store = storage.RestartStore(tmp_path / "store", 10**7, 2, 0)
    generation = save(store, carry, config)
    rewrite(generation, lambda payload: payload.pop("temf_schema_version"))
    restored, _ = store.read(generation, expected_identity=config)
    exact(carry, restored["carries"]["d01"])
