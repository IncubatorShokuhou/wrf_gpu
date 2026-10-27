"""Joined branch-owner/parameter-owner tests through actual restart consumers."""
from dataclasses import replace
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf import runtime_parameters as rp
from gpuwrf.surface_params import controls as sc


@pytest.fixture
def initialized():
    path = Path(__file__).resolve().parents[1] / "v025/restart/test_restart_store.py"
    spec = importlib.util.spec_from_file_location("params_restart_fixture", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper.initialized.__wrapped__()


def parameter_namelist(namelist, value=.12):
    # These tests target persistence, not admission/physics (covered separately).
    return replace(namelist, diffusion_parameters={"diff_6th_factor": jnp.float32(value)})


@pytest.mark.usefixtures("research_diffusion_policy")
def test_pickle_consumer_authenticates_saved_and_requested_values(initialized, tmp_path):
    from gpuwrf.io.restart import read_restart, write_restart
    grid, original, carry = initialized
    nl = parameter_namelist(original)
    path = tmp_path / "controlled.pkl"
    write_restart(carry, nl, grid, 1, path)
    restored, saved, _, step = read_restart(path, namelist=nl)
    assert step == 1 and rp.restart_parameter_digests(saved) == rp.restart_parameter_digests(nl)
    for a, b in zip(jax.tree.leaves(carry), jax.tree.leaves(restored), strict=True):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    with pytest.raises(ValueError, match="runtime parameter"):
        read_restart(path, namelist=parameter_namelist(original, .24))
    with pytest.raises(ValueError, match="runtime parameter"):
        read_restart(path, namelist=original)
    with path.open("rb") as f:
        payload = pickle.load(f)
    assert payload.pop("runtime_parameter_digests") == rp.restart_parameter_digests(nl)
    with path.open("wb") as f:
        pickle.dump(payload, f)
    with pytest.raises(ValueError, match="runtime parameter"):
        read_restart(path)  # missing auth must also reject the embedded ON namelist


def test_pickle_consumer_ctopo_and_off_compatibility(initialized, tmp_path):
    from gpuwrf.io.restart import read_restart, write_restart
    from gpuwrf.io.topo_wind_static import TopoWindStatic
    grid, original, carry = initialized
    path = tmp_path / "legacy.pkl"
    write_restart(carry, original, grid, 1, path)
    with path.open("rb") as f:
        payload = pickle.load(f)
    assert "runtime_parameter_digests" not in payload
    read_restart(path, namelist=original)
    topo = TopoWindStatic(jnp.full((grid.ny, grid.nx), 2., jnp.float32), None)
    nl = replace(original, topo_wind_static=topo)
    write_restart(carry, nl, grid, 1, path)
    read_restart(path, namelist=nl)
    changed = replace(nl, topo_wind_static=topo._replace(ctopo=topo.ctopo * 2))
    with pytest.raises(ValueError, match="surface-control"):
        read_restart(path, namelist=changed)


@pytest.mark.usefixtures("research_diffusion_policy")
def test_restart_identity_records_each_domain_family(initialized, tmp_path):
    from types import SimpleNamespace
    from gpuwrf.integration.nested_pipeline import NestedPipelineConfig
    from gpuwrf.runtime.restart_store import nested_identity, RestartStore
    grid, original, carry = initialized
    cfg = NestedPipelineConfig(tmp_path, tmp_path / "out", tmp_path / "proof", hours=3, max_dom=2)
    start = datetime(2026, 7, 25, 18, tzinfo=timezone.utc)
    nl = parameter_namelist(original)
    bundles = {"d01": SimpleNamespace(namelist=nl), "d02": SimpleNamespace(namelist=original)}
    identity = nested_identity(cfg, bundles, start)
    assert identity["diffusion_parameters"] == {
        "d01": rp.restart_parameter_digests(nl)["diffusion_parameters"], "d02": None}
    store = RestartStore(tmp_path / "checkpoint", 10**8, 2, 0)
    generation, _ = store.save({"d01": carry, "d02": carry}, {"d01": 1, "d02": 1},
                               {"d01": 54., "d02": 54.}, identity)
    store.read(generation, expected_identity=identity)
    bundles["d01"] = SimpleNamespace(namelist=parameter_namelist(original, .24))
    changed = nested_identity(cfg, bundles, start)
    assert changed["diffusion_parameters"] != identity["diffusion_parameters"]
    assert changed["namelists"] == identity["namelists"]  # branch can admit ONLY the declared digest change
    with pytest.raises(ValueError, match="identity"):
        store.read(generation, expected_identity=changed)


@pytest.mark.usefixtures("research_diffusion_policy")
def test_branch_static_view_excludes_only_declared_operands(initialized):
    from gpuwrf.runtime.aot_cheap_key import canonical_digest
    from gpuwrf.io.topo_wind_static import TopoWindStatic
    _, original, _ = initialized
    assert rp.restart_static_namelist(original) is original
    a, b = parameter_namelist(original), parameter_namelist(original, .24)
    views = []
    def canon(nl):
        views.append(rp.restart_static_namelist(nl))  # retain tokens: pointer reuse must not fake equality
        return canonical_digest(views[-1])
    assert canon(a) == canon(b)
    assert canon(a) != canon(replace(b, dampcoef=.7))
    assert canon(a) != canon(replace(b, diffusion_parameters={"dampcoef": jnp.float32(.12)}))
    assert canon(a) != canon(replace(b, diffusion_parameters={"diff_6th_factor": jnp.ones(2, jnp.float32)}))
    topo = TopoWindStatic(jnp.ones((2, 3), jnp.float32), jnp.ones((2, 3), jnp.float32))
    a = replace(a, topo_wind_static=topo)
    b = replace(a, topo_wind_static=topo._replace(ctopo=topo.ctopo * 2))
    assert canon(a) == canon(b) and sc.restart_controls_digest(a) != sc.restart_controls_digest(b)
    assert canon(a) != canon(replace(a, topo_wind_static=topo._replace(ctopo2=topo.ctopo2 * 2)))
    assert rp.restart_parameter_digests({"diffusion_parameters": a.diffusion_parameters}) == rp.restart_parameter_digests(a)
