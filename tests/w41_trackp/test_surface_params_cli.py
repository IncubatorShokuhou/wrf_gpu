"""`gpuwrf surface-params` basis/expand/eki: the NumPy expansion equals the model-side expansion bytewise."""
from __future__ import annotations

import json

import numpy as np
import pytest

from w41_trackp_fixtures import pytestmark, _fixture, _namelist  # noqa: F401
from gpuwrf.surface_params import cli as spc
from gpuwrf.surface_params import controls as sc


@pytest.fixture()
def basis(tmp_path):
    _land, _forcing, static = _fixture()
    nl = _namelist(static)
    path = tmp_path / "basis.npz"
    manifest = spc.write_basis({"d01": nl}, path)
    return nl, path, manifest


def test_basis_expand_matches_model_expansion(basis, tmp_path):
    nl, path, manifest = basis
    assert all(v["active"] for v in manifest["domains"]["d01"]["controls"].values())
    out = tmp_path / "c.npz"
    rc = spc.main_args(["expand", "--basis", str(path), "--output", str(out), "--set", "albsat=1.1",
                        "--class", "z0mvt:5=1.5", "--class", "z0mvt:12=0.6"])
    assert rc == 0
    loaded, _ = sc.load_controls(out)
    want = sc.expand_controls(nl, {"albsat": 1.1, "z0mvt": {5: 1.5, 12: 0.6}}, device=False)
    assert sorted(loaded["d01"]) == sorted(want)
    for name in want:
        assert loaded["d01"][name].tobytes() == want[name].tobytes(), name
    assert spc.main_args(["expand", "--basis", str(path), "--output", str(out), "--set", "z0mvt=1.2"]) == 2  # exists
    assert spc.main_args(["expand", "--basis", str(path), "--output", str(tmp_path / "x.npz"),
                          "--set", "z0mvt=500"]) == 2  # out of bounds


def test_eki_init_update_members(basis, tmp_path):
    nl, path, _ = basis
    spec = {"parameters": [
        {"name": "z0:forest", "control": "z0mvt", "classes": [2, 5], "prior_sd_log": 0.4, "bounds": [0.5, 2.0]},
        {"name": "z0:all", "control": "z0mvt", "classes": "all", "prior_sd_log": 0.2, "bounds": [0.5, 2.0]},
        {"name": "alb:soil", "control": "albdry", "prior_sd_log": 0.1, "bounds": [0.8, 1.25]}]}
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    d0 = tmp_path / "it0"
    assert spc.main_args(["eki", "init", "--spec", str(spec_path), "--basis", str(path), "--members", "6",
                          "--seed", "3", "--output-dir", str(d0)]) == 0
    state = json.loads((d0 / "eki_state.json").read_text())
    theta = np.asarray(state["theta"])
    assert theta.shape == (6, 3) and len(state["member_controls"]) == 6
    # member 2's control file == model-side expansion of its multipliers (products of overlapping terms)
    m = np.exp(theta[2])
    vt = np.asarray(nl.noahmp_static.ivgtyp)
    z0_field = np.where(np.isin(vt, [2, 5]), m[0], 1.0) * m[1]
    want = sc.expand_controls(nl, {"z0mvt": z0_field, "albdry": m[2]}, device=False)
    got, _ = sc.load_controls(state["member_controls"][2])
    for name in want:
        assert got["d01"][name].tobytes() == want[name].tobytes(), name
    obs_path = tmp_path / "obs.npz"
    np.savez(obs_path, y=np.array([1.0, 2.0, 3.0]), sigma=np.array([0.1, 0.1, 0.1]))
    pred = tmp_path / "pred.npy"
    np.save(pred, theta @ np.array([[1.0, 0.0, 0.5], [0.2, 1.0, 0.0], [0.0, 0.3, 1.0]]))
    d1 = tmp_path / "it1"
    assert spc.main_args(["eki", "update", "--state", str(d0 / "eki_state.json"), "--predictions", str(pred),
                          "--obs", str(obs_path), "--dt", "0.5", "--output-dir", str(d1)]) == 0
    new = json.loads((d1 / "eki_state.json").read_text())
    assert new["iteration"] == 1 and len(new["history"]) == 1
    assert not np.allclose(np.asarray(new["theta"]), theta)
    th = np.asarray(new["theta"])
    assert np.all(th[:, 0] <= np.log(2.0) + 1e-12) and np.all(th[:, 0] >= np.log(0.5) - 1e-12)
