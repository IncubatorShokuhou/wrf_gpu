"""Host surface control ``ctopo`` (topo_wind CTOPO excess factor): same program, value reaches YSU, fail closed.

CTOPO is already a traced namelist child (``TopoWindStatic``), so the control replaces its VALUE at bind time and
adds no program input: the lowered physics step is the same text for any factor, and only the YSU-updated state
changes. Form: CTOPO' = 1 + f (CTOPO - 1), rounded once to REAL (cells without a correction keep exactly 1).
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/"tests"))

from gpuwrf.io.topo_wind_static import TopoWindStatic  # noqa: E402
from gpuwrf.surface_params import cli as spc  # noqa: E402
from gpuwrf.surface_params import controls as sc  # noqa: E402

CTOPO = np.array([[1.0, 1.0, 1.0], [1.5, 2.0, 1.0], [0.0, 0.5, 3.0]], np.float32)  # incl. valley ramp 0 / 0.5


def _namelist(bl=1, static=True):
    from test_v013_operational_smoke import _grid, _namelist as base
    grid = _grid(nz=12, ny=3, nx=3)
    nl = base(grid, mp_physics=0, cu_physics=0, bl_pbl_physics=bl, sf_sfclay_physics=1, sf_surface_physics=0,
              use_noahmp=False, ra_lw_physics=0, ra_sw_physics=0)
    return dataclasses.replace(nl, topo_wind_static=TopoWindStatic(jnp.asarray(CTOPO), None) if static else None)


def test_excess_form_keeps_uncorrected_cells_and_factor_one_is_byte_identical():
    nl = _namelist()
    one = sc.expand_controls(nl, {"ctopo": 1.0}, device=False)["ctopo"]
    assert one.dtype == np.float32 and one.tobytes() == CTOPO.tobytes()
    out = sc.expand_controls(nl, {"ctopo": 0.8}, device=False)["ctopo"]
    want = (1.0+0.8*(CTOPO.astype(np.float64)-1.0)).astype(np.float32)
    assert out.tobytes() == want.tobytes()
    assert np.all(out[CTOPO == 1.0] == 1.0) and not np.array_equal(out, CTOPO)
    field = sc.expand_controls(nl, {"ctopo": np.full((3, 3), 0.9)}, device=False)["ctopo"]  # per-cell factors too
    assert field.tobytes() == (1.0+0.9*(CTOPO.astype(np.float64)-1.0)).astype(np.float32).tobytes()
    with pytest.raises(ValueError, match="outside"):  # 1 + 1.5 (0 - 1) < 0: negative drag factor refused
        sc.expand_controls(nl, {"ctopo": 1.5}, device=False)


def test_bind_replaces_only_the_traced_child_and_lowers_the_same_program():
    from gpuwrf.runtime import aot_cheap_key as ck
    from gpuwrf.runtime import operational_mode as om
    from gpuwrf.runtime.operational_state import initial_operational_carry, seed_history_diagnostics
    from test_v013_operational_smoke import _base_state

    nl = _namelist()
    field = sc.expand_controls(nl, {"ctopo": 0.8}, device=False)["ctopo"]
    bound = sc.bind_domain_controls(nl, {"d01": {"ctopo": field}}, "d01")
    assert bound.surface_controls is None  # no clock-base field: the default program of the case
    assert np.asarray(bound.topo_wind_static.ctopo).tobytes() == field.tobytes()
    assert bound.topo_wind_static.ctopo2 is None and bound.topo_wind_static.ctopo_first is None
    assert jax.tree_util.tree_structure(bound) == jax.tree_util.tree_structure(nl)
    assert ck.static_config_hash(bound) == ck.static_config_hash(nl)
    assert [(x.shape, x.dtype) for x in jax.tree.leaves(bound)] == [(x.shape, x.dtype) for x in jax.tree.leaves(nl)]
    carry = initial_operational_carry(_base_state(nl.grid))
    if carry.history_diagnostics is None:
        carry = carry.replace(history_diagnostics=seed_history_diagnostics(carry.state))
    step = lambda c, n: om._physics_step_forcing(c, n, 0., run_radiation=False, first_timestep=True)  # noqa: E731
    assert jax.jit(lambda c, n: step(c, n).carry).lower(carry, nl).as_text() == \
        jax.jit(lambda c, n: step(c, n).carry).lower(carry, bound).as_text()
    a, b = step(carry, nl), step(carry, bound)
    changed = [n for n, (x, y) in enumerate(zip(jax.tree.leaves(a.state), jax.tree.leaves(b.state)))
               if not np.array_equal(np.asarray(x), np.asarray(y))]
    assert changed, "the ctopo control never reached the YSU surface drag"
    same = sc.bind_domain_controls(nl, {"d01": {"ctopo": CTOPO.copy()}}, "d01")  # factor 1: identical results
    c = step(carry, same)
    assert all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(jax.tree.leaves(a.state), jax.tree.leaves(c.state)))


def test_ctopo_with_traced_controls_and_bind_refusals():
    nl = _namelist()
    with pytest.raises(ValueError, match="no start_em CTOPO"):
        sc.expand_controls(_namelist(static=False), {"ctopo": 0.8}, device=False)
    with pytest.raises(ValueError, match="does not read CTOPO"):
        sc.expand_controls(_namelist(bl=5), {"ctopo": 0.8}, device=False)
    fed = dataclasses.replace(nl, topo_wind_static=TopoWindStatic(jnp.asarray(CTOPO), None, jnp.asarray(CTOPO), None))
    with pytest.raises(ValueError, match="two-way feedback"):
        sc.expand_controls(fed, {"ctopo": 0.8}, device=False)
    with pytest.raises(ValueError, match="per-vegetation-class"):
        sc.expand_controls(nl, {"ctopo": {5: 1.2}}, device=False)
    field = {"d01": {"ctopo": CTOPO.copy()}}
    with pytest.raises(ValueError, match="B>1"):
        sc.bind_domain_controls(nl, field, "d01", allow_host=False)
    assert sc.has_host_controls(field) and not sc.has_host_controls({"d01": {"z0mvt": CTOPO}})
    with pytest.raises(ValueError, match="dtype"):
        sc.bind_domain_controls(nl, {"d01": {"ctopo": CTOPO.astype(np.float64)}}, "d01")
    with pytest.raises(ValueError, match="shape"):
        sc.bind_domain_controls(nl, {"d01": {"ctopo": CTOPO[:2]}}, "d01")


def test_basis_and_expand_for_a_topo_wind_domain_without_noah(tmp_path):
    nl = _namelist()
    manifest = spc.write_basis({"d01": nl}, tmp_path/"basis.npz")
    assert manifest["domains"]["d01"]["controls"].keys() == {"ctopo"}
    basis, manifest = spc.load_basis(tmp_path/"basis.npz")
    got = spc.expand_from_basis(basis, manifest, {"d01": {"ctopo": 0.8}})["d01"]["ctopo"]
    assert got.tobytes() == sc.expand_controls(nl, {"ctopo": 0.8}, device=False)["ctopo"].tobytes()
    spec = {"parameters": [{"name": "drag", "control": "ctopo", "domains": ["d01"]}]}
    terms = spc.expand_from_basis(basis, manifest, spc.member_multipliers(spec, [np.log(0.8)]))["d01"]["ctopo"]
    assert np.array_equal(terms, (1.0+np.exp(np.log(0.8))*(CTOPO.astype(np.float64)-1.0)).astype(np.float32))
    with pytest.raises(ValueError, match="per-vegetation-class"):
        spc.expand_from_basis(basis, manifest, {"d01": {"ctopo": [((5,), 1.2)]}})
    with pytest.raises(ValueError, match="no controllable domain"):
        spc.write_basis({"d01": _namelist(static=False)}, tmp_path/"none.npz")


SWISS = Path("<DATA_ROOT>/wrf_gpu2/archive/v040/v-prep/F20_20261016/Swiss/input")


@pytest.mark.skipif(not (SWISS/"namelist.input").is_file(), reason="local Swiss case not present")
def test_cli_refuses_host_controls_with_checkpoint_or_resume(tmp_path, capsys):
    from gpuwrf.cli import main
    controls = tmp_path/"ctopo.npz"
    sc.save_controls(controls, {"d01": {"ctopo": np.ones((3, 3), np.float32)}})
    base = ["run", "--input-dir", str(SWISS), "--output-dir", str(tmp_path/"out"), "--hours", "3",
            "--surface-controls", str(controls), "--dry-run"]
    assert main([*base, "--checkpoint-dir", str(tmp_path/"ck"), "--checkpoint-interval-steps", "10"]) != 0
    assert "cannot be combined with checkpoint/resume" in capsys.readouterr().err
    traced = tmp_path/"z0.npz"
    sc.save_controls(traced, {"d01": {"z0mvt": np.ones((3, 3), np.float32)}})
    base[base.index(str(controls))] = str(traced)
    main([*base, "--checkpoint-dir", str(tmp_path/"ck2"), "--checkpoint-interval-steps", "10"])
    assert "cannot be combined with checkpoint/resume" not in capsys.readouterr().err


def test_nested_two_way_feedback_refuses_host_controls():
    """_load_domains refuses before the initial feedback would recompute CTOPO over the bound value (source pin +
    the predicate the pipeline calls)."""
    import inspect

    from gpuwrf.integration import nested_pipeline as pipeline
    source = inspect.getsource(pipeline._load_domains)
    guard = source.index("has_host_controls(config.surface_controls)")
    assert guard < source.index("initialize_two_way_domains(")
    assert "allow_host=False" in inspect.getsource(pipeline._bind_batch_surface_controls)
    path = {"d01": {"ctopo": CTOPO}, "d02": {"z0mvt": CTOPO}}
    assert sc.has_host_controls(path)
    assert json.dumps(sorted(sc.HOST_CONTROLS)) == '["ctopo"]'
