"""Track P traced surface controls: table-edit oracle, active consumers, validation, files, namelist aux.

The oracle: a control set built by ``expand_controls`` must make Noah-MP produce BYTE-identical output to the
released code path run on an MPTABLE (or wrfinput SHDMAX) edited the same way — i.e. a traced control is exactly
"the table value WRF reads at that consumption site", for every control. A perturbation must reach the named
consumer (fluxes/state change where the parameter acts, nothing changes where it cannot act).
"""
from __future__ import annotations

import dataclasses
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from w41_trackp_fixtures import SHAPE, VEG, pytestmark, _fixture, _namelist  # noqa: F401
from gpuwrf.physics.noahmp.noahmp_driver import build_energy_params, noah_mp_step  # noqa: E402
from gpuwrf.surface_params import controls as sc  # noqa: E402


def _bytes(tree):
    return [np.asarray(x).tobytes() for x in jax.tree.leaves(tree)]


def _run(land, forcing, static, ep, rp):
    return noah_mp_step(land, forcing, static, 60.0, energy_params=ep, rad_params=rp)


def _table_edit(static, name, factor_by_veg):
    """Released code path on an edited table (the WRF view: MPTABLE row x factor)."""
    p = static.parameters
    if name == "shdmax":
        m = np.ones(SHAPE)
        for cat, f in factor_by_veg.items():
            m[VEG == cat] = f
        return static.replace(shdmax=jnp.asarray(np.asarray(static.shdmax, np.float64) * m))
    if name in ("eg", "albsat", "albdry"):  # uniform tables (EG(1), soil colour 4): uniform factor only
        (factor,) = set(factor_by_veg.values())
        return static.replace(parameters=p._replace(**{name: jnp.asarray(np.asarray(getattr(p, name)) * factor)}))
    table = np.array(getattr(p, name), dtype=np.float64)
    for cat, f in factor_by_veg.items():
        table[cat] = table[cat] * f
    return static.replace(parameters=p._replace(**{name: jnp.asarray(table)}))


EDITS = {
    "z0mvt": {5: 1.5, 12: 0.6},
    "vcmx25": {2: 0.7, 10: 1.3},
    "mp": {5: 1.2, 7: 0.8},
    "rhol": {10: 1.1, 12: 0.9},
    "rhos": {2: 1.2},
    "laim": {10: 1.4, 5: 0.8},
    "saim": {12: 1.5},
    "shdmax": {7: 1.3, 2: 0.9},
    "eg": {2: 0.98, 5: 0.98, 10: 0.98, 12: 0.98, 7: 0.98},
    "albsat": {2: 1.2, 5: 1.2, 10: 1.2, 12: 1.2, 7: 1.2},
    "albdry": {2: 0.85, 5: 0.85, 10: 0.85, 12: 0.85, 7: 0.85},
}


@pytest.mark.parametrize("name", sorted(EDITS))
def test_control_equals_table_edit_bytes(name):
    # ALBDRY caps GROUNDALB only on dry soil (min(ALBSAT + INC, ALBDRY), INC = max(0.11 - 0.4 SMC, 0))
    land, forcing, static = _fixture(smois=0.05 if name == "albdry" else 0.25)
    nl = _namelist(static)
    edited = _table_edit(static, name, EDITS[name])
    ep_e, rp_e = build_energy_params(edited, SHAPE)
    want = _run(land, forcing, edited, ep_e, rp_e)
    mult = EDITS[name] if name not in ("eg", "albsat", "albdry") else next(iter(EDITS[name].values()))
    controls = sc.expand_controls(nl, {name: mult})
    s2, ep2, rp2 = sc.apply_surface_controls(static, nl.noahmp_energy_params, nl.noahmp_rad_params, controls)
    got = _run(land, forcing, s2, ep2, rp2)
    assert _bytes(got) == _bytes(want), f"{name}: traced control differs from the edited-table program"
    base = _run(land, forcing, static, nl.noahmp_energy_params, nl.noahmp_rad_params)
    assert _bytes(got) != _bytes(base), f"{name}: perturbation did not reach any Noah-MP output"


def test_prior_controls_are_byte_copies_and_identity():
    land, forcing, static = _fixture()
    nl = _namelist(static)
    prior = sc.prior_controls(nl)
    assert sorted(prior) == sorted(sc.CONTROLS)
    for name in ("z0mvt", "eg", "vcmx25", "mp"):
        assert np.asarray(prior[name]).tobytes() == np.asarray(getattr(nl.noahmp_energy_params, name)).tobytes()
    for name in ("albsat", "albdry", "rhol", "rhos"):
        assert np.asarray(prior[name]).tobytes() == np.asarray(getattr(nl.noahmp_rad_params, name)).tobytes()
    s2, ep2, rp2 = sc.apply_surface_controls(static, nl.noahmp_energy_params, nl.noahmp_rad_params, prior)
    base = _run(land, forcing, static, nl.noahmp_energy_params, nl.noahmp_rad_params)
    assert _bytes(_run(land, forcing, s2, ep2, rp2)) == _bytes(base)


CONSUMERS = {  # control -> (factor, output that must change, output that must NOT change)
    "z0mvt": (1.5, "znt", None),
    "albsat": (1.3, "albedo", None),
    "rhol": (1.2, "albedo", None),
    "eg": (0.95, "emiss", None),
    "laim": (1.5, "lai", None),
    "saim": (1.5, "sai", "lai"),
    "vcmx25": (0.5, "lh", "albedo"),
    "mp": (0.5, "lh", "albedo"),
    "shdmax": (0.5, "hfx", None),
}


@pytest.mark.parametrize("name", sorted(CONSUMERS))
def test_control_reaches_named_consumer(name):
    land, forcing, static = _fixture()
    nl = _namelist(static)
    factor, changes, invariant = CONSUMERS[name]
    base_land, base_flux = _run(land, forcing, static, nl.noahmp_energy_params, nl.noahmp_rad_params)
    s2, ep2, rp2 = sc.apply_surface_controls(static, nl.noahmp_energy_params, nl.noahmp_rad_params,
                                             sc.expand_controls(nl, {name: factor}))
    new_land, new_flux = _run(land, forcing, s2, ep2, rp2)
    pick = lambda land_, flux, key: np.asarray(getattr(flux, key, None) if hasattr(flux, key) else getattr(land_, key))  # noqa: E731
    assert not np.array_equal(pick(base_land, base_flux, changes), pick(new_land, new_flux, changes))
    if invariant is not None:
        assert np.array_equal(pick(base_land, base_flux, invariant), pick(new_land, new_flux, invariant))


def test_validation_refuses_bad_or_inactive_controls():
    _land, _forcing, static = _fixture()
    nl = _namelist(static)
    good = sc.expand_controls(nl, {"z0mvt": 1.0}, device=False)
    with pytest.raises(ValueError, match="unknown"):
        sc.validate_controls({"z0": good["z0mvt"]}, nl)
    with pytest.raises(ValueError, match="shape"):
        sc.validate_controls({"z0mvt": good["z0mvt"][:1]}, nl)
    with pytest.raises(ValueError, match="dtype"):
        sc.validate_controls({"z0mvt": good["z0mvt"].astype(np.float16)}, nl)
    bad = good["z0mvt"].copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        sc.validate_controls({"z0mvt": bad}, nl)
    with pytest.raises(ValueError, match="outside"):
        sc.expand_controls(nl, {"z0mvt": 1.0e3})
    with pytest.raises(ValueError, match="negative absorption"):  # in bounds, but rho + tau > 1 (nir)
        sc.expand_controls(nl, values={"rhol": np.full((2,) + SHAPE, 0.9)})
    with pytest.raises(ValueError, match="empty"):
        sc.validate_controls({}, nl)
    from gpuwrf.io.noahmp_options import NoahMPOptions
    nl_dyn = _namelist(static, NoahMPOptions(dveg=2))
    with pytest.raises(ValueError, match="no active consumer"):
        sc.expand_controls(nl_dyn, {"laim": 1.1})
    nl_jarvis = _namelist(static, NoahMPOptions(opt_crs=2))
    with pytest.raises(ValueError, match="no active consumer"):
        sc.expand_controls(nl_jarvis, {"vcmx25": 1.1})
    with pytest.raises(ValueError, match="Noah-MP only"):
        sc.expand_controls(types.SimpleNamespace(use_noahmp=False), {"z0mvt": 1.0})


def test_file_roundtrip_and_tamper(tmp_path):
    _land, _forcing, static = _fixture()
    nl = _namelist(static)
    controls = sc.expand_controls(nl, {"z0mvt": {5: 2.0}, "laim": 0.9}, device=False)
    path = tmp_path / "controls.npz"
    manifest = sc.save_controls(path, {"d01": controls}, provenance={"test": True})
    loaded, man2 = sc.load_controls(path)
    assert man2 == manifest
    assert sc.controls_digest(loaded["d01"]) == sc.controls_digest(controls) == manifest["domains"]["d01"]["digest"]
    data = dict(np.load(path))
    data["d01/z0mvt"] = data["d01/z0mvt"] * 1.01
    np.savez(tmp_path / "tampered.npz", **data)
    with pytest.raises(ValueError, match="digest mismatch"):
        sc.load_controls(tmp_path / "tampered.npz")
    bound = sc.bind_domain_controls(_Bindable(nl), path, "d01")
    assert sc.controls_digest(bound.surface_controls) == sc.controls_digest(controls)
    assert sc.bind_domain_controls(_Bindable(nl), path, "d02").surface_controls is None


@dataclasses.dataclass(frozen=True)
class _Bindable:
    base: object
    surface_controls: object = None

    def __getattr__(self, key):
        return getattr(self.base, key)


def test_traced_controls_guard_and_restart_digest():
    _land, _forcing, static = _fixture()
    nl = _namelist(static)
    controls = sc.expand_controls(nl, {"z0mvt": 1.2})
    holder = types.SimpleNamespace(surface_controls=controls)
    assert sc.traced_controls(holder, types.SimpleNamespace(surface_controls=controls)) is controls
    with pytest.raises(ValueError, match="traced clock base"):
        sc.traced_controls(holder, None)
    assert sc.traced_controls(types.SimpleNamespace(surface_controls=None), None) is None
    other = sc.expand_controls(nl, {"z0mvt": 1.3})
    rec = sc.restart_controls_digest(holder)
    sc.validate_controls_restart(rec, holder)
    with pytest.raises(ValueError, match="differ"):
        sc.validate_controls_restart(rec, types.SimpleNamespace(surface_controls=other))
    with pytest.raises(ValueError, match="differ"):
        sc.validate_controls_restart(None, holder)  # pre-Track-P checkpoint, controls requested
    with pytest.raises(ValueError, match="differ"):
        sc.validate_controls_restart(rec, {"surface_controls": None})


def test_aux_holder_is_content_blind():
    from gpuwrf.runtime.aot_cheap_key import canonical_digest
    _land, _forcing, static = _fixture()
    nl = _namelist(static)
    a = sc.controls_aux(sc.expand_controls(nl, {"z0mvt": 1.2, "laim": 0.8}))
    b = sc.controls_aux(sc.expand_controls(nl, {"z0mvt": 0.7, "laim": 1.1}))
    c = sc.controls_aux(sc.expand_controls(nl, {"z0mvt": 0.7}))
    assert a == b and hash(a) == hash(b) and canonical_digest(a) == canonical_digest(b)
    assert a != c and canonical_digest(a) != canonical_digest(c)
    assert sc.controls_digest(a.value) != sc.controls_digest(b.value)


def test_namelist_aux_roundtrip_and_clock_base():
    from datetime import datetime, timezone

    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.runtime import aot_cheap_key as ck
    from gpuwrf.runtime.operational_mode import OperationalNamelist, build_clock_base

    _land, _forcing, static = _fixture()
    nl = _namelist(static)
    a = sc.expand_controls(nl, {"z0mvt": 1.2, "laim": 0.9})
    b = sc.expand_controls(nl, {"z0mvt": 0.8, "laim": 1.1})
    base = OperationalNamelist(GridSpec.canary_3km_template(), None, None, 18., mp_physics=0, ra_sw_physics=0,
                               ra_lw_physics=0, sf_sfclay_physics=0, cu_physics=0,
                               time_utc=datetime(2026, 6, 1, tzinfo=timezone.utc))
    na, nb = dataclasses.replace(base, surface_controls=a), dataclasses.replace(base, surface_controls=b)
    # released treedef unchanged without controls; with controls: same treedef/static key for any values
    assert jax.tree_util.tree_structure(base) == jax.tree_util.tree_structure(dataclasses.replace(base))
    assert jax.tree_util.tree_structure(na) == jax.tree_util.tree_structure(nb)
    assert jax.tree_util.tree_structure(na) != jax.tree_util.tree_structure(base)
    assert ck.static_config_hash(na) == ck.static_config_hash(nb) != ck.static_config_hash(base)
    assert len(jax.tree.leaves(na)) == len(jax.tree.leaves(base))  # values are not namelist leaves
    back = jax.tree_util.tree_unflatten(*reversed(jax.tree_util.tree_flatten(na)))
    assert sc.controls_digest(back.surface_controls) == sc.controls_digest(a)
    cb = build_clock_base(na)
    assert cb.surface_controls is not None and sc.controls_digest(cb.surface_controls) == sc.controls_digest(a)
    assert build_clock_base(base).surface_controls is None
    cba, cbb = build_clock_base(na), build_clock_base(nb)
    assert jax.tree_util.tree_structure(cba) == jax.tree_util.tree_structure(cbb)
    assert [x.shape for x in jax.tree.leaves(cba)] == [x.shape for x in jax.tree.leaves(cbb)]


def test_namelist_roundtrip_with_topo_wind_and_controls():
    """Both optional namelist entries together (merge seam with w41-topowind): LIFO unflatten restores each one."""
    from datetime import datetime, timezone

    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.runtime.operational_mode import OperationalNamelist

    _land, _forcing, static = _fixture()
    ctl = sc.expand_controls(_namelist(static), {"z0mvt": 1.2})
    base = OperationalNamelist(GridSpec.canary_3km_template(), None, None, 18., mp_physics=0, ra_sw_physics=0,
                               ra_lw_physics=0, sf_sfclay_physics=0, cu_physics=0,
                               time_utc=datetime(2026, 6, 1, tzinfo=timezone.utc))
    topo = jnp.arange(6.0).reshape(2, 3)  # any traced child stands in for TopoWindStatic here
    for nl in (dataclasses.replace(base, topo_wind_static=topo, surface_controls=ctl),
               dataclasses.replace(base, topo_wind_static=topo),
               dataclasses.replace(base, surface_controls=ctl)):
        back = jax.tree_util.tree_unflatten(*reversed(jax.tree_util.tree_flatten(nl)))
        assert (back.topo_wind_static is None) == (nl.topo_wind_static is None)
        if nl.topo_wind_static is not None:
            assert np.array_equal(np.asarray(back.topo_wind_static), np.asarray(topo))
        assert (back.surface_controls is None) == (nl.surface_controls is None)
        if nl.surface_controls is not None:
            assert sc.controls_digest(back.surface_controls) == sc.controls_digest(ctl)
    both = dataclasses.replace(base, topo_wind_static=topo, surface_controls=ctl)
    assert len(jax.tree.leaves(both)) == len(jax.tree.leaves(dataclasses.replace(base, topo_wind_static=topo)))
