"""Restart authentication must cover the bound CTOPO operand, without model edits."""
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwrf.surface_params import controls as sc


@dataclass(frozen=True)
class Namelist:
    bl_pbl_physics: int = 1
    topo_wind_static: object = None
    surface_controls: object = None
    noahmp_energy_params: object = None


def namelist():
    from gpuwrf.io.topo_wind_static import TopoWindStatic
    return Namelist(topo_wind_static=TopoWindStatic(np.array([[1., 2., 4.]], np.float32), None))


def test_host_control_restart_hashes_actual_consumer_and_refuses_changes():
    base = namelist()
    fields = {"ctopo": np.array([[1., 3., 7.]], np.float32)}
    bound = sc.bind_domain_controls(base, {"d01": fields}, "d01")
    assert bound.surface_controls is None  # no new clock-base operand
    assert sc.restart_controls_digest(bound) == sc.controls_digest(fields)
    assert sc.restart_controls_digest(base) != sc.restart_controls_digest(bound)
    recorded = sc.restart_controls_digest(bound)
    sc.validate_controls_restart(recorded, bound)
    for other in (base, replace(bound, topo_wind_static=None)):
        with pytest.raises(ValueError, match="differ"):
            sc.validate_controls_restart(recorded, other)
    with pytest.raises(ValueError, match="differ"):
        sc.validate_controls_restart(None, bound)  # old, unauthenticated host operand


def test_mixed_surface_digest_and_legacy_non_topographic_compatibility():
    traced = {"z0mvt": np.array([[.1, .2, .3]], np.float32)}
    topo = namelist().topo_wind_static
    mixed = SimpleNamespace(surface_controls=traced, topo_wind_static=topo)
    expected = sc.controls_digest({**traced, "ctopo": topo.ctopo})
    assert sc.restart_controls_digest(mixed) == expected
    assert sc.restart_controls_digest(vars(mixed)) == expected
    assert sc.restart_controls_digest({"surface_controls": traced}) == sc.controls_digest(traced)
    assert sc.restart_controls_digest({}) is None
    assert sc.restart_controls_digest(None) is None


def test_digest_uses_bound_array_instead_of_source_file(tmp_path):
    fields = {"ctopo": np.array([[1., 3., 7.]], np.float32)}
    source = tmp_path / "controls.npz"
    sc.save_controls(source, {"d01": fields})
    bound = sc.bind_domain_controls(namelist(), source, "d01")
    recorded = sc.restart_controls_digest(bound)
    sc.save_controls(source, {"d01": {"ctopo": fields["ctopo"] * 2}})
    assert sc.restart_controls_digest(bound) == recorded
    rebound = sc.bind_domain_controls(namelist(), source, "d01")
    with pytest.raises(ValueError, match="differ"):
        sc.validate_controls_restart(recorded, rebound)
