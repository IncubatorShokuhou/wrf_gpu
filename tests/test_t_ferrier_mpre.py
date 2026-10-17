"""SSiB mixed radiation honours WRF's resolved has_reqc/i/s selection."""
from dataclasses import replace

import jax
import numpy as np
import pytest


@pytest.mark.parametrize("use_mp_re", [0, 1])
def test_ssib_two_radiation_calls_without_mp_radii(cpu_pallas_interpret, tmp_path, monkeypatch, use_mp_re):
    from test_s2_ssib_wiring import _case
    from gpuwrf.physics import rrtmg_mp_re
    from gpuwrf.runtime import operational_mode as op

    # Match the CLI profile that revealed c40. This import-time flag alone
    # must not override physics_init's LW AND SW selection predicate.
    monkeypatch.setattr(rrtmg_mp_re, "_MP_RE", True)
    _, state, nl, _ = _case(tmp_path)
    nl = replace(nl, use_mp_re=use_mp_re)
    carry = op._initial_carry_for_run(state, nl)
    assert not op.mp_re_active(nl)
    assert all(getattr(carry.state, n) is None for n in ("re_cloud", "re_ice", "re_snow"))
    clock = op.build_clock_base(nl)
    # Actual runtime refresh -> SSiB -> actual Dudhia and RRTMG-LW solvers.
    # Two forcing calls exercise the caller before and after held radiation
    # updates, with the unseeded radius fields WRF specifies for this pair.
    for step in (1, 2):
        carry = op._refresh_rrtmg_driver(carry, nl, (step-1)*nl.dt_s, True, clock)
        jax.block_until_ready(carry.rthraten)
        for leaf in jax.tree.leaves((carry.rthraten, carry.radiation_diagnostics)):
            assert np.isfinite(np.asarray(leaf)).all()
        assert all(getattr(carry.state, n) is None for n in ("re_cloud", "re_ice", "re_snow"))


def test_ssib_forwards_qualified_selection(cpu_pallas_interpret, tmp_path, monkeypatch):
    """Capture the builder boundary, including an explicitly qualified pair."""
    from test_s2_ssib_wiring import _case
    from gpuwrf.coupling import ssib_surface_hook as hook, physics_couplers as coupling
    from gpuwrf.physics import rrtmg_mp_re

    monkeypatch.setattr(rrtmg_mp_re, "_MP_RE", True)
    _, state, nl, carry = _case(tmp_path)
    seen = []
    monkeypatch.setattr(hook, "ssib_dudhia_columns", lambda *args: (None, None, None))
    class BoundaryReached(Exception):
        pass
    def capture(*args, **kw):
        seen.append((kw["use_mp_re"], kw["mp_physics"]))
        raise BoundaryReached
    monkeypatch.setattr(coupling, "_rrtmg_column_inputs", capture)
    for sw, use, mp, expected in ((1,1,8,0), (1,0,8,0), (4,1,8,1), (4,1,5,0)):
        selected = replace(nl, ra_sw_physics=sw, use_mp_re=use, mp_physics=mp)
        with pytest.raises(BoundaryReached):
            hook.ssib_radiation(state, carry.ssib_land, nl.ssib_static, selected, 0., None)
        assert seen[-1] == (expected, mp)
