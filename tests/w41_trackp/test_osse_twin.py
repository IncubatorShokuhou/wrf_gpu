"""P3 OSSE: EKI recovers a known two-group roughness truth through the traced controls of the real Noah-MP,
with ONE compiled member-vmapped land program for every member and iteration (no recompile per parameter set)."""
from __future__ import annotations

import numpy as np

from w41_trackp_fixtures import pytestmark  # noqa: F401  (needs the pristine MPTABLE)
from gpuwrf.surface_params.osse import TwinConfig, run_twin


def test_twin_recovers_two_group_roughness():
    cfg = TwinConfig(hours=8.0, dt_s=600.0, members=12, iterations=3, sigma_t2=0.2, sigma_hfx=10.0)
    out = run_twin(cfg)
    truth, est = np.asarray(out["truth_multipliers"]), np.asarray(out["estimate_multipliers"])
    sd = np.asarray(out["estimate_sd_log"])
    assert np.all(np.abs(np.log(est / truth)) < np.maximum(3.0 * sd, 0.05)), (truth, est, sd)
    assert np.all(np.abs(est / truth - 1.0) < 0.25), (truth, est)
    # normalised misfit = chi^2 / m of noisy synthetic obs: its floor is ~1 (noise), so the gate is "decreases and
    # ends inside the chi^2 noise band", not a fixed reduction factor (the prior sits only ~0.3 above the floor)
    misfit, m = out["misfit_history"], out["observations"]
    assert misfit[-1] < misfit[0], misfit
    assert abs(misfit[-1] - 1.0) < 3.0 * np.sqrt(2.0 / m), (misfit, m)
    assert out["jit_compiles"] == 1 and out["forward_calls"] == cfg.iterations + 2
