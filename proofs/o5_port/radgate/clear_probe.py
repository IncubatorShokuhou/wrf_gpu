"""LW clear-sky truth check of rrtmg_precision_gate under both McICA overlaps (o5-port, measurement only)."""
import json, sys
from pathlib import Path
import numpy as np
tree = Path(sys.argv[1]); sys.path.insert(0, str(tree / "tests/v025/b_phys"))
import importlib.util
spec = importlib.util.spec_from_file_location("g", tree / "tests/v025/b_phys/rrtmg_precision_gate.py"); g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
import jax
from gpuwrf.kernels import rad_mcica
truth = json.loads((tree / "tests/v025/b_rad/data/lw_driver_clear_flux.json").read_text())
print("truth keys", sorted(k for k in truth if k not in ("fluxes",)), {k: truth[k] for k in truth if k not in ("fluxes", "fields")})
expected = np.asarray(truth["fluxes"])
out = {"imported_maxrand": bool(rad_mcica._MAXRAND), "backend": jax.default_backend()}
for maxrand in (True, False):
    with g.overlap(maxrand):
        state, _ = g.T1.load_lw_fixture_state()
        res = jax.block_until_ready(g.LW.solve_rrtmg_lw_column(state, with_clear_sky=True))
    rec = {}
    for i, field in enumerate(truth["fields"]):
        a = np.asarray(getattr(res, field)); e = expected[..., i]
        err = np.abs(a - e); allow = truth["flux_tolerance"]["atol"] + truth["flux_tolerance"]["rtol"] * np.abs(e)
        rec[field] = dict(max_abs=float(err.max()), passed=bool(np.all(err <= allow)))
    out[f"maxrand={maxrand}"] = rec
print(json.dumps(out, indent=1))
