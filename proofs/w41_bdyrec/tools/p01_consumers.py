"""P01: windowed boundary consumers == full stack (bitwise), NaN outside the window."""
import os, sys, itertools
sys.path.insert(0, sys.argv[1] + "/src")
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from gpuwrf.coupling import boundary_apply as ba
from gpuwrf.coupling.boundary_records import lateral_window_scope
from gpuwrf.kernels import dyn_boundary_fp32 as kb
assert jax.devices()[0].platform == "cpu"
rng = np.random.default_rng(1)
R, W, I = 13, 4, 21600.0
leaf = rng.standard_normal((R, 4, 5, 6, 9)).astype(np.float32) * 10 + 280
def window_of(base):
    rows = np.minimum(np.arange(base, base + W), R - 1)
    return jnp.asarray(leaf[rows]), jnp.asarray(np.array([base, R - 2], np.int32))
leads = sorted({0.0, 54.0, I, I + 18.0, 2 * I - 18, 5.5 * I, (R - 1) * I, (R - 1) * I + 18.0, (R - 1) * I + 54.0,
                3 * I + 1.0, 7 * I})
bad = 0; checked = 0; poisoned = 0
funcs = {
  "interp_legacy": lambda l, lead: ba.interpolate_boundary_leaf(l, lead, I),
  "tend": lambda l, lead: ba.boundary_tendency_leaf(l, lead, I),
  "root_rate": lambda l, lead: jnp.stack(ba._root_record_value_rate_real4(l, lead, I)),
  "scalar_rate": lambda l, lead: jnp.stack(ba._scalar_record_value_rate_real4(l, lead, I)),
  "interp_native": lambda l, lead: kb.interpolate_leaf(l, lead, I) if not hasattr(l, "_w") else None,
}
for name, fn in funcs.items():
    for lead in leads:
        full = np.asarray(jax.jit(fn)(jnp.asarray(leaf), jnp.asarray(lead)) if name != "interp_native" else kb.interpolate_leaf(jnp.asarray(leaf), jnp.asarray(lead), I))
        g = lead / I
        for base in range(0, R - 1):
            wl, ww = window_of(base)
            if name == "interp_native":
                out = np.asarray(kb.interpolate_leaf(wl, jnp.asarray(lead), I, ww))
            else:
                def windowed(l, lead_v, w, fn=fn):
                    with lateral_window_scope(w):
                        return fn(l, lead_v)
                out = np.asarray(jax.jit(windowed)(wl, jnp.asarray(lead), ww))
            nan = np.isnan(out).all()
            if nan:
                poisoned += 1
                continue
            checked += 1
            if not np.array_equal(out.view(np.uint8), full.view(np.uint8)):
                bad += 1
                print("MISMATCH", name, lead, base, np.abs(out - full).max())
print({"checked": checked, "poisoned": poisoned, "bad": bad})
