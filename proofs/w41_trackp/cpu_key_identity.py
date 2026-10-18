"""Track P CPU gate: traced surface controls change values, never the program (real Swiss d01, CPU).

usage: JAX_PLATFORMS=cpu python cpu_key_identity.py <src_root> <out.json> [steps]
Arms on ONE loaded case (REAL CLI loader nested_pipeline._load_domains, Pallas interpreted for load, trace and
execution as O1_INTERPRET_EXEC=1; tridiagonal_solve -> pure-JAX Thomas, E105):
  OFF    released program (no controls)
  PRIOR  all 11 controls = byte copies of the bound table values
  A, B   two different perturbed parameter sets (per-class/uniform multipliers)
Recorded per arm: location-stripped jaxpr sha of the production step operational_mode._advance_chunk_fori (E151),
aot_cheap_key.cheap_key of the exact AOT call contract (domain_tree), jit cache size, and after ``steps`` executed
steps a sha256 per carry leaf. Gates: jaxpr(PRIOR)=jaxpr(A)=jaxpr(B); cheap_key(A)=cheap_key(B)=cheap_key(PRIOR);
one jit compile serves PRIOR, A and B; carry(PRIOR) == carry(OFF) bitwise; carry(A) != carry(OFF), != carry(B).
"""
import dataclasses
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

SRC, OUT = str(Path(sys.argv[1]).resolve()), sys.argv[2]
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 2
sys.path.insert(0, SRC + "/src")
sys.path.insert(0, SRC + "/proofs/o1_nlbind")
import jax  # noqa: E402
import numpy as np  # noqa: E402

import cpu_safe  # noqa: E402

cpu_safe.install()
from gpuwrf.contracts import state as state_contract  # noqa: E402
from gpuwrf.integration import nested_pipeline as pipeline  # noqa: E402
from gpuwrf.runtime import aot_cheap_key as ck  # noqa: E402
from gpuwrf.runtime import operational_mode as om  # noqa: E402
from gpuwrf.surface_params import controls as sc  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402

assert jax.devices()[0].platform == "cpu"
assert om.__file__.startswith(SRC), om.__file__
state_contract._gpu_device = lambda: jax.devices("cpu")[0]
CASE = Path(os.environ.get("TRACKP_CASE", "<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu"))
LANE = Path("<USER_HOME>/wrf_gpu2_lanes/w41-trackp")
_real_pallas_call = pl.pallas_call


def _interpret_pallas_call(*a, **k):
    k["interpret"] = True
    k.pop("compiler_params", None)
    return _real_pallas_call(*a, **k)


pl.pallas_call = _interpret_pallas_call


def canon(jaxpr) -> str:
    text = str(jaxpr)
    text = re.sub(r"name_and_src_info=[^\]\n]*", "", text)
    return re.sub(r"/[^ \n:]*\.py:\d+(:\d+)?", "", text)


def leaf_hashes(tree):
    out = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        arr = np.ascontiguousarray(np.asarray(leaf))
        out[jax.tree_util.keystr(path)] = hashlib.sha256(arr.tobytes()).hexdigest()[:16]
    return out


t0 = time.time()
with tempfile.TemporaryDirectory(dir=LANE) as tmp:
    case = Path(tmp) / "case"
    case.mkdir()
    for f in CASE.iterdir():
        (case / f.name).symlink_to(f)
    cfg = pipeline.NestedPipelineConfig(case, Path(tmp) / "out", Path(tmp) / "proof", hours=1, max_dom=1)
    _h, bundles, _x, _rs, _dt, carries = pipeline._load_domains(cfg, ("d01",))
ns_off, carry0 = bundles["d01"].namelist, carries["d01"]
om._resolve_operational_suite(ns_off)
cadence = int(ns_off.radiation_cadence_steps)
names = sorted(sc.CONTROLS)
vt = np.asarray(ns_off.noahmp_static.ivgtyp)
cats = [int(c) for c, n in zip(*np.unique(vt, return_counts=True))]
# two parameter sets (physically bounded): per-class roughness, soil/leaf albedo, emissivity, LAI, stomata, FVEG
mult_a = {"z0mvt": {c: 1.5 for c in cats[: max(1, len(cats) // 2)]}, "albsat": 1.2, "albdry": 1.2, "rhol": 0.9,
          "rhos": 1.0, "eg": 0.99, "laim": 1.3, "saim": 1.0, "vcmx25": 0.8, "mp": 1.1, "shdmax": 0.9}
mult_b = {"z0mvt": 0.7, "albsat": 0.9, "albdry": 0.95, "rhol": 1.05, "rhos": 0.95, "eg": 1.0, "laim": 0.8,
          "saim": 1.2, "vcmx25": 1.2, "mp": 0.9, "shdmax": 1.0}
arms = {
    "OFF": ns_off,
    "PRIOR": dataclasses.replace(ns_off, surface_controls=sc.prior_controls(ns_off, names)),
    "A": dataclasses.replace(ns_off, surface_controls=sc.expand_controls(ns_off, mult_a)),
    "B": dataclasses.replace(ns_off, surface_controls=sc.expand_controls(ns_off, mult_b)),
}
out = {"src": SRC, "case": str(CASE), "steps": STEPS, "load_s": round(time.time() - t0, 1),
       "grid_shape": list(sc.grid_shape(ns_off)), "veg_categories": cats, "mult_a": repr(mult_a),
       "mult_b": repr(mult_b), "arms": {}}
start = jax.numpy.asarray(1, jax.numpy.int32)
kwargs = {"n_steps": 1, "cadence": cadence}
for arm, nl in arms.items():
    rec = {}
    t1 = time.time()
    cb = om.build_clock_base(nl)
    rec["controls_digest"] = sc.controls_digest(nl.surface_controls)
    jx = jax.make_jaxpr(lambda c, nl=nl, cb=cb: om._advance_chunk_fori(c, nl, 1, cb, n_steps=1, cadence=cadence))(carry0)
    text = canon(jx)
    rec["jaxpr_sha"] = hashlib.sha256(text.encode()).hexdigest()
    rec["cheap_key"] = ck.cheap_key(om._advance_chunk_fori, (carry0, nl, start, cb), kwargs, nl)
    rec["trace_s"] = round(time.time() - t1, 1)
    c = jax.tree.map(lambda x: x.copy() if hasattr(x, "copy") else x, carry0)
    t2 = time.time()
    for i in range(STEPS):
        c = om._advance_chunk_fori(c, nl, i + 1, cb, n_steps=1, cadence=cadence)
        jax.block_until_ready(c)
    rec["exec_s"] = round(time.time() - t2, 1)
    rec["jit_cache_size_after"] = int(om._advance_chunk_fori._cache_size())
    rec["leaf_sha"] = leaf_hashes(c)
    land = c.noahmp_land
    rec["land_means"] = {k: float(np.nanmean(np.asarray(getattr(land, k), np.float64)))
                         for k in ("znt", "albedo", "emiss", "lai", "t_skin", "qsfc")}
    rec["state_means"] = {k: float(np.mean(np.asarray(getattr(c.state, k), np.float64)))
                          for k in ("hfx", "lh", "theta", "u", "qv") if getattr(c.state, k, None) is not None}
    rec["nonfinite"] = int(sum(int((~np.isfinite(np.asarray(x))).sum()) for x in jax.tree.leaves(c)
                               if np.issubdtype(np.asarray(x).dtype, np.floating)))
    out["arms"][arm] = rec
    Path(OUT).write_text(json.dumps(out, indent=1, default=str))
    print(arm, {k: v for k, v in rec.items() if k != "leaf_sha"}, flush=True)

a = out["arms"]
diff = lambda x, y: sorted(k for k in a[x]["leaf_sha"] if a[x]["leaf_sha"][k] != a[y]["leaf_sha"][k])  # noqa: E731
out["gates"] = {
    "jaxpr_controls_equal": a["PRIOR"]["jaxpr_sha"] == a["A"]["jaxpr_sha"] == a["B"]["jaxpr_sha"],
    "jaxpr_off_differs": a["OFF"]["jaxpr_sha"] != a["PRIOR"]["jaxpr_sha"],
    "cheap_key_controls_equal": a["PRIOR"]["cheap_key"] == a["A"]["cheap_key"] == a["B"]["cheap_key"] is not None,
    "one_compile_for_controls": a["B"]["jit_cache_size_after"] == a["PRIOR"]["jit_cache_size_after"],
    "prior_bitwise_off": diff("PRIOR", "OFF") == [],
    "prior_vs_off_differing_leaves": diff("PRIOR", "OFF"),
    "a_vs_off_differing_leaves": len(diff("A", "OFF")),
    "a_vs_b_differing_leaves": len(diff("A", "B")),
    "finite": all(r["nonfinite"] == 0 for r in a.values()),
}
out["gates"]["PASS"] = (out["gates"]["jaxpr_controls_equal"] and out["gates"]["jaxpr_off_differs"]
                        and out["gates"]["cheap_key_controls_equal"] and out["gates"]["one_compile_for_controls"]
                        and out["gates"]["prior_bitwise_off"] and out["gates"]["a_vs_off_differing_leaves"] > 0
                        and out["gates"]["a_vs_b_differing_leaves"] > 0 and out["gates"]["finite"])
Path(OUT).write_text(json.dumps(out, indent=1, default=str))
print("GATES", out["gates"], flush=True)
