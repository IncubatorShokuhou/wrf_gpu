"""icloud=3 port vs verbatim WRF 4.7.1 cal_cldfra3 on censused real columns.

Gates (pre-registered):
  L  literal: op-by-op port (jax.disable_jit, host-libm log10f) == no-FMA pristine, BYTE on
     cldfra/qc/qi for every row.
  C1 compiled: jitted port (host log10f via pure_callback, everything else XLA-compiled)
     within |FMA build - base| + sum_v max|base(v +- 1 ulp) - base|, 0 elements outside.
  C2 disclosed: XLA's own log10 inside the fused small-condensate formula
     MIN(0.99, 0.1*(11+log10(qc+qi))) vs the correctly rounded literal over 4e6 log-uniform
     samples of the whole branch domain (1e-10, 1e-6): max ulp / abs of cf reported; the
     full product (XLA log10) outside-C1 count is reported too.
  (History: a first composition max(FMA, ULP, libm+-1ulp) left 27, sum left 10 elements
  outside, all in this log10 branch -> C1/C2 split; receipts OG1_max / OG1_sum.)
  K  census: every cal_cldfra3/find_cloudLayers/adjust_* branch active on >= 1 row.
usage: python oracle_gate.py FIXTURE.npz BUILD_DIR RECEIPT.json [SAVEPOINT.npz]
  SAVEPOINT freezes the pristine outputs, the C1 envelope and cal_cldfra2 (verbatim) for tests.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(HERE)]
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import oracle  # noqa: E402
from gpuwrf.physics import cloud_fraction as cf_mod  # noqa: E402

NAMES = ("qv", "qc", "qi", "qs", "dz", "p", "t")
OUTS = ("cldfra", "qc", "qi")


def host_log10(x, direction=None):
    # glibc 2.43 log10f is correctly rounded == f64 log10 rounded once to binary32.
    y = np.log10(np.asarray(x, np.float64)).astype(np.float32)
    if direction is not None:
        y = np.nextafter(y, np.float32(direction)).astype(np.float32)
    return jnp.asarray(y)


def literal_port(args, g, direction=None):
    saved = cf_mod._LOG10
    cf_mod._LOG10 = lambda x: host_log10(x, direction)
    try:
        with jax.disable_jit():
            return [np.asarray(a) for a in cf_mod.wrf_cal_cldfra3(*args, gridkm=g)]
    finally:
        cf_mod._LOG10 = saved


def main(fixture: Path, build: Path, receipt_path: Path, savepoint: Path | None = None) -> int:
    data = np.load(fixture)
    rows = {n: data[n] for n in (*NAMES, "xland", "gridkm")}
    info = {"base": oracle.build(build / "base")}
    fma_flags = ["-O2", "-mfma", "-ffp-contract=fast", "-ffree-form", "-ffree-line-length-none",
                 "-finit-real=zero", "-finit-integer=0", "-finit-logical=false"]
    saved = oracle.FLAGS
    oracle.FLAGS = fma_flags
    try:
        info["fma"] = oracle.build(build / "fma")
    finally:
        oracle.FLAGS = saved
    result = {"rows": int(rows["qv"].shape[0]), "nz": int(rows["qv"].shape[1]), "groups": {}}
    totals = {"L_diff": 0, "C_outside": 0, "C_exact": 0, "elements": 0}
    census = {}
    n_rows = rows["qv"].shape[0]
    frozen = {f"{o}{suffix}": np.zeros(rows["qv"].shape, np.float32 if suffix == "" else np.float64)
              for o in OUTS for suffix in ("", "_bound")}
    for g in np.unique(rows["gridkm"]):
        sel = rows["gridkm"] == g
        cols = {n: rows[n][sel] for n in (*NAMES, "xland")}
        base = oracle.run(build / "base", cols, float(g), "base")
        fma = oracle.run(build / "fma", cols, float(g), "fma")
        spread = {o: np.abs(fma[o].astype(np.float64) - base[o]) for o in OUTS}
        ulp = {o: np.zeros_like(spread[o]) for o in OUTS}
        for n in NAMES:
            worst = {o: np.zeros_like(spread[o]) for o in OUTS}
            for direction in (-np.inf, np.inf):
                moved = dict(cols, **{n: np.nextafter(cols[n], np.float32(direction)).astype(np.float32)})
                out = oracle.run(build / "base", moved, float(g), f"ulp_{n}")
                for o in OUTS:
                    worst[o] = np.maximum(worst[o], np.abs(out[o].astype(np.float64) - base[o]))
            for o in OUTS:
                ulp[o] += worst[o]
        args = [jnp.asarray(cols[n]) for n in (*NAMES, "xland")]
        literal = literal_port(args, float(g))
        libm = {o: np.zeros_like(spread[o]) for o in OUTS}
        for direction in (-np.inf, np.inf):
            for o, v in zip(OUTS, literal_port(args, float(g), direction)):
                libm[o] = np.maximum(libm[o], np.abs(v.astype(np.float64) - base[o]))
        bound = {o: spread[o] + ulp[o] for o in OUTS}
        saved = cf_mod._LOG10
        cf_mod._LOG10 = lambda x: jax.pure_callback(lambda v: np.asarray(host_log10(v)),
                                                    jax.ShapeDtypeStruct(x.shape, jnp.float32), x,
                                                    vmap_method="broadcast_all")
        try:
            compiled = jax.jit(lambda *a, g=float(g): cf_mod.wrf_cal_cldfra3(
                *a, gridkm=g, return_diagnostics=True))(*args)
        finally:
            cf_mod._LOG10 = saved
        product = jax.jit(lambda *a, g=float(g): cf_mod.wrf_cal_cldfra3(*a, gridkm=g))(*args)
        group = {}
        for o, lit, com, prod in zip(OUTS, literal, compiled[:3], product):
            lit, com, prod = np.asarray(lit), np.asarray(com), np.asarray(prod)
            assert np.isfinite(lit).all() and np.isfinite(com).all() and np.isfinite(base[o]).all()
            ldiff = int((lit.view(np.int32) != base[o].view(np.int32)).sum())
            delta = np.abs(com.astype(np.float64) - base[o])
            outside = delta > bound[o]
            pdelta = np.abs(prod.astype(np.float64) - base[o])
            group[o] = {"product_xla_log10_outside_C1": int((pdelta > bound[o]).sum()),
                        "product_xla_log10_max_abs": float(pdelta.max()), "L_diff": ldiff, "C_exact": int((delta == 0).sum()), "C_outside": int(outside.sum()),
                        "C_max_abs": float(delta.max()), "bound_max": float(bound[o].max()),
                        "fma_spread_max": float(spread[o].max()), "ulp_env_max": float(ulp[o].max()),
                        "libm_env_max": float(libm[o].max()),
                        "oracle_changed_vs_input": int((base[o] != cols[o if o != "cldfra" else "qc"]).sum())
                        if o != "cldfra" else int((base[o] > 0).sum())}
            totals["L_diff"] += ldiff
            totals["C_outside"] += int(outside.sum())
            totals["C_exact"] += int((delta == 0).sum())
            totals["elements"] += delta.size
        group["qv_untouched"] = bool((base["qv"] == cols["qv"]).all())
        for o in OUTS:
            frozen[o][sel] = base[o]
            frozen[f"{o}_bound"][sel] = bound[o]
        result["groups"][f"{float(g):.7g}"] = group
        diag = {k: np.asarray(v) for k, v in compiled[3].items()}
        for k, v in diag.items():
            if v.dtype == bool:
                census[k] = census.get(k, 0) + int(v.any(axis=-1).sum() if v.ndim == 2 else v.sum())
    info["cldfra2"] = oracle.build_cldfra2(build / "c2")
    frozen["cldfra2_qc_qi"] = oracle.run_cldfra2(build / "c2", rows["qc"], rows["qi"], True, True).reshape(n_rows, -1)
    frozen["cldfra2_qc_only"] = oracle.run_cldfra2(build / "c2", rows["qc"], rows["qi"], True, False).reshape(n_rows, -1)
    if savepoint is not None:
        np.savez_compressed(savepoint, **frozen, fixture_sha256=np.array(
            hashlib.sha256(fixture.read_bytes()).hexdigest()), provenance=np.array(json.dumps(info)))
    meta = json.loads(str(data["meta"]))
    result.update(totals=totals, census_rows_with_branch=census,
                  edits={e: sum(m["edit"] == e for m in meta) for e in sorted({m["edit"] for m in meta})},
                  sources=sorted({m["source"] for m in meta}), oracle=info,
                  fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
                  port_sha256=hashlib.sha256(Path(cf_mod.__file__).read_bytes()).hexdigest())
    needed = ("full", "small", "rh", "ice_multi", "ice_single", "h2o_multi", "h2o_single", "lwp_scaled",
              "iwp_scaled", "top_cleared")
    rng = np.random.default_rng(11)
    x = (10.0 ** rng.uniform(-10, -6, 4_000_000)).astype(np.float32)
    x = x[(x > np.float32(1e-10)) & (x < np.float32(1e-6))]
    f = np.float32
    exact = np.minimum(f(0.99), f(0.1) * (f(11.0) + np.log10(x.astype(np.float64)).astype(f)))
    xla = np.asarray(jax.jit(lambda v: jnp.minimum(f(0.99), f(0.1) * (f(11.0) + jnp.log10(v))))(jnp.asarray(x)))
    ulps = np.abs(xla.view(np.int32).astype(np.int64) - exact.view(np.int32).astype(np.int64))
    result["C2_small_branch_xla_log10"] = {"samples": int(x.size), "max_ulp_cf": int(ulps.max()),
                                           "max_abs_cf": float(np.abs(xla - exact).max()),
                                           "ulp_histogram": np.bincount(ulps).tolist()}
    result["gates"] = {"L_byte": totals["L_diff"] == 0, "C1_envelope": totals["C_outside"] == 0,
                       "K_census": all(census.get(k, 0) > 0 for k in needed)}
    receipt_path.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"gates": result["gates"], "totals": totals, "census": census}))
    return 0 if all(result["gates"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main(*(Path(a).resolve() for a in sys.argv[1:5])))
