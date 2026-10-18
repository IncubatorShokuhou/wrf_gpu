"""Regime-censused real columns for the icloud=3 oracle (E154).

Sources (wrf_gpu2-owned original CPU-WRF 4.7.1 history, no ALISIOS data root):
  Swiss RD11 24 h (Jan 2023, 3 km, Thompson/MYNN): winter stratus, ice, snow.
  v-prep c42 (WN3 0227 Canary 9/3 km, Thompson/MYNN): trade-inversion Sc/Cu.
Caller view = radiation_driver at first_rk_step_part1: T = t_phy (dry theta * Exner of
full p), P = P_HYD, DZ8W = (PH+PHB)/g differences (g = 9.81), XLAND.
Adversarial rows are real columns with one documented edit each, to activate branches the
real sample does not reach (LWP/IWP > 1 kg m-2 rescale, warm tc >= 25 C, cold snow-only).
usage: python build_fixtures.py OUT.npz
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src")]
from gpuwrf.io.netcdf_lock import Dataset  # noqa: E402

SWISS = Path("<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu")
C42 = Path("<DATA_ROOT>/wrf_gpu2/archive/v040/v-prep/cpu_references/c42")
FRAMES = [(SWISS, "wrfout_d01_2023-01-15_06:00:00", 3000.), (SWISS, "wrfout_d01_2023-01-15_12:00:00", 3000.),
          (SWISS, "wrfout_d01_2023-01-15_18:00:00", 3000.), (SWISS, "wrfout_d01_2023-01-16_00:00:00", 3000.),
          (C42, "wrfout_d02_2026-02-28_06:00:00", 3000.), (C42, "wrfout_d02_2026-02-28_12:00:00", 3000.),
          (C42, "wrfout_d02_2026-02-28_18:00:00", 3000.), (C42, "wrfout_d01_2026-02-28_12:00:00", 9000.)]
NAMES = ("qv", "qc", "qi", "qs", "dz", "p", "t")


def read_frame(path: Path) -> dict:
    f = np.float32
    with Dataset(path) as ds:
        get = lambda n: np.asarray(ds[n][0], f)  # noqa: E731
        p_full = get("P") + get("PB")
        theta = get("T") + f(300.)
        rcp = f(287.) / f(1004.5)  # module_model_constants rcp = r_d/cp, cp = 7*r_d/2
        t = theta * (p_full / f(100000.)) ** rcp
        z_w = (get("PH") + get("PHB")) / f(9.81)
        cols = dict(qv=get("QVAPOR"), qc=get("QCLOUD"), qi=get("QICE"), qs=get("QSNOW"),
                    dz=z_w[1:] - z_w[:-1], p=get("P_HYD"), t=t)
        out = {k: np.moveaxis(v, 0, -1).reshape(-1, v.shape[0]) for k, v in cols.items()}
        out["xland"] = get("XLAND").reshape(-1)
    return out


def main(out_path: Path) -> None:
    import jax
    from gpuwrf.physics.cloud_fraction import wrf_cal_cldfra3, wrf_gridkm

    rng = np.random.default_rng(20261017)
    rows, meta = [], []
    for root, name, dx in FRAMES:
        cols = read_frame(root / name)
        g = float(wrf_gridkm(dx, dx))
        _, _, _, diag = jax.jit(lambda *a, g=g: wrf_cal_cldfra3(*a, gridkm=g, return_diagnostics=True))(
            *(cols[n] for n in NAMES), cols["xland"])
        diag = {k: np.asarray(v) for k, v in diag.items()}
        flags = {k: v.any(axis=-1) if v.ndim == 2 else v for k, v in diag.items()
                 if k in ("small", "ice_multi", "ice_single", "h2o_multi", "h2o_single", "lwp_scaled", "iwp_scaled")}
        flags["stratus_qc"] = (cols["qc"][:, :12] > 1e-6).any(axis=-1)
        flags["ice_qi"] = (cols["qi"] >= 1e-7).any(axis=-1)
        flags["snow_cold"] = ((cols["qs"] > 1e-6) & (cols["t"] < 273.)).any(axis=-1)
        flags["ocean"] = cols["xland"] > 1.5
        flags["clear"] = ~(diag["full"] | diag["small"]).any(axis=-1)
        picked = set()
        for key, mask in flags.items():
            idx = np.flatnonzero(mask)
            picked.update(rng.choice(idx, size=min(8, idx.size), replace=False).tolist())
        picked.update(rng.choice(cols["qv"].shape[0], size=12, replace=False).tolist())
        for c in sorted(picked):
            rows.append({n: cols[n][c] for n in (*NAMES, "xland")} | {"gridkm": g})
            meta.append({"source": f"{root.name}/{name}", "column": int(c), "edit": "none"})
    # Adversarial real-structured edits (one per row, documented). lwp_alt: alternating
    # supersaturated/dry warm layers -> many single-level water decks, LWP > 1 rescale.
    # iwp_stretch: same aloft with dz x20 -- NON-PHYSICAL arithmetic stress, the only way
    # adjust_cloudFinal's IWP > 1 branch is reachable (0.05*qvs*cfr per deck is tiny).
    import jax.numpy as jnp
    from gpuwrf.physics.cloud_fraction import _rsif, _rslf

    def qvs_of(p, t, blend):
        w = np.asarray(_rslf(jnp.asarray(p), jnp.asarray(t)))
        if not blend:
            return w
        i = np.asarray(_rsif(jnp.asarray(p), jnp.asarray(t)))
        tc = t - np.float32(273.15)
        return np.where(tc >= -12, w, np.where(tc < -35, i, w - (w - i) * (np.float32(-12.) - tc) / np.float32(23.)))

    base = [i for i, m in enumerate(meta) if m["source"].startswith("c42/wrfout_d02") and rows[i]["xland"] > 1.5]
    odd = (np.arange(rows[0]["qv"].size) % 2 == 1)
    for j, i in enumerate(base[:12]):
        r = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in rows[i].items()}
        kind = ("lwp_alt", "iwp_stretch", "warm25", "snow_only")[j % 4]
        if kind == "lwp_alt":
            r["t"][:14] += np.float32(6.)
            alt = np.where(odd, np.float32(1.02), np.float32(0.45))
            r["qv"][:14] = (alt * qvs_of(r["p"], r["t"], False))[:14].astype(np.float32)
            r["qc"][:14] = r["qi"][:14] = np.float32(0.)
        elif kind == "iwp_stretch":
            r["t"][17:34] += np.float32(8.)
            alt = np.where(odd, np.float32(1.44), np.float32(0.3))
            r["qv"][17:34] = (alt * qvs_of(r["p"], r["t"], True))[17:34].astype(np.float32)
            r["qc"][17:34] = r["qi"][17:34] = r["qs"][17:34] = np.float32(0.)
            r["dz"][17:34] *= np.float32(20.)
        elif kind == "warm25":
            r["t"][:6] += np.float32(14.)  # tc >= 25 C near the surface
        else:
            r["qc"][:] = np.float32(0.)
            r["qi"][:] = np.float32(0.)
            r["qs"][8:16] = np.float32(3.e-6)
            r["t"][8:16] = np.minimum(r["t"][8:16], np.float32(268.))
        rows.append(r)
        meta.append(dict(meta[i], edit=kind))
    arrays = {n: np.stack([r[n] for r in rows]).astype(np.float32) for n in (*NAMES, "xland", "gridkm")}
    np.savez_compressed(out_path, **arrays, meta=np.array(json.dumps(meta)))
    sources = sorted({(str(root / name)) for root, name, _ in FRAMES})
    receipt = {"rows": len(rows), "nz": int(arrays["qv"].shape[1]),
               "sources_sha256": {s: hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in sources},
               "npz_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest()}
    out_path.with_suffix(".json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps({k: receipt[k] for k in ("rows", "nz", "npz_sha256")}))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
