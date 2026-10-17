#!/usr/bin/env python3
"""Analytic cost model for multi-GPU wrf_gpu (lane o2-multigpu M4). Pure Python; every output number is [I].

Deep-halo decomposition (stage 1, src/gpuwrf/parallel): each of P = px*py tiles runs the unmodified step on
(owned + 2H per internal axis) columns, H = k*R + 5 + 1 with R = 24 + 1 + ns//2 + ns + 1 (tile_driver.required_halo);
every k steps each tile sends one message per neighbour (<= 8) of H x edge x B_col bytes (B_col = exchanged carry
bytes per column; live-leaf exchange skips write-before-read scratch). No compute/communication overlap in stage 1:
  T_step(P) = max_tile(t_col * cols_tile) + max_tile(sum_msgs hops*(L + bytes/B)) / k
  E_strong(P) = T_step(1) / (P * T_step(P))
Ensemble mode (M1): independent members, no messages: E ~ 1 minus host/launch overhead [U].
Calibration inputs are [I]: t_col from measured RTX 5090 PROD warm roots (LW9 A51/A55: ~0.82 us per
column-step incl. host gaps) scaled by HBM bandwidth for other GPUs; fabric numbers from vendor ceilings x an
effective fraction (FABRIC_SCENARIOS.md). Real multi-GPU performance is UNMEASURED.

  python3 scripts/runpod/cost_model.py --out cost_model.json [--bcol 4000] [--markdown cost_model.md]
"""
from __future__ import annotations

import argparse
import itertools
import json
import math

# domain: (nx, ny, nz, acoustic substeps ns, dt_s)   [ns: WRF auto sound steps of the namelist]
DOMAINS = {
    "PROD_d01_120x70": (120, 70, 44, 4, 54.0),
    "PROD_d02_267x117": (267, 117, 44, 4, 18.0),
    "WN3_d03_112x94": (112, 94, 44, 4, 6.0),
    "Swiss_mirror10_420x420": (420, 420, 44, 4, 18.0),
    "Large_1000x1000": (1000, 1000, 44, 4, 18.0),
    "Large_2000x2000": (2000, 2000, 44, 4, 18.0),
}
# GPU: (us per column-step at nz=44 [I], HBM GB/s, memory GiB)
GPUS = {"RTX5090": (0.82, 1792, 32), "RTX4090": (0.82 * 1792 / 1008, 1008, 24), "H100_SXM": (0.82 * 1792 / 3350, 3350, 80)}
# fabric: (effective GB/s per direction, latency us, hops per message)  [I]
FABRICS = {
    "pcie4_hoststaged_4090": (20.0 * 0.6, 15.0, 2),   # GeForce: no P2P -> D2H + H2D through pinned host memory
    "pcie4_p2p": (25.0 * 0.8, 10.0, 1),
    "pcie5_p2p": (64.0 * 0.5, 10.0, 1),
    "nvlink4_h100": (450.0 * 0.5, 2.0, 1),
    "ib_ndr400_rdma": (50.0 * 0.5, 5.0, 1),
    "zero_latency_inf_bw": (1e12, 0.0, 1),            # control: compute + redundancy only
}


def required_halo(ns: int, k: int = 1, bdy: int = 5, shadow_radius: int = 0) -> int:
    return k * (24 + 1 + ns // 2 + ns + 1 + shadow_radius) + bdy + 1  # = tile_driver.required_halo


def split(n: int, p: int) -> list[int]:
    b, e = divmod(n, p)
    return [b + (1 if i < e else 0) for i in range(p)]


def factorizations(p: int):
    return [(a, p // a) for a in range(1, p + 1) if p % a == 0]


def tiles(nx: int, ny: int, px: int, py: int, h: int):
    """[(cols_tile, [(neighbour message cells)])] per tile; message cells = halo cells received from that neighbour."""
    xs, ys = split(nx, px), split(ny, py)
    x0s = [sum(xs[:i]) for i in range(px)]
    y0s = [sum(ys[:i]) for i in range(py)]
    out = []
    for iy, ix in itertools.product(range(py), range(px)):
        ex = (max(0, x0s[ix] - h), min(nx, x0s[ix] + xs[ix] + h))
        ey = (max(0, y0s[iy] - h), min(ny, y0s[iy] + ys[iy] + h))
        msgs = []
        for jy, jx in itertools.product(range(py), range(px)):
            if (jx, jy) == (ix, iy):
                continue
            ox = max(0, min(ex[1], x0s[jx] + xs[jx]) - max(ex[0], x0s[jx]))
            oy = max(0, min(ey[1], y0s[jy] + ys[jy]) - max(ey[0], y0s[jy]))
            if ox and oy:
                msgs.append(ox * oy)
        out.append(((ex[1] - ex[0]) * (ey[1] - ey[0]), msgs, xs[ix] * ys[iy]))
    return out


HALO_OVERRIDE: int | None = None  # --halo: measured instead of the analytic bound (exactness only as verified)
SHADOW_RADIUS = 0  # --shadow-length-m/--dx-m: max_steps + interpolation neighbour, before dynamics


def evaluate(domain: str, gpu: str, fabric: str, p: int, *, bcol: float, k: int = 1, compute_scale: float = 1.0) -> dict:
    """Best px x py factorization of P for one scenario. Stage 1 serializes communication after compute (no
    overlap): this IS the serialized-communication control; the zero_latency_inf_bw fabric is the other control."""
    nx, ny, nz, ns, dt = DOMAINS[domain]
    t_col = GPUS[gpu][0] * 1e-6 * compute_scale
    bw, lat, hops = FABRICS[fabric]
    bound = required_halo(ns, k, shadow_radius=SHADOW_RADIUS)
    h = bound if HALO_OVERRIDE is None else HALO_OVERRIDE * k
    t1 = t_col * nx * ny
    best = None
    for px, py in factorizations(p):
        if px > nx or py > ny:
            continue
        ts = tiles(nx, ny, px, py, h)
        comp = max(c for c, _m, _o in ts) * t_col
        comm = max(sum(hops * (lat * 1e-6 + cells * bcol / (bw * 1e9)) for cells in m) for _c, m, _o in ts) / k
        step = comp + comm
        red = max(c / o for c, _m, o in ts)
        mem_gib = max(c for c, _m, _o in ts) * bcol * 2.5 / 2**30  # carry + temps ~2.5x exchanged bytes [I]
        rec = {"domain": domain, "gpu": gpu, "fabric": fabric, "P": p, "px": px, "py": py, "halo": h, "k": k,
               "dependency_bound": bound, "shadow_radius": SHADOW_RADIUS, "halo_below_bound": h < bound,
               "t_step_s": step, "t_compute_s": comp, "t_comm_s": comm, "redundancy": red,
               "msg_mb_max": max((max(m) if m else 0) * bcol / 1e6 for _c, m, _o in ts),
               "efficiency": t1 / (p * step), "speedup": t1 / step, "s_per_fc_hour": step * 3600 / dt,
               "single_gpu_s_per_fc_hour": t1 * 3600 / dt, "mem_gib_tile_I": mem_gib,
               "fits_gpu_memory": mem_gib <= GPUS[gpu][2] * 0.85}
        if best is None or rec["t_step_s"] < best["t_step_s"]:
            best = rec
    return best or {"domain": domain, "gpu": gpu, "fabric": fabric, "P": p, "infeasible": True}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bcol", type=float, default=7892.0,
                    help="exchanged carry bytes per column (default 7892 [M CPU]: release carry, live-leaf set, PROD d01 "
                         "nz=44, proofs/o2_multigpu/release_bytes_probe.py; all horizontal leaves 10184)")
    ap.add_argument("--halo", type=int, default=None, help="per-step halo override (e.g. a measured radius); default analytic")
    ap.add_argument("--shadow-length-m", type=float, default=None, help="enabled topographic shadow length (omit if disabled)")
    ap.add_argument("--dx-m", type=float, default=3000.0, help="dx of the selected shadow configuration")
    ap.add_argument("--out", default=None)
    ap.add_argument("--markdown", default=None)
    a = ap.parse_args()
    global HALO_OVERRIDE, SHADOW_RADIUS
    HALO_OVERRIDE = a.halo
    if a.shadow_length_m is not None:
        if not math.isfinite(a.dx_m) or a.dx_m <= 0 or not math.isfinite(a.shadow_length_m) or a.shadow_length_m < 0:
            ap.error("shadow length must be finite/nonnegative and dx finite/positive")
        SHADOW_RADIUS = int(a.shadow_length_m / a.dx_m + 1.0) + 1
    rows = []
    for domain, gpu, fabric, p, k, scale in itertools.product(DOMAINS, GPUS, FABRICS, (1, 2, 4, 8, 16, 64), (1, 2),
                                                                (0.5, 1.0, 2.0)):
        if k == 2 and scale != 1.0:
            continue
        r = evaluate(domain, gpu, fabric, p, bcol=a.bcol, k=k, compute_scale=scale)
        r["compute_scale"] = scale
        rows.append(r)
    out = {"label": "[I] analytic, uncalibrated on multi-GPU hardware", "bcol_bytes": a.bcol, "domains": DOMAINS,
           "shadow_length_m": a.shadow_length_m, "dx_m": a.dx_m, "shadow_radius": SHADOW_RADIUS,
           "gpus": GPUS, "fabrics": FABRICS, "rows": rows}
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
    lines = ["| domain | GPU | fabric | P | px×py | halo | redundancy | msg MB | comm ms | step ms | E_strong | s/fc-h (1 GPU) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("infeasible") or r["compute_scale"] != 1.0 or r["k"] != 1:
            continue
        if r["gpu"] != "RTX4090" and r["fabric"] not in ("nvlink4_h100", "ib_ndr400_rdma"):
            continue
        if r["gpu"] == "RTX4090" and r["fabric"] not in ("pcie4_hoststaged_4090", "zero_latency_inf_bw"):
            continue
        if r["gpu"] == "H100_SXM" and r["fabric"] not in ("nvlink4_h100", "ib_ndr400_rdma"):
            continue
        if r["gpu"] == "RTX5090":
            continue
        lines.append(f"| {r['domain']} | {r['gpu']} | {r['fabric']} | {r['P']} | {r['px']}×{r['py']} | {r['halo']} | "
                     f"{r['redundancy']:.2f} | {r['msg_mb_max']:.1f} | {1e3 * r['t_comm_s']:.1f} | {1e3 * r['t_step_s']:.1f} | "
                     f"{r['efficiency']:.2f} | {r['s_per_fc_hour']:.1f} ({r['single_gpu_s_per_fc_hour']:.1f}) |")
    text = "\n".join(lines)
    if a.markdown:
        with open(a.markdown, "w") as f:
            f.write(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
