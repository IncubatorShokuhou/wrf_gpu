"""CPU-side admission math for optimize members on one GPU (no JAX, no device query).

Nested members (max_dom > 1) are sized by the launcher's C-auto memory plans. Single-domain
members have no C-auto plan (the CLI keeps its single-domain allocator), so the launcher needs a
fixed per-member pool (``--pool-gib``). The default pool below is an estimate [I] scaled from
measured anchors [M]; the GPU smoke measures the real per-process peak and replaces it.
"""
from __future__ import annotations

import math

GIB = 1024**3
MIB = 1024**2
# Measured anchors (v0.4.0 sealed bundle C-auto plan 11d2ed32, WN3 0227 3-nest, release defaults):
# d01 step program 120x70x44: argument 182 + output 168 + temp 503 - alias 168 MiB = 685 MiB
# footprint, generated code 123 MiB; whole 3-nest live peak 2.81 GiB vs executable footprints
# 1.99 GiB -> init/IO live factor 1.41. Outside-pool CUDA context/modules: >= 1 GiB (E97 824 MiB,
# A58 963 MiB). Desktop on the dev RTX 5090: ~4 GiB (A58 3,977 MiB).
ANCHOR = {"columns": 120*70, "levels": 44, "footprint_bytes": 685*MIB, "code_bytes": 123*MIB,
          "live_factor": 2.81/1.99, "source": "release bundle v0.4.0 C-auto plan 11d2ed32 (WN3 0227 d01 program)"}
PHYSICS_MARGIN = 1.5  # temps differ by scheme; non-default physics was never sized on this geometry
SAFETY = 1.20         # gpu_allocator.SAFETY_FACTOR
POOL_HEADROOM = 0.5*GIB
OUTSIDE_POOL = 1.0*GIB
HOST_COLD_GB = 16.0   # launcher default host need when nothing was measured


# Measured GPU members (lane w41-optimize GPU smoke 2026-10-17, RTX 5090, release-default physics, cu 0 and 1):
# WN3 d01 grid (c60 input, 3 h, two members in parallel, 1.0 GiB pool): per-process nvidia-smi peak 1,712 / 1,744 MiB,
# host VmHWM 6.4 GB while compiling. Swiss d01: cold KF member 6.0 GB host, warm (bundle AOT) member 2.8 GB host,
# per-process peak 2,214-2,216 MiB at a 1.5 GiB pool (pool-bound). A measured grid uses 1.25x the pool that sufficed.
MEASURED = {(120, 70, 44): {"pool_sufficed_gib": 1.0, "per_process_peak_mib": 1744, "host_cold_gb": 6.4,
                            "source": "proofs/w41_optimize GPU smoke arm B (WN3 0227 d01, cu0/cu1 parallel)"}}


def member_pool(geometry):
    """Single-domain member pool in GiB (rounded up to 0.25), with the arithmetic or the measurement."""
    columns = int(geometry["west_east"])*int(geometry["south_north"])
    levels = int(geometry["bottom_top"])
    measured = MEASURED.get((int(geometry["west_east"]), int(geometry["south_north"]), levels))
    if measured is not None:
        gib = math.ceil(measured["pool_sufficed_gib"]*1.25*4)/4
        return {"pool_gib": gib, "outside_pool_gib": OUTSIDE_POOL/GIB, "need_gib": gib+OUTSIDE_POOL/GIB,
                "columns": columns, "levels": levels, "measured": measured,
                "formula": "ceil4(1.25 x the measured pool that sufficed); need = pool + 1 GiB",
                "label": "[M] anchored on a GPU measurement of this grid (release-default physics, cu 0/1); "
                         "other physics +25 % margin [I]"}
    scale = columns*levels/(ANCHOR["columns"]*ANCHOR["levels"])
    step = max(0.25*GIB, ANCHOR["footprint_bytes"]*scale)
    live = step*ANCHOR["live_factor"]*PHYSICS_MARGIN
    pool = (live+ANCHOR["code_bytes"])*SAFETY+POOL_HEADROOM
    gib = math.ceil(pool/GIB*4)/4
    return {"pool_gib": gib, "outside_pool_gib": OUTSIDE_POOL/GIB, "need_gib": gib+OUTSIDE_POOL/GIB,
            "columns": columns, "levels": levels, "scale_vs_anchor": round(scale, 4),
            "formula": "ceil4(((max(0.25 GiB, 685 MiB*scale)*1.41*1.5)+123 MiB)*1.2+0.5 GiB); need = pool + 1 GiB",
            "anchor": ANCHOR["source"], "label": "[I] estimate; GPU smoke replaces it with measured per-process peaks"}


def admission(need_gib, *, members, gpu_visible_gib=31.35, desktop_gib=4.0, margin_gib=0.0,
              host_available_gb=None, host_gb_per_member=HOST_COLD_GB, host_reserve_gb=8.0):
    """How many members the launcher can run at once (same FIFO rule as scripts/parallel_cases.py)."""
    vram = max(0, math.floor((gpu_visible_gib-desktop_gib-margin_gib)/need_gib))
    result = {"members": members, "vram_parallel": min(members, vram),
              "vram_rule": f"floor(({gpu_visible_gib} visible - {desktop_gib} desktop - {margin_gib} margin) / {need_gib} need)"}
    if host_available_gb is not None:
        host = max(0, math.floor((host_available_gb-host_reserve_gb)/host_gb_per_member))
        result |= {"host_parallel": min(members, host),
                   "host_rule": f"floor(({host_available_gb} available - {host_reserve_gb} reserve) / {host_gb_per_member} per member)"}
        result["parallel"] = min(result["vram_parallel"], result["host_parallel"])
    else:
        result["parallel"] = result["vram_parallel"]
    return result
