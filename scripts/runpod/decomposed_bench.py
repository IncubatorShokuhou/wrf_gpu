#!/usr/bin/env python3
"""Multi-GPU domain-decomposition benchmark + correctness check (lane o2-multigpu; written for a 2x RTX 4090 box).

One process, P GPUs. Loads a real case (default: the bundled Swiss example), optionally crops / mirror-tiles it
into a large synthetic-from-real domain, then:
  1. single-GPU baseline on GPU 0 (compile excluded, --steps timed)        [skip with --no-single]
  2. decomposed run over px x py GPUs (deep halo, --transport device_put|ppermute, --exchange-every k)
  3. compares the gathered decomposed state with the single-GPU state (bitwise leaves / max diffs)
  4. microbenchmarks the transport alone at the real message sizes, and a raw GPU0->GPU1 copy
Writes OUT/bench.json (+ optional nsys: wrap the command, see RUNBOOK.md). Every number it writes is [M] for
that machine; nothing here is a release claim.

  python3 scripts/runpod/decomposed_bench.py --out runs/bench_swiss10 --mirror 10x10 --px 2 --py 1 --steps 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", type=Path, default=REPO / "examples" / "switzerland_d01")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--crop", default=None, help="x0:x1,y0:y1 window of the loaded domain")
    ap.add_argument("--mirror", default=None, help="MXxMY mirror tiling (e.g. 10x10 -> 420x420 from Swiss 42x42)")
    ap.add_argument("--px", type=int, default=2)
    ap.add_argument("--py", type=int, default=1)
    ap.add_argument("--halo", type=int, default=None, help="default: selected-configuration dependency bound")
    ap.add_argument("--unsafe-halo-probe", action="store_true", help="allow a deliberately undersized halo for a negative test")
    ap.add_argument("--exchange-every", type=int, default=1)
    ap.add_argument("--steps", type=int, default=20, help="timed steps (multiple of --exchange-every)")
    ap.add_argument("--start-step", type=int, default=2, help="1-based first step (2: skip the radiation step 1)")
    ap.add_argument("--transport", default="device_put", choices=["device_put", "ppermute"])
    ap.add_argument("--exchange", default="live", choices=["live", "all"], help="halo leaves: those the step reads, or all")
    ap.add_argument("--no-single", action="store_true", help="skip the single-GPU baseline/compare (domain too big)")
    ap.add_argument("--no-compare", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true", help="smoke-test the script on CPU virtual devices "
                    "(XLA_FLAGS=--xla_force_host_platform_device_count=P, JAX_PLATFORMS=cpu); timings meaningless")
    ap.add_argument("--host-global", action="store_true", help="keep the global carry in host RAM (big domains: GPU 0 "
                    "then holds only its tile); implies --no-single")
    a = ap.parse_args()
    if a.exchange_every < 1 or a.steps < 1 or a.steps % a.exchange_every:
        ap.error("--steps must be positive and divisible by --exchange-every")
    a.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("GPUWRF_WRF_ROOT", os.environ.get("GPUWRF_WRF_ROOT", ""))
    sys.path.insert(0, str(REPO / "src"))
    t_start = time.perf_counter()

    import jax
    import numpy as np

    from gpuwrf.parallel import decomposition as dc
    from gpuwrf.parallel.harness import compare, finite_summary, load_case, run_single
    from gpuwrf.parallel.tile_driver import TRANSPORTS, TileRuntime, configuration_halo, message_rounds, release_step_live_leaves
    from gpuwrf.runtime import operational_mode as om

    devs = jax.devices()
    nproc = a.px * a.py
    rec: dict = {"argv": sys.argv, "jax": jax.__version__, "devices": [str(d) for d in devs],
                 "device_kinds": [getattr(d, "device_kind", "") for d in devs], "platform": devs[0].platform,
                 "env": {k: v for k, v in os.environ.items() if k.startswith(("GPUWRF_", "XLA_", "JAX_", "CUDA_", "NCCL_"))}}
    if a.allow_cpu and devs[0].platform == "cpu":
        from gpuwrf.contracts import state as state_contract

        state_contract._gpu_device = lambda: jax.devices("cpu")[0]
    elif devs[0].platform != "gpu":
        print(f"need GPUs (or --allow-cpu), JAX sees {devs}", file=sys.stderr)
        return 2
    if len(devs) < nproc:
        print(f"need {nproc} GPUs, JAX sees {devs}", file=sys.stderr)
        return 2
    mirror = tuple(int(v) for v in a.mirror.lower().split("x")) if a.mirror else None
    carry, nl, meta = load_case(a.input, a.out / "scratch", crop=a.crop, mirror=mirror, device=devs[0])
    rec["case"] = meta
    cadence = int(nl.radiation_cadence_steps)
    clock = om.build_clock_base(nl)
    k = a.exchange_every
    bound = configuration_halo(nl, k)
    halo = a.halo if a.halo is not None else bound
    if nproc > 1 and halo < bound and not a.unsafe_halo_probe:
        ap.error(f"halo {halo} is below selected-configuration bound {bound}; use --unsafe-halo-probe only for negative tests")
    rec["dependency_bound"] = bound
    rec["unsafe_halo_probe"] = bool(a.unsafe_halo_probe)
    spec = dc.DecompositionSpec.build(nl.grid.nx, nl.grid.ny, a.px, a.py, halo,
        bdy_width=int(getattr(getattr(nl, "boundary_config", None), "spec_bdy_width", 5)))
    rec["spec"] = spec.describe()
    if a.host_global:
        a.no_single = True
        carry = jax.device_get(carry)  # NumPy leaves: tiles are sliced on the host and placed on their GPU
    carry0 = [np.asarray(x) for x in jax.tree_util.tree_leaves(carry)] if not a.no_single else []

    ref = None
    if not a.no_single:
        c1, t1 = run_single(carry, nl, start_step=a.start_step, steps=a.steps, k=k, clock=clock, cadence=cadence, warmup=True)
        rec["single_gpu"] = t1
        ref = [np.asarray(x) for x in jax.tree_util.tree_leaves(c1)]
        del c1

    live = release_step_live_leaves(carry, nl, clock, cadence) if a.exchange == "live" else None
    if live is not None:
        rec["live_leaves"] = int(sum(live))
        rec["all_leaves"] = len(live)
    t0 = time.perf_counter()
    rt = TileRuntime.create(carry, nl, spec, devs[:nproc], cadence=cadence, exchange_every=k, clock_base=clock,
                            transport=a.transport, live=live, start_step=a.start_step)
    rec["create_s"] = round(time.perf_counter() - t0, 3)
    rec["message_bytes"] = {f"{s}->{d}": b for (s, d), b in rt.message_bytes().items()}
    # Compile on the existing tile buffers, then restore the original carry. Both arms time EXACTLY the
    # same start..start+steps-1 window; no second full tile set is retained during warm-up.
    t0 = time.perf_counter()
    rt.advance(a.start_step, k)
    rt.block()
    rec["decomposed_compile_and_first_interval_s"] = round(time.perf_counter() - t0, 3)
    rt.scatter(carry, start_step=a.start_step)
    rt.block()
    t0 = time.perf_counter()
    rt.advance(a.start_step, a.steps)
    rt.block()
    run_s = time.perf_counter() - t0
    rec["decomposed"] = {"run_s": round(run_s, 3), "s_per_step": run_s / a.steps, "transport": a.transport,
                         "start_step": a.start_step, "steps": a.steps, "stop_step_exclusive": a.start_step + a.steps,
                         "cadence": cadence}
    if ref is not None:
        rec["speedup_vs_single"] = rec["single_gpu"]["s_per_step"] / rec["decomposed"]["s_per_step"]
        rec["parallel_efficiency"] = rec["speedup_vs_single"] / nproc
    # Always check every owned output after timing, including --no-single/--no-compare (T4).
    got_tree = rt.gather()
    rec["finite"] = finite_summary(got_tree)
    if ref is not None and not a.no_compare:
        rec["compare"] = compare(jax.tree_util.tree_leaves(got_tree), ref, rt.kinds, carry0)
    del got_tree
    passed = rec["finite"]["verdict"] == "FINITE" and rec.get("compare", {}).get("verdict", "BITWISE") == "BITWISE"
    rec["acceptance"] = "PASS" if passed else "FAIL"
    # No speed claim from an unqualified negative probe, even if one sampled window happens to match.
    if a.unsafe_halo_probe:
        rec.pop("speedup_vs_single", None)
        rec.pop("parallel_efficiency", None)

    # transport microbenchmark at the real message sizes (round lengths), 20 repetitions
    rounds = message_rounds(spec, rt.plan)
    msgs = {r: {d: jax.device_put(np.zeros(rt.lengths[(s, d)], np.uint8), devs[s]) for (s, d) in rt.plan if s == r}
            for r in range(nproc)}
    inbox = [dict() for _ in range(nproc)]
    tr = TRANSPORTS[a.transport](devs[:nproc], rounds, rt.lengths)
    tr.route(msgs, inbox)
    jax.block_until_ready(inbox)
    t0 = time.perf_counter()
    for _ in range(20):
        tr.route(msgs, inbox)
    jax.block_until_ready(inbox)
    rec["transport_only_s_per_exchange"] = (time.perf_counter() - t0) / 20
    big = jax.device_put(np.zeros((16 if devs[0].platform == "cpu" else 256) * 2**20, np.uint8), devs[0])
    copy_target = devs[min(1, len(devs) - 1)]
    jax.block_until_ready(jax.device_put(big, copy_target))
    t0 = time.perf_counter()
    for _ in range(5):
        jax.block_until_ready(jax.device_put(big, copy_target))
    rec["raw_copy_gpu0_to_gpu1_GBps"] = 5 * big.nbytes / (time.perf_counter() - t0) / 1e9
    rec["process_s"] = round(time.perf_counter() - t_start, 1)
    name = f"bench_{a.px}x{a.py}_{a.transport}_{a.exchange}_k{k}.json"
    (a.out / name).write_text(json.dumps(rec, indent=1, default=str) + "\n")
    summary = {k_: rec.get(k_) for k_ in ("single_gpu", "decomposed", "speedup_vs_single", "parallel_efficiency",
                                          "transport_only_s_per_exchange", "raw_copy_gpu0_to_gpu1_GBps")}
    summary["compare_verdict"] = rec.get("compare", {}).get("verdict")
    print(json.dumps(summary, default=str))
    rt.close()
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
