"""CPU changed-artifact cost, precision and early-return comparison (no GPU claim).

Usage: python component_cost.py BASELINE_SOURCE_FILE OUTPUT_JSON
Baseline source must be extracted from an immutable git commit. Both variants
use original full Swiss CPU-WRF fields and the same dynamic arguments/options.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import sys
import time

import jax
import jax.numpy as jnp
import netCDF4
import numpy as np

from gpuwrf.assimilation import fdda_spectral as new

jax.config.update("jax_enable_x64", True)
assert jax.devices()[0].platform == "cpu"
spec = importlib.util.spec_from_file_location("baseline_spectral_cost", sys.argv[1])
old = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old
spec.loader.exec_module(old)

cpu = Path("<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu")
with netCDF4.Dataset(cpu / "wrfout_d01_2023-01-15_00:00:00") as ds:
    a = lambda n: np.asarray(ds[n][0], np.float32)
    q = a("QVAPOR")
    th = (a("THM") + np.float32(300)) / (np.float32(1) + np.float32(np.float32(461.6)/np.float32(287)) * q)
    values = [a("U"), a("V"), th, a("PH"), q, a("PBLH"), a("HGT"),
              (a("PHB") + a("PH")) / np.float32(9.81)]
with netCDF4.Dataset("<USER_HOME>/wrf_gpu2_lanes/s1-specnudge/cpu_wrf/wrffdda_d01") as ds:
    pair = {f"{name}_{suffix.lower()}": jnp.asarray(ds[f"{variable}_NDG_{suffix}"][0], jnp.float32)
            for name, variable in (("u", "U"), ("v", "V"), ("t", "T"), ("ph", "PH"), ("q", "Q"))
            for suffix in ("OLD", "NEW")}
values = tuple(jnp.asarray(a, jnp.float32) for a in values)
options = dict(interval_m=60, end_h=6, guv=3e-4, gt=3e-4, gph=3e-4, gq=0,
               xwavenum=2, ywavenum=2)
report = {"platform": "cpu", "geometry": list(values[2].shape), "options": options,
          "gpu_cost_peak_vram_E41": "UNKNOWN; no GPU used", "variants": {}}
outputs = {}
for label, module in (("old_select_after_fft", old), ("new_early_return", new)):
    opts = module.SpectralFddaOptions(**options)
    fn = jax.jit(lambda pair, *args: module.spectral_nudging(opts, pair, *args, jnp.float32))
    start = time.perf_counter()
    executable = fn.lower(pair, *values, 18.).compile()
    compile_s = time.perf_counter() - start
    memory = executable.memory_analysis()
    hlo = executable.as_text()
    forbidden = [line for line in hlo.splitlines() if re.search(r"= f64\[[^\]]+\]", line)]
    assert not forbidden, forbidden[:5]
    result = {"compile_s": compile_s, "compiler_memory_estimates": {
        n: int(getattr(memory, n)) for n in ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes")},
        "fft_HLO_ops": sum(bool(re.search(r"\bfft\(", line)) for line in hlo.splitlines()),
        "non_scalar_f64_results": len(forbidden), "timings": {}}
    for name, lead in (("active", 18.), ("past_end", 7 * 3600.)):
        for _ in range(3):
            jax.block_until_ready(executable(pair, *values, lead))
        times = []
        for _ in range(20):
            start = time.perf_counter()
            value = executable(pair, *values, lead)
            jax.block_until_ready(value)
            times.append(time.perf_counter() - start)
        result["timings"][name] = {"median_s": float(np.median(times)), "min_s": min(times), "n": len(times)}
        outputs[(label, name)] = [np.asarray(a) for a in value]
    report["variants"][label] = result
for name in ("active", "past_end"):
    same = [a.tobytes() == b.tobytes() for a, b in zip(outputs[("old_select_after_fft", name)],
                                                    outputs[("new_early_return", name)])]
    report[name + "_component_byte_equal"] = same
    report[name + "_max_abs_regression_delta"] = [float(np.abs(a.astype(np.float64) - b).max())
        for a, b in zip(outputs[("old_select_after_fft", name)], outputs[("new_early_return", name)])]
    # This is regression/attribution evidence only. Fidelity is the independent
    # pristine-WRF fixture gate, not an old-versus-new GPU port comparison.
Path(sys.argv[2]).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
