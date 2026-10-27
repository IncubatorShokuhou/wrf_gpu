"""t0 land gate: actual native loader + land-history mapping vs the CPU-WRF twin t0.

Usage: JAX_PLATFORMS=cpu python land_t0_probe.py SOURCE_TREE CASE CPU_T0_DIR OUTPUT_DIR
Loads the real 3-nest cycled case through nested_pipeline._load_domains
(cycling=.true. namelist), maps the initial Noah-MP carry exactly as the
history writer does at own step 0 (land_history_diagnostics), and compares
every Noah-MP state of the CPU-WRF t0 history. Report-only: BYTE flags and
raw differences, no tolerances. Host CPU evidence, not a GPU claim.
"""
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from netCDF4 import Dataset

FIELDS = ("TV", "TG", "TAH", "EAH", "CANLIQ", "CANICE", "FWET", "SNEQVO", "ALBOLD", "XSAI", "LAI",
          "ISNOW", "TSNO", "SNICE", "SNLIQ", "ZSNSO", "SH2O", "SMOIS", "TSLB", "SNOW", "SNOWH", "TAUSS")


def main():
    assert os.environ["JAX_PLATFORMS"] == "cpu"
    root, case, reference, output = map(Path, sys.argv[1:5])
    sys.path[:0] = [str(root / "src"), str(root / "proofs/o1_nlbind")]
    import cpu_safe
    cpu_safe.install()
    import jax
    assert jax.devices()[0].platform == "cpu"
    from gpuwrf.backends import pallas as facade
    facade.cuda_native = lambda: False
    from gpuwrf.contracts import state as contract
    contract._gpu_device = lambda: jax.devices("cpu")[0]
    from gpuwrf.integration import nested_pipeline as pipeline
    from gpuwrf.io.land_history import land_history_diagnostics, load_land_history_inputs
    from gpuwrf.config.paths import wrf_run_dir
    output.mkdir(exist_ok=False, parents=True)
    config = pipeline.NestedPipelineConfig(case, output / "unused_run", output / "unused_proofs", hours=3, max_dom=3)
    started = time.perf_counter()
    _h, _b, _s, _r, _dt, carries = pipeline._load_domains(config, ("d01", "d02", "d03"))
    loaded = time.perf_counter() - started
    records = []
    for domain, carry in carries.items():
        land = carry.noahmp_land
        inputs = load_land_history_inputs(case / f"wrfinput_{domain}", table_dir=wrf_run_dir())
        fields, _land = land_history_diagnostics(land, land, inputs, own_step=0)
        fields = dict(fields)
        fields["TSK"] = np.asarray(carry.state.t_skin)
        with Dataset(reference / f"wrfout_{domain}_2026-05-04_00:00:00") as ds:
            ds.set_auto_maskandscale(False)
            landmask = np.asarray(ds["LANDMASK"][0]) > .5
            for name in FIELDS + ("TSK",):
                if name not in ds.variables:
                    records.append({"domain": domain, "field": name, "status": "absent in CPU t0"})
                    continue
                expected = np.asarray(ds[name][0])
                if name not in fields:
                    records.append({"domain": domain, "field": name, "status": "not mapped by port t0 history"})
                    continue
                actual = np.asarray(fields[name]).astype(expected.dtype).reshape(expected.shape)
                delta = actual.astype(float) - expected.astype(float)
                cell = landmask if delta.ndim == 2 else np.broadcast_to(landmask, delta.shape)
                where = np.unravel_index(np.argmax(np.abs(delta)), delta.shape)
                records.append({"domain": domain, "field": name, "byte_equal": expected.tobytes() == actual.tobytes(),
                                "different_cells": int(np.count_nonzero(expected != actual)),
                                "different_land_cells": int(np.count_nonzero((expected != actual) & cell)),
                                "max_abs": float(np.max(np.abs(delta))),
                                "max_abs_land": float(np.max(np.abs(np.where(cell, delta, 0.)))),
                                "worst_index": list(map(int, where)),
                                "wrf_value": float(expected[where]), "port_value": float(actual[where])})
    report = {"platform": jax.default_backend(), "source": str(root), "case": str(case), "reference": str(reference),
              "load_s": loaded, "records": records,
              "byte_equal": sum(bool(r.get("byte_equal")) for r in records),
              "compared": sum("byte_equal" in r for r in records),
              "scope": "actual 3-nest loader + land_history_diagnostics own_step 0 vs CPU-WRF t0; report-only, no tolerance",
              "GPU_qualified": False}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))
    for r in records:
        if not r.get("byte_equal"):
            print(json.dumps(r))


if __name__ == "__main__":
    main()
