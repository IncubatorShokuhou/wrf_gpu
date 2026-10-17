"""Subprocess runner: release-default MYNN kernel vs the pristine public call on every fixture set.

Usage (clean env, release flags resolved by ``import gpuwrf`` like the CLI):
  JAX_PLATFORMS=cpu GPUWRF_FAST_DEFAULTS=1 PYTHONPATH=src python f20_public_kernel.py <oracle> <workdir> <out.json>
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import f20_oracle as O  # noqa: E402
import f20_port as F  # noqa: E402

FIXTURE = HERE / "fixtures/f20_columns.npz"


def load_sets():
    arrays = np.load(FIXTURE)
    manifest = json.loads(FIXTURE.with_suffix(".json").read_text())
    sets = {}
    for name, meta in manifest.items():
        if not isinstance(meta, dict):
            continue
        cols = {k.split("__", 1)[1]: arrays[k].copy() for k in arrays.files if k.startswith(name + "__")}
        cols["qsfc"] = cols["qv"][:, 0].copy()  # dead in mynnedmf (ch/qsfc/znt only declared/printed)
        cols["znt"] = np.full_like(cols["tsk"], np.float32(0.1))
        sets[name] = (cols, meta)
    return sets


def main(oracle, workdir, out):
    import jax
    assert jax.devices()[0].platform == "cpu"
    assert os.environ.get("GPUWRF_MYNN_FP32_COLUMNS") == "1", "release MYNN flags were not resolved"
    result = {"flags": {k: v for k, v in sorted(os.environ.items()) if k.startswith("GPUWRF_MYNN")}}
    for name, (cols, meta) in load_sets().items():
        wrf, _ = O.run(oracle, Path(workdir) / name, cols, meta["dt"], meta["dx"])
        row = {"wrf_maxmf": wrf["maxmf"].tolist(), "wrf_pblh": wrf["pblh"].tolist(),
               "wrf_rth0": wrf["rthblten"][:, 0].tolist()}
        for mode in ("face", "mass"):
            port = F.run_kernel(cols, meta["dt"], meta["dx"], w_mode=mode)
            row[mode + "_maxmf"] = port["maxmf"].tolist()
            row[mode + "_pblh"] = port["pblh"].tolist()
            row[mode + "_rth0"] = port["rthblten"][:, 0].tolist()
        result[name] = row
    Path(out).write_text(json.dumps(result) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:4])
