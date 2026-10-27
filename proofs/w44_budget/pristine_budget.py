"""Reference-only MYNN budgets from the unmodified public WRF driver.

Reuse the committed F20 real-column fixture and its public-driver caller. Only
the caller's output list, diagnostic initialization and tke_budget switch change;
the original WRF objects remain unmodified. This does not test the GPU port.
The extra allocated top plane is retained and checked, rather than hidden by
the older F20 reader's mass-level slice.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

EXTRA = ("dqke", "qwt", "qshear", "qbuoy", "qdiss", "edmf_qt", "edmf_thl",
         "edmf_ent", "edmf_qc", "sub_thl3d", "sub_sqv3d", "det_thl3d", "det_sqv3d")
ZEROS = ("sub_thl3d", "sub_sqv3d", "det_thl3d", "det_sqv3d")
BUDGET = ("dqke", "qwt", "qshear", "qbuoy", "qdiss")


def check_reference(on):
    assert all(np.isfinite(v).all() for v in on.values()), "non-finite reference"
    assert all(not np.count_nonzero(on[k]) for k in ZEROS), "original env_subs=false"
    assert all(not np.count_nonzero(on[k][:, -1]) for k in BUDGET), "unused budget top"
    # Every frozen F20 set excites every budget term. Reject a disabled switch
    # or dropped output before issuing a usable-reference verdict.
    assert all(np.count_nonzero(on[k]) for k in BUDGET), "inactive/dropped budget term"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(left, right):
    assert left.dtype == right.dtype and left.shape == right.shape
    assert np.isfinite(left).all() and np.isfinite(right).all()
    delta = left.astype(np.float64) - right.astype(np.float64)
    return dict(byte_equal=left.tobytes() == right.tobytes(),
                max_abs=float(np.max(np.abs(delta))),
                rms=float(np.sqrt(np.mean(delta * delta))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True, help="immutable model snapshot")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--edmf", type=int, choices=(0,1), default=1)
    args = p.parse_args()
    assert os.environ.get("JAX_PLATFORMS") == "cpu"
    assert not Path("/tmp/wrf_gpu2_quiet").exists(), "QUIET"
    source = args.source.resolve()
    helper = source / "tests/v025/t_cfix_mynn/f20_oracle.py"
    spec = importlib.util.spec_from_file_location("w44_original_oracle", helper)
    O = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(O)
    assert O.CALLER.count('bl_mynn_edmf=1') == 1
    original_caller, original_out3 = O.CALLER.replace('bl_mynn_edmf=1',f'bl_mynn_edmf={args.edmf}'), O.OUT3
    fixture = helper.parent / "fixtures/f20_columns.npz"
    manifest_path = fixture.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    record = dict(schema="wrf_gpu2.w44_budget.pristine_reference.v1",
                  scope="original WRF only; no port, compiled GPU identity or feature qualification",
                  runner_sha256=sha(__file__), source=str(source), edmf=args.edmf,
                  helper_sha256=sha(helper), fixture_sha256=sha(fixture),
                  manifest_sha256=sha(manifest_path), cases={})
    output = args.out / "result.json"
    output.write_text(json.dumps(record, indent=2) + "\n")
    executables = {}
    for arm in ("plain", "export_off", "export_on"):
        O.CALLER, O.OUT3 = original_caller, original_out3
        if arm != "plain":
            anchor = "  pblh=0.; maxwidth=0.; maxmf=0.; ztop_plume=0.; kpbl=1\n"
            assert O.CALLER.count(anchor) == 1
            O.CALLER = O.CALLER.replace(anchor, anchor + "  " + "; ".join(f"{n}=0." for n in EXTRA) + "\n")
            anchor = "     exch_h, qc_bl, cldfra_bl\n"
            assert O.CALLER.count(anchor) == 1
            O.CALLER = O.CALLER.replace(anchor, "     exch_h, qc_bl, cldfra_bl, &\n     " + ", ".join(EXTRA) + "\n")
            O.OUT3 = (*original_out3, *EXTRA)
        if arm == "export_on":
            assert O.CALLER.count("tke_budget=0") == 1
            O.CALLER = O.CALLER.replace("tke_budget=0", "tke_budget=1")
        executables[arm] = (O.build(args.out / arm), O.OUT3)
    with np.load(fixture) as f:
        for name, meta in manifest.items():
            if not isinstance(meta, dict):
                continue
            cols = {k.split("__", 1)[1]: f[k].copy() for k in f.files if k.startswith(name + "__")}
            assert all(np.isfinite(v).all() for v in cols.values())
            cols["qsfc"] = cols["qv"][:, 0].copy()  # dead in original core; same F20 caller
            cols["znt"] = np.full_like(cols["tsk"], np.float32(.1))
            runs = {}
            for arm, (exe, fields) in executables.items():
                O.OUT3 = fields
                work = args.out / "runs" / name / arm
                runs[arm], _ = O.run(exe, work, cols, meta["dt"], meta["dx"])
                raw = np.frombuffer((work / "out.bin").read_bytes(), np.float32)
                n, nz = cols["th"].shape
                top = raw[n * len(O.OUT2):].reshape(len(fields), nz + 1, n)[:, -1]
                assert np.isfinite(top).all() and not np.count_nonzero(top), (name, arm, "unused top")
            identities = {arm: {key: compare(runs["plain"][key], runs[arm][key])
                                for key in (*O.OUT2, *original_out3)}
                          for arm in ("export_off", "export_on")}
            assert all(v["byte_equal"] for rows in identities.values() for v in rows.values()), name
            on = runs["export_on"]
            check_reference(on)
            residual = on["dqke"].astype(np.float64) - (
                on["qshear"].astype(np.float64) + on["qbuoy"] + on["qwt"] - on["qdiss"])
            np.savez(args.out / (name + ".npz"), **on)
            record["cases"][name] = dict(input_provenance=meta, columns=n, levels=nz,
                identity=identities, nonfinite=0, zero_sub_det=True, zero_unused_top=True,
                active_plume_columns=int(np.count_nonzero(np.any(on["edmf_a"] > 0, axis=1))),
                active_plume_cells=int(np.count_nonzero(on["edmf_a"] > 0)),
                fields={k: dict(nonzero=int(np.count_nonzero(v)), max_abs=float(np.max(np.abs(v))))
                        for k, v in on.items()},
                closure_reference_only=dict(max_abs=float(np.max(np.abs(residual))),
                    rms=float(np.sqrt(np.mean(residual * residual))),
                    note="pristine residual, no invented zero-closure bound or port claim"),
                output_sha256=sha(args.out / (name + ".npz")))
            output.write_text(json.dumps(record, indent=2) + "\n")
            print(name, "BYTE", record["cases"][name]["active_plume_columns"], flush=True)
    record["verdict"] = "ORIGINAL_REFERENCE_ONLY_PASS"
    output.write_text(json.dumps(record, indent=2) + "\n")
    assert not any(name == "jax" or name.startswith("gpuwrf") for name in sys.modules)


if __name__ == "__main__":
    main()
