"""Freeze the F3a original-WRF witnesses for the cycling init tests (CPU only).

Run once, offline (generator, not collected by pytest), with JAX_PLATFORMS=cpu
(gpuwrf import pulls JAX in; no JAX computation is used for any reference value):
  PYTHONPATH=src python tests/data/w44_warmstart/generate_f3a_witnesses.py <work_dir>

1. rebalance_cycl: the UNCHANGED WRF body (proofs/w44_warmstart/rebalance_oracle.py
   build()/native_call(), gfortran -O0 -ffp-contract=off) on two real cloudy column
   sets: the cycled native May d01 input (the critic's cloudy input) and a Swiss
   CPU-WRF 2023-01-15 12Z state with significant QC/QR/QI/QS/QG. Both theta modes.
2. MYNN cycling init: the WS11 original-WRF witness (40 cases, copied verbatim)
   plus NEW real-structured ice+snow columns (Swiss CPU-WRF 15Z) run through the
   byte-unchanged proofs/w44_warmstart/mynn_init_oracle.py (pristine WRF objects)
   in a private root whose only difference is the column fixture JSON.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import netCDF4
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
MAY = Path("<DATA_ROOT>/wrf_gpu2/scratch/w44-warmstart/native_may04_00_newforcing/wrfinput_d01")
SWISS_DIR = Path("<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu")
SWISS_INPUT = SWISS_DIR / "wrfinput_d01"
SWISS_REBAL = SWISS_DIR / "wrfout_d01_2023-01-15_12:00:00"
SWISS_MYNN = SWISS_DIR / "wrfout_d01_2023-01-15_15:00:00"
WS11 = Path("<USER_HOME>/wrf_gpu2_lanes/w44-warmstart/WS11_mynn")
PRISTINE = Path("<USER_HOME>/src/wrf_pristine")
REBAL_NAMES = ("T", "THM", "MU", "MUB", "PB", "PHB", "ALB", "C1F", "C2F", "C1H", "C2H", "C3F", "C4F",
               "C3H", "C4H", "DNW", "RDNW", "RDN", "P_TOP")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(path, names, extra=None):
    with netCDF4.Dataset(path) as ds:
        out = {n: np.asarray(ds[n][0], np.float32) for n in names}
        if extra:
            for n in extra:
                out[n] = np.asarray(ds[n][0], np.float32)
    return out


def select_rebalance_columns(f, moist, per_species=16, clear=8):
    ny, nx = f["MU"].shape
    chosen = []
    for name in moist[1:]:
        colmax = f[name].max(axis=0).ravel()
        order = np.argsort(-colmax, kind="stable")
        chosen += [int(c) for c in order[:per_species] if colmax[c] > 0]
    total = sum(f[n] for n in moist[1:]).max(axis=0).ravel()
    chosen += [int(c) for c in np.flatnonzero(total == 0)[:clear]]
    chosen = sorted(set(chosen))
    return [(c // nx, c % nx) for c in chosen]


def crop_columns(f, columns):
    jj = np.array([c[0] for c in columns])
    ii = np.array([c[1] for c in columns])
    out = {}
    for name, value in f.items():
        if value.ndim == 3:
            out[name] = np.ascontiguousarray(value[:, jj, ii][:, None, :])
        elif value.ndim == 2:
            out[name] = np.ascontiguousarray(value[jj, ii][None, :])
        else:
            out[name] = value.copy()
    return out


def rebalance_witnesses(work):
    oracle = load_module("w44_rebalance_oracle", ROOT / "proofs/w44_warmstart/rebalance_oracle.py")
    from gpuwrf.io.cycling_init import rebalance_cycling
    (work / "rebalance_build").mkdir()
    native, binary, source = oracle.build(PRISTINE, work / "rebalance_build")
    moist = oracle.MOIST
    cases = {}
    may = read(MAY, REBAL_NAMES + moist)
    swiss = read(SWISS_REBAL, tuple(n for n in REBAL_NAMES if n != "ALB") + moist)
    base = read(SWISS_INPUT, ("ALB", "PB", "PHB", "MUB"))
    for name in ("PB", "PHB", "MUB"):
        # Resolved base state is static; the 12Z history carries the input's bytes.
        assert swiss[name].tobytes() == base[name].tobytes(), name
    swiss["ALB"] = base["ALB"]
    data, provenance = {}, {}
    for case, fields, src in (("may_d01", may, MAY), ("swiss_d01_12z", swiss, SWISS_REBAL)):
        columns = select_rebalance_columns(fields, moist)
        crop = crop_columns(fields, columns)
        record = {"source": str(src), "source_sha256": sha(src), "columns_jdx_idx": columns,
                  "n_columns": len(columns), "layout": "(nz, 1, n_columns) gathered whole columns",
                  "species_max": {n: float(crop[n].max()) for n in moist},
                  "species_nonzero_columns": {n: int((crop[n].max(axis=0) > 0).sum()) for n in moist}}
        if case == "swiss_d01_12z":
            record["ALB_source"] = str(SWISS_INPUT)
            record["ALB_source_sha256"] = sha(SWISS_INPUT)
        for key, value in crop.items():
            data[f"rebal__{case}__in__{key}"] = value
        record["oracle_vs_port"] = {}
        for mode in (0, 1):
            reference = oracle.native_call(native, crop, mode)
            candidate = rebalance_cycling(crop, moist_fields=moist, use_theta_m=mode)
            mutant = rebalance_cycling(crop, moist_fields=moist[:1], use_theta_m=mode)
            checks = {}
            for out in oracle.OUTPUTS:
                if out == "PH0":
                    continue
                ref = reference[out]
                data[f"rebal__{case}__wrf{mode}__{out}"] = ref
                limit = oracle.TOLERANCES[out]
                bound = limit["atol"] + limit["rtol"] * np.abs(ref)
                err = np.abs(candidate[out].astype(float) - ref)
                merr = np.abs(mutant[out].astype(float) - ref)
                checks[out] = {"byte_equal": candidate[out].tobytes() == ref.tobytes(),
                               "max_abs_error": float(err.max()), "rule_pass": bool(np.all(err <= bound)),
                               "qv_only_sum_max_abs_error": float(merr.max()),
                               "qv_only_sum_rule_pass": bool(np.all(merr <= bound))}
            record["oracle_vs_port"][f"use_theta_m={mode}"] = checks
        provenance[case] = record
    binary.unlink()
    meta = {"wrf_source": "dyn_em/start_em.F rebalance_cycl (body extracted unchanged)", **source,
            "oracle_script": "proofs/w44_warmstart/rebalance_oracle.py",
            "oracle_script_sha256": sha(ROOT / "proofs/w44_warmstart/rebalance_oracle.py"),
            "tolerances": oracle.TOLERANCES, "moist": list(moist), "cases": provenance}
    return data, meta


def swiss_mynn_fixture(n_columns=12):
    names = ("U", "V", "W", "T", "QVAPOR", "P", "PB", "PH", "PHB", "QCLOUD", "QICE", "QSNOW")
    with netCDF4.Dataset(SWISS_MYNN) as ds:
        f = {n: np.asarray(ds[n][0], np.float32) for n in names}
        s = {n: np.asarray(ds[n][0], np.float32) for n in ("UST", "XLAND", "TSK", "HFX", "QFX", "PSFC")}
        qsq = np.asarray(ds["QSQ"][0], np.float32) if "QSQ" in ds.variables else np.zeros_like(f["T"])
        dx = float(ds.DX)
    qi, qs = f["QICE"], f["QSNOW"]
    both = ((qi > 1e-6) & (qs > 1e-6)).any(axis=0)
    score = np.where(both, qs.max(axis=0) + qi.max(axis=0), -1.0)
    nx = score.shape[1]
    order = np.argsort(-score.ravel(), kind="stable")[:n_columns]
    assert np.all(score.ravel()[order] > 0)
    columns = [(int(c // nx), int(c % nx)) for c in order]
    f32 = np.float32
    state = {k: [] for k in ("u", "v", "w", "theta", "qv", "tke", "p", "rho", "dz", "qc", "qi", "qs", "qsq", "exner")}
    flux = {k: [] for k in ("ustar", "theta_flux", "qv_flux", "fltv", "xland", "t_skin", "rhosfc")}
    for j, i in columns:
        u = f32(.5) * (f["U"][:, j, i] + f["U"][:, j, i + 1])
        v = f32(.5) * (f["V"][:, j, i] + f["V"][:, j + 1, i])
        w = f32(.5) * (f["W"][:-1, j, i] + f["W"][1:, j, i])
        theta = f["T"][:, j, i] + f32(300)
        qv = f["QVAPOR"][:, j, i]
        p = f["P"][:, j, i] + f["PB"][:, j, i]
        phi = f["PH"][:, j, i] + f["PHB"][:, j, i]
        dz = (phi[1:] - phi[:-1]) / f32(9.81)
        exner = (p / f32(100000)) ** f32(2. / 7.)
        rho = p / (f32(287) * theta * exner * (f32(1) + f32(.608) * qv))
        for key, value in (("u", u), ("v", v), ("w", w), ("theta", theta), ("qv", qv), ("tke", np.zeros_like(qv)),
                           ("p", p), ("rho", rho), ("dz", dz), ("qc", f["QCLOUD"][:, j, i]), ("qi", qi[:, j, i]),
                           ("qs", qs[:, j, i]), ("qsq", qsq[:, j, i]), ("exner", exner)):
            state[key].append([float(x) for x in np.asarray(value, np.float32)])
        rhosfc = s["PSFC"][j, i] / (f32(287) * s["TSK"][j, i])
        tflux = s["HFX"][j, i] / (rhosfc * f32(1004.5))
        qflux = s["QFX"][j, i] / rhosfc
        for key, value in (("ustar", s["UST"][j, i]), ("theta_flux", tflux), ("qv_flux", qflux),
                           ("fltv", tflux + f32(.608) * theta[0] * qflux), ("xland", s["XLAND"][j, i]),
                           ("t_skin", s["TSK"][j, i]), ("rhosfc", rhosfc)):
            flux[key].append(float(np.float32(value)))
    fixture = {"provenance": {"source": str(SWISS_MYNN), "source_sha256": sha(SWISS_MYNN),
                              "columns_jdx_idx": columns,
                              "derivation": "mass-point U/V/W averages, theta=T+300, p=P+PB, dz=dPHI/9.81, exner=(p/1e5)^(2/7) REAL; tke overwritten by oracle"},
               "rows": [{"domain": "swiss_d01_15z", "dx": dx, "state": state, "flux": flux}]}
    return fixture, columns


def parse_mynn_oracle(directory):
    lines = (directory / "input.txt").read_text().splitlines()
    nz, nb = (int(float(x)) for x in lines[0].split()[:2])
    scalars, arrays, offset = [], [], 1
    for _ in range(nb):
        scalars.append(list(map(float, lines[offset].split())))
        offset += 1
        arrays.append([list(map(float, row.split())) for row in lines[offset:offset + nz]])
        offset += nz
    report = json.loads((directory / "report.json").read_text())
    cold = np.array([r["mode"] == "cold" for r in report["records"]])
    assert len(cold) == nb
    truth = np.load(directory / "expected.npz")["values"].astype(np.float32)
    return (np.asarray(scalars, np.float32), np.asarray(arrays, np.float32), truth, cold, report)


def mynn_witnesses(work):
    oracle_script = ROOT / "proofs/w44_warmstart/mynn_init_oracle.py"
    wrapper = ROOT / "tests/v025/b_phys/mynn_cold_oracle.f90"
    fixture, columns = swiss_mynn_fixture()
    private = work / "mynn_private_root"
    (private / "proofs/w44_warmstart").mkdir(parents=True)
    (private / "tests/v025/b_phys/fixtures").mkdir(parents=True)
    shutil.copy2(oracle_script, private / "proofs/w44_warmstart/mynn_init_oracle.py")
    shutil.copy2(wrapper, private / "tests/v025/b_phys/mynn_cold_oracle.f90")
    (private / "tests/v025/b_phys/fixtures/mynn_closure_carries_wrf.json").write_text(json.dumps(fixture))
    out = work / "mynn_swiss_oracle"
    run = subprocess.run([sys.executable, str(private / "proofs/w44_warmstart/mynn_init_oracle.py"), str(out)],
                         capture_output=True, text=True, timeout=600)
    (work / "mynn_swiss_oracle.log").write_text(run.stdout + run.stderr)
    run.check_returncode()
    data, sets = {}, {}
    for tag, directory in (("ws11", WS11), ("swiss_ice_snow", out)):
        scalars, arrays, truth, cold, report = parse_mynn_oracle(directory)
        for key, value in (("scalars", scalars), ("arrays", arrays), ("truth", truth), ("cold", cold)):
            data[f"mynn__{tag}__{key}"] = value
        sets[tag] = {"oracle_dir": str(directory), "cases": int(len(cold)), "cold_cases": int(cold.sum()),
                     "input_txt_sha256": sha(directory / "input.txt"),
                     "expected_npz_sha256": sha(directory / "expected.npz"),
                     "report_json_sha256": sha(directory / "report.json"),
                     "source_fixture": report["fixture"], "source_fixture_sha256": report["fixture_sha256"],
                     "wrf_source": report["source"], "wrf_source_sha256": report["source_sha256"],
                     "selection_clear_source_sha256": report["selection_clear_source_sha256"],
                     "pristine_objects": report["objects"], "compile_command": report["command"],
                     "qi_max": float(arrays[:, :, 10].max()), "qs_max": float(arrays[:, :, 11].max()),
                     "qc_max": float(arrays[:, :, 9].max()),
                     "cases_with_qi_and_qs": int(((arrays[:, :, 10] > 0).any(1) & (arrays[:, :, 11] > 0).any(1)).sum())}
    sets["swiss_ice_snow"]["columns_jdx_idx"] = columns
    sets["swiss_ice_snow"]["source"] = str(SWISS_MYNN)
    sets["swiss_ice_snow"]["source_sha256"] = sha(SWISS_MYNN)
    sets["swiss_ice_snow"]["oracle_script_sha256_in_private_root"] = sha(private / "proofs/w44_warmstart/mynn_init_oracle.py")
    sets["swiss_ice_snow"]["oracle_script_sha256_repo"] = sha(oracle_script)
    sets["swiss_ice_snow"]["wrapper_sha256"] = sha(wrapper)
    meta = {"slots": ["u", "v", "w", "theta", "qv", "tke", "p", "rho", "dz", "qc", "qi", "qs", "qsq", "exner"],
            "scalars": ["ustar", "theta_flux", "qv_flux", "fltv", "xland", "t_skin", "rhosfc", "dx"],
            "truth": ["QKE", "QSQ", "EL"],
            "wrf_source_lines": "phys/module_bl_mynnedmf.F driver init 618-747 (thl/thlv resolved-only, snow excluded :675-686) + mym_initialize",
            "limits_source": "proofs/w44_warmstart/compare_mynn_init.py LIMITS (frozen WS11 gate)",
            "sets": sets}
    return data, meta


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=False)
    rebal, rebal_meta = rebalance_witnesses(work)
    mynn, mynn_meta = mynn_witnesses(work)
    npz = HERE / "f3a_wrf_witnesses.npz"
    np.savez_compressed(npz, **rebal, **mynn)
    meta = {"schema": 1, "generator": "tests/data/w44_warmstart/generate_f3a_witnesses.py",
            "generator_sha256": sha(Path(__file__)), "npz": npz.name, "npz_sha256": sha(npz),
            "rebalance": rebal_meta, "mynn_init": mynn_meta,
            "jax_imported_by_generator": "jax" in sys.modules}
    (HERE / "f3a_wrf_witnesses.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({"npz_bytes": npz.stat().st_size, "rebalance": {k: v["oracle_vs_port"] for k, v in rebal_meta["cases"].items()},
                      "jax": meta["jax_imported_by_generator"]}, indent=1))


if __name__ == "__main__":
    main()
