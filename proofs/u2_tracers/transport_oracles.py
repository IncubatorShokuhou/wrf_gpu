"""Original WRF advection on E205-admitted per-RK tracer operands; no JAX.

The capture JSON must accept complete first-step production-carry byte identity.
Plain RK1/2 use full advect_scalar; RK3 uses full advect_scalar_pd. The latter
is evaluated with both true split WRF masses and the captured folded caller,
holding its already-preloaded qold identical to separate these two effects.
Outputs are diagnostics, not a changed tolerance or coupled forecast gate.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

import numpy as np


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(module, directory, mode):
    # Reuse the existing source extractor and wrapper; change compiler options
    # before its sole compile, never the extracted original routine body.
    def compile_original(command, **kwargs):
        retained = [flag for flag in command[1:] if not
                    (flag.startswith("-O") or flag.startswith("-march=")
                     or flag.startswith("-ffp-contract="))]
        arithmetic = (["-O0", "-ffp-contract=off"] if mode == "literal" else
                      ["-O2", "-ffp-contract=fast", "-march=native", "-mfma"])
        command[:] = [os.environ.get("FC", "<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/x86_64-conda-linux-gnu-gfortran"),
                      *arithmetic, "-finit-real=zero", "-finit-integer=0",
                      "-finit-logical=false", *retained]
        return subprocess.run(command, **kwargs)
    module.subprocess = SimpleNamespace(run=compile_original)
    return module.build(directory, contraction="off" if mode == "literal" else "fast")


def compare(actual, literal, fma):
    actual, literal, fma = (np.asarray(value) for value in (actual, literal, fma))
    if not all(np.isfinite(value).all() for value in (actual, literal, fma)):
        raise ValueError("nonfinite captured or original tendency; comparison refused")
    error = np.abs(actual.astype("f8") - literal.astype("f8"))
    spread = np.abs(fma.astype("f8") - literal.astype("f8"))
    excess = error - spread
    imax = tuple(int(i) for i in np.unravel_index(np.argmax(error), error.shape))
    iexcess = tuple(int(i) for i in np.unravel_index(np.argmax(excess), excess.shape))
    targets = {}
    for index in ((0,19,17),(0,23,24)):
        if len(index) == error.ndim and all(i < size for i,size in zip(index,error.shape)):
            targets[str(index)] = {"native": float(actual[index]), "literal": float(literal[index]),
                                  "fma": float(fma[index]), "error": float(error[index]),
                                  "spread": float(spread[index]), "excess": float(excess[index])}
    return {"values": int(error.size), "finite": bool(all(np.isfinite(v).all() for v in (actual,literal,fma))),
            "byte_equal_literal": bool(actual.dtype == literal.dtype and actual.tobytes() == literal.tobytes()),
            "byte_equal_fma": bool(actual.dtype == fma.dtype and actual.tobytes() == fma.tobytes()),
            "literal_nonbyte": int(np.count_nonzero(actual.view("u4") != literal.view("u4"))),
            "fma_nonbyte": int(np.count_nonzero(actual.view("u4") != fma.view("u4"))),
            "max_abs_error": float(error[imax]), "error_argmax": list(imax),
            "max_abs_original_fma_spread": float(spread.max()),
            "outside_pointwise_original_fma_spread": int(np.count_nonzero(excess > 0)),
            "max_excess": float(excess[iexcess]), "excess_argmax": list(iexcess),
            "at_error_argmax": {"native": float(actual[imax]), "literal": float(literal[imax]),
                                "fma": float(fma[imax]), "spread": float(spread[imax])},
            "coupled_argmax_targets": targets}


def flux_comparison(captured, case, repo, dx, dy):
    """History retained RK3 fields are directly paired with captured RK3."""
    nc = load_module("transport_netcdf_lock", repo / "src/gpuwrf/io/netcdf_lock.py")
    frames = sorted(case.glob("wrfout_d01_*"))
    if len(frames) != 4:
        raise ValueError("original flux case must contain four three-step history files")
    path = frames[1]
    with nc.Dataset(path) as ds:
        original = {key: np.asarray(ds.variables[name][0], dtype=np.float32)
                    for key,name in (("ru","RU_M"),("rv","RV_M"),("rom","WW_M"),("mut","MUTS"))}
    port = {key:captured["rk3_" + key] for key in original}
    if any(port[key].shape != original[key].shape for key in original):
        raise ValueError("captured/original flux shapes differ")
    summaries = {key:compare(port[key],original[key],original[key]) for key in original}
    c1 = captured["rk3_c1"].astype("f8")[:,None,None]
    c2 = captured["rk3_c2"].astype("f8")[:,None,None]
    dt = float(captured["rk3_dt_rk"])
    normalized = {}

    def faces(plane, axis):
        left = np.take(plane, [0], axis=axis)
        right = np.take(plane, [-1], axis=axis)
        pad = np.concatenate((left,plane,right), axis=axis)
        lo = [slice(None)] * 2; hi = [slice(None)] * 2
        lo[axis]=slice(None,-1);hi[axis]=slice(1,None)
        return .5*(pad[tuple(lo)]+pad[tuple(hi)])

    for key,axis,map_key,spacing,mass_map_key in (("ru",1,"msfuy",dx,"msfty"),
                                                ("rv",0,"msfvx",dy,"msftx")):
        mass_port = c1 * faces(port["mut"].astype("f8"),axis)[None] + c2
        mass_original = c1 * faces(original["mut"].astype("f8"),axis)[None] + c2
        effective_port = port[key].astype("f8") / mass_port
        effective_original = original[key].astype("f8") / mass_original
        # This quotient is an inferred effective mass-flux velocity. It is
        # not the arithmetic average of physical small-step wind velocities.
        velocity_delta = (effective_port - effective_original) * captured[map_key][None]
        cfl_delta = (effective_port - effective_original) * dt / spacing * faces(captured[mass_map_key],axis)[None]
        normalized[key] = (velocity_delta, cfl_delta)
        summaries[key]["effective_flux_velocity_delta_m_s"] = {
            "max_abs":float(np.max(np.abs(velocity_delta))),
            "argmax":list(map(int,np.unravel_index(np.argmax(np.abs(velocity_delta)),velocity_delta.shape)))}
        summaries[key]["inferred_pd_courant_delta"] = {
            "max_abs":float(np.max(np.abs(cfl_delta))),
            "argmax":list(map(int,np.unravel_index(np.argmax(np.abs(cfl_delta)),cfl_delta.shape)))}
    spots = {}
    for z,y,x in ((0,19,17),(0,23,24)):
        entry = {key:float(captured["rk3_"+key][z,y,x]) for key in ("q","qold","frozen","adv")}
        entry["mut_port"] = float(port["mut"][y,x]);entry["mut_original"] = float(original["mut"][y,x])
        entry["faces"] = {}
        for key,indices in (("ru",((z,y,x),(z,y,x+1))),
                            ("rv",((z,y,x),(z,y+1,x))),
                            ("rom",((z,y,x),(z+1,y,x)))):
            entry["faces"][key] = [{"index":list(index),"port":float(port[key][index]),
                                    "original":float(original[key][index]),
                                    "delta":float(np.float64(port[key][index])-np.float64(original[key][index]))}
                                   for index in indices]
            if key in normalized:
                velocity_delta, cfl_delta = normalized[key]
                for face,index in zip(entry["faces"][key],indices):
                    face["effective_flux_velocity_delta_m_s"] = float(velocity_delta[index])
                    face["inferred_pd_courant_delta"] = float(cfl_delta[index])
        spots[str((z,y,x))] = entry
    return {"scope":"first-step final-RK3 only; both target positions sampled in this step",
            "frame":str(path),"frame_sha256":sha_file(path),"fields":summaries,"spots":spots,
            "normalization":"[I] effective velocity=flux/(c1*adjacent currentMUTS mean+c2)*face map; Courant=flux/mass*dt/dx*adjacent mass-map mean"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dx-m", type=float, required=True)
    parser.add_argument("--dy-m", type=float, required=True)
    parser.add_argument("--original-flux-case", type=Path)
    args = parser.parse_args()
    admission = json.loads(args.capture.read_text())
    if admission.get("accepted") is not True or admission.get("unequal_leaves") or not admission.get("leaves"):
        raise ValueError("E205 capture rejected or incomplete; no captured operand may be used")
    stages = admission.get("stages", [1, 2, 3])
    if not isinstance(stages, list) or not stages or len(set(stages)) != len(stages) or any(
            not isinstance(stage, int) or stage not in (1, 2, 3) for stage in stages):
        raise ValueError("capture stages must be a nonempty unique subset of [1,2,3]")
    if args.dx_m != args.dy_m or args.dx_m <= 0:
        raise ValueError("existing plain wrapper requires positive equal DX/DY")
    capture_npz = args.capture.with_suffix(".npz")
    with np.load(capture_npz) as archive:
        captured = {name: archive[name] for name in archive.files}
    if any(not np.isfinite(value).all() for value in captured.values()):
        raise ValueError("nonfinite admitted operand; original comparison refused")
    repo = Path(__file__).resolve().parents[2]
    plain = load_module("transport_plain_original", repo / "tests/v025/b_core/scalar_oracle.py")
    pd = load_module("transport_pd_original", repo / "tests/v025/b_core/pd_scalar_oracle.py")
    # Verify the plain routine's executable map usage, rather than assuming
    # actual maps conformal. Its sole executable map field is MSFTX, so passing
    # it as all six wrapper arguments is valid even when other maps differ.
    routine = re.search(r"(?ims)^SUBROUTINE advect_scalar\s*\(.*?^END SUBROUTINE advect_scalar\s*$", plain.SOURCE.read_text()).group(0)
    body = routine[routine.index("! definition of flux operators"):]
    unused = [name for name in ("msfty","msfux","msfuy","msfvx","msfvy")
              if re.search(r"(?i)\b" + name + r"\b", body)]
    if unused:
        raise ValueError(f"plain all-map wrapper no longer admissible: {unused}")
    libs = {}
    manifests = {}
    needed = ([("plain",plain)] if any(stage < 3 for stage in stages) else []) + (
        [("pd",pd)] if 3 in stages else [])
    for name,module in needed:
        for mode in ("literal","fma"):
            directory = args.build_dir / (name + "_" + mode)
            libs[name,mode] = build(module, directory, mode)
            manifests[name + "_" + mode] = json.loads((directory / "manifest.json").read_text())
    output_arrays = {}
    rows = []
    for rk in stages:
        a = {name[len(f"rk{rk}_"):]: value for name,value in captured.items() if name.startswith(f"rk{rk}_")}
        observed = a["adv"]
        if rk < 3:
            operands = {name:a[name] for name in ("q","ru","rv","rom","fzm","fzp","rdzw","c1","c2")}
            operands.update(mu=a["mut"],map=captured["msftx"])
            references = {mode: plain.evaluate(libs["plain",mode],operands,1./args.dx_m)
                          for mode in ("literal","fma")}
            rows.append({"rk":rk,"caller":"plain current-stage q", "comparison": compare(observed,**references)})
            for mode,array in references.items(): output_arrays[f"rk{rk}_{mode}"] = array
        else:
            operands = {name:a[name] for name in ("q","qold","ru","rv","rom","mut","mub","fzm","fzp","rdzw","c1","c2")}
            operands.update(muold=a["old_pert"],**dict(zip(("mx","my","ux","uy","vx","vy"),
                                (captured[n] for n in ("msftx","msfty","msfux","msfuy","msfvx","msfvy")))))
            large_dt = float(a["dt_rk"])
            for caller in ("split_original", "folded_captured"):
                selected = dict(operands)
                if caller == "folded_captured":
                    selected.update(mub=np.zeros_like(a["mub"]),muold=a["mu_old"])
                references = {mode: pd.evaluate(libs["pd",mode],selected,1./args.dx_m,1./args.dy_m,large_dt)
                              for mode in ("literal","fma")}
                rows.append({"rk":rk,"caller":caller,"qold":"identical captured preloaded qold; preload error held fixed",
                             "comparison":compare(observed,**references)})
                for mode,array in references.items():output_arrays[f"rk{rk}_{caller}_{mode}"] = array
            split = output_arrays["rk3_split_original_literal"]
            folded = output_arrays["rk3_folded_captured_literal"]
            rows.append({"rk":rk,"caller":"split_vs_folded_mass_only", "comparison":compare(folded,split,split)})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez(args.output.with_suffix(".npz"),**output_arrays)
    receipt = {"scope":"matched admitted per-RK operands; diagnostic only, no coupled acceptance/tolerance change",
               "capture_json_sha256":sha_file(args.capture),"capture_npz_sha256":sha_file(capture_npz),
               "capture_admission":admission,"generator_sha256":sha_file(Path(__file__)),
               "stages": stages,
               "original_source_sha256":sha_file(plain.SOURCE),
               "wrapper_sha256":{name:sha_file(repo / f"tests/v025/b_core/{name}_scalar_oracle.py")
                                 for name in ("pd",)} | {"plain":sha_file(repo / "tests/v025/b_core/scalar_oracle.py")},
               "compiler_manifests":manifests,"dx_m":args.dx_m,"dy_m":args.dy_m,
               "plain_map_admission":"original executable body only reads MSFTX; other five maps absent",
               "maps_max_msftx_minus_msfty":float(np.max(np.abs(captured["msftx"]-captured["msfty"]))),
               "rows":rows,"reference_npz_sha256":sha_file(args.output.with_suffix(".npz"))}
    if args.original_flux_case:
        if 3 not in stages:
            raise ValueError("direct retained-flux comparison requires captured RK3")
        receipt["original_flux_comparison"] = flux_comparison(captured,args.original_flux_case,repo,args.dx_m,args.dy_m)
    args.output.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
