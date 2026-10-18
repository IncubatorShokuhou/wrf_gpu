"""Pristine WRF 4.7.1 start_em CTOPO/CTOPO2 oracle vs gpuwrf.io.topo_wind_static (lane w41-topowind).

No JAX. Extracts dyn_em/start_em.F:1535-1628 verbatim (sha-checked), builds it inside
ctopo_oracle_wrapper.F90 with the pristine FCOPTIM (-O2 -ftree-vectorize -funroll-loops,
no FMA), an -O0 no-contraction variant and an FMA variant (spread only), and runs it on
REAL inputs:
  WN3 20260227_18z_a1 d01/d02/d03: HT = CPU-WRF history HGT at t=0 (= grid%ht after the
    nest terrain blending), XLAND/VAR/MAPFAC/COSALPHA/SINALPHA from wrfinput, VAR_SSO from a
    lane geogrid run with the NCAR varsso_10m dataset (the ALISIOS GEOGRID.TBL lowres path);
  Swiss RD11 d01 (42x42, 3 km Alps): wrfinput fields + lane geogrid VAR_SSO.
topo_wind 1 and 2 on every domain, plus a periodic-x/y variant (halo wrap) on WN3 d03.
Writes <out>/CTOPO_ORACLE.json and the test fixture tests/fixtures/w41_topowind/ctopo_oracle.npz.
usage: python build_ctopo_oracle.py <lane_build_dir> <out_dir>
"""
from pathlib import Path
import ctypes, hashlib, json, re, subprocess, sys

import numpy as np
import netCDF4

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
from gpuwrf.io.topo_wind_static import start_em_ctopo  # noqa: E402

BUILD, OUT = Path(sys.argv[1]), Path(sys.argv[2])
START_EM = Path("<USER_HOME>/src/wrf_pristine/WRF/dyn_em/start_em.F")
START_EM_SHA = "4b6c98a06fb2b2c4472af73415099e311502747dc5543d9e9b1323497d83ddb4"
LINES = (1535, 1628)
FC = "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"
VARIANTS = {
    "wrf_fcoptim": ["-O2", "-ftree-vectorize", "-funroll-loops"],
    "O0_nocontract": ["-O0", "-ffp-contract=off"],
    "fma": ["-O2", "-mfma", "-ffp-contract=fast"],
}
SCR = Path("<DATA_ROOT>/wrf_gpu2/scratch/w41-topowind")
WN3 = Path("<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run")
SWISS = Path("<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu")
CASES = {
    "wn3_d01": (WN3 / "wrfinput_d01", WN3 / "wrfout_d01_2026-02-28_00:00:00", SCR / "geogrid_wn3/geo_em.d01.nc"),
    "wn3_d02": (WN3 / "wrfinput_d02", WN3 / "wrfout_d02_2026-02-28_00:00:00", SCR / "geogrid_wn3/geo_em.d02.nc"),
    "wn3_d03": (WN3 / "wrfinput_d03", WN3 / "wrfout_d03_2026-02-28_00:00:00", SCR / "geogrid_wn3/geo_em.d03.nc"),
    "swiss_d01": (SWISS / "wrfinput_d01", SWISS / "wrfinput_d01", SCR / "geogrid_swiss/geo_em.d01.nc"),
}
# Real-structured adversarial variant (E154/E102): every real land cell has sigma_sso > e,
# so the 'land, sqrt(VAR_SSO) <= 2.718 -> 1' branch needs scaled VAR_SSO (sigma straddles e).
SCALED = {"wn3_d03_sigma_small": ("wn3_d03", 3.0e-5)}
CASES.update({k: CASES[v[0]] for k, v in SCALED.items()})
FIXTURE_CASES = ("wn3_d03", "swiss_d01", "wn3_d03_sigma_small")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(name, flags):
    out = BUILD / name
    out.mkdir(parents=True, exist_ok=True)
    so = out / "ctopo.so"
    subprocess.run([FC, "-shared", "-fPIC", *flags, "-ffree-form", "-ffree-line-length-none", "-I", str(BUILD),
                    str(HERE / "ctopo_oracle_wrapper.F90"), "-o", str(so)], cwd=out, check=True, timeout=300)
    asm = subprocess.run(["objdump", "-d", str(so)], capture_output=True, text=True, check=True).stdout
    fn = ctypes.CDLL(str(so)).ctopo_probe
    arr = np.ctypeslib.ndpointer(dtype=np.float32, flags="F_CONTIGUOUS")
    fn.argtypes = [ctypes.c_int] * 5 + [arr] * 8 + [ctypes.c_float] * 2 + [arr] * 3
    fn.restype = None
    return fn, dict(flags=flags, sha256=sha(so), fma_instructions=len(re.findall(r"\svfn?m(?:add|sub)\d", asm)),
                    logf_calls=len(re.findall(r"call.*<logf@plt>", asm)))


def load_case(name):
    if name in SCALED:
        base, factor = SCALED[name]
        c = load_case(base)
        c["var_sso"] = (c["var_sso"] * np.float32(factor)).astype(np.float32)
        c["provenance"] = dict(c["provenance"], var_sso_scale=factor)
        return c
    wi, hist, geo = CASES[name]
    with netCDF4.Dataset(wi) as w, netCDF4.Dataset(hist) as h, netCDF4.Dataset(geo) as g:
        f = lambda ds, v: np.asarray(ds[v][0], np.float32)
        return dict(ht=f(h, "HGT"), xland=f(w, "XLAND"), var2d=f(w, "VAR"), var_sso=f(g, "VAR_SSO"),
                    msftx=f(w, "MAPFAC_MX"), msfty=f(w, "MAPFAC_MY"), cosa=f(w, "COSALPHA"), sina=f(w, "SINALPHA"),
                    rdx=np.float32(1.0 / float(w.DX)), rdy=np.float32(1.0 / float(w.DY)),
                    provenance={"wrfinput": str(wi), "ht_source": str(hist), "var_sso_geo_em": str(geo),
                                "wrfinput_sha256": sha(wi), "geo_em_sha256": sha(geo)})


def run_oracle(fn, c, topo_wind, periodic):
    F = lambda a: np.asfortranarray(a.T)  # (ny,nx) C -> Fortran (nx,ny)
    ny, nx = c["ht"].shape
    outs = [np.zeros((nx, ny), np.float32, order="F") for _ in range(3)]
    fn(nx, ny, topo_wind, int(periodic), int(periodic), F(c["ht"]), F(c["xland"]), F(c["var_sso"]), F(c["var2d"]),
       F(c["msftx"]), F(c["msfty"]), F(c["cosa"]), F(c["sina"]), float(c["rdx"]), float(c["rdy"]), *outs)
    return [o.T.copy() for o in outs]


def census(c, topo_wind, lap, ctopo, ctopo2):
    land = c["xland"] < np.float32(1.5)
    out = {"cells": int(land.size), "land": int(land.sum()), "water": int((~land).sum())}
    if topo_wind == 1:
        sig = np.sqrt(np.where(land, c["var_sso"], 1)).astype(np.float32)
        out.update(land_sigma_le_e=int((land & (sig <= np.float32(2.718))).sum()),
                   land_log=int((land & (sig > np.float32(2.718))).sum()),
                   lap_keep=int((lap > -10).sum()), lap_blend=int(((lap <= -10) & (lap >= -20)).sum()),
                   lap_ramp=int(((lap < -20) & (lap >= -30)).sum()), lap_zero=int((lap < -30).sum()),
                   ctopo_min=float(ctopo.min()), ctopo_max=float(ctopo.max()), ctopo2_lt1=int((ctopo2 < 1).sum()))
    else:
        raw = (c["var2d"] * np.float32(0.4)) / np.float32(200.0) + np.float32(1.175)
        out.update(land_capped=int((land & (raw > np.float32(1.575))).sum()),
                   land_uncapped=int((land & (raw <= np.float32(1.575))).sum()),
                   ctopo_min=float(ctopo.min()), ctopo_max=float(ctopo.max()))
    return out


def main():
    assert sha(START_EM) == START_EM_SHA
    BUILD.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    block = "".join(START_EM.read_text().splitlines(keepends=True)[LINES[0] - 1:LINES[1]])
    assert block.lstrip().startswith("IF ( .NOT. config_flags%restart ) THEN") and block.rstrip().endswith("END IF")
    (BUILD / "start_em_ctopo_block.inc").write_text(block)
    variants = {name: build(name, flags) for name, flags in VARIANTS.items()}
    report = {"start_em": str(START_EM), "start_em_sha256": START_EM_SHA, "lines": LINES,
              "block_sha256": hashlib.sha256(block.encode()).hexdigest(), "compiler": FC,
              "variants": {k: v[1] for k, v in variants.items()}, "cases": {}}
    fixture = {}
    ok = True
    for name in CASES:
        c = load_case(name)
        report["cases"][name] = {"provenance": c["provenance"], "shape": list(c["ht"].shape), "runs": {}}
        for topo_wind in (1, 2):
            for periodic in ((False, True) if name == "wn3_d03" else (False,)):
                if name in SCALED and topo_wind == 2:
                    continue
                key = f"tw{topo_wind}{'_periodic' if periodic else ''}"
                port = start_em_ctopo(topo_wind, c["ht"], c["xland"], var_sso=c["var_sso"], var2d=c["var2d"],
                                      periodic_x=periodic, periodic_y=periodic)
                run = {}
                for vname, (fn, _) in variants.items():
                    ctopo, ctopo2, lap = run_oracle(fn, c, topo_wind, periodic)
                    d = {"ctopo_byte_equal": bool(np.array_equal(ctopo.view(np.uint32), port[0].view(np.uint32))),
                         "ctopo2_byte_equal": bool(np.array_equal(ctopo2.view(np.uint32), port[1].view(np.uint32))),
                         "ctopo_max_abs": float(np.abs(ctopo - port[0]).max()),
                         "ctopo2_max_abs": float(np.abs(ctopo2 - port[1]).max())}
                    run[vname] = d
                    if vname != "fma":
                        ok &= d["ctopo_byte_equal"] and d["ctopo2_byte_equal"]
                    if vname == "wrf_fcoptim":
                        ref = (ctopo, ctopo2, lap)
                run["census"] = census(c, topo_wind, ref[2], ref[0], ref[1])
                report["cases"][name]["runs"][key] = run
                if name in FIXTURE_CASES:
                    fixture[f"{name}_{key}_ctopo"], fixture[f"{name}_{key}_ctopo2"] = ref[0], ref[1]
        if name in FIXTURE_CASES:
            for field in ("ht", "xland", "var_sso", "var2d"):
                fixture[f"{name}_{field}"] = c[field]
    report["PASS_byte_exact_wrf_and_O0"] = bool(ok)
    (OUT / "CTOPO_ORACLE.json").write_text(json.dumps(report, indent=1))
    fx = REPO / "tests/fixtures/w41_topowind"
    fx.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(fx / "ctopo_oracle.npz", **fixture)
    print(json.dumps({k: {r: (v["wrf_fcoptim"], v["census"]) for r, v in c["runs"].items()}
                      for k, c in report["cases"].items()}, indent=0)[:6000])
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
