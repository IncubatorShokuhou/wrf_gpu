"""Prepare topo_wind composite cases (lane w41-topowind) — CPU only, no JAX.

Each case = an admitted v0.4.0 composite + topo_wind, inputs unchanged except that
topo_wind=1 needs VAR_SSO: LANE COPIES of wrfinput get VAR_SSO from a lane geogrid run
with the NCAR varsso_10m dataset (GEOGRID.TBL lowres path) and GOT_VAR_SSO=1. real.exe
copies VAR_SSO verbatim from met_em (module_initialize_real.F:4891-4904 only sets the
flag), so this equals rerunning WPS/real with varsso present. Originals are never touched.
  c02tw1  Swiss 3 km, YSU (bl=1, sf=1, Kessler, KF), topo_wind=1
  c07tw1  Swiss 3 km, Shin-Hong (bl=11, mp9, RRTMG, cu16), topo_wind=1
  c02tw2  Swiss 3 km, YSU, topo_wind=2 (VAR; no input change)
  wn3tw   WN3 0227 3-nest, YSU+sfclay1 on all domains, topo_wind = 2,1,1
The CPU dir gets the pristine serial WRF 4.7.1 (private copy, only the two E104 debug
OPEN literals redirected, like proofs/v040/final_validation/prepare.py).
usage: python prepare_cases.py <out_root> [case ...]
"""
from pathlib import Path
import hashlib, json, shutil, sys

import numpy as np
import netCDF4

ROOT = Path(sys.argv[1])
PRISTINE = Path("<USER_HOME>/src/wrf_pristine/WRF")
VPREP = Path("<USER_HOME>/wrf_gpu2_lanes/v-prep")
SCR = Path("<DATA_ROOT>/wrf_gpu2/scratch/w41-topowind")
WN3 = Path("<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run")
DMPAR = Path("<DATA_ROOT>/canairy_meteo/artifacts/wrf_src/WRF/install_gen2_dmpar/run")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
CASES = {
    "c02tw1": dict(base="c02", topo_wind=[1], var_sso={"d01": SCR / "geogrid_swiss/geo_em.d01.nc"}),
    "c07tw1": dict(base="c07", topo_wind=[1], var_sso={"d01": SCR / "geogrid_swiss/geo_em.d01.nc"}),
    "c02tw2": dict(base="c02", topo_wind=[2], var_sso={}),
    # 24 h (D6 window), no restart files; everything else = the ALISIOS WN3 0227 namelist.
    "wn3tw": dict(base="wn3", topo_wind=[2, 1, 1], physics={"bl_pbl_physics": [1, 1, 1], "sf_sfclay_physics": [1, 1, 1]},
                  time_control={"run_hours": [24], "end_month": [3, 3, 3], "end_day": [1, 1, 1],
                                "restart_interval": [100000]},
                  var_sso={d: SCR / f"geogrid_wn3/geo_em.{d}.nc" for d in ("d02", "d03")}),
}


def set_physics(text, key, values, group="physics"):
    """Replace or add one &group entry in a namelist.input text (keeps every other byte)."""
    line = f" {key} = " + ", ".join(str(v) for v in values) + ","
    out, done, in_phys = [], False, False
    for raw in text.splitlines():
        s = raw.strip().lower()
        if s.startswith("&" + group):
            in_phys = True
        elif in_phys and s.startswith(key + " ") or (in_phys and s.startswith(key + "=")):
            raw, done = line, True
        elif in_phys and s == "/":
            if not done:
                out.append(line)
                done = True
            in_phys = False
        out.append(raw)
    assert done, key
    return "\n".join(out) + "\n"


def inject_var_sso(src, dst, geo_em):
    shutil.copy2(src, dst)
    with netCDF4.Dataset(geo_em) as g:
        vs = np.asarray(g["VAR_SSO"][0], np.float32)
    with netCDF4.Dataset(dst, "r+") as w:
        assert w["VAR_SSO"].shape[1:] == vs.shape
        w["VAR_SSO"][0] = vs
        w["GOT_VAR_SSO"][0] = 1
    return dict(source=str(src), source_sha256=sha(src), geo_em=str(geo_em), geo_em_sha256=sha(geo_em),
                var_sso_max=float(vs.max()), copy_sha256=sha(dst))


def private_executable(case):
    raw = (PRISTINE / "main/wrf.exe").read_bytes()
    result, patches = raw, []
    for index, old in enumerate((b"<DATA_ROOT>/wrf_gpu2/wrf_truth/em_quarter_ss_center_dump.txt",
                                 b"<DATA_ROOT>/wrf_gpu2/wrf_truth/em_grav2d_x_touchdown_dump.txt")):
        new = str(case / f"d{index}").encode()
        assert len(new) <= len(old) and result.count(old) == 1
        patches.append(dict(offset=result.index(old), old=old.decode(), new=new.decode()))
        result = result.replace(old, new.ljust(len(old), b" "))
    path = case / "cpu/wrf.exe"
    path.write_bytes(result)
    path.chmod(0o755)
    return dict(original=str(PRISTINE / "main/wrf.exe"), original_sha256=sha(PRISTINE / "main/wrf.exe"),
                private_sha256=sha(path), patches=patches, arithmetic_bytes_unchanged=True)


def prepare(name):
    spec = CASES[name]
    case = ROOT / name
    assert not list((case / "cpu").glob("wrfout_*")), "never overwrite reference frames"
    inp, cpu = case / "input", case / "cpu"
    inp.mkdir(parents=True, exist_ok=True)
    cpu.mkdir(exist_ok=True)
    if spec["base"] == "wn3":
        src_dir, domains = WN3, ("d01", "d02", "d03")
        text = (WN3 / "namelist.input").read_text()
        extra = [f"wrfbdy_d01"] + [f"wrflowinp_{d}" for d in domains]
    else:
        src_dir, domains = VPREP / spec["base"] / "input", ("d01",)
        text = (src_dir / "namelist.input").read_text()
        extra = ["wrfbdy_d01"]
    for key, values in spec.get("physics", {}).items():
        text = set_physics(text, key, values)
    for key, values in spec.get("time_control", {}).items():
        text = set_physics(text, key, values, group="time_control")
    text = set_physics(text, "topo_wind", spec["topo_wind"])
    (inp / "namelist.input").write_text(text)
    record = dict(case=name, base=spec["base"], namelist_sha256=sha(inp / "namelist.input"), inputs={})
    for d in domains:
        target = inp / f"wrfinput_{d}"
        if target.exists() or target.is_symlink():
            target.unlink()
        src = (src_dir / f"wrfinput_{d}").resolve()
        if d in spec["var_sso"]:
            record["inputs"][d] = inject_var_sso(src, target, spec["var_sso"][d])
        else:
            target.symlink_to(src)
            record["inputs"][d] = dict(source=str(src), source_sha256=sha(src), symlink=True)
    for f in extra:
        target = inp / f
        if not target.exists():
            target.symlink_to((src_dir / f).resolve())
    run_dir = DMPAR if spec["base"] == "wn3" else PRISTINE / "run"
    if spec["base"] == "wn3":
        for table in ("freezeH2O.dat", "qr_acr_qg_V4.dat", "qr_acr_qsV2.dat"):
            if not (cpu / table).exists():
                (cpu / table).symlink_to(WN3 / table)
    for item in run_dir.iterdir():
        if item.is_file() and item.name not in ("namelist.input", "wrf.exe", "real.exe", "ndown.exe"):
            target = cpu / item.name
            if not target.exists():
                target.symlink_to(item.resolve())
    for item in inp.iterdir():
        target = cpu / item.name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(item.resolve() if item.is_symlink() else item)
    for table in ("MPTABLE.TBL",):
        target = cpu / table
        if not target.exists():
            target.symlink_to(PRISTINE / "phys/noahmp/parameters" / table)
    if spec["base"] == "wn3":
        # ALISIOS' dmpar WRF 4.7.1 build (the WN3 reference executable, D15/D26), 4 ranks.
        exe = cpu / "wrf.exe"
        if not exe.exists():
            exe.symlink_to(DMPAR / "wrf.exe")
        record["executable"] = dict(path=str(DMPAR / "wrf.exe"), sha256=sha(DMPAR / "wrf.exe"), ranks=4)
    else:
        record["executable"] = private_executable(case)
    (case / "PREPARE.json").write_text(json.dumps(record, indent=1) + "\n")
    print(name, json.dumps({d: r.get("var_sso_max", "symlink") for d, r in record["inputs"].items()}))


if __name__ == "__main__":
    for name in sys.argv[2:] or list(CASES):
        prepare(name)
