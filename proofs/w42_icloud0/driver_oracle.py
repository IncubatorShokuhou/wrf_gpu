"""w42-icloud0 Z01: the actual pristine WRF V4.7.1 radiation_driver with icloud = 0 vs icloud = 1 on real WN3 columns.

Derived from proofs/s1_mpfix/full_radiation_oracle.py (same generated-caller technique): the Fortran CALL is generated
from radiation_driver's own declaration contract, the driver and RRTMG come from the retained pristine WRF library
(module_radiation_driver.F compiled from the pristine source into the lane build; never edited, never run in the pristine
tree), solve_em is not invoked and the WRFGPU2 oracle dumps are disabled (all outputs live in WORK).

Operands: 16 real CPU-WRF WN3 0227 d02/d03 columns of the registered RE01 fixture (2 per regime: liquid thin ocean /
thick / land and MYNN-SGS-only daytime, clear daytime, ice / mixed / snow at night), Thompson radii (has_reqc/i/s = 1),
mp_physics = 8, cldovrlp = 2, ghg_input = 0, o3input = 0, slope_rad = topo_shading = 0 (the s1_mpfix caller controls).
Arms per column: icloud = 0 (the F7 intervention) and icloud = 1 (cal_cldfra1 control, icloud_bl = 0).

Output (JSON): per column and arm the scalars SCALARS and the profiles PROFILES as the driver returns them.
usage: python driver_oracle.py WORK OUT_JSON
"""
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WRF = ROOT / "data/wrf_pristine/WRF"
ENV = Path("<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build")
FC = ENV / "bin/x86_64-conda-linux-gnu-gfortran"
FIXTURE = ROOT / "tests/v025/fid_q2/fixtures/rrtmg_mp_re_0227_v1.npz"
COLUMNS = (0, 84, 3, 87, 6, 90, 12, 108, 9, 93)  # daytime liq_thin_ocean, liq_thick, liq_land, sgs_only, clear
NIGHT = ("ice", "mixed", "snow")                    # + the first two columns of each night regime
SCALARS = ("glw", "swdown", "gsw", "olr", "solconx", "coszen", "lwupt", "swupt", "lwupb", "lwdnb", "swdnb", "swddir",
           "swddif", "swddni", "swdownc", "lwdnbc", "lwuptc", "swdnbc", "swuptc", "lwcf", "swcf", "cldt")
PROFILES = ("cldfra", "rthratenlw", "rthratensw", "rthratenlwc", "rthratenswc")


def main(work: Path, out: Path):
    work.mkdir(parents=True, exist_ok=True)
    fx = np.load(FIXTURE)
    cat = fx["category"]
    columns = list(COLUMNS) + [int(i) for c in NIGHT for i in np.flatnonzero(cat == c)[:2]]
    source = WRF / "phys/module_radiation_driver.F"
    text = subprocess.check_output([str(FC), "-E", "-cpp", "-DEM_CORE=1", "-P", "-ffree-form", str(source)], text=True)
    statements, statement = [], ""
    for line in text.splitlines():
        line = line.split("!")[0].strip()
        if not line or line.startswith("#"):
            continue
        statement += " " + line.lstrip("&").rstrip("&")
        if not line.endswith("&"):
            statements.append(statement.strip())
            statement = ""
    sig = next(s for s in statements if re.match(r"SUBROUTINE radiation_driver\s*\(", s, re.I))
    args = [v.strip().lower() for v in sig[sig.index("(") + 1:-1].split(",")]

    def split(s):
        depth, start, parts = 0, 0, []
        for i, c in enumerate(s):
            depth += (c == "(") - (c == ")")
            if c == "," and depth == 0:
                parts.append(s[start:i].strip())
                start = i + 1
        return parts + [s[start:].strip()]

    decl = {}
    for s in statements:
        if "::" not in s:
            continue
        left, right = s.split("::", 1)
        if not re.match(r"(real|integer|logical)\b", left, re.I):
            continue
        for name in split(right):
            name = re.split(r"[=(]", name)[0].strip().lower()
            if name in args and name not in decl:
                decl[name] = left
    assert set(args) <= decl.keys()
    assert {*SCALARS, *PROFILES, "icloud", "icloud_bl"} <= set(args)
    # WRF's grid caller supplies every argument (several OPTIONAL ones are read without PRESENT guards).
    selected = args
    params = dict(nz=44, ids=1, ide=2, jds=1, jde=2, kds=1, kde=45, ims=1, ime=1, jms=1, jme=1, kms=1, kme=45, kts=1,
                  kte=44, num_tiles=1, alevsiz=1, levsiz=59, no_src_types=1, n_ozmixm=12, n_aerosolc=1, paerlev=1,
                  cam_abs_dim1=1, cam_abs_dim2=1)
    declarations = ["integer,parameter::" + ",".join(f"{k}={v}" for k, v in params.items()),
                    f"integer::k,case,ic,arm,ncol={len(columns)}"]
    initial = []
    for name in selected:
        if name in params:
            continue
        attrs = [s for s in split(decl[name]) if not s.lower().startswith(("intent", "optional"))]
        declarations.append(",".join(attrs) + " :: " + name)
        initial.append(f"{name}={'.false.' if attrs[0].lower().startswith('logical') else '0'}")
    known = {"lw_physics": 4, "sw_physics": 4, "mp_physics": 8, "cldovrlp": 2, "idcor": 0, "itimestep": 1,
             "julday": 59, "julyr": 2026, "yr": 2026, "nphs": 1, "stepra": 1, "ra_call_offset": 0, "id": 1,
             "cu_physics": 0, "swint_opt": 0, "sf_surface_physics": 4, "aer_type": 1, "wif_input_opt": 1, "radt": 30.,
             "dt": 54., "xland": 2., "xlat": 28., "xlong": -16., "cen_lat": 28., "tsk": 290., "emiss": .98,
             "albedo": .08, "gmt": 0., "julian": 59.5, "xtime": 720., "p_top": float(fx["in_p_top"]),
             "cam_abs_freq_s": 21600., "aer_ssa_val": 1., "aer_angexp_val": 1.3, "aer_asy_val": .85, "i_start": 1,
             "i_end": 1, "j_start": 1, "j_end": 1, "has_reqc": 1, "has_reqi": 1, "has_reqs": 1}
    initial += [f"{n}={v}" for n, v in known.items()]
    initial += [f"f_{n}=.true." for n in ("qv", "qc", "qr", "qi", "qs", "qg")]
    read_names = ["t", "p", "pi", "rho", "qv", "qc", "qi", "qs", "qr", "qg", "re_cloud", "re_ice", "re_snow"]
    initial += [f"read(10){n}(1,1:nz,1)" for n in read_names]
    initial += ["read(10)p8w,t8w,dz8w", "read(10)tsk,emiss,albedo,xland,xlat,xlong,julian,xtime"]
    initial += ["do k=1,kme", "z(1,k,1)=sum(dz8w(1,1:k,1))", "enddo"]
    initial += [f"{n}(1,kme,1)={n}(1,nz,1)" for n in read_names]
    program = ("program icloud0_radiation\nuse module_radiation_driver,only:radiation_driver\n"
               "use module_ra_rrtmg_lw,only:rrtmg_lwinit\nuse module_ra_rrtmg_sw,only:rrtmg_swinit\nimplicit none\n"
               + "\n".join(declarations)
               + "\nopen(20,file='output.bin',access='stream',form='unformatted',status='replace',convert='native')\n"
               "do ic=1,ncol\ndo arm=0,1\n"
               "open(10,file='input.bin',access='stream',form='unformatted',status='old',convert='native')\n"
               "do case=1,ic\n" + "\n".join(initial) + "\nenddo\nclose(10)\n"
               "icloud=arm\n"
               "if (ic==1.and.arm==0)then\n"
               "call rrtmg_lwinit(p_top,.true.,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,i_start(1),i_end(1),"
               "j_start(1),j_end(1),kts,kte)\n"
               "call rrtmg_swinit(.true.,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,i_start(1),i_end(1),"
               "j_start(1),j_end(1),kts,kte)\nendif\n"
               "call radiation_driver( &\n" + ", &\n".join(f"{n}={n}" for n in selected) + ")\n"
               "write(20)ic,icloud\n"
               "write(20)" + ",".join(SCALARS) + "\n"
               "write(20)" + ",".join(f"{n}(1,1:nz,1)" for n in PROFILES) + "\n"
               "enddo\nenddo\nclose(20)\nend program\n")
    (work / "driver.f90").write_text(program)
    flags = ["-w", "-ffree-line-length-none", "-fconvert=big-endian", "-frecord-marker=4", "-O0", "-g", "-fcheck=all"]
    incs = [f"-I{WRF / p}" for p in ("phys", "frame", "share", "main", "inc")]
    (work / "module_radiation_driver.F").write_bytes(source.read_bytes())
    run = lambda cmd: subprocess.run(cmd, cwd=work, check=True)  # noqa: E731
    run([str(FC), "-c", *flags, *incs, "-cpp", "-DEM_CORE=1", "-ffree-form", "module_radiation_driver.F", "-o",
         "module_radiation_driver.o"])
    run([str(FC), "-c", *flags, *incs, "driver.f90", "-o", "driver.o"])
    libs = [WRF / p for p in ("main/libwrflib.a", "external/fftpack/fftpack5/libfftpack.a",
                              "external/io_grib1/libio_grib1.a", "external/io_grib_share/libio_grib_share.a",
                              "external/io_int/libwrfio_int.a")]
    run([str(FC), *flags, "driver.o", "module_radiation_driver.o", *map(str, libs), f"-L{WRF}/external/esmf_time_f90",
         "-lesmf_time", str(WRF / "frame/module_internal_header_util.o"), str(WRF / "frame/pack_utils.o"),
         f"-L{WRF}/external/io_netcdf", "-lwrfio_nf", f"-L{ENV}/lib", "-lnetcdff", "-lnetcdf",
         f"-Wl,-rpath,{ENV}/lib", "-o", "driver"])
    for name in ("RRTMG_LW_DATA", "RRTMG_SW_DATA", "CAMtr_volume_mixing_ratio"):
        link = work / name
        if not link.exists():
            link.symlink_to(WRF / "run" / name)
    mapping = {"t": "in_t", "p": "in_p_hyd", "pi": "in_pi", "rho": "in_rho", "qv": "in_qv", "qc": "in_qc_rad",
               "qi": "in_qi_rad", "qs": "in_qs", "qr": "in_qr", "qg": "in_qg", "re_cloud": "col_re_cloud",
               "re_ice": "col_re_ice", "re_snow": "col_re_snow"}
    with (work / "input.bin").open("wb") as stream:
        for idx in columns:
            for name in read_names:
                np.asarray(fx[mapping[name]][idx], np.float32).tofile(stream)
            for name in ("in_p_hyd_w", "in_t8w", "in_dz8w"):
                np.asarray(fx[name][idx], np.float32).tofile(stream)
            np.asarray([fx[n][idx] for n in ("in_tsk", "in_emiss", "in_albedo", "in_xland", "in_xlat", "in_xlong",
                                             "in_julian")] + [fx["in_tau"][idx] * 60.], np.float32).tofile(stream)
    env = {k: v for k, v in os.environ.items() if not k.startswith("WRFGPU2_ORACLE")}
    subprocess.run([str(work / "driver")], cwd=work, env=env, check=True)
    raw = (work / "output.bin").read_bytes()
    offset, rows = 0, []
    for _ in range(2 * len(columns)):
        head = np.frombuffer(raw, offset=offset, count=2, dtype="<i4")
        offset += 8
        scalars = np.frombuffer(raw, offset=offset, count=len(SCALARS), dtype="<f4")
        offset += 4 * len(SCALARS)
        profiles = np.frombuffer(raw, offset=offset, count=len(PROFILES) * 44, dtype="<f4").reshape(len(PROFILES), 44)
        offset += 4 * profiles.size
        assert np.isfinite(scalars).all() and np.isfinite(profiles).all()
        rows.append(dict(column=columns[int(head[0]) - 1], icloud=int(head[1]),
                         scalars=dict(zip(SCALARS, map(float, scalars))),
                         profiles={n: p.tolist() for n, p in zip(PROFILES, profiles)}))
    assert offset == len(raw)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    receipt = dict(scope="actual pristine WRF V4.7.1 radiation_driver, icloud 0 vs 1, real WN3 0227 RE01 columns",
                   source_sha256=sha(source), library_sha256=sha(WRF / "main/libwrflib.a"),
                   generated_driver_sha256=hashlib.sha256(program.encode()).hexdigest(),
                   real_operand_fixture_sha256=sha(FIXTURE), input_columns=columns,
                   categories=[str(cat[i]) for i in columns], mapping=mapping, caller_controls=known,
                   scalars=list(SCALARS), profiles=list(PROFILES), rows=rows)
    out.write_text(json.dumps(receipt, separators=(",", ":")) + "\n")
    print(f"PASS pristine radiation_driver {len(rows)} calls ({len(columns)} columns x icloud 0/1), finite")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
