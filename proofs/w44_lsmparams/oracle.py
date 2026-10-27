"""Changed-table pristine WRF REAL oracle, caller flags and exact-decimal output.

Builds a private wrapper linked to unchanged original WRF objects. Only the
wrapper's Jarvis selection and output precision/budget dump are extended.
Never writes the pristine tree. Run in the lane's artifact directory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_driver(repo, out, wrf):
    out.mkdir(parents=True, exist_ok=True)
    compiler = Path("<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran")
    env = dict(os.environ, LD_LIBRARY_PATH=str(compiler.parent.parent / "lib"))
    original = repo / "proofs/noahmp/noahmp_offline_driver.F90"
    source = original.read_text()
    source = source.replace("integer :: nsteps, istep", "integer :: nsteps, istep, w2_crs")
    needle = "  call noahmp_options(4, 1, 1, 3,"
    assert source.count(needle) == 1
    source = source.replace(needle, "  w2_crs=1\n  call get_environment_variable('W2_OPT_CRS',envv)\n"
                            "  if(len_trim(envv)>0) read(envv,*) w2_crs\n"
                            "  call noahmp_options(4, w2_crs, 1, 3,")
    source = source.replace("ES16.8", "ES25.16")
    marker = "    write(IOUT,'(A)') 'ENDCOL'"
    assert source.count(marker) == 1
    source = source.replace(marker,
        "    write(IOUT,'(A,2ES25.16)') 'W2_CLOSURE ', &\n"
        "       (sav+sag)-(fira+fsh+fcev+fgev+fctr+ssoil+canhs)+pah,canhs\n" + marker)
    driver = out / "w2_driver.F90"
    driver.write_text(source)
    recipe = (repo / "proofs/noahmp/build_driver.sh").read_text()
    stubs = []
    for name in ("_wrfstubs", "_modstubs"):
        prefix = f"cat > {name}.F90 <<'EOF'\n"
        body = recipe.split(prefix, 1)[1].split("\nEOF", 1)[0] + "\n"
        path = out / f"{name}.F90"
        path.write_text(body)
        stubs.append(path)
    phys = wrf / "phys"
    objects = [phys / (name + ".o") for name in ("module_sf_noahmplsm", "module_sf_noahmpdrv",
               "module_sf_gecros", "module_sf_noahmp_glacier", "module_sf_noahmp_groundwater")]
    executable = out / "w2_driver.exe"
    flags = ["-w", "-ffree-form", "-ffree-line-length-none", "-fconvert=big-endian",
             "-frecord-marker=4", "-O2", "-I" + str(phys)]
    subprocess.run([str(compiler), *flags, *map(str, stubs), str(driver), *map(str, objects),
                    "-o", str(executable)], cwd=out, env=env, check=True, timeout=120)
    provenance = {"compiler": str(compiler), "flags": flags, "real_bits": 32,
                  "wrapper_original_sha256": sha(original), "wrapper_sha256": sha(driver),
                  "changes": "caller opt_crs selector, lossless REAL decimal dumps, closure dump; physics object bodies unchanged",
                  "objects": {str(f): sha(f) for f in objects}, "executable_sha256": sha(executable)}
    (out / "build.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return executable, env, provenance


def edited_tables(repo, wrf, out, name, factor):
    editor = module("lsm_mptable_edit", repo / "proofs/w41_trackp/edit_mptable.py")
    result = {}
    for table in ("MPTABLE.TBL", "SOILPARM.TBL", "GENPARM.TBL"):
        text = (wrf / "run" / table).read_text()
        if table == "MPTABLE.TBL" and name in {"rsmin", "hvt", "hvb", "z0sno", "snow_emis"}:
            group = "noahmp_global_parameters" if name in {"z0sno", "snow_emis"} else "noahmp_modis_parameters"
            key = "RS" if name == "rsmin" else name.upper()
            start, end = editor._block(text, group)
            text = text[:start] + editor._edit_row(text[start:end], key, lambda _: factor) + text[end:]
        if table == "SOILPARM.TBL" and name in {"smcmax", "dksat", "bexp", "psisat"}:
            column = {"bexp": 1, "smcmax": 4, "psisat": 6, "dksat": 7}[name]
            lines = text.splitlines(keepends=True)
            start = next(i for i, line in enumerate(lines) if line.strip() == "STAS") + 2
            ncat = int(lines[start - 1].split(",")[0])
            for index in range(start, start + ncat):
                cells = lines[index].split(",")
                cells[column] = " " + repr(float(cells[column]) * factor)
                lines[index] = ",".join(cells)
            text = "".join(lines)
        path = out / table
        path.write_text(text)
        result[table] = sha(path)
    return result


def write_columns(path, rows):
    lines = ["MODIFIED_IGBP_MODIS_NOAH", f"{len(rows)} 1 4 {rows[0]['dt']}",
             " ".join(map(str, rows[0]["zsoil"]))]
    for c in rows:
        s, f = c["state_in"], c["forcing"]
        records = [
            [c["vegtyp"], c["isltyp"]],
            [c["lat_rad"], c["julian"], 365, f["cosz"], c["dx"], f["dz8w"], f["zlvl"]],
            [c["shdfac"], c["shdmax"], c["tbot"]],
            [f[k] for k in ("sfctmp", "sfcprs", "psfc", "uu", "vv", "q2", "qc", "soldn", "lwdn")],
            [f[k] for k in ("prcpconv", "prcpnonc", "prcpsnow", "prcpgrpl", "prcphail")],
            s["stc"], s["smc"], s["sh2o"],
            [s[k] for k in ("tv", "tg", "tah", "eah", "canliq", "canice", "fwet", "qsfc")],
            [c["wrf"]["phen_out"]["lai"], c["wrf"]["phen_out"]["sai"],
             s["snowh"], s["sneqv"], s["sneqvo"], s["albold"], s["tauss"]],
            [s["isnow"]], s["zsnso"], s["snice"], s["snliq"],
            [s[k] for k in ("cm", "ch", "smcwtd")],
        ]
        lines.extend(" ".join(map(str, record)) for record in records)
    path.write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--wrf", type=Path, default=Path("<USER_HOME>/src/wrf_pristine/WRF"))
    p.add_argument("--rows", choices=("source", "frozen"), default="source",
                   help="frozen: retained real frozen-soil columns (critic F5), soil arms + unchanged base")
    a = p.parse_args()
    sys.path[:0] = [str(a.repo / "src"), str(a.repo / "proofs/w44_lsmparams")]
    import jax
    from columns import frozen_soil_columns, source_columns
    assert jax.devices()[0].platform == "cpu"
    executable, env, build = build_driver(a.repo, a.out / "build", a.wrf)
    parser = module("lsm_parse_wrf", a.repo / "proofs/noahmp/build_noahmp_savepoints.py")
    rows = source_columns() if a.rows == "source" else frozen_soil_columns()
    factors = {"smcmax": 1.1, "dksat": 1.1, "bexp": 1.1, "psisat": 1.1,
               "hvt": 1.1, "hvb": .01, "rsmin": 1.1, "z0sno": 1.1, "snow_emis": .99}
    if a.rows == "frozen":
        # Same factors as the source arms; "base" = unchanged tables (activity census reference).
        factors = {"base": 1.0, **{name: factors[name] for name in ("smcmax", "dksat", "bexp", "psisat")}}
    report = {"build": build, "columns": rows, "arms": {}}
    for name, factor in factors.items():
        arm = a.out / name
        arm.mkdir(parents=True, exist_ok=True)
        tables = edited_tables(a.repo, a.wrf, arm, "none" if name == "base" else name, factor)
        write_columns(arm / "noahmp_columns.in", rows)
        run_env = dict(env, NOAHMP_WIRE_SOIL="1", NOAHMP_WIRE_FICEOLD="1", NOAHMP_NSTEPS="1",
                       W2_OPT_CRS="2" if name == "rsmin" else "1")
        result = subprocess.run([str(executable)], cwd=arm, env=run_env,
                                capture_output=True, text=True, timeout=60)
        (arm / "stdout.log").write_text(result.stdout)
        (arm / "stderr.log").write_text(result.stderr)
        assert result.returncode == 0 and "NOAHMP_OFFLINE_OK" in result.stdout, (name, result.returncode, result.stdout[-2000:])
        parser.HERE = arm
        values = parser._parse_savepoints()
        budget = [list(map(float, line.split()[1:])) for line in (arm / "noahmp_savepoints.out").read_text().splitlines()
                  if line.startswith("W2_CLOSURE")]
        assert len(values) == len(rows) == len(budget)
        report["arms"][name] = {"factor": factor, "opt_crs": int(run_env["W2_OPT_CRS"]),
                                  "tables": tables, "input_sha256": sha(arm / "noahmp_columns.in"),
                                  "output_sha256": sha(arm / "noahmp_savepoints.out"),
                                  "oracle": values, "closure": budget}
    (a.out / "oracle.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"arms": list(report["arms"]), "columns": len(rows), "out": str(a.out),
                      "physics": "unchanged original WRF objects", "platform": "cpu"}), flush=True)


if __name__ == "__main__":
    main()
