"""Stage short original CPU-WRF Swiss control/spectral runs, without reference writes.

Usage: python prepare_cpu_cases.py LANE_ROOT
WRF executable is an exact lane-owned copy of RD11's original CPU executable.
Thompson writable lookup files are copied, never symlinked. History/reference
outputs are excluded. E104 dump paths in instrumented binaries are refused.
"""
from pathlib import Path
import hashlib
import json
import re
import shutil
import sys
import os
import netCDF4

CPU = Path("<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu")
FDDA = Path("<USER_HOME>/wrf_gpu2_lanes/o2-fdda/case/wrffdda_d01")
TABLES = Path(os.environ.get("SPECTRAL_WRF_TABLE_DIR", "<USER_HOME>/src/wrf_pristine/WRF/test/em_real/oracle_run"))
TABLE_NAMES = ("qr_acr_qg_V4.dat", "qr_acr_qsV2.dat", "freezeH2O.dat")


def edit(text, group, values):
    match = re.search(rf"&{group}\b(.*?)/", text, flags=re.S | re.I)
    body = match.group(1) if match else "\n"
    for key, value in values.items():
        line = f" {key} = {value},"
        pat = rf"^\s*{key}\s*=.*$"
        body = re.sub(pat, line, body, flags=re.M | re.I) if re.search(pat, body, flags=re.M | re.I) else body + line + "\n"
    new = f"&{group}{body}/"
    return text[:match.start()] + new + text[match.end():] if match else text + "\n" + new + "\n"


def main():
    root = Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=True)
    exe = (CPU / "wrf.exe").read_bytes()
    if b"em_quarter_ss_center_dump" in exe or b"em_grav2d_x_touchdown_dump" in exe:
        raise RuntimeError("instrumented WRF executable: relocate E104 paths before running")
    receipt = dict(cpu_source=str((CPU / "wrf.exe").resolve()), executable_sha256=hashlib.sha256(exe).hexdigest(),
                   instrumented_dump_paths=False, analysis_source=str(FDDA), source_case=str(CPU), steps=3, dt_s=18.)
    # o2's writer has the correct real.exe field layout but omits WRF metadata.
    # Preserve all analysis values and supply a truthful V4-compatible header;
    # never use force_use_old_data or modify the shared original file.
    shutil.copyfile(FDDA, root / "wrffdda_d01")
    with netCDF4.Dataset(CPU / "wrfinput_d01") as original, netCDF4.Dataset(root / "wrffdda_d01", "a") as analysis:
        analysis.setncatts({k: original.getncattr(k) for k in original.ncattrs() if k != "TITLE"})
        analysis.TITLE = "WRF V4.7.1-compatible auxinput10 analyses assembled from original CPU-WRF history"
        analysis.GPUWRF_ANALYSIS_PROVENANCE = str(FDDA)
        # ext_ncd_read_field returns BEFORE reading values if FieldType is absent.
        # WRF_REAL=104 (external/io_netcdf/wrf_io_flags.h). A SUCCESS log alone
        # does not prove analyses were ingested; the debug/restart probe does.
        for name, variable in analysis.variables.items():
            if name == "Times":
                continue
            variable.FieldType = 104
            variable.MemoryOrder = "XYZ" if variable.ndim == 4 else "XY"
            variable.stagger = {"U": "X", "V": "Y", "PH": "Z"}.get(name.split("_NDG_")[0], "")
    for arm, code in (("control", 0), ("spectral", 2)):
        case = root / arm
        case.mkdir()
        for p in CPU.iterdir():
            if p.name.startswith(("wrfout", "wrfrst", "rsl.", "namelist", "wrf.exe", "wrffdda")):
                continue
            if p.name in TABLE_NAMES:
                shutil.copyfile(p, case / p.name)
            elif p.is_file():
                (case / p.name).symlink_to(p)
        for name in TABLE_NAMES:
            if (TABLES / name).is_file() and not (case / name).exists():
                shutil.copyfile(TABLES / name, case / name)
        (case / "wrf.exe").write_bytes(exe)
        (case / "wrf.exe").chmod(0o755)
        (case / "wrffdda_d01").symlink_to(root / "wrffdda_d01")
        text = (CPU / "namelist.input").read_text()
        text = edit(text, "time_control", dict(run_days=0, run_hours=0, run_minutes=0, run_seconds=54,
            end_day=15, end_hour=0, end_minute=0, end_second=54, history_interval=0, history_interval_s=18,
            history_interval_m=0, frames_per_outfile=1))
        text = edit(text, "fdda", dict(grid_fdda=code, gfdda_inname="'wrffdda_d<domain>'", gfdda_interval_m=60,
            gfdda_end_h=6, io_form_gfdda=2, guv="3e-4", gt="3e-4", gph="3e-4", gq=0,
            xwavenum=2, ywavenum=2, fgdt=0))
        (case / "namelist.input").write_text(text)
    (root / "PREPARATION.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
