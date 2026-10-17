"""Stage and run an original CPU-WRF Swiss 3 h diagnostics reference, lane-local.

Invocation must carry the QUIET guard and assigned affinity/nice policy.
The exact WN3 executable used by RD11 is copied, hash checked, and rejected
if the known shared-output instrumentation path is present (E104).
Only lane-local namelist, logs and outputs are written; original inputs stay
read-only. Run as one MPI rank on the lane's single assigned CPU core.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

SOURCE = Path("<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu")
RUN = Path(sys.argv[1]).resolve()
LANE = Path("<USER_HOME>/wrf_gpu2_lanes/u2-diag")
assert RUN.is_relative_to(LANE) and not RUN.exists()
assert not Path("/tmp/wrf_gpu2_quiet").exists()
RUN.mkdir()
exe = (SOURCE / "wrf.exe").resolve()
exe_bytes = exe.read_bytes()
if b"wrf_truth/em_quarter_ss_center_dump.txt" in exe_bytes:
    raise RuntimeError("E104: executable contains shared dump instrumentation; use a private rebuilt executable")
sha = hashlib.sha256(exe_bytes).hexdigest()
shutil.copy2(exe, RUN / "wrf.exe")
assert hashlib.sha256((RUN / "wrf.exe").read_bytes()).hexdigest() == sha
for path in SOURCE.iterdir():
    if path.name in ("wrf.exe", "namelist.input") or path.name.startswith(("wrfout", "wrfrst", "rsl.", "namelist.output")):
        continue
    if path.is_symlink() or path.name.startswith(("wrfinput", "wrfbdy", "wrflowinp")):
        (RUN / path.name).symlink_to(path.resolve())
text = (SOURCE / "namelist.input").read_text()
for key,value in dict(run_days=0,run_hours=3,end_day=15,end_hour=3).items():
    text = re.sub(rf"(^\s*{key}\s*=).*",rf"\g<1> {value},",text,flags=re.M)
text = text.replace("&time_control", "&time_control\n nwp_diagnostics = 1,\n iofields_filename = 'diagnostics_iofields.txt',",1)
text = text.replace("&physics", "&physics\n prec_acc_dt = 60.,\n do_radar_ref = 1,\n swint_opt = 1,",1)
(RUN / "namelist.input").write_text(text)
(RUN / "diagnostics_iofields.txt").write_text("+:h:0:SWDDNI,SWDDIF,REFL_10CM,REFD_MAX\n")
cmd = ["<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/mpirun", "--bind-to", "none", "-np", "1", "./wrf.exe"]
receipt = dict(executable=str(exe),executable_sha256=sha,identical_copy=True,
               known_shared_dump_path_absent=True,source_inputs=str(SOURCE),
               argv=cmd,affinity=sorted(os.sched_getaffinity(0)),
               options=dict(nwp_diagnostics=1,prec_acc_dt=60.,do_radar_ref=1,swint_opt=1),
               namelist_sha256=hashlib.sha256(text.encode()).hexdigest(),
               timing_valid=False,timing_reason="shared CPU core; science reference only")
(RUN / "receipt.json").write_text(json.dumps(receipt,indent=2))
start=time.monotonic()
with (RUN / "wrf.log").open("w") as log:
    process = subprocess.Popen(cmd,cwd=RUN,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                               env={**os.environ,"OMP_NUM_THREADS":"1"})
    try:
        rc = process.wait(timeout=600)
    except subprocess.TimeoutExpired:
        # PRTE can put ranks into separate process groups under this owned SID.
        # Terminate only that session, never a name-based process search (E31).
        for path in Path("/proc").glob("[0-9]*/stat"):
            try:
                info=path.read_text().rsplit(")",1)[1].split()
                if int(info[3]) == process.pid:
                    os.kill(int(path.parent.name),signal.SIGKILL)
            except (FileNotFoundError,ProcessLookupError,PermissionError):
                pass
        process.wait()
        rc=124
receipt.update(rc=rc,wall_s=time.monotonic()-start)
logs="\n".join(p.read_text(errors="replace") for p in RUN.glob("rsl.error.*"))
receipt["success"]="SUCCESS COMPLETE WRF" in logs and rc == 0
receipt["frames"]=[p.name for p in sorted(RUN.glob("wrfout*"))]
(RUN / "receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt))
assert receipt["success"] and len(receipt["frames"]) == 4
