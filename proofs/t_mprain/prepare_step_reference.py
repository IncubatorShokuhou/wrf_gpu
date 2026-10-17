"""Prepare an E104-safe original CPU-WRF one-hour/every-step reference.

Read existing s1-gpuqual inputs/reference inventory only. Changes are run clock
and history selection/cadence; physics and input fields are unchanged.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import sys

from gpuwrf.io.netcdf_lock import Dataset

assert os.environ.get("JAX_PLATFORMS") == "cpu"
assert not Path("/tmp/wrf_gpu2_quiet").exists()
root, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
out.mkdir(parents=True, exist_ok=False)
case = Path("<USER_HOME>/wrf_gpu2_lanes/s1-gpuqual/mp16")
wrf = root / "data/wrf_pristine/WRF"
for item in (wrf / "run").iterdir():
    if item.is_file() and item.name not in ("namelist.input", "wrf.exe", "real.exe", "ndown.exe"):
        (out / item.name).symlink_to(item.resolve())
for name in ("wrfinput_d01", "wrfbdy_d01"):
    (out/name).symlink_to((case / "input" / name).resolve())
text = (case / "input/namelist.input").read_text()
edits = dict(run_days=0, run_hours=1, end_day=15, end_hour=1, history_interval=0, frames_per_outfile=201)
for key,value in edits.items():
    text,n = re.subn(rf"(?im)^(\s*{key}\s*=)[^\n]*", lambda m:m[1]+f" {value},", text)
    assert n == 1, key
text=text.replace("&time_control\n", "&time_control\n history_interval_s = 18,\n iofields_filename = 'history_fields.txt',\n")
(out/"namelist.input").write_text(text)
keep={"Times", "T", "THM", "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP", "RAINNC", "SNOWNC", "GRAUPELNC"}
with Dataset(case/"cpu/wrfout_d01_2023-01-15_00:00:00") as ds:
    drop=sorted(set(ds.variables)-keep)
(out/"history_fields.txt").write_text("\n".join("-:h:0:"+",".join(drop[i:i+8]) for i in range(0,len(drop),8))+"\n")
raw=(wrf/"main/wrf.exe").read_bytes()
patched=raw
patches=[]
for i,name in enumerate(("em_quarter_ss_center_dump.txt", "em_grav2d_x_touchdown_dump.txt")):
    old=("<DATA_ROOT>/wrf_gpu2/wrf_truth/"+name).encode()
    new=f"<USER_HOME>/wrf_gpu2_lanes/t-mprain/d{i}.txt".encode()
    assert old in patched and len(new)<=len(old)
    patched=patched.replace(old,new.ljust(len(old),b" "))
    patches.append(dict(old=old.decode(),new=new.decode()))
(out/"wrf.exe").write_bytes(patched)
(out/"wrf.exe").chmod(0o755)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt=dict(original_executable_sha256=hashlib.sha256(raw).hexdigest(),
             private_executable_sha256=sha(out/"wrf.exe"), patches=patches,
             physics="mp16 MYNN5 RRTMG4 NoahMP4 Swiss42x42x44 dt18; existing gpuqual inputs",
             clock_edits=edits, history_interval_s=18, inputs={n:sha(out/n) for n in ("wrfinput_d01","wrfbdy_d01","namelist.input")})
(out/"manifest.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt,indent=2))
