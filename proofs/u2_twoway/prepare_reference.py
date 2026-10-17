"""Stage an isolated 108-second original CPU-WRF feedback=1 reference.

Uses existing WN3 input copies, not ALISIOS station/truth data. No shared WRF
executable, case or dump is modified. Run the staged wrf.exe separately.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import shutil
import sys

SOURCE = Path("<USER_HOME>/wrf_gpu2_lanes/s1-nestspec/coupled/mp40")
EXE = Path("<DATA_ROOT>/canairy_meteo/artifacts/wrf_src/WRF/install_gen2_dmpar/bin/wrf")
target = Path(sys.argv[1]).resolve()
target.mkdir(exist_ok=False, parents=True)
raw = EXE.read_bytes()
assert all(s not in raw for s in (b"WRFGPU2_DUMP", b"<DATA_ROOT>/wrf_gpu2/wrf_truth", b"em_quarter_ss_center_dump"))
shutil.copy2(EXE,target/"wrf.exe")
plan = json.loads((SOURCE/"PLAN.json").read_text())
for name in ("wrfinput_d01","wrfinput_d02","wrfbdy_d01"):
    shutil.copy2(SOURCE/name,target/name)
    assert hashlib.sha256((target/name).read_bytes()).hexdigest()==plan["input_files"][name]["sha256"]
for file in SOURCE.iterdir():
    if file.is_symlink() and not file.name.startswith(("wrf", "rsl")):
        (target/file.name).symlink_to(file.resolve())
# Reuse pristine Thompson setup tables instead of regenerating them for every
# isolated reference. These are private copies; the source is never writable.
table_cache = {}
for name in ("qr_acr_qg_V4.dat","qr_acr_qsV2.dat","freezeH2O.dat"):
    source = Path(plan["source_case"])/name
    shutil.copy2(source,target/name)
    table_cache[name] = {"source":str(source),"sha256":hashlib.sha256((target/name).read_bytes()).hexdigest()}
text = (SOURCE/"namelist.input").read_text()
for key,value in (("mp_physics","8, 8"),("feedback","1"),("smooth_option","2")):
    pattern = rf"^\s*{key}\s*=.*$"
    assert re.search(pattern,text,flags=re.M|re.I),key
    text = re.sub(pattern,f" {key} = {value},",text,flags=re.M|re.I)
(target/"namelist.input").write_text(text)
plan.update(created_utc=datetime.now(timezone.utc).isoformat(),case=str(target),mp_physics=8,
            feedback=1,smooth_option=2,namelist_sha256=hashlib.sha256(text.encode()).hexdigest(),
            executable_sha256=hashlib.sha256(raw).hexdigest(),thompson_setup_tables=table_cache,
            execution="CPU on granted lane cores nice19 MPI1 OMP1, timeout600; no GPU")
(target/"PLAN.json").write_text(json.dumps(plan,indent=2)+"\n")
print(json.dumps(plan,indent=2))
