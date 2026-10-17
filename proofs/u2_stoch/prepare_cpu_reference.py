"""Private CPU-WRF 3-step reference case with an isolated existing debug path.

Only the executable's literal FILE= string changes; .text bytes are sealed
identical. No source tree or reference case is written. Run the returned
private executable separately under the lane CPU affinity and timeout.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
from datetime import datetime,timedelta


def _section(blob, name):
    assert blob[:6] == b"\x7fELF\x02\x01"
    shoff = struct.unpack_from("<Q", blob, 40)[0]
    entsize, count, names_idx = struct.unpack_from("<HHH", blob, 58)
    headers = [struct.unpack_from("<IIQQQQIIQQ", blob, shoff+i*entsize) for i in range(count)]
    names_hdr = headers[names_idx]
    names = blob[names_hdr[4]:names_hdr[4]+names_hdr[5]]
    for row in headers:
        start = row[0]
        if names[start:names.index(b"\0",start)].decode() == name:
            return blob[row[4]:row[4]+row[5]]
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vertical", type=int, choices=(0,1), default=0)
    parser.add_argument("--tables-from", type=Path)
    parser.add_argument("--max-dom",type=int,choices=(1,2),default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    wrf = Path("<USER_HOME>/src/wrf_pristine/WRF")
    source = wrf/"main/wrf.exe"
    original = source.read_bytes()
    old = b"<DATA_ROOT>/wrf_gpu2/wrf_truth/em_quarter_ss_center_dump.txt"
    new = b"<USER_HOME>/wrf_gpu2_lanes/u2-stoch/WRF_DUMP.txt"
    assert len(new) <= len(old) and original.count(old) > 0
    patched = original.replace(old, new.ljust(len(old),b" "))
    assert _section(original,".text") == _section(patched,".text")
    executable = args.output/"wrf.exe"
    executable.write_bytes(patched)
    executable.chmod(0o755)
    for path in args.case.glob("wrfinput_d[0-9][0-9]"):
        (args.output/path.name).symlink_to(path.resolve())
    for path in args.case.glob("wrfbdy_d[0-9][0-9]"):
        (args.output/path.name).symlink_to(path.resolve())
    for path in args.case.glob("wrflowinp_d[0-9][0-9]"):
        (args.output/path.name).symlink_to(path.resolve())
    for path in (wrf/"run").iterdir():
        if path.is_file() and (path.name.endswith((".TBL","_DATA",".formatted")) or path.name in {"tr49t67","CAMtr_volume_mixing_ratio"}):
            target = args.output/path.name
            if not target.exists():
                target.symlink_to(path.resolve())
    if args.tables_from:
        for name in ("qr_acr_qg_V4.dat","qr_acr_qsV2.dat","freezeH2O.dat"):
            path = args.tables_from/name
            assert path.is_file(), path
            (args.output/name).symlink_to(path.resolve())
    text = (args.case/"namelist.input").read_text()
    def first(key):
        match=re.search(rf"^\s*{key}\s*=\s*([^,\n]+)",text,re.M|re.I)
        assert match is not None,key
        return int(match.group(1))
    dt=first("time_step")
    seconds=3*dt
    start=datetime(*[first("start_"+key) for key in ("year","month","day","hour","minute","second")])
    end=start+timedelta(seconds=seconds)
    changes = {"run_days":0,"run_hours":0,"run_minutes":0,"run_seconds":seconds,"max_dom":args.max_dom,
               "history_interval":0,"restart_interval":0,"frames_per_outfile":1}
    changes.update({"end_"+key:",".join([str(getattr(end,key))]*args.max_dom)
                    for key in ("year","month","day","hour","minute","second")})
    for key, value in changes.items():
        text, n = re.subn(rf"^\s*{key}\s*=.*$",f" {key} = {value},",text,flags=re.M|re.I)
        assert n == 1, key
    cadence=dt if args.max_dom==1 else dt//3
    restart=dt if args.max_dom==1 else seconds
    text = text.replace("&time_control",f"&time_control\n history_interval_s={','.join([str(cadence)]*args.max_dom)},\n restart_interval_s={restart},")
    text += "\n&stoch\n skebs=1,\n sppt=1,\n spp=1,\n rand_perturb=1,\n nens=7,\n"
    for key in ("skebs_vertstruc","sppt_vertstruc","rand_pert_vertstruc","vertstruc_spp_conv","vertstruc_spp_pbl","vertstruc_spp_lsm"):
        text += f" {key}={args.vertical},\n"
    for key in ("lengthscale_sppt","lengthscale_rand_pert","lengthscale_spp_conv","lengthscale_spp_pbl","lengthscale_spp_lsm"):
        text += f" {key}=15000.,\n"
    text += "/\n"
    (args.output/"namelist.input").write_text(text)
    result = {"source_executable":str(source),"source_sha256":hashlib.sha256(original).hexdigest(),
              "private_sha256":hashlib.sha256(patched).hexdigest(),"text_sha256":hashlib.sha256(_section(original,".text")).hexdigest(),
              "debug_file_literal_only":True,"debug_literal_instances":original.count(old),
              "source_case":str(args.case),"vertical":args.vertical,"steps":3,"dt":dt,"domains":args.max_dom,
              "stoch":"skebs/sppt/spp/rand_perturb=1,nens7,gaussianLength15km"}
    (args.output/"reference_manifest.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
