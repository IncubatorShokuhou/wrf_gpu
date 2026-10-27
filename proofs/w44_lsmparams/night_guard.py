"""Strict booked-window admission, CPU-only and before any GPU/lock use."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess


def admit(start, end, now, need):
    pattern = r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ"
    if not start or not end or not re.fullmatch(pattern, start) or not re.fullmatch(pattern, end):
        raise ValueError("GA_START_UTC/GA_END_UTC must both be explicit UTC Z timestamps")
    begin, finish = (int(datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()) for s in (start, end))
    if now < begin or finish - now < need:
        raise ValueError("booked window is future, expired or shorter than declared need")
    return {"start_epoch": begin, "end_epoch": finish, "now_epoch": now, "need_seconds": need}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--need", type=int, default=2400)
    args = parser.parse_args()
    assert args.need > 0
    try:
        raw = subprocess.check_output(["bash", "<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh"],
                                      text=True, timeout=10).strip()
        if not raw.isdigit():
            raise ValueError("real UTC helper did not return an integer epoch")
        result = admit(os.environ.get("GA_START_UTC"), os.environ.get("GA_END_UTC"), int(raw), args.need)
        print(json.dumps(result))
    except (ValueError, subprocess.SubprocessError) as exc:
        print("REFUSED_WINDOW (GPU NOT STARTED): " + str(exc))
        raise SystemExit(4)
