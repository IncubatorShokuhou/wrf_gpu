"""Durable, host-only phase records for bounded preparation processes."""
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import time
import subprocess


class PhaseJournal:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        epoch = float(subprocess.check_output(['<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh'],
            text=True, timeout=5, env={**os.environ, 'JAX_PLATFORMS':'cpu'}))
        self.utc_origin = datetime.fromtimestamp(epoch, timezone.utc)

    def write(self, phase, event, **details):
        row = dict(phase=phase, event=event, pid=os.getpid(),
                   utc=(self.utc_origin + timedelta(seconds=time.monotonic()-self.started)).isoformat(),
                   elapsed_s=time.monotonic() - self.started, **details)
        with self.path.open("a") as stream:
            stream.write(json.dumps(row, default=str) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    @contextmanager
    def phase(self, phase_label, **details):
        started = time.monotonic()
        self.write(phase_label, "begin", **details)
        result = {}
        try:
            yield result
        except BaseException as error:
            self.write(phase_label, "error", seconds=time.monotonic() - started,
                       error=f"{type(error).__name__}: {error}", **details)
            raise
        else:
            self.write(phase_label, "end", seconds=time.monotonic() - started,
                       result=result, **details)


def write_receipt(path, receipt):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".pending")
    with temporary.open("w") as stream:
        json.dump(receipt, stream, indent=2, default=str)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
