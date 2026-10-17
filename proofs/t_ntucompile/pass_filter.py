"""Run a probe with a hard wall cap; keep only MLIR pass headers / progress lines.

usage: pass_filter.py <timeout_s> <log> cmd...   (MLIR_ENABLE_DUMP=1 if PASSDUMP=1)
The child runs in its own session; on expiry the whole process group is killed,
so no orphaned compiler survives the cap (cf. t_p3mp53/pass_trace.py).
"""
import os
import signal
import subprocess
import sys
import threading
import time

timeout = float(sys.argv[1])
env = dict(os.environ)
if env.get('PASSDUMP') == '1':
    env['MLIR_ENABLE_DUMP'] = '1'
with open(sys.argv[2], 'x') as log:
    t0 = time.monotonic()
    proc = subprocess.Popen(sys.argv[3:], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env=env, start_new_session=True)
    expired = threading.Event()

    def watchdog():
        if proc.poll() is None and not done.wait(timeout):
            expired.set()
            os.killpg(proc.pid, signal.SIGKILL)

    done = threading.Event()
    threading.Thread(target=watchdog, daemon=True).start()
    for raw in proc.stdout:
        line = raw.decode(errors='replace')
        if any(w in line for w in ('IR Dump', 'LOWER', 'START', 'DONE', 'Error', 'error', 'Traceback',
                                   'COMPILED', 'PROBE')):
            log.write(f'{time.monotonic() - t0:9.2f} {line[:400].rstrip()}\n')
            log.flush()
    rc = proc.wait()
    done.set()
    if expired.is_set():
        rc = 124
    log.write(f'PROCESS_RC {rc} wall {time.monotonic() - t0:.2f}\n')
sys.exit(rc)
