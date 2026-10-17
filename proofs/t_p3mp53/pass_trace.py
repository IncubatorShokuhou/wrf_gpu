"""Filter MLIR pass headers from the unchanged native compile pipeline."""
import os
from pathlib import Path
import subprocess
import sys

with Path(sys.argv[1]).open('x') as log:
    process = subprocess.Popen(sys.argv[2:], stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True,
        env={**os.environ, 'MLIR_ENABLE_DUMP': '1'})
    for line in process.stdout:
        if any(word in line for word in ('IR Dump', 'LOWER', 'START', 'DONE',
                                         'Error', 'Traceback', 'COMPILED')):
            log.write(line)
            log.flush()
            print(line, end='', flush=True)
    rc = process.wait()
    log.write(f'PROCESS_RC {rc}\n')
sys.exit(rc)
