"""Bounded manager allocation: exact new nightly DONE, never stale permission."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import re

EXPECTED_ISSUE = '20261014_12z'
# Manager/alisios_mgr 2026-10-14T23:34Z explicitly opens the allocation NOW.
OPEN = datetime(2026, 10, 14, 23, 34, 40, tzinfo=timezone.utc)
CLOSE = datetime(2026, 10, 15, 18, 45, tzinfo=timezone.utc)
NIGHTLY = Path('<DATA_ROOT>/alisios/state/nightly_9nest/LAST_RUN.json')


def nightly_status(now, nightly):
    if now < OPEN:
        return 'WAIT_ALLOCATION_OPEN'
    if now >= CLOSE:
        return 'ALLOCATION_CLOSED'
    if nightly.get('issue') != EXPECTED_ISSUE or nightly.get('verdict') != 'DONE':
        return 'WAIT_EXACT_NIGHTLY_DONE'
    return 'READY'


def main():
    assert os.environ.get('JAX_PLATFORMS') == 'cpu'
    events = []
    for line in Path('<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md').read_text().splitlines():
        match = re.match(r'^\S+\s+manager:\s+.*?\bGPU (STOP|RESUME)\b', line)
        if match:
            events.append((match.group(1), line))
    if events and events[-1][0] == 'STOP':
        print('GPU_STOP: ' + events[-1][1])
        return 3
    try:
        nightly = json.loads(NIGHTLY.read_text())
    except (OSError, ValueError):
        nightly = {}
    status = nightly_status(datetime.now(timezone.utc), nightly)
    print(status + ': issue=' + str(nightly.get('issue')) + ' verdict=' + str(nightly.get('verdict')))
    return 0 if status == 'READY' else 3


if __name__ == '__main__':
    raise SystemExit(main())
