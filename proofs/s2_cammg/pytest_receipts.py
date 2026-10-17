"""Small per-test receipts survive a bounded shared-consumer arm timeout."""
import json
import os
from pathlib import Path


def pytest_runtest_logreport(report):
    if report.when!='call' and not report.failed:
        return
    target=os.environ.get('CAMMG_PYTEST_RECEIPT')
    if target:
        with Path(target).open('a') as out:
            out.write(json.dumps(dict(nodeid=report.nodeid,when=report.when,
                outcome=report.outcome,duration=report.duration,
                failure=str(report.longrepr) if report.failed else None))+'\n')
