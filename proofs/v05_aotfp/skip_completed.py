"""Resume the uncompleted suffix of an ordered, timed-out engineering suite."""
import json,os
from pathlib import Path
import pytest
@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config,items):
    n=int(os.environ['V05_COMPLETED_PREFIX'])
    record={'completed_prefix':n,'total':len(items),'completed_nodeids':[i.nodeid for i in items[:n]],'pending_nodeids':[i.nodeid for i in items[n:]]}
    Path(os.environ['V05_RESUME_RECORD']).write_text(json.dumps(record,indent=2)+'\n')
    config.hook.pytest_deselected(items=items[:n])
    items[:]=items[n:]

def pytest_runtest_logreport(report):
    if report.when == 'call' or report.failed:
        with Path(os.environ['V05_RESUME_RECORD']+'.jsonl').open('a') as out:
            out.write(json.dumps({'nodeid':report.nodeid,'when':report.when,'outcome':report.outcome})+'\n')
