"""Release compiled test artifacts between shared-consumer cases (no assertion edits)."""
import gc
import json
import os
from pathlib import Path


def pytest_runtest_logreport(report):
    with Path(os.environ["MP11_TEST_EVENTS"]).open("a") as output:
        output.write(json.dumps(dict(node=report.nodeid, when=report.when,
                                     outcome=report.outcome, duration=report.duration)) + "\n")


def pytest_runtest_teardown(item, nextitem):
    import jax
    jax.clear_caches()
    gc.collect()
