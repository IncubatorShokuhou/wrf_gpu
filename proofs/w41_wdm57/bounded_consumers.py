"""Bound executable retention between unchanged shared-consumer assertions."""
import gc
import json
import os
from pathlib import Path
import resource


def pytest_runtest_setup(item):
    if os.environ.get("WDM57_BASELINE_X64") == "1":
        import jax
        jax.config.update("jax_enable_x64", True)


def pytest_runtest_logreport(report):
    import jax
    with Path(os.environ["WDM57_TEST_EVENTS"]).open("a") as output:
        output.write(json.dumps(dict(node=report.nodeid, when=report.when,
                                     outcome=report.outcome, duration=report.duration,
                                     x64_enabled=bool(jax.config.jax_enable_x64),
                                     rss_hwm_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)) + "\n")


def pytest_runtest_teardown(item, nextitem):
    import jax
    jax.clear_caches()
    gc.collect()
