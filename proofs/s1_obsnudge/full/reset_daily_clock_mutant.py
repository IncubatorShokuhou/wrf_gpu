"""Regression sensitivity: restore the production daily clock reset bug.

Run one fast daily host-loop test with pytest -p and this directory on
PYTHONPATH. This changes the actual stepper's call boundary, never assertions.
"""


def pytest_configure(config):
    from gpuwrf.integration.daily_pipeline import _RucDailyStepper

    original = _RucDailyStepper.__call__

    def reset(self, state, namelist, hours):
        self.next_step = 1
        return original(self, state, namelist, hours)

    _RucDailyStepper.__call__ = reset
