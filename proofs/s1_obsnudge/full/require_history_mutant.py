"""Reintroduce the reviewed unconditional-history output route in-process."""


def pytest_configure(config):
    from gpuwrf.integration import nested_pipeline

    def require_history(state, namelist, run_start, *, history_diagnostics=None, **kwargs):
        return nested_pipeline._history_surface_diagnostics_for_output(
            state, namelist, run_start, history=history_diagnostics, **kwargs)

    nested_pipeline._noahmp_surface_diagnostics_for_output = require_history
