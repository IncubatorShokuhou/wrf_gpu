"""Offline station verification; NumPy/pandas only, no model execution.

See docs/station_verification.md for units, support and sampling definitions.
"""

__all__ = ["verify", "read_observations", "read_stations", "score_sources"]


def __getattr__(name):
    # CLI parser/help must not import NumPy/pandas/netCDF4 merely to register.
    from importlib import import_module

    modules = {"verify": "engine", "read_observations": "observations",
               "read_stations": "observations", "score_sources": "scores"}
    if name not in modules:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{modules[name]}"), name)
    globals()[name] = value
    return value
