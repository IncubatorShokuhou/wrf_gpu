"""Coupling exports loaded on demand, keeping configuration admission backend-dark."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "BoundaryConfig": (".boundary_apply", "BoundaryConfig"),
    "apply_lateral_boundaries": (".boundary_apply", "apply_lateral_boundaries"),
    "build_initial_state": (".driver", "build_initial_state"),
    "run_forecast_segment": (".driver", "run_forecast_segment"),
    "run_to_output_leads": (".driver", "run_to_output_leads"),
    **{name: (".physics_couplers", name) for name in (
        "mynn_adapter", "rrtmg_adapter", "surface_adapter", "thompson_adapter",
    )},
}

__all__ = [
    "BoundaryConfig",
    "apply_lateral_boundaries",
    "build_initial_state",
    "mynn_adapter",
    "rrtmg_adapter",
    "run_forecast_segment",
    "run_to_output_leads",
    "surface_adapter",
    "thompson_adapter",
]


def __getattr__(name):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = target
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
