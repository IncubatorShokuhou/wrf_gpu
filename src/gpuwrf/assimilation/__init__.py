"""Device-resident data-assimilation hooks.

Exports resolve lazily (PEP 562) so that importing a host-only submodule such
as ``gpuwrf.assimilation.letkf`` does not import JAX or the dynamics core
through ``data_assimilation`` (a1-da critique M11). The public names and their
objects are unchanged.
"""

_EXPORTS = (
    "DataAssimilationConfig",
    "DigitalFilterConfig",
    "NudgingComponent",
    "NudgingRates",
    "add_dry_physics_tendencies",
    "apply_nudging_rates",
    "data_assimilation_dry_tendencies",
    "data_assimilation_rates",
    "dfi_filter_coefficients",
    "digital_filter_initialize",
)

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        from . import data_assimilation

        value = getattr(data_assimilation, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
