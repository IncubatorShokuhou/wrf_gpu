"""Run-length / time-step alignment shared by the nested pipeline and the CLI pre-flight.

JAX-free on purpose: ``gpuwrf run --dry-run`` (and the pre-flight of a real run) must refuse a forecast length whose
seconds are not an exact multiple of every domain's time step BEFORE any model initialisation -- the nested pipeline
used to refuse it only after init (FINDINGS E187/E225).
"""

from __future__ import annotations

from typing import Any


def domain_list_value(value: Any, index: int, default: Any) -> Any:
    """``value[index]`` of a WRF list-valued namelist entry (last entry repeats), scalar passthrough."""

    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        value = value[index] if index < len(value) else value[-1]
    return default if value is None else value


def dt_by_domain(run, names: tuple[str, ...]) -> dict[str, float]:
    """Per-domain model timestep from the namelist (root time_step / ratio chain).

    WRF sets the child timestep from ``parent_time_step / parent_grid_ratio``; the
    Canary nests use a fixed integer ratio so the result is exact.
    """

    nml = run.namelist
    root_dt = domain_list_value(nml.get("domains", {}).get("time_step"), 0, None)
    if root_dt is None:
        root_dt = domain_list_value(nml.get("time_control", {}).get("time_step"), 0, None)
    if root_dt is None:
        raise ValueError("namelist has no domains/time_control time_step for the root domain")
    dt: dict[str, float] = {names[0]: float(root_dt)}
    for name in names[1:]:
        grid = run.grid(name)
        parent = f"d{int(grid.parent_id):02d}"
        ratio = int(grid.parent_grid_ratio)
        if ratio <= 1:
            raise ValueError(f"{name}: parent_grid_ratio must be > 1 for a child, got {ratio}")
        if parent not in dt:
            raise ValueError(f"{name}: parent {parent} not loaded before child (bad domain order)")
        dt[name] = dt[parent] / float(ratio)
    return dt


def steps_by_domain(hours, dt_s: dict[str, float], names: tuple[str, ...]) -> dict[str, int]:
    """Exact step count of every domain for ``hours``; ValueError when any domain does not align.

    The root is checked first with the historical ``root dt`` wording, then every child.
    """

    steps: dict[str, int] = {}
    for index, name in enumerate(names):
        raw = float(hours) * 3600.0 / float(dt_s[name])
        rounded = int(round(raw))
        if abs(raw - rounded) > 1.0e-9:
            label = "root" if index == 0 else name
            raise ValueError(
                f"hours={hours} does not align with {label} dt={dt_s[name]}s (would need {raw} steps)"
            )
        steps[name] = rounded
    return steps


def preflight_dt_by_domain(run, names: tuple[str, ...]) -> dict[str, float] | None:
    """Per-domain dt for the CLI pre-flight, or ``None`` when the root ``time_step`` is unavailable.

    The root is resolved first; the child ratio chain is added only when every child grid is readable and
    consistent, otherwise the root alone is checked (the pipeline reports a bad chain itself after init).
    """

    errors = (OSError, KeyError, AttributeError, ValueError)
    try:
        root = dt_by_domain(run, names[:1])
    except errors:
        return None
    if len(names) == 1:
        return root
    try:
        return dt_by_domain(run, names)
    except errors:
        return root
