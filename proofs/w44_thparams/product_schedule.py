"""Use the product history/sync schedule in root probes, without writing files.

A non-None output callback is essential: the scheduler otherwise allows a
fused cascade even when a child history alarm lies inside its three substeps.
This helper changes proof harnesses only; it does not implement model physics.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def observe_output(name, step, state):
    """Preserve scheduling semantics; root probes do not claim output IO cost."""
    return None


def product_schedule(input_dir, names, dt_by_domain, *, hours=3):
    """Return ``(runtime_kwargs, JSON_receipt)`` from real product helpers.

    ``dt_by_domain`` must be the map returned by the probe's real _load_domains.
    Keep the full forecast horizon when replaying individual root steps so the
    scheduler sees the same future history alarms as the product driver.
    """
    from gpuwrf.io.gen2_accessor import Gen2Run
    from gpuwrf.integration.nested_pipeline import (
        _nested_event_aware_fusion_k_from_env,
        _nested_sync_mode_from_env,
        _output_alarm_steps_by_domain,
        _output_cadence_steps_by_domain,
    )

    names = tuple(names)
    run = Gen2Run(Path(input_dir))
    cadence, intervals = _output_cadence_steps_by_domain(run, names, dt_by_domain)
    totals = {}
    for name in names:
        raw = float(hours) * 3600.0 / float(dt_by_domain[name])
        total = int(round(raw))
        if total <= 0 or abs(raw - total) > 1.e-9:
            raise ValueError(f"hours={hours} does not align with {name} dt")
        totals[name] = total
    all_alarms, overrides = _output_alarm_steps_by_domain(
        names, dt_by_domain, intervals, totals)
    block, sync = _nested_sync_mode_from_env()
    fusion_k = _nested_event_aware_fusion_k_from_env()
    kwargs = dict(output=observe_output, output_cadence_steps=cadence,
                  output_alarm_steps=overrides, block_between=block,
                  root_sync_cadence=sync, event_aware_fusion_k=fusion_k)
    receipt = dict(
        input_dir=str(run.path),
        namelist_sha256=hashlib.sha256((run.path / "namelist.input").read_bytes()).hexdigest(),
        hours=hours, dt_s=dt_by_domain, total_steps=totals,
        history_interval_minutes=intervals, output_cadence_steps=cadence,
        all_output_alarm_steps=all_alarms, output_alarm_steps=overrides,
        block_between=block, root_sync_cadence=sync,
        event_aware_fusion_k=fusion_k,
        output_observer="non-None no-op; scheduling only, no IO cost claim")
    return kwargs, receipt
