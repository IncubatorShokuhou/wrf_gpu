"""Opt-in ensemble data assimilation (LETKF) for wrf_gpu — lane a1-da probe.

Offline analysis step between forecasts. Never imported by the forecast
program (runtime/integration/dynamics/physics/kernels). See
``proofs/a1_da/DESIGN.md``. Modules: ``core`` (LETKF algebra), ``grid``
(analysis grid, localisation tables), ``obs`` (observation ingest/QC),
``operators`` (H(x)), ``state_io`` (member frames, rebalance, wrfinput),
``analysis`` / ``cli`` (orchestration).
"""
