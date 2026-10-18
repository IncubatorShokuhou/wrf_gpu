"""Track P: Noah-MP surface parameters as traced per-case controls (opt-in, default off).

Design: ``.agent/decisions/DESIGN-SURFACE-PARAM-ESTIMATION.md`` (+ Astra critique, staged plan K1).
A control is a per-cell VALUE field that replaces one gathered Noah-MP table parameter (MPTABLE Z0MVT,
EG, VCMX25, MP, ALBSAT/ALBDRY, RHOL/RHOS, LAIM/SAIM) or the wrfinput FVGMAX field (SHDMAX) at the
place where the land model reads it. The fields ride the TRACED per-domain clock base, so changing their
values never recompiles (same cheap key / AOT executable), and an absent control set leaves the released
program unchanged. ``controls`` holds the catalogue, host expansion, validation and file format;
``eki`` holds the derivative-free ensemble Kalman inversion scaffold (host fp64, no JAX).
"""

from gpuwrf.surface_params.controls import (  # noqa: F401
    CONTROLS,
    apply_surface_controls,
    controls_digest,
    expand_controls,
    load_controls,
    prior_controls,
    save_controls,
    validate_controls,
)
