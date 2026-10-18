"""Variable localisation for the LETKF — OPT-IN (lane w41-da-ens; R3 item 3, Kang et al. 2011 JGR 116 D09110).

R3 (4)/BA: in the winter case 0228 the two surface stations' co-located 10-m winds update T2 through N = 10 sampling
noise in the wind–temperature cross covariance (correlation noise ≈ 1/√9 ≈ 0.33 [I]), amplified by the global ρ:
T2 OmB 0.444 → OmA 0.520 with all obs, 0.442 with T2 alone. Horizontal localisation cannot separate co-located
variables. Kang et al. zero the physically weak cross-variable covariances; in the LETKF this is a separate local
transform per STATE-VARIABLE GROUP, with each observation's localised inverse error variance multiplied by a coupling
factor c(obs variable, state group) ∈ [0, 1] (c = 0: the obs does not update that group; c = 1: unchanged).

Tables (obs var codes of ``letkf.obs``: 1 T, 2 QV, 3 U, 4 V sonde; 11 T2, 12 Q2, 13 U10, 14 V10 surface):

* ``"surface_split"`` — sonde obs update every group; surface T2/Q2 update only the thermodynamic group, surface
  U10/V10 only the wind group (the R3 design item; keeps the well-sampled upper-air multivariate coupling).
* ``"strict"`` — every obs updates only its own group (thermo: T/QV/T2/Q2, wind: U/V/U10/V10).

Groups: ``wind`` = U, V, U10, V10; ``thermo`` = T, QVAPOR, T2, Q2, PSFC (any other analysed field → ``thermo``).
The per-group transforms are computed by ``analysis.analyse(var_loc=...)`` (da-core: one transform per distinct
factor vector); :func:`as_var_loc` converts a table into that dict / the config ``localisation.variables`` section.
Cost: one transform set per group (2× the weight computation and memory of the plain LETKF).
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

GROUPS = {"wind": ("U", "V", "U10", "V10"), "thermo": ("T", "QVAPOR", "T2", "Q2", "PSFC")}
_SONDE = (1, 2, 3, 4)
TABLES = {
    "surface_split": {**{v: {"wind": 1.0, "thermo": 1.0} for v in _SONDE},
                      11: {"wind": 0.0, "thermo": 1.0}, 12: {"wind": 0.0, "thermo": 1.0},
                      13: {"wind": 1.0, "thermo": 0.0}, 14: {"wind": 1.0, "thermo": 0.0}},
    "strict": {1: {"wind": 0.0, "thermo": 1.0}, 2: {"wind": 0.0, "thermo": 1.0},
               3: {"wind": 1.0, "thermo": 0.0}, 4: {"wind": 1.0, "thermo": 0.0},
               11: {"wind": 0.0, "thermo": 1.0}, 12: {"wind": 0.0, "thermo": 1.0},
               13: {"wind": 1.0, "thermo": 0.0}, 14: {"wind": 1.0, "thermo": 0.0}},
}


def resolve_table(table) -> dict:
    if isinstance(table, str):
        if table not in TABLES:
            raise ValueError(f"unknown variable-localisation table {table!r}; known {sorted(TABLES)}")
        return TABLES[table]
    return {int(k): {str(g): float(c) for g, c in v.items()} for k, v in dict(table).items()}


def group_of(name: str) -> str:
    for group, names in GROUPS.items():
        if name in names:
            return group
    return "thermo"


def coupling(obs_var, table, group: str) -> np.ndarray:
    """(n_obs,) factor c(obs var, group); obs vars missing from the table couple fully (1)."""
    t = resolve_table(table)
    return np.array([float(t.get(int(v), {}).get(group, 1.0)) for v in np.asarray(obs_var)], np.float64)


_KIND = {1: "sonde", 2: "sonde", 3: "sonde", 4: "sonde", 11: "surface", 12: "surface", 13: "surface", 14: "surface"}
_VAR = {1: "T", 2: "QV", 3: "U", 4: "V", 11: "T2", 12: "Q2", 13: "U10", 14: "V10"}


def as_var_loc(table, names) -> dict:
    """The table as ``analysis.analyse(var_loc=...)`` / config ``localisation.variables``:
    {state: {'kind:VAR': factor}} for every analysed state name in ``names`` (factors 1 omitted)."""
    t = resolve_table(table)
    out = {}
    for name in names:
        g = group_of(name)
        row = {f"{_KIND[code]}:{_VAR[code]}": float(f[g]) for code, f in sorted(t.items()) if float(f[g]) != 1.0}
        if row:
            out[name] = row
    return out


def describe(table) -> Mapping:
    t = resolve_table(table)
    return {"groups": {g: list(v) for g, v in GROUPS.items()}, "coupling": {str(k): v for k, v in sorted(t.items())}}
