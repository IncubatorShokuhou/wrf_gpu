"""WRF lake-model entrypoints: REAL32 carry and explicit DOUBLE internals.

WRF reference: ``phys/module_sf_lake.F`` dispatches ``Lake``/``LakeMain`` and
initializes with ``lakeini``. The model carries lake depth/category state, snow
layers, lake-water temperatures, soil/snow ice/liquid water, lake ice fraction,
Monin-Obukhov flux state, hydrology and tridiagonal thermal solves.

The initializer and full Lake/LakeMain timestep are CPU-oracle-qualified. CLI
selection requires an ingested lake bundle with Noah-MP/Thompson/MYNN. GPU and
coupled-forecast qualification remain pending; see ``docs/lake-model.md``.
"""

from __future__ import annotations

from typing import NamedTuple

LAKE_SOURCE = "<USER_HOME>/src/wrf_pristine/WRF/phys/module_sf_lake.F"
REGISTRY_SOURCE = "<USER_HOME>/src/wrf_pristine/WRF/Registry/Registry.EM_COMMON"

LAKE_CARRY_MEMBERS = (
    "lakedepth2d",
    "savedtke12d",
    "snowdp2d",
    "h2osno2d",
    "snl2d",
    "t_grnd2d",
    "t_lake3d",
    "lake_icefrac3d",
    "z_lake3d",
    "dz_lake3d",
    "t_soisno3d",
    "h2osoi_ice3d",
    "h2osoi_liq3d",
    "h2osoi_vol3d",
    "z3d",
    "dz3d",
    "zi3d",
    "watsat3d",
    "csol3d",
    "tkmg3d",
    "tkdry3d",
    "tksatu3d",
    "lakemask",
)


class LakeState(NamedTuple):
    """Minimal future-port handle for the lake carry/static payload."""

    carry_members: tuple[str, ...]
    static_tables: object = None
    carry: object = None


def lake_step(*args, **kwargs):
    """Full REAL32 WRF Lake/LakeMain column timestep (lazy component import)."""
    from gpuwrf.physics.lake_step import lake_step as advance
    return advance(*args,**kwargs)


def lake_initial_state(*args, **kwargs):
    """Standalone REAL32 WRF ``lakeini`` component; see its typed inputs/config.

    Import lazily so the released operational path and its trace are unchanged.
    """
    from gpuwrf.physics.lake_initialization import lake_initial_state as initialize

    return initialize(*args, **kwargs)


__all__ = [
    "LAKE_SOURCE",
    "REGISTRY_SOURCE",
    "LAKE_CARRY_MEMBERS",
    "LakeState",
    "lake_step",
    "lake_initial_state",
]
