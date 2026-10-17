"""Manifest schema shared by fixture generation and metadata-only upgrades."""
from __future__ import annotations


def variables():
    units = {
        "lakedepth2d": "m", "savedtke12d": "W m-1 K-1", "snowdp2d": "m",
        "h2osno2d": "mm H2O", "snl2d": "negative layer count", "t_grnd2d": "K",
        "t_lake3d": "K", "lake_icefrac3d": "1", "z_lake3d": "m", "dz_lake3d": "m",
        "t_soisno3d": "K", "h2osoi_ice3d": "kg m-2", "h2osoi_liq3d": "kg m-2",
        "h2osoi_vol3d": "m3 m-3", "z3d": "m", "dz3d": "m", "zi3d": "m",
        "watsat3d": "m3 m-3", "csol3d": "J m-3 K-1", "tkmg3d": "W m-1 K-1",
        "tkdry3d": "W m-1 K-1", "tksatu3d": "W m-1 K-1",
    }
    sizes = (1,)*6 + (10,)*4 + (15,)*6 + (16,) + (10,)*5
    result = {name: dict(name=name, units=unit, shape=[size], dtype="float32",
                         absolute_tolerance=0.0, relative_tolerance=2e-7,
                         staggering="column; layer axis last",
                         timestep_relative_tolerance=2e-6, timestep_absolute_tolerance=1e-10)
              for (name, unit), size in zip(units.items(), sizes)}
    result["meta"] = dict(name="meta", units="veg,soil,xland,xice,lakemask,depthflag,lakepoint",
                          shape=[7], dtype="float32", absolute_tolerance=0.0, relative_tolerance=0.0,
                          staggering="column")
    result["flux"] = dict(name="flux", units="W m-2,W m-2,W m-2,K,kg m-2 s-1,K,K,kg kg-1,1",
                          members=["HFX","LH","GRDFLX","TSK","QFX","T2","TH2","Q2","ALBEDO"],
                          shape=[9], dtype="float32", absolute_tolerance=1e-6, relative_tolerance=2e-6,
                          staggering="column", qualification="CPU-qualified full Lake REAL32 saved interface")
    result["qsat"] = dict(name="qsat", units="K,Pa,Pa,Pa K-1,kg kg-1,K-1", shape=[8,6],
                          dtype="float64", absolute_tolerance=0.0, relative_tolerance=2e-14,
                          staggering="column", qualification="oracle only; candidate discarded")
    return result
