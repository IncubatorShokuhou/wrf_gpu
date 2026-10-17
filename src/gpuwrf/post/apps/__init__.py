"""Offline wind/solar/agriculture/fire/aviation application products.

Numerical submodules use only NumPy. xarray and the post registry are imported
only for dataset/CLI use; pvlib is an optional dependency for solar geometry.
"""
from __future__ import annotations

PRODUCT_NAMES = ("app_wind", "app_solar", "app_icing", "app_fire", "app_fwi", "app_frost", "app_visibility")


def register_products():
    """Register with the u1-post core; safe to call more than once."""
    from ..registry import PRODUCTS, register
    from . import dataset as d

    thermo = ("T", "P", "PB", "QVAPOR")
    height = ("PH", "PHB", "HGT")
    wind = (*height, "U", "V")
    specs = {
        "app_wind": (d.wind_product, (*wind, "U10", "V10"), thermo),
        "app_solar": (d.solar_product, ("SWDOWN", "XLAT", "XLONG", "HGT", "T2"), ("SWDDNI", "SWDDIF")),
        "app_icing": (d.icing_product, (*thermo, "QCLOUD", "QRAIN"), ()),
        "app_fire": (d.fire_product, (*wind, *thermo), ()),
        "app_fwi": (d.fwi_product, ("T2", "PSFC", "Q2", "U10", "V10", "XLAT", "RAINNC", "RAINC"),
                    ("RAINSH", "I_RAINNC", "I_RAINC")),
        "app_frost": (d.frost_product, ("T2", "TSK"), ()),
        "app_visibility": (d.visibility_product, (*thermo, "QCLOUD", "QRAIN", "QICE", "QSNOW"), ()),
    }
    for name, (runner, required, optional) in specs.items():
        if name not in PRODUCTS:
            register(name, runner, required, optional)
