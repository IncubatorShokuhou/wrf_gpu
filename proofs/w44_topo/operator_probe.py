"""Proof adapter for the validated WRF shading reference; not a forecast entry."""
from gpuwrf.physics.topographic_shading import topographic_shadow, toposhad_solar_geometry

__all__ = ["topographic_shadow", "toposhad_solar_geometry"]
