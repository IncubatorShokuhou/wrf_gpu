"""Deletion control: remove the existing layout helper's barrier at test time."""
import pytest


@pytest.fixture(autouse=True)
def disable_firewall(monkeypatch):
    from gpuwrf.kernels import layout_pin
    monkeypatch.setattr(layout_pin, "small_grid_firewall", lambda value, shape: value)
