"""WRF digital filter initialization (``&dfi_control``), lane o2-dfi.

``filter`` and ``control`` are pure Python/NumPy (safe for the pre-JAX CLI admission);
``driver`` runs the TDFI stages with the operational step (imports JAX).
"""
