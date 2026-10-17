"""Backend-dark GPU refusal for unqualified 3-D Smagorinsky."""
from collections.abc import Mapping

from gpuwrf.backends.platform import target_backend

REFUSAL = (
    "km_opt=3 (3-D Smagorinsky) is CPU-ONLY: its only GPU 24 h attempt became "
    "non-finite at 1 h; no completed CPU-WRF reference is available. "
    "See proofs/v040/gpuqual/km3/D6.json. Use JAX_PLATFORMS=cpu or default diff_opt=1/km_opt=4."
)


def selection_errors(options, *, backend=None):
    source = options.get("dynamics", options) if isinstance(options, Mapping) else options
    value = source.get("km_opt", 4) if isinstance(source, Mapping) else getattr(source, "km_opt", 4)
    values = value if isinstance(value, (list, tuple)) else (value,)
    if (backend or target_backend()) != "cpu" and any(v == 3 for v in values):
        return [REFUSAL]
    return []
