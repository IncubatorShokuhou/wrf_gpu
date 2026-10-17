"""Backend capabilities, with a backend-dark bootstrap query.

CUDA policy must be decided before JAX initializes (allocator/XLA flags),
whereas kernel dispatch uses the actual PJRT client, never the generic ``gpu``
alias which also describes AMD devices.
"""
from __future__ import annotations

from functools import lru_cache
import os
from typing import Mapping


def target_backend(environ: Mapping[str, str] | None = None) -> str:
    """Best-effort bootstrap target; never imports JAX or invokes a vendor tool."""
    env = os.environ if environ is None else environ
    pin = (env.get("JAX_PLATFORMS", "") or env.get("JAX_PLATFORM_NAME", ""))
    first = pin.strip().lower().split(",")[0].strip()
    if first and first != "gpu":
        return first
    # An explicit AMD environment/device takes precedence over an installed
    # NVIDIA utility. Both vendors can coexist on a workstation.
    if "ROCR_VISIBLE_DEVICES" in env or "HIP_VISIBLE_DEVICES" in env or os.path.exists("/dev/kfd"):
        return "rocm"
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return "cpu" if visible.strip() in {"", "-1"} else "cuda"
    if any(os.path.exists(p) for p in ("/dev/nvidia0", "/dev/nvidiactl")):
        return "cuda"
    return "unknown" if first == "gpu" else "cpu"


@lru_cache(maxsize=1)
def runtime_backend() -> str:
    """Actual default device capability; called only when device work begins."""
    import jax

    device = jax.devices()[0]
    return device_backend(device)


def device_backend(device) -> str:
    """Classify an already selected device, including non-default devices."""
    if device.platform != "gpu":
        return device.platform
    version = str(getattr(getattr(device, "client", None), "platform_version", "")).lower()
    kind = str(getattr(device, "device_kind", "")).lower()
    if "cuda" in version or "nvidia" in kind:
        return "cuda"
    if "rocm" in version or "amd" in kind:
        return "rocm"
    return "gpu"  # Unknown accelerator: use portable arithmetic.


def cuda_native() -> bool:
    return runtime_backend() == "cuda"
