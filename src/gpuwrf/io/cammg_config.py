"""Backend-dark admission for CPU-only scheme fallbacks.

mp_physics=11 (CAM-MG) was refused on GPU in the v0.4.0 fallback because XLA's
bundled Triton miscompiled its native column (zero RAINNCV/SNOWNCV). The column
now compiles with stock Triton (gpuwrf.kernels._triton_kernel_call, lane
t-cammgws), so GPU_REFUSED_MP_PHYSICS is empty; the pre-JAX/API hooks remain.
"""
from collections.abc import Mapping
import os
from pathlib import Path
import re

from gpuwrf.backends.platform import target_backend

GPU_REFUSED_MP_PHYSICS = frozenset()  # was {11} in the CPU-only fallback (t-mp11gate)

REFUSAL = (
    "mp_physics=11 (CAM-MG) is CPU-ONLY: the native GPU kernel returns zero "
    "RAINNCV/SNOWNCV. See KNOWN_ISSUES.md#cam-mg-gpu-precipitation; fix planned "
    "for v0.4.1. Set JAX_PLATFORMS=cpu or select another microphysics scheme."
)


def selection_errors(options, *, backend=None):
    source = options.get("physics", options) if isinstance(options, Mapping) else options
    value = source.get("mp_physics", 8) if isinstance(source, Mapping) else getattr(source, "mp_physics", 8)
    values = value if isinstance(value, (list, tuple)) else (value,)
    if (backend or target_backend()) != "cpu" and any(v in GPU_REFUSED_MP_PHYSICS for v in values):
        return [REFUSAL]
    return []


def validate_selection(options):
    errors = _gpu_selection_errors(options)
    if errors:
        raise ValueError("; ".join(errors))


def _gpu_selection_errors(options):
    from gpuwrf.io.km3_config import selection_errors as km3_errors
    return selection_errors(options) + km3_errors(options, backend=target_backend())


def validate_input_dir(input_dir):
    """Admit every domain before API device initialization; CPU is a no-op."""
    if target_backend() == "cpu":
        return
    from gpuwrf.io.namelist_parser import parse_namelist
    validate_selection(parse_namelist(Path(input_dir) / "namelist.input"))


def bootstrap_refusal(argv):
    """Follow the FLG/GFDL pre-JAX CLI admission mechanism."""
    if len(argv) < 2 or argv[1] != "run" or target_backend() == "cpu":
        return None

    # Abbreviations are intentionally unsupported here: refuse before JAX
    # rather than guess argparse's resolution against the full option set.
    arguments = argv[2:]
    if "--" in arguments:
        arguments = arguments[:arguments.index("--")]
    for arg in arguments:
        spelling = arg.split("=", 1)[0]
        if spelling.startswith("--") and any(
            name.startswith(spelling) and name != spelling
            for name in ("--namelist", "--input-dir")
        ):
            return "GPU admission requires full --namelist / --input-dir spellings; abbreviations are refused before JAX."

    def option(name):
        # argparse store actions take the LAST occurrence, including = forms.
        for index in range(len(arguments) - 1, -1, -1):
            arg = arguments[index]
            if arg.startswith(name + "="):
                return arg.split("=", 1)[1]
            if arg == name and index + 1 < len(arguments):
                return arguments[index + 1]
        return None

    namelist = option("--namelist")
    input_directory = option("--input-dir")
    # --namelist validates an override, but loaders still read the case's own
    # namelist. Admit both, plus distinct ensemble inputs, before JAX imports.
    paths = [Path(namelist)] if namelist is not None else []
    if input_directory is not None:
        paths.append(Path(input_directory) / "namelist.input")
    try:
        batch = int(os.environ.get("GPUWRF_BATCH_ENSEMBLE", "1"))
    except ValueError:
        batch = 1  # Ordinary configuration validation handles malformed values.
    if batch > 1:
        raw = os.environ.get("GPUWRF_BATCH_INPUT_DIRS", "")
        paths.extend(Path(p.strip()) / "namelist.input"
                     for p in re.split(r"[,\n" + re.escape(os.pathsep) + "]", raw) if p.strip())
    from gpuwrf.io.namelist_parser import parse_namelist
    for path in dict.fromkeys(paths):
        if not path.is_file():
            continue
        try:
            config = parse_namelist(path)
        except (ValueError, OSError):
            continue  # Ordinary CLI parsing handles malformed input.
        errors = _gpu_selection_errors(config)
        if errors:
            return errors[0]
    return None
