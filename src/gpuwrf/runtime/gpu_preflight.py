"""CPU-side GPU run preflight guards.

The nested forecast path is a long, expensive JAX/XLA job. These checks run
before any forecast compile/device work so an unsafe local GPU environment fails
closed with an actionable message instead of OOMing mid-integration.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable, Mapping, Sequence, TextIO

LOCK_FILE = Path("/tmp/wrf_gpu2_gpu.lock")
HOLDER_FILE = Path(f"{LOCK_FILE}.holder")
DEFAULT_MIN_FREE_VRAM_GIB = 24.0
DEFAULT_MIN_FREE_VRAM_FRACTION = 0.50
REQUIRE_GPU_LOCK_ENV = "GPUWRF_REQUIRE_GPU_LOCK"

_FALSEY = {"0", "false", "no", "off", ""}
_TRUTHY = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class VramSnapshot:
    """One ``nvidia-smi`` memory row for the selected GPU."""

    index: str
    name: str
    free_mib: int
    total_mib: int
    used_mib: int
    uuid: str = ""

    @property
    def free_gib(self) -> float:
        return self.free_mib / 1024.0

    @property
    def total_gib(self) -> float:
        return self.total_mib / 1024.0

    @property
    def used_gib(self) -> float:
        return self.used_mib / 1024.0


@dataclass(frozen=True)
class VramThreshold:
    """Resolved free-VRAM gate for one selected GPU."""

    min_free_gib: float
    absolute_floor_gib: float
    fraction: float
    fractional_gib: float | None
    source: str


class GpuPreflightError(RuntimeError):
    """Raised when a nested GPU run should not start."""

    def __init__(self, message: str, payload: Mapping[str, object]) -> None:
        super().__init__(message)
        self.payload = dict(payload)


def env_bool(name: str, default: bool, environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    raw = env.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _FALSEY:
        return False
    if value in _TRUTHY:
        return True
    return default


def _explicit_min_free_vram_gib(environ: Mapping[str, str] | None = None) -> float | None:
    env = os.environ if environ is None else environ
    raw = env.get("GPUWRF_MIN_FREE_VRAM_GIB")
    if raw is None or not raw.strip():
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def min_free_vram_fraction(environ: Mapping[str, str] | None = None) -> float:
    """Return the fractional card-memory floor for nested-run preflight."""

    env = os.environ if environ is None else environ
    raw = env.get("GPUWRF_MIN_FREE_VRAM_FRACTION")
    if raw is None or not raw.strip():
        return DEFAULT_MIN_FREE_VRAM_FRACTION
    try:
        return min(1.0, max(0.0, float(raw)))
    except ValueError:
        return DEFAULT_MIN_FREE_VRAM_FRACTION


def _c_auto_threshold(env: Mapping[str, str]) -> VramThreshold | None:
    """Case-sized need when the CLI applied a C-auto plan (budget admitted pre-re-exec).

    cuda_async preallocates the pool at backend initialization, which precedes
    this check on the CLI path; a reserved pool then needs only the outside room.
    """
    try:
        budget = int(env["_GPUWRF_C_AUTO_BUDGET"])
        headroom = int(env["_GPUWRF_C_AUTO_HEADROOM"])
    except (KeyError, ValueError):
        return None
    bridge = sys.modules.get("jax._src.xla_bridge")
    reserved = any(name != "cpu" for name in (getattr(bridge, "_backends", None) or {}))
    need = (headroom + (0 if reserved else budget)) / 1024.0**3
    source = "C-auto plan: outside-pool headroom" + ("" if reserved else " + pool budget")
    return VramThreshold(min_free_gib=need, absolute_floor_gib=need, fraction=0.0,
                         fractional_gib=None, source=source)


def resolve_min_free_vram_threshold(
    environ: Mapping[str, str] | None = None,
    *,
    total_gib: float | None = None,
) -> VramThreshold:
    """Resolve the fail-closed free-VRAM threshold for the selected GPU."""

    env = os.environ if environ is None else environ
    explicit = _explicit_min_free_vram_gib(env)
    if explicit is not None:
        return VramThreshold(
            min_free_gib=explicit,
            absolute_floor_gib=explicit,
            fraction=0.0,
            fractional_gib=None,
            source="GPUWRF_MIN_FREE_VRAM_GIB explicit override",
        )
    planned = _c_auto_threshold(env)
    if planned is not None:
        return planned
    fraction = min_free_vram_fraction(env)
    fractional_gib = None if total_gib is None else max(0.0, fraction * max(0.0, total_gib))
    min_free = DEFAULT_MIN_FREE_VRAM_GIB
    if fractional_gib is not None:
        min_free = max(DEFAULT_MIN_FREE_VRAM_GIB, fractional_gib)
    return VramThreshold(
        min_free_gib=min_free,
        absolute_floor_gib=DEFAULT_MIN_FREE_VRAM_GIB,
        fraction=fraction,
        fractional_gib=fractional_gib,
        source="max(default absolute floor, GPUWRF_MIN_FREE_VRAM_FRACTION * total VRAM)",
    )


def min_free_vram_gib(
    environ: Mapping[str, str] | None = None,
    *,
    total_gib: float | None = None,
) -> float:
    """Return the resolved fail-closed nested-run free-VRAM threshold."""

    return resolve_min_free_vram_threshold(environ, total_gib=total_gib).min_free_gib


def force_gpu_run_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Whether the operator explicitly bypassed lock/headroom failures."""

    return env_bool("GPUWRF_FORCE_GPU_RUN", False, environ)


def gpu_lock_required(
    environ: Mapping[str, str] | None = None,
    *,
    lock_file: Path = LOCK_FILE,
) -> tuple[bool, str]:
    """Whether the nested preflight demands the ``scripts/with_gpu_lock.sh`` holder proof.

    Only where the shared dev lock infrastructure exists (``lock_file``) or when
    ``GPUWRF_REQUIRE_GPU_LOCK=1``: a public machine without it runs the VRAM /
    C-auto headroom check only. ``GPUWRF_REQUIRE_GPU_LOCK=0`` cannot switch the
    proof off where the lock file exists (unlocked GPU use breaks locked runs).
    """
    env = os.environ if environ is None else environ
    if env_bool(REQUIRE_GPU_LOCK_ENV, False, env):
        return True, f"{REQUIRE_GPU_LOCK_ENV}=1"
    if Path(lock_file).exists():
        return True, f"dev GPU lock infrastructure present ({lock_file})"
    return False, f"no dev GPU lock infrastructure ({lock_file} absent), {REQUIRE_GPU_LOCK_ENV} unset"


def read_gpu_lock_holder(holder_file: Path = HOLDER_FILE) -> str:
    try:
        text = holder_file.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    return text


def _fd_points_to_lock(fd_text: str, lock_file_text: str) -> bool:
    try:
        fd = int(fd_text)
    except (TypeError, ValueError):
        return False
    if fd < 0:
        return False
    proc_fd = Path("/proc/self/fd") / str(fd)
    try:
        target = proc_fd.resolve(strict=True)
        lock_file = Path(lock_file_text).resolve(strict=False)
    except OSError:
        return False
    return target == lock_file


def gpu_lock_status(
    environ: Mapping[str, str] | None = None,
    *,
    holder_file: Path = HOLDER_FILE,
    lock_file: Path = LOCK_FILE,
) -> dict[str, object]:
    """Best-effort advisory check for a ``scripts/with_gpu_lock.sh`` holder.

    This is a cooperative same-box environment/holder-token check, not a hard
    proof that this process owns the kernel ``flock`` mutex.
    """

    env = os.environ if environ is None else environ
    holder = read_gpu_lock_holder(holder_file)
    fd_text = env.get("GPUWRF_GPU_LOCK_FD", "")
    lock_file_text = env.get("GPUWRF_GPU_LOCK_FILE", str(lock_file))
    token = env.get("GPUWRF_GPU_LOCK_TOKEN", "")
    held_flag = env.get("GPUWRF_GPU_LOCK_HELD", "") == "1"
    fd_ok = _fd_points_to_lock(fd_text, lock_file_text)
    token_ok = bool(token and token in holder)
    ok = bool(held_flag and fd_ok and token_ok)
    reason = ""
    if not held_flag:
        reason = "GPUWRF_GPU_LOCK_HELD is not set by scripts/with_gpu_lock.sh"
    elif not fd_ok:
        reason = f"GPU lock fd {fd_text!r} does not point to {lock_file_text}"
    elif not token_ok:
        reason = "GPU lock token is absent from the holder file"
    return {
        "ok": ok,
        "reason": reason,
        "holder": holder,
        "lock_file": str(lock_file),
        "holder_file": str(holder_file),
        "fd": fd_text,
        "token_present": bool(token),
    }


def _visible_device_tokens(environ: Mapping[str, str]) -> list[str] | None:
    raw = environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    return [token.strip() for token in raw.split(",") if token.strip()]


def _first_target_device_token(environ: Mapping[str, str]) -> str:
    tokens = _visible_device_tokens(environ)
    if tokens is None:
        return "0"
    if not tokens or tokens[0].lower() in {"-1", "none", "void", "nodevfiles"}:
        raise RuntimeError("CUDA_VISIBLE_DEVICES hides all CUDA GPUs")
    return tokens[0]


def _parse_nvidia_smi_memory_row(row: str) -> VramSnapshot:
    parts = [part.strip() for part in row.split(",")]
    if len(parts) < 6:
        raise RuntimeError(f"unexpected nvidia-smi memory row: {row!r}")
    try:
        free_mib = int(parts[2])
        total_mib = int(parts[3])
        used_mib = int(parts[4])
    except ValueError as exc:
        raise RuntimeError(f"could not parse nvidia-smi memory row: {row!r}") from exc
    return VramSnapshot(
        index=parts[0],
        name=", ".join(part for part in parts[5:] if part),
        free_mib=free_mib,
        total_mib=total_mib,
        used_mib=used_mib,
        uuid=parts[1],
    )


def _device_token_matches(snapshot: VramSnapshot, token: str) -> bool:
    if snapshot.index == token:
        return True
    if snapshot.uuid == token:
        return True
    return bool(snapshot.uuid and snapshot.uuid.startswith(token))


def select_nvidia_smi_snapshot(
    rows: Sequence[str],
    environ: Mapping[str, str] | None = None,
) -> VramSnapshot:
    """Select the ``nvidia-smi`` row for the GPU JAX will see as device 0."""

    env = os.environ if environ is None else environ
    snapshots = [_parse_nvidia_smi_memory_row(row) for row in rows if row.strip()]
    if not snapshots:
        raise RuntimeError("nvidia-smi returned no GPU rows")
    target = _first_target_device_token(env)
    for snapshot in snapshots:
        if _device_token_matches(snapshot, str(target)):
            return snapshot
    choices = ", ".join(
        f"index={snapshot.index} uuid={snapshot.uuid or '<unknown>'}" for snapshot in snapshots
    )
    raise RuntimeError(
        f"CUDA_VISIBLE_DEVICES selects GPU {target!r}, but nvidia-smi returned only: {choices}"
    )


def query_nvidia_smi_memory(environ: Mapping[str, str] | None = None) -> VramSnapshot:
    """Query free memory on the CUDA-visible GPU JAX will use first."""

    env = os.environ if environ is None else environ
    exe = shutil.which("nvidia-smi")
    if exe is None:
        raise RuntimeError("nvidia-smi not found on PATH")
    proc = subprocess.run(
        [
            exe,
            "--query-gpu=index,uuid,memory.free,memory.total,memory.used,name",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout).strip()
        raise RuntimeError(f"nvidia-smi failed with rc={proc.returncode}: {err}")
    rows = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return select_nvidia_smi_snapshot(rows, env)


def _snapshot_payload(snapshot: VramSnapshot | None, error: str | None) -> dict[str, object]:
    if snapshot is None:
        return {"ok": False, "error": error or "unknown"}
    return {
        "ok": True,
        "index": snapshot.index,
        "name": snapshot.name,
        "uuid": snapshot.uuid,
        "free_mib": snapshot.free_mib,
        "used_mib": snapshot.used_mib,
        "total_mib": snapshot.total_mib,
        "free_gib": round(snapshot.free_gib, 3),
        "used_gib": round(snapshot.used_gib, 3),
        "total_gib": round(snapshot.total_gib, 3),
    }


def _format_status(payload: Mapping[str, object]) -> str:
    lock = payload.get("lock") if isinstance(payload.get("lock"), Mapping) else {}
    vram = payload.get("vram") if isinstance(payload.get("vram"), Mapping) else {}
    threshold = payload.get("vram_threshold") if isinstance(payload.get("vram_threshold"), Mapping) else {}
    holder = lock.get("holder") if isinstance(lock, Mapping) else ""
    if isinstance(vram, Mapping) and vram.get("ok"):
        free = vram.get("free_gib")
        total = vram.get("total_gib")
        device = vram.get("name")
        index = vram.get("index")
        vram_text = f"free_vram={free} GiB / {total} GiB on {device} (index={index})"
    else:
        reason = vram.get("error") if isinstance(vram, Mapping) else "unknown"
        vram_text = f"free_vram=unavailable ({reason})"
    if isinstance(threshold, Mapping):
        fractional = threshold.get("fractional_gib", "n/a")
        threshold_text = (
            f"threshold={threshold.get('min_free_gib')} GiB "
            f"(source={threshold.get('source')}; floor={threshold.get('absolute_floor_gib')} GiB; "
            f"fraction={threshold.get('fraction')}; fractional={fractional} GiB)"
        )
    else:
        threshold_text = "threshold=<unknown>"
    return f"holder={holder or '<empty>'}; {vram_text}; {threshold_text}"


def _threshold_payload(threshold: VramThreshold) -> dict[str, object]:
    payload: dict[str, object] = {
        "min_free_gib": round(threshold.min_free_gib, 3),
        "absolute_floor_gib": round(threshold.absolute_floor_gib, 3),
        "fraction": round(threshold.fraction, 6),
        "source": threshold.source,
    }
    if threshold.fractional_gib is not None:
        payload["fractional_gib"] = round(threshold.fractional_gib, 3)
    return payload


def run_nested_gpu_preflight(
    *,
    force: bool = False,
    environ: Mapping[str, str] | None = None,
    query_memory: Callable[[], VramSnapshot] = query_nvidia_smi_memory,
    holder_file: Path = HOLDER_FILE,
    lock_file: Path = LOCK_FILE,
    stderr: TextIO | None = None,
) -> dict[str, object]:
    """Fail closed without free-VRAM headroom, and without the GPU-lock proof where it is required.

    The lock proof applies only where :func:`gpu_lock_required` says so (dev lock
    infrastructure present or ``GPUWRF_REQUIRE_GPU_LOCK=1``); a public machine runs
    the VRAM / C-auto headroom check only.
    """

    env = os.environ if environ is None else environ
    err = sys.stderr if stderr is None else stderr
    from gpuwrf.backends.platform import target_backend
    backend = target_backend(env)
    if backend != "cuda":
        return {"schema": "GpuwrfNestedGpuPreflight", "skipped": True,
                "backend": backend, "reason": "non-CUDA backend"}
    force = bool(force or force_gpu_run_enabled(env))
    lock_required, lock_policy = gpu_lock_required(env, lock_file=lock_file)
    lock = dict(gpu_lock_status(env, holder_file=holder_file, lock_file=lock_file),
                required=lock_required, policy=lock_policy)

    snapshot: VramSnapshot | None = None
    memory_error: str | None = None
    try:
        snapshot = query_memory()
    except Exception as exc:  # noqa: BLE001 - surfaced in fail-closed payload
        memory_error = f"{type(exc).__name__}: {exc}"
    threshold = resolve_min_free_vram_threshold(
        env,
        total_gib=snapshot.total_gib if snapshot is not None else None,
    )

    payload: dict[str, object] = {
        "schema": "GpuwrfNestedGpuPreflight",
        "schema_version": 1,
        "status": "PASS",
        "forced": bool(force),
        "min_free_vram_gib": float(threshold.min_free_gib),
        "min_free_vram_fraction": float(threshold.fraction),
        "vram_threshold": _threshold_payload(threshold),
        "lock": lock,
        "vram": _snapshot_payload(snapshot, memory_error),
        "action": (
            ("Run nested GPU forecasts through `scripts/with_gpu_lock.sh --label <name> -- <cmd>`; "
             if lock_required else "Free GPU memory used by other processes; ")
            + "set GPUWRF_MIN_FREE_VRAM_GIB for an explicit GiB threshold override or "
            "GPUWRF_MIN_FREE_VRAM_FRACTION to tune the card-relative threshold; set "
            "GPUWRF_FORCE_GPU_RUN=1 or pass --force-gpu-run only for a deliberate override."
        ),
    }

    lock_held = bool(lock.get("ok"))
    failures: list[str] = []
    advisories: list[str] = []
    if lock_required and not lock_held:
        failures.append(str(lock.get("reason") or "GPU lock is not held"))
    if snapshot is None:
        failures.append(f"could not query free VRAM before the run ({memory_error})")
    elif snapshot.free_gib < threshold.min_free_gib:
        vram_msg = (
            f"free VRAM {snapshot.free_gib:.2f} GiB is below "
            f"resolved threshold {threshold.min_free_gib:.2f} GiB ({threshold.source})"
        )
        # cuda_async (and any preallocating XLA allocator) reserves its device pool
        # during backend init -- BEFORE this nvidia-smi reading -- so the "used" VRAM
        # we observe is OUR OWN pool, not another process. When the GPU lock is
        # verifiably held, with_gpu_lock.sh exclusively owns the card and already
        # checked free VRAM *pre-launch* (before this process allocated its pool); a
        # low reading here is therefore a false-negative, not real contention.
        # Downgrade it to an advisory so cuda_async runs are not spuriously killed
        # rc=75. (If the pool is genuinely too large for the run, allocation fails
        # later with a real OOM -- surfaced honestly at that point, not pre-empted.)
        if lock_held:
            advisories.append(
                vram_msg
                + " -- advisory only: GPU lock held (exclusive); reading reflects our"
                " own allocator pool, not contention"
            )
        else:
            failures.append(vram_msg)

    if advisories:
        payload["advisories"] = advisories

    if failures and not force:
        payload["status"] = "FAIL"
        payload["failures"] = failures
        message = "nested GPU preflight failed: " + "; ".join(failures)
        print(f"gpuwrf: {message} ({_format_status(payload)})", file=err)
        raise GpuPreflightError(message, payload)

    if failures and force:
        payload["status"] = "FORCED"
        payload["failures"] = failures
        print(f"gpuwrf: WARNING: forcing nested GPU run despite preflight failures: {'; '.join(failures)}", file=err)
    else:
        if advisories:
            print(
                "gpuwrf: nested GPU preflight ADVISORY (lock held, not fatal): "
                + "; ".join(advisories),
                file=err,
            )
        print(f"gpuwrf: nested GPU preflight PASS: {_format_status(payload)}", file=err)
    return payload


# --- CUDA architecture gate ------------------------------------------------------------------
# The release fast paths are Pallas/Triton kernels; XLA compiles Triton only for compute
# capability >= 8.0 (measured on a GTX 1080 Ti, cc 6.1: ``FAILED_PRECONDITION: Triton support is
# only enabled for Ampere GPUs (compute capability 8.0) and up``), and the legacy defaults still
# contain Pallas kernels. Refuse such a GPU before any model work instead of failing deep inside
# an XLA compile. Pure Python on the run's control path: no traced program changes.
MIN_CUDA_COMPUTE_CAPABILITY: tuple[int, int] = (8, 0)
UNSUPPORTED_GPU_ENV = "GPUWRF_ALLOW_UNSUPPORTED_GPU"
UNSUPPORTED_GPU_EXIT_CODE = 69  # sysexits EX_UNAVAILABLE: the hardware cannot run this program
NVIDIA_SMI_TIMEOUT_S = 15.0  # architecture query budget; a timeout is recorded as "unknown", never refused


class UnsupportedGpuError(GpuPreflightError):
    """Raised when the selected CUDA GPU is older than :data:`MIN_CUDA_COMPUTE_CAPABILITY`."""


def parse_compute_capability(value: object) -> tuple[int, int] | None:
    """``"6.1"``/``"12.0"``/``(8, 6)`` -> ``(major, minor)``; anything unreadable -> ``None``."""

    if isinstance(value, (tuple, list)) and len(value) == 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None
    text = str(value or "").strip()
    major, _, minor = text.partition(".")
    if not major.isdigit() or (minor and not minor.isdigit()):
        return None
    return int(major), int(minor or 0)


def unsupported_gpu_opted_in(environ: Mapping[str, str] | None = None) -> bool:
    return env_bool(UNSUPPORTED_GPU_ENV, False, environ)


def unsupported_gpu_message(name: str, capability: tuple[int, int]) -> str:
    gpu = name.strip() or "the selected NVIDIA GPU"
    return (
        f"unsupported GPU: {gpu} has CUDA compute capability {capability[0]}.{capability[1]}; gpuwrf "
        "requires compute capability >= 8.0 (NVIDIA Ampere sm_80 or newer). Its fast paths are "
        "Pallas/Triton kernels and XLA compiles Triton only for sm_80+, so this run would fail later "
        "inside the XLA compile (FAILED_PRECONDITION). Alternatives: run on an sm_80+ NVIDIA GPU, or use "
        "the portable CPU path `JAX_PLATFORMS=cpu gpuwrf run ...` (much slower; see docs/portability.md). "
        f"To start on this GPU anyway (unsupported, slow, expected to fail where Triton is needed), set "
        f"{UNSUPPORTED_GPU_ENV}=1."
    )


def check_cuda_architecture(
    *,
    name: str,
    compute_capability: object,
    environ: Mapping[str, str] | None = None,
    stderr: TextIO | None = None,
    source: str = "",
) -> dict[str, object]:
    """Gate one CUDA device by compute capability; raise :class:`UnsupportedGpuError` below sm_80.

    An unreadable capability is reported as ``unknown`` and not refused (the compile would then
    give the authoritative answer). With ``GPUWRF_ALLOW_UNSUPPORTED_GPU=1`` an old GPU is admitted
    with a loud 'unsupported, slow' warning.
    """

    env = os.environ if environ is None else environ
    capability = parse_compute_capability(compute_capability)
    minimum = MIN_CUDA_COMPUTE_CAPABILITY
    payload: dict[str, object] = {
        "schema": "GpuwrfCudaArchitecture",
        "name": str(name),
        "compute_capability": (f"{capability[0]}.{capability[1]}" if capability else None),
        "minimum_compute_capability": f"{minimum[0]}.{minimum[1]}",
        "source": source,
        "opt_in_env": UNSUPPORTED_GPU_ENV,
    }
    if capability is None:
        payload["status"] = "unknown"
        return payload
    if capability >= minimum:
        payload["status"] = "supported"
        return payload
    if unsupported_gpu_opted_in(env):
        payload["status"] = "unsupported_opt_in"
        err = sys.stderr if stderr is None else stderr
        gpu = str(name).strip() or "the selected NVIDIA GPU"
        print(f"gpuwrf: WARNING: UNSUPPORTED, SLOW: {gpu} (compute capability {capability[0]}.{capability[1]} "
              f"< {minimum[0]}.{minimum[1]}) admitted only because {UNSUPPORTED_GPU_ENV}=1; no support, "
              "performance or fidelity claim, and the default Pallas/Triton kernels will still fail.", file=err)
        return payload
    payload["status"] = "unsupported"
    raise UnsupportedGpuError(unsupported_gpu_message(str(name), capability), payload)


def query_nvidia_smi_compute_capability(environ: Mapping[str, str] | None = None) -> tuple[str, str]:
    """``(name, compute_cap)`` of the CUDA-visible GPU JAX will use first (backend-dark)."""

    env = os.environ if environ is None else environ
    exe = shutil.which("nvidia-smi")
    if exe is None:
        raise RuntimeError("nvidia-smi not found on PATH")
    try:
        proc = subprocess.run(
            [exe, "--query-gpu=index,uuid,compute_cap,name", "--format=csv,noheader,nounits"],
            check=False, capture_output=True, text=True, timeout=NVIDIA_SMI_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        # A hung driver/nvidia-smi must not hang the run start: the caller records "unknown".
        raise RuntimeError(f"nvidia-smi timed out after {NVIDIA_SMI_TIMEOUT_S:g} s") from exc
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout).strip()
        raise RuntimeError(f"nvidia-smi failed with rc={proc.returncode}: {err}")
    rows = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return select_nvidia_smi_compute_capability(rows, env)


def select_nvidia_smi_compute_capability(
    rows: Sequence[str], environ: Mapping[str, str] | None = None
) -> tuple[str, str]:
    """Select ``index, uuid, compute_cap, name`` rows like :func:`select_nvidia_smi_snapshot`."""

    env = os.environ if environ is None else environ
    parsed = []
    for row in rows:
        parts = [part.strip() for part in row.split(",", 3)]
        if len(parts) == 4:
            parsed.append(parts)
    if not parsed:
        raise RuntimeError("nvidia-smi returned no GPU rows")
    target = str(_first_target_device_token(env))
    for index, uuid, capability, name in parsed:
        if index == target or uuid == target or bool(uuid and uuid.startswith(target)):
            return name, capability
    raise RuntimeError(f"CUDA_VISIBLE_DEVICES selects GPU {target!r}, which nvidia-smi does not list")


def run_gpu_architecture_preflight(
    *,
    environ: Mapping[str, str] | None = None,
    query: Callable[[Mapping[str, str]], tuple[str, str]] | None = None,
    stderr: TextIO | None = None,
) -> dict[str, object]:
    """CLI gate before any model work: refuse a CUDA GPU below sm_80 (see :func:`check_cuda_architecture`).

    Backend-dark (``nvidia-smi``, no JAX init), skipped for non-CUDA targets. A failed query is
    recorded as ``unknown``; the device-level check in ``contracts.state`` still applies.
    """

    env = os.environ if environ is None else environ
    from gpuwrf.backends.platform import target_backend
    backend = target_backend(env)
    if backend != "cuda":
        return {"schema": "GpuwrfCudaArchitecture", "status": "skipped", "backend": backend}
    try:
        name, capability = (query or query_nvidia_smi_compute_capability)(env)
    except Exception as exc:  # noqa: BLE001 - recorded, not fatal (no capability evidence)
        return {"schema": "GpuwrfCudaArchitecture", "status": "unknown", "source": "nvidia-smi",
                "error": f"{type(exc).__name__}: {exc}"}
    return check_cuda_architecture(name=name, compute_capability=capability, environ=env,
                                   stderr=stderr, source="nvidia-smi")


def check_jax_device_architecture(device, environ: Mapping[str, str] | None = None,
                                  stderr: TextIO | None = None) -> dict[str, object]:
    """Device-level gate for programmatic runs (state allocation): CUDA devices only."""

    from gpuwrf.backends.platform import device_backend
    if device_backend(device) != "cuda":
        return {"schema": "GpuwrfCudaArchitecture", "status": "skipped", "platform": str(device.platform)}
    return check_cuda_architecture(name=str(getattr(device, "device_kind", "")),
                                   compute_capability=getattr(device, "compute_capability", None),
                                   environ=environ, stderr=stderr, source="jax-device")


__all__ = [
    "DEFAULT_MIN_FREE_VRAM_GIB",
    "DEFAULT_MIN_FREE_VRAM_FRACTION",
    "GpuPreflightError",
    "MIN_CUDA_COMPUTE_CAPABILITY",
    "NVIDIA_SMI_TIMEOUT_S",
    "UNSUPPORTED_GPU_ENV",
    "UNSUPPORTED_GPU_EXIT_CODE",
    "UnsupportedGpuError",
    "check_cuda_architecture",
    "check_jax_device_architecture",
    "parse_compute_capability",
    "query_nvidia_smi_compute_capability",
    "run_gpu_architecture_preflight",
    "select_nvidia_smi_compute_capability",
    "unsupported_gpu_message",
    "unsupported_gpu_opted_in",
    "VramSnapshot",
    "VramThreshold",
    "env_bool",
    "force_gpu_run_enabled",
    "gpu_lock_required",
    "gpu_lock_status",
    "min_free_vram_fraction",
    "min_free_vram_gib",
    "query_nvidia_smi_memory",
    "read_gpu_lock_holder",
    "resolve_min_free_vram_threshold",
    "run_nested_gpu_preflight",
    "select_nvidia_smi_snapshot",
]
