"""Strict, host-only JSON input for opt-in passive tracers (no JAX imports)."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import struct


# Output uses the species name verbatim. Reserve existing WRF output/state names
# as well as the entire moisture (Q*) and built-in tracer (TR17_*) namespaces.
_RESERVED_NAMES = frozenset("""
U V W T P PB PH PHB MU MUB EXNER RHO RHO_D THETA THM TK
U10 V10 T2 TH2 PSFC HGT HGT_M XLAT XLONG XLAT_U XLONG_U XLAT_V XLONG_V
XTIME TIMES ITIMESTEP Z ZNU ZNW DN DNW RDN RDNW FNM FNP CFN CFN1
P_TOP P_HYD HYDRO RAINC RAINNC RAINSH RAINNCV SNOWNC GRAUPELNC HAILNC
SNOW SNOWH SNOWC ACSNOW SST TSK TSLB SMOIS SH2O CANWAT SEAICE
LANDMASK XLAND LU_INDEX IVGTYP ISLTYP ISWATER ISLAKE ISICE VEGFRA LAI
ALBEDO EMISS GLW SWDOWN HFX LH GRDFLX UST PBLH MOL RMOL ZNT
MAPFAC_M MAPFAC_U MAPFAC_V MAPFAC_MX MAPFAC_MY MAPFAC_UX MAPFAC_UY
MAPFAC_VX MAPFAC_VY COSALPHA SINALPHA F E SINALPHA COSALPHA
AKHS AKMS XKH XKM TKE TKE_PBL CLDFRA MAX_MSTFX MAX_MSTFY MAX_MSTFZ
MASS_TRACERS NC NR NI NS NG NH NN TRACER TRACERS NUM_TRACERS
C1F C1H C2F C2H C3F C3H C4F C4H CANICE CANLIQ CF1 CF2 CF3 CLAT
CLDFRA_BL COSZEN DTAUX3D DTAUY3D DZS ISNOW OLR RDX RDY REFL_10CM
SFROFF SNEQVO SNICE SNLIQ SR SWNORM TSNO UDROFF ZS ZSNSO
LWDNB LWDNBC LWDNT LWDNTC LWUPB LWUPBC LWUPT LWUPTC
SWDNB SWDNBC SWDNT SWDNTC SWUPB SWUPBC SWUPT SWUPTC
SWDDIR SWDDNI SWDDIF GSW PREC_ACC_C PREC_ACC_NC SNOW_ACC_NC
WSPD10MAX W_UP_MAX W_DN_MAX ISEEDARRAY_SPP_CONV ISEEDARRAY_SPP_LSM
ISEEDARRAY_SPP_PBL ISEEDARR_SKEBS ISEEDARR_SPPT
""".split())
_NAME_RE = re.compile(r"[A-Z][A-Z0-9_]*\Z")


@dataclass(frozen=True)
class TracerSpecies:
    """A kg/kg-dry-air scalar and optional analytic loss coefficients."""

    name: str
    decay_rate_s: float = 0.0
    deposition_velocity_m_s: float = 0.0
    wet_scavenging_per_mm: float = 0.0


@dataclass(frozen=True)
class Emission:
    """Lowest-layer source on a half-open mass-grid rectangle.

    A point has a 1x1 rectangle and ``rate_kg_s`` is its total mass rate.
    An area uses ``flux_kg_m2_s`` independently in each covered grid cell.
    """

    species_index: int
    start_s: float
    end_s: float
    x0: int
    x1: int
    y0: int
    y1: int
    rate_kg_s: float = 0.0
    flux_kg_m2_s: float = 0.0
    is_point: bool = False


@dataclass(frozen=True)
class UserTracerConfig:
    species: tuple[TracerSpecies, ...]
    emissions: tuple[Emission, ...]
    source_sha256: str


def _object(value: object, context: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return value


def _keys(value: dict, allowed: set[str], required: set[str], context: str) -> None:
    unknown = value.keys() - allowed
    missing = required - value.keys()
    if unknown:
        raise ValueError(f"{context}: unknown keys {sorted(unknown)}")
    if missing:
        raise ValueError(f"{context}: missing keys {sorted(missing)}")


def _number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a finite nonnegative number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{context} must be finite") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{context} must be a finite nonnegative number")
    # Host float64 finiteness alone does not protect the REAL32 model operands:
    # an overflowing emission rate can produce Inf * 0 outside its time window.
    # Validate the conversion without rounding or clamping the stored input.
    try:
        real32 = struct.unpack("!f", struct.pack("!f", number))[0]
    except OverflowError as exc:
        raise ValueError(f"{context} must remain finite in REAL32") from exc
    if not math.isfinite(real32):
        raise ValueError(f"{context} must remain finite in REAL32")
    return number


def _index(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer")
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_config(path: str | Path, *, shape: tuple[int, int, int]) -> UserTracerConfig:
    """Read version-1 JSON; reject invalid inputs rather than clamp indices.

    ``shape`` is the local domain mass-grid shape ``(nz, ny, nx)``. The SHA256
    identifies the exact input bytes, including whitespace, for run manifests.
    No model arrays are allocated by this function.
    """
    if len(shape) != 3 or any(
        isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in shape
    ):
        raise ValueError("shape must contain three positive integers (nz, ny, nx)")
    _, ny, nx = shape
    raw = Path(path).read_bytes()
    try:
        root = _object(json.loads(raw, object_pairs_hook=_unique_object), "config")
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"invalid tracer JSON in {path}: {exc}") from exc
    _keys(root, {"version", "species", "emissions"}, {"version", "species"}, "config")
    if type(root["version"]) is not int or root["version"] != 1:
        raise ValueError("tracer config version must be integer 1")
    if not isinstance(root["species"], list) or not root["species"]:
        raise ValueError("species must be a nonempty list")
    species = []
    indices = {}
    coefficients = {"decay_rate_s", "deposition_velocity_m_s", "wet_scavenging_per_mm"}
    for i, entry in enumerate(root["species"]):
        context = f"species[{i}]"
        entry = _object(entry, context)
        _keys(entry, coefficients | {"name"}, {"name"}, context)
        name = entry["name"]
        if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
            raise ValueError(f"{context}.name must match [A-Z][A-Z0-9_]*")
        if name in _RESERVED_NAMES or name.startswith(("Q", "TR17_")):
            raise ValueError(f"{context}.name {name!r} is reserved by WRF")
        if name in indices:
            raise ValueError(f"duplicate tracer species {name!r}")
        indices[name] = i
        species.append(TracerSpecies(name, **{
            key: _number(entry.get(key, 0), f"{context}.{key}") for key in coefficients
        }))

    entries = root.get("emissions", [])
    if not isinstance(entries, list):
        raise ValueError("emissions must be a list")
    emissions = []
    common = {"species", "start_s", "end_s"}
    for i, entry in enumerate(entries):
        context = f"emissions[{i}]"
        entry = _object(entry, context)
        is_point = "x" in entry or "y" in entry or "rate_kg_s" in entry
        geometry = {"x", "y", "rate_kg_s"} if is_point else {
            "x0", "x1", "y0", "y1", "flux_kg_m2_s"
        }
        _keys(entry, common | geometry, common | geometry, context)
        name = entry["species"]
        if not isinstance(name, str) or name not in indices:
            raise ValueError(f"{context}: unknown species {name!r}")
        start = _number(entry["start_s"], f"{context}.start_s")
        end = _number(entry["end_s"], f"{context}.end_s")
        if end <= start:
            raise ValueError(f"{context}: require end_s > start_s >= 0")
        if is_point:
            x0 = _index(entry["x"], f"{context}.x")
            y0 = _index(entry["y"], f"{context}.y")
            x1, y1 = x0 + 1, y0 + 1
            rate = _number(entry["rate_kg_s"], f"{context}.rate_kg_s")
            flux = 0.0
        else:
            x0, x1, y0, y1 = (
                _index(entry[key], f"{context}.{key}") for key in ("x0", "x1", "y0", "y1")
            )
            rate = 0.0
            flux = _number(entry["flux_kg_m2_s"], f"{context}.flux_kg_m2_s")
        if not (0 <= x0 < x1 <= nx and 0 <= y0 < y1 <= ny):
            raise ValueError(f"{context}: emission extent lies outside mass grid ({ny}, {nx})")
        emissions.append(Emission(indices[name], start, end, x0, x1, y0, y1, rate, flux, is_point))
    return UserTracerConfig(tuple(species), tuple(emissions), sha256(raw).hexdigest())
