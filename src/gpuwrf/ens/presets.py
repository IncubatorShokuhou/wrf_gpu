"""Ensemble presets: WRF stochastic-physics members and a qualified multiphysics table.

Host-only (no JAX): member namelists are surgical text edits of the source case
namelist, re-parsed and checked so that only the intended groups change. The
model's own pre-JAX stochastic admission (:func:`stochastic_refusals`) refuses
invalid scheme/physics combinations before anything is written.

Stochastic members follow WRF's ensemble convention: identical ``iseed_*`` and
a distinct ``nens`` per member (README.namelist: "nens ... should be different
for each member"); WRF's ``rand_seed`` mixes ``iseed``, ``nens`` and the start
date into each stream's seed. Berner et al. (2011, Mon. Wea. Rev. 139,
1972-1995) compare such stochastic (SKEBS) ensembles with multiphysics ones.
"""
from __future__ import annotations

from datetime import datetime
from functools import lru_cache
import importlib.util
from numbers import Integral
from pathlib import Path
import re

from gpuwrf.io.stochastic_config import SEED_DEFAULTS, SEED_KEYS, SWITCHES, bind_stochastic, stochastic_refusals

PRESET_MODES = ("stochastic", "multiphysics")
SCHEMES = ("skebs", "sppt", "spp")
DEFAULT_SCHEMES = ("skebs", "sppt")
# Seed streams each scheme advances (stochastic_config.SEED_KEYS order).
_STREAMS = {"skebs": ("iseed_skebs",), "sppt": ("iseed_sppt",),
            "spp": ("iseed_spp_conv", "iseed_spp_pbl", "iseed_spp_lsm")}
_GAUSSIAN = {"sppt": ("sppt",), "spp": ("spp_conv", "spp_pbl", "spp_lsm")}
_EVIDENCE = "v0.4.0 final validation, Swiss d01 3 km 24 h, GPU vs original CPU-WRF frozen D6 PASS"
# Rows are complete &physics edits of one D6-PASS v0.4.0 composite (v-prep case id), all with
# Noah-MP (sf_surface_physics=4) and otherwise the default Swiss namelist. e41 = that composite's
# cubin-gate verdict (FAIL rows are D6-PASS but carry the disclosed opt-in E41 flags).
MULTIPHYSICS = (
    ("P00", dict(mp_physics=8, cu_physics=0, bl_pbl_physics=5, sf_sfclay_physics=5, ra_lw_physics=4, ra_sw_physics=4), "c00", "PASS"),
    ("P01", dict(mp_physics=3, cu_physics=3, bl_pbl_physics=3, sf_sfclay_physics=3, ra_lw_physics=4, ra_sw_physics=1), "c04", "PASS"),
    ("P02", dict(mp_physics=8, cu_physics=10, bl_pbl_physics=5, sf_sfclay_physics=5, ra_lw_physics=4, ra_sw_physics=4), "c30", "PASS"),
    ("P03", dict(mp_physics=8, cu_physics=11, bl_pbl_physics=5, sf_sfclay_physics=5, ra_lw_physics=4, ra_sw_physics=4), "c31", "PASS"),
    ("P04", dict(mp_physics=8, cu_physics=0, bl_pbl_physics=9, sf_sfclay_physics=5, ra_lw_physics=3, ra_sw_physics=3), "c28", "PASS"),
    ("P05", dict(mp_physics=24, cu_physics=93, bl_pbl_physics=5, sf_sfclay_physics=5, ra_lw_physics=4, ra_sw_physics=4,
                 use_mp_re=1), "c17", "PASS"),
    ("P06", dict(mp_physics=50, cu_physics=0, bl_pbl_physics=5, sf_sfclay_physics=5, ra_lw_physics=4, ra_sw_physics=4,
                 use_mp_re=1), "c21", "PASS"),
    ("P07", dict(mp_physics=1, cu_physics=1, bl_pbl_physics=1, sf_sfclay_physics=1, ra_lw_physics=1, ra_sw_physics=1), "c02", "FAIL"),
    ("P08", dict(mp_physics=4, cu_physics=5, bl_pbl_physics=7, sf_sfclay_physics=1, ra_lw_physics=4, ra_sw_physics=4), "c05", "FAIL"),
    ("P09", dict(mp_physics=6, cu_physics=6, bl_pbl_physics=8, sf_sfclay_physics=1, ra_lw_physics=4, ra_sw_physics=4), "c06", "FAIL"),
    ("P10", dict(mp_physics=2, cu_physics=2, bl_pbl_physics=2, sf_sfclay_physics=2, ra_lw_physics=4, ra_sw_physics=2), "c03", "FAIL"),
    ("P11", dict(mp_physics=9, cu_physics=16, bl_pbl_physics=11, sf_sfclay_physics=1, ra_lw_physics=4, ra_sw_physics=4,
                 use_mp_re=1), "c07", "FAIL"),
)


def wrf_seed_words(iseed, nens, start):
    """Host copy of physics.stochastic_rng.seed_array (WRF rand_seed, GNU eight words)."""
    if any(isinstance(v, bool) or not isinstance(v, Integral) or not -(2**31) <= v < 2**31 for v in (iseed, nens)):
        raise ValueError("iseed and nens must be WRF INTEGER(4)")
    wrap = lambda value: (int(value) + 2**31) % 2**32 - 2**31
    fctime = wrap(start.year * (start.month*100 + start.day) + start.hour)
    mix = fctime + wrap(iseed*nens*1000000)
    mod = lambda divisor: (abs(mix) % divisor) * (-1 if mix < 0 else 1)
    group = [wrap(iseed+nens*1000000), mod(19211), mod(71209), mod(11279)]
    return tuple(group*2)


def _first(value):
    return value[0] if isinstance(value, list) else value


def _format(value):
    if isinstance(value, bool):
        return ".true." if value else ".false."
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return ", ".join(_format(v) for v in value)
    return str(value)


def _set_keys(text, group, values):
    """Replace or append ``key = value`` lines inside one existing &group (other bytes kept)."""
    out, inside, seen, done = [], False, False, set()
    start = re.compile(rf"^\s*&{group}\s*$", re.I)
    for line in text.splitlines():
        code = line.split("!")[0]
        if re.match(rf"^\s*&{group}\b", line, re.I):
            if not start.match(line) or seen:
                raise ValueError(f"&{group}: put the group name on its own line, once")
            inside = seen = True
        elif inside and code.strip() == "/":
            out.extend(f" {k} = {_format(v)}," for k, v in values.items() if k not in done)
            done.update(values)
            inside = False
        elif inside and code.strip():
            match = re.match(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*=", code)
            if match is None or code.count("=") > 1 or "/" in code:
                raise ValueError(f"&{group}: unsupported continuation/multi-assignment line {line!r}")
            key = match.group(1).lower()
            if key in values:
                if key in done:
                    raise ValueError(f"&{group}: {key} assigned twice")
                out.append(f" {key} = {_format(values[key])},")
                done.add(key)
                continue
        out.append(line)
    if inside:
        raise ValueError(f"unterminated &{group}")
    if not seen:
        out += [f"&{group}", *(f" {k} = {_format(v)}," for k, v in values.items()), "/"]
    return "\n".join(out) + "\n"


@lru_cache(maxsize=1)
def _spectral_setup():
    """The model's host-only spectral setup, loaded by path so that offline ens never imports JAX."""
    path = Path(__file__).resolve().parents[1]/"physics"/"stochastic_setup.py"
    spec = importlib.util.spec_from_file_location("gpuwrf_ens_stochastic_setup", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_spectra(config):
    """Run the model's d01 initialize_spectrum for every active pattern (it refuses degenerate bands).

    Original WRF 4.7.1 does not refuse: for a Gaussian scale far above the domain size its REAL
    normalization underflows, the spectral coefficients become NaN and the clipped pattern is the
    constant +cutoff (v-prep c41, Swiss 126 km domain: PATTERN_SPP_PBL = 0.3, RAND_PERT = 0.09).
    """
    cfg = bind_stochastic(config, "d01")
    if cfg is None:
        return
    domains = config.get("domains", {})
    try:
        nx, ny, nz = (int(_first(domains[k])) for k in ("e_we", "e_sn", "e_vert"))
        dx, dy, dt = float(_first(domains["dx"])), float(_first(domains["dy"])), int(_first(domains["time_step"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("stochastic presets need &domains e_we/e_sn/e_vert/dx/dy/time_step") from exc
    for index, pattern in enumerate(cfg.patterns):
        if pattern is None or not cfg.produce[index]:
            continue
        try:
            _spectral_setup().initialize_spectrum(nx, ny, nz, dx=dx, dy=dy, dt=dt, **pattern.setup_kwargs())
        except ValueError as exc:
            name = pattern.variable if index > 1 else "SKEBS"
            raise ValueError(f"stochastic pattern {name} (lengthscale {pattern.lengthscale/1000:g} km) has no finite WRF "
                             f"spectral normalization on this {nx}x{ny} x {dx/1000:g} km grid ({exc}); original WRF "
                             "would run with a constant clipped pattern. Pass a smaller --lengthscale-km") from None


def _stochastic_values(config, schemes, nens, lengthscale_km, max_dom):
    """The &stoch keys a member sets; refuses unresolved Gaussian scales."""
    unknown = sorted(set(schemes) - set(SCHEMES))
    if unknown or not schemes or len(set(schemes)) != len(schemes):
        raise ValueError(f"schemes must be distinct entries of {','.join(SCHEMES)}, got {','.join(schemes) or 'none'}")
    base = {str(k).lower(): v for k, v in (config.get("stoch") or {}).items()}
    if any(k in base for k in (*SWITCHES, "nens")):
        raise ValueError("source namelist already sets &stoch switches/nens; the preset owns them (remove them first)")
    values = {scheme: [1]*max_dom for scheme in schemes}
    values["nens"] = nens
    if lengthscale_km is not None:
        for scheme in schemes:
            for group in _GAUSSIAN.get(scheme, ()):
                values["lengthscale_" + group] = float(lengthscale_km)*1000.
    return values


def _row(row_id):
    for row in MULTIPHYSICS:
        if row[0] == row_id:
            return row
    raise ValueError(f"unknown multiphysics row {row_id}; table rows {','.join(r[0] for r in MULTIPHYSICS)}")


def build_members(mode, base_text, config, *, members=None, schemes=None, lengthscale_km=None, nens_start=1,
                  rows=None, start=None):
    """Return ``[(namelist_text, info)]`` for a preset; raises ValueError before anything is written."""
    if mode not in PRESET_MODES:
        raise ValueError(f"preset mode must be one of {PRESET_MODES}")
    if lengthscale_km is not None and not (isinstance(lengthscale_km, (int, float)) and 0 < float(lengthscale_km) < 1e5):
        raise ValueError("--lengthscale-km must be positive and finite")
    max_dom = int(_first(config.get("domains", {}).get("max_dom", 1)))
    if mode == "stochastic":
        if rows:
            raise ValueError("--physics-rows belongs to --mode multiphysics")
        schemes = tuple(DEFAULT_SCHEMES if schemes is None else schemes)
        if isinstance(members, bool) or not isinstance(members, int) or members < 2:
            raise ValueError("stochastic preset needs --members >= 2")
        plan = [(None, None)] * members
    else:
        schemes = tuple(schemes or ())
        if rows:
            if len(set(rows)) != len(rows):
                raise ValueError("duplicate multiphysics row")
            plan = [_row(r) for r in rows]
            if members is not None and members != len(plan):
                raise ValueError("--members disagrees with --physics-rows")
        else:
            if isinstance(members, bool) or not isinstance(members, int) or members < 2:
                raise ValueError("multiphysics preset needs --members >= 2 (or --physics-rows)")
            if members > len(MULTIPHYSICS):
                raise ValueError(f"multiphysics table has {len(MULTIPHYSICS)} qualified rows; --members {members} refused")
            plan = list(MULTIPHYSICS[:members])
        if len(plan) < 2:
            raise ValueError("multiphysics preset needs at least two rows")
        plan = [(r, r[1]) for r in plan]
    if isinstance(nens_start, bool) or not isinstance(nens_start, int) or nens_start < 1 \
            or nens_start + len(plan) - 1 > 2146:
        raise ValueError("member nens must stay in 1..2146 (WRF iseed+nens*1000000 is INTEGER(4))")
    base_stoch = {str(k).lower(): v for k, v in (config.get("stoch") or {}).items()}
    seeds = {k: int(_first(base_stoch.get(k, d))) for k, d in zip(SEED_KEYS, SEED_DEFAULTS)}
    sf_surface = _first(config.get("physics", {}).get("sf_surface_physics"))
    out, used = [], {}
    for index, (row, physics) in enumerate(plan):
        text, info = base_text, {}
        if physics is not None:
            text = _set_keys(text, "physics", {k: [v]*max_dom for k, v in physics.items()})
            info["physics_row"] = row[0]
            info["physics"] = dict(physics)
            info["evidence"] = f"{_EVIDENCE} (v-prep {row[2]}); E41 {row[3]}"
            if sf_surface != 4:
                info["qualification_note"] = (f"row qualified with Noah-MP sf_surface_physics=4; this case uses "
                                              f"{sf_surface}: UNQUALIFIED combination")
        if schemes:
            nens = nens_start + index
            stoch = _stochastic_values(config, schemes, nens, lengthscale_km, max_dom)
            text = _set_keys(text, "stoch", stoch)
            info["stoch"] = stoch
            info["nens"] = nens
            words = {}
            for scheme in schemes:
                for key in _STREAMS[scheme]:
                    word = wrf_seed_words(seeds[key], nens, start)
                    if word in used:
                        raise ValueError(f"member {index} {key} seed equals {used[word]}: streams would be identical")
                    used[word] = f"member {index} {key}"
                    words[key] = list(word)
            info["seed_words"] = words
        out.append((text, info))
    return out


def check_member_config(mode, base, member, schemes):
    """Re-parsed member namelist may differ from the source only in the preset's groups."""
    allowed = {"stoch"} if mode == "stochastic" else {"physics", "stoch"} if schemes else {"physics"}
    groups = set(base) | set(member)
    changed = {g for g in groups if base.get(g) != member.get(g)}
    if not changed <= allowed:
        raise ValueError(f"preset changed unexpected namelist groups {sorted(changed - allowed)}")
    if base.get("physics", {}).keys() - member.get("physics", {}).keys():
        raise ValueError("preset dropped a &physics key")
    errors = stochastic_refusals(member)
    if errors:
        raise ValueError("stochastic admission refused: " + "; ".join(errors))
    check_spectra(member)
    if "spp" in schemes and _first(member.get("physics", {}).get("bl_pbl_physics")) != 5:
        # WRF 4.7.1 SPP perturbs only MYNN; its cumulus/LSM consumers are upstream no-ops
        # (docs/stochastic.md), so such members would carry no perturbation at all.
        raise ValueError("spp perturbs only MYNN (bl_pbl_physics=5) in WRF 4.7.1; members would be unperturbed")
    if schemes:
        cfg = bind_stochastic(member, "d01")
        if cfg is None:
            raise ValueError("preset member has no active stochastic pattern")


def parse_start(config):
    clock = config.get("time_control", {})
    try:
        return datetime(*(int(_first(clock[k])) for k in ("start_year", "start_month", "start_day", "start_hour")))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("namelist start_year/month/day/hour required for stochastic seeds") from exc
