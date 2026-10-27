"""WRF array views for the shared input/checkpoint consistency checks.

Only representation conversion lives here; scientific checks are owned by
io.wrfinput_consistency. No files are copied and no arrays are sent to a device.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import numpy as np

STATE_TO_WRF = {"u": "U", "v": "V", "w": "W", "qv": "QVAPOR", "qc": "QCLOUD",
                "qr": "QRAIN", "qi": "QICE", "qs": "QSNOW", "qg": "QGRAUP",
                "Ni": "QNICE", "Nr": "QNRAIN", "Ns": "QNSNOW", "Ng": "QNGRAUPEL",
                "qh": "QHAIL", "Nc": "QNCLOUD", "Nn": "QNDROP", "Nh": "QNHAIL",
                "qvolg": "QVGRAUPEL", "qvolh": "QVHAIL", "nwfa": "QNWFA", "nifa": "QNIFA",
                "qke": "QKE", "p_perturbation": "P", "ph_perturbation": "PH", "t_skin": "TSK",
                "mu_perturbation": "MU"}
LAND_TO_WRF = {"smois": "SMOIS", "tslb": "TSLB", "sh2o": "SH2O", "tsno": "TSNO",
               "snice": "SNICE", "snliq": "SNLIQ", "snowh": "SNOWH", "sneqv": "SNOW",
               "canliq": "CANLIQ", "canice": "CANICE", "tv": "TV", "tg": "TG", "tah": "TAH",
               "lai": "LAI", "sai": "SAI", "t_skin": "TSK"}


def _finite_payload(payload):
    import jax
    count = 0
    for value in jax.tree.leaves(payload):
        if isinstance(value, np.ndarray) and value.dtype.kind in "fc":
            if not np.all(np.isfinite(value)):
                raise ValueError("checkpoint branch contains NaN/Inf, including retained carry")
            count += 1
    return count


def checkpoint_validator(input_dir, bundles):
    def validate(name, payload, edits, waivers, original):
        count = _finite_payload(payload)
        controls = getattr(getattr(bundles.get(name), "namelist", None), "surface_controls", None)
        controlled_cap = None if controls is None else controls.get("smcmax")
        cap_report = None
        if controlled_cap is not None:
            from gpuwrf.surface_params.lsmparams import validate_retained_porosity
            cap_report = validate_retained_porosity(bundles[name].namelist, payload.get("noahmp_land_fields"))
        # An untouched model state can be supersaturated. Empty branches do not
        # reinterpret it as a new balanced real.exe analysis or repair it.
        if not any(edits.values()):
            if waivers:
                raise ValueError("consistency waivers require an actual domain override")
            return {"passed": True, "finite_arrays": count, "scope": "unchanged checkpoint bytes",
                    **({"controlled_soil_porosity": cap_report} if cap_report is not None else {})}
        from gpuwrf.io.wrfinput_consistency import validate_fields
        from gpuwrf.io.netcdf_lock import Dataset
        from gpuwrf.io.wrfrst_netcdf import _dry_theta

        state, land = payload["state_fields"], payload.get("noahmp_land_fields")
        fields, variable_dimensions = {}, {}
        with Dataset(Path(input_dir) / f"wrfinput_{name}") as ds:
            ds.set_auto_maskandscale(False)
            dimensions = {key: len(value) for key, value in ds.dimensions.items()}
            attributes = {key: ds.getncattr(key) for key in ds.ncattrs()}
            for key in ("Times", "HGT", "LANDMASK", "XLAND", "LU_INDEX", "IVGTYP", "ISLTYP",
                        "P_TOP", "C3F", "C4F", "C3H", "C4H"):
                if key in ds.variables:
                    fields[key] = np.asarray(ds[key][:])
                    variable_dimensions[key] = ds[key].dimensions
        xyz = ("Time", "bottom_top", "south_north", "west_east")
        xy = ("Time", "south_north", "west_east")
        for leaf, key in STATE_TO_WRF.items():
            if state.get(leaf) is not None:
                fields[key] = np.asarray(state[leaf])[None]
                dims = xy if fields[key].ndim == 3 else xyz
                if key == "U":
                    dims = xyz[:-1] + ("west_east_stag",)
                elif key == "V":
                    dims = xyz[:2] + ("south_north_stag", "west_east")
                elif key in {"W", "PH"}:
                    dims = ("Time", "bottom_top_stag", "south_north", "west_east")
                variable_dimensions[key] = dims
        fields["T"] = (_dry_theta(SimpleNamespace(**state)) - 300)[None]
        variable_dimensions["T"] = xyz
        base = payload["extra_fields"].get("base_state")
        if base is None:
            raise ValueError("checkpoint overrides require the native persistent base state (E78)")
        for leaf, key in (("pb", "PB"), ("phb", "PHB"), ("mub", "MUB")):
            fields[key] = np.asarray(getattr(base, leaf))[None]
            variable_dimensions[key] = xyz if key == "PB" else (xy if key == "MUB" else ("Time", "bottom_top_stag", "south_north", "west_east"))
        if land is not None:
            for leaf, key in LAND_TO_WRF.items():
                if land.get(leaf) is not None:
                    fields[key] = np.asarray(land[leaf])[None]
                    dims = xy
                    if fields[key].ndim == 4:
                        layer = "soil_layers_stag" if key in {"SMOIS", "TSLB", "SH2O"} else "snow_layers"
                        dimensions[layer] = fields[key].shape[1]
                        dims = ("Time", layer, "south_north", "west_east")
                    variable_dimensions[key] = dims
        if "t_skin" in edits.get("state", {}):
            if "t_skin" in edits.get("land", {}) and not np.array_equal(state["t_skin"], land["t_skin"]):
                raise ValueError("conflicting state/land t_skin overrides; use one target or equal values")
            fields["TSK"] = np.asarray(state["t_skin"])[None]
        changed = {}
        for group, key, mapping in (("state", "state_fields", STATE_TO_WRF), ("land", "noahmp_land_fields", LAND_TO_WRF)):
            for leaf in edits.get(group, {}):
                target = mapping.get(leaf, "T")
                cells = np.asarray(payload[key][leaf]) != np.asarray(original[key][leaf])
                changed[target] = cells[None] | changed.get(target, False)
        # A qv edit also changes dry theta at fixed theta_m. The checker sees
        # the actual dry representation used for its temperature/RH dependency.
        if "qv" in edits.get("state", {}) or "theta" in edits.get("state", {}):
            before = _dry_theta(SimpleNamespace(**original["state_fields"]))
            changed["T"] = (fields["T"][0] != before - 300)[None]
        return validate_fields(fields, {"dimensions": dimensions, "variable_dimensions": variable_dimensions,
                                       "attributes": attributes},
                               changed_fields=changed, waivers=waivers,
                               surface_physics=int(bundles[name].namelist.sf_surface_physics), checkpoint=True,
                               **({"effective_smcmax": controlled_cap} if controlled_cap is not None else {}))
    return validate
