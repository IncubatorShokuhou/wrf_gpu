"""Exact WRF Registry inventory for the V044 MYNN budget program (no JAX)."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex

FIELDS = {
    "qSHEAR": ("tke_budget", "Existing turbulence qshear, then WRF surface boundary + vertical destagger.", "module_bl_mynnedmf.F:2877,1141-1157"),
    "qBUOY": ("tke_budget", "Existing qbuoy plus half plume TKE production, then WRF surface boundary + destagger.", "module_bl_mynnedmf.F:2885-2887,1142-1157"),
    "qDISS": ("tke_budget", "Existing _mym_predict_qke qdiss; set unused top to zero.", "module_bl_mynnedmf.F:3170,1156"),
    "qWT": ("tke_budget", "Existing _mym_predict_qke qwt for default edmf_tke=0; set unused top to zero.", "module_bl_mynnedmf.F:3153-3169,1155"),
    "Dtke": ("tke_budget", "New cheap post-solve difference: 0.5*(qke_new-qke_old)/dt; unused top zero.", "module_bl_mynnedmf.F:756-757,1147,1157"),
    "edmf_a": ("bl_mynn_output", "Existing plume output; retain without a new plume solve.", "module_bl_mynnedmf.F:6472,6489"),
    "edmf_w": ("bl_mynn_output", "Existing plume output under release flags; retain without a new plume solve.", "module_bl_mynnedmf.F:6473,6484"),
    "edmf_qt": ("bl_mynn_output", "Existing plume output; retain without a new plume solve.", "module_bl_mynnedmf.F:6474,6485"),
    "edmf_thl": ("bl_mynn_output", "Existing plume output; retain without a new plume solve.", "module_bl_mynnedmf.F:6475,6486"),
    "edmf_ent": ("bl_mynn_output", "New per-plume ENT export and area-weighted sum in native plume kernel (ENT currently transient).", "module_bl_mynnedmf.F:6476,6487"),
    "edmf_qc": ("bl_mynn_output", "Existing plume output; retain without a new plume solve.", "module_bl_mynnedmf.F:6477,6488"),
    "sub_thl3D": ("bl_mynn_output", "Exact zero: pristine compile-time env_subs=.false.; never synthesize subsidence.", "module_bl_mynnedmf.F:343,815,6517"),
    "sub_sqv3D": ("bl_mynn_output", "Exact zero: pristine compile-time env_subs=.false.; never synthesize subsidence.", "module_bl_mynnedmf.F:343,816,6517"),
    "det_thl3D": ("bl_mynn_output", "Exact zero: pristine compile-time env_subs=.false.; never synthesize detrainment.", "module_bl_mynnedmf.F:343,819,6517"),
    "det_sqv3D": ("bl_mynn_output", "Exact zero: pristine compile-time env_subs=.false.; never synthesize detrainment.", "module_bl_mynnedmf.F:343,820,6517"),
    "RTHBLTEN": ("gpuwrf budget program", "Existing raw dry-theta PBL source from mynn_adapter_with_source_leaves.", "module_pbl_driver.F:1720-1875"),
    "RQVBLTEN": ("gpuwrf budget program", "Existing raw mixing-ratio PBL source after WRF post_run conversion.", "module_pbl_driver.F:1720-1875"),
    "RQCBLTEN": ("gpuwrf budget program", "Existing raw cloud-water PBL source (cloudmix=1).", "module_pbl_driver.F:1720-1875"),
    "RQIBLTEN": ("gpuwrf budget program", "Existing raw cloud-ice PBL source (cloudmix=1).", "module_pbl_driver.F:1720-1875"),
    "RUBLTEN": ("gpuwrf budget program", "Existing raw mass-point PBL source before A-to-C coupling; GWDO may subsequently add to WRF RUBLTEN.", "module_pbl_driver.F:1720-1875"),
    "RVBLTEN": ("gpuwrf budget program", "Existing raw mass-point PBL source before A-to-C coupling; GWDO may subsequently add to WRF RVBLTEN.", "module_pbl_driver.F:1720-1875"),
}


def inventory(wrf):
    wrf = Path(wrf).resolve()
    registry = wrf / "Registry/Registry.EM_COMMON"
    found = {}
    for line_no, line in enumerate(registry.read_text().splitlines(), 1):
        if not line.startswith("state"):
            continue
        # Unrelated Registry declarations contain continued/unquoted syntax
        # that is not POSIX shell text. Tokenize only the requested rows.
        raw = line.split()
        if len(raw) < 3 or raw[2].lower() not in ({n.lower() for n in FIELDS} | {"dqke"}):
            continue
        parts = shlex.split(line)
        if len(parts) > 10 and parts[8] in FIELDS:
            name = parts[8]
            switch, cost, source = FIELDS[name]
            found[name] = dict(registry_name=parts[2], dimensions=parts[3],
                               staggering=parts[6], description=parts[9], units=parts[10],
                               registry_line=line_no, switch=switch, computation=cost, source=source)
    if set(found) != set(FIELDS):
        raise RuntimeError(f"Registry inventory mismatch: {set(FIELDS) - set(found)}")
    source_paths = [registry, wrf / "phys/module_bl_mynnedmf.F",
                    wrf / "phys/module_bl_mynnedmf_driver.F", wrf / "phys/module_pbl_driver.F"]
    return dict(schema="wrf_gpu2.w44_budget.inventory.v1", wrf_root=str(wrf),
                sources={str(p.relative_to(wrf)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
                switches=dict(tke_budget="WRF Registry.EM_COMMON:2473/3189; bl_mynn_tkebudget does not exist",
                              bl_mynn_output="WRF Registry.EM_COMMON:2483/3191"),
                fields={name: found[name] for name in FIELDS},
                notes=["TKE Registry units are m2 s-2; original :1138 explicitly describes rates m2 s-3. Preserve Registry metadata.",
                       "qSHEAR/qBUOY/qDISS/qWT have Registry Z staggering, but the driver copies kts:kte after budget destaggering. Verify writer allocation against original CPU output.",
                       "Diagnostic values are not yet validated. Source inventory is not a numerical oracle.",
                       "Budget sum uses qSHEAR+qBUOY+qWT-qDISS. Compare closure residual to pristine, including clipping/surface/density conventions, rather than forcing closure."])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrf-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(inventory(args.wrf_root), indent=2) + "\n")
