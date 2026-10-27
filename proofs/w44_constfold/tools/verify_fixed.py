"""Post-fix check on the RELEASE program: for every fixed site, WRF's REAL bits are live in the flag-ON program and
the old Python-fold bits are gone from that file's literals; with the flag OFF the old bits are back (= main).

usage: python verify_fixed.py ORACLE_JSON OUT_JSON ON_PREFIX[,ON_PREFIX] OFF_PREFIX[,OFF_PREFIX]
"""
import json
import sys
from collections import defaultdict

import numpy as np


def b32(x):
    return int(np.asarray(np.float32(x)).view(np.uint32))


oracle = json.load(open(sys.argv[1]))
out_path = sys.argv[2]
on_prefixes, off_prefixes = sys.argv[3].split(","), sys.argv[4].split(",")

# (oracle label, release file, legacy Python value as the old code folded it)
SITES = [
    ("mynn_qsat_blend_chi_den", "physics/mynn_edmf.py", (273.15 - 6.0) - 240.0),
    ("mynn_xl_blend_cpv_cice", "physics/mynn_sgs_cloud.py", 4.0 * 461.6 - 2106.0),
    ("mynn_xl_blend_chi_den", "physics/mynn_sgs_cloud.py", 273.15 - 240.0),
    ("mynn_q1_rh_den", "physics/mynn_sgs_cloud.py", 1.0 - 0.83),
    ("mynn_level2_one_m_c2", "physics/mynn_pbl.py", 1.0 - 0.729),
    ("mynnsl_inv_27315", "physics/surface_layer.py", 1.0 / 273.15),
    ("mynnsl_ep_3", "physics/surface_layer.py", 1.0 - 287.0 / 461.6),
    ("noahmp_vkc2", "physics/noahmp/energy.py", 0.4 * 0.4),
    ("noahmp_rsurf_den", "physics/noahmp/energy.py", 2.71828 - 1.0),
    ("noahmp_snowh2o_ratio", "physics/noahmp/snow.py", 0.4 / (1.0 - 0.4)),
    ("kf_rhh_slope", "kernels/phys_kf_column.py", 0.2 / 1000.0),
    ("rrtmg_sw_bpade", "kernels/rad_sw_band_sums.py", 1.0 / 0.278),
    ("acoustic_dampmag_dts4p5", "kernels/dyn_acoustic_fp32.py", 4.5 * 0.2),
    ("acoustic_dampmag_dts18", "kernels/dyn_acoustic_fp32.py", 18.0 * 0.2),
    ("noahmp_class_alb_arg_dt54", "physics/noahmp/energy_radiation.py", -0.01 * 54.0 / 3600.0),
    ("noahmp_snow_ret_dt54", "physics/noahmp/snow.py", 5.0e-5 * 54.0),
]


def file_bits(prefixes):
    acc = defaultdict(set)
    for pre in prefixes:
        for rec in json.load(open(pre + ".literals.json")):
            if rec["dtype"] == "float32" and rec["frame"]:
                acc[rec["frame"].rsplit(":", 1)[0].replace("src/gpuwrf/", "")].add(rec["bits"])
    return acc


on, off = file_bits(on_prefixes), file_bits(off_prefixes)
rows, ok_all = [], True
for label, path, legacy in SITES:
    real_bits, legacy_bits = oracle[label]["bits"], b32(legacy)
    row = {"site": label, "file": path, "real_bits": hex(real_bits), "legacy_bits": hex(legacy_bits),
           "real_in_on": real_bits in on[path], "legacy_in_on": legacy_bits in on[path],
           "legacy_in_off": legacy_bits in off[path], "real_in_off": real_bits in off[path],
           "file_seen_on": path in on, "file_seen_off": path in off}
    # A site is verified when ON carries WRF's bits and OFF carries the old bits; the old bits may legitimately
    # survive in ON only if that value is also a different literal of the same file (reported, checked by hand).
    row["verified"] = row["real_in_on"] and row["legacy_in_off"] and not row["real_in_off"]
    row["clean_on"] = not row["legacy_in_on"]
    ok_all &= row["verified"]
    rows.append(row)
    print(f"{label:28s} {path:34s} ON real {row['real_in_on']!s:5s} ON legacy {row['legacy_in_on']!s:5s} "
          f"OFF legacy {row['legacy_in_off']!s:5s} OFF real {row['real_in_off']!s:5s} -> {'OK' if row['verified'] else 'CHECK'}")
json.dump({"on": on_prefixes, "off": off_prefixes, "all_verified": bool(ok_all), "sites": rows},
          open(out_path, "w"), indent=1)
print("all verified:", ok_all)
