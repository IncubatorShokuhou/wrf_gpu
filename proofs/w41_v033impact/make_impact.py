"""Assemble IMPACT.json for lane w41-v033impact from the probe/cross-check/raw-diff/exec/GPU artifacts (no model code)."""
import json
import time
from pathlib import Path

import numpy as np

L = Path("<USER_HOME>/wrf_gpu2_lanes/w41-v033impact")
METRICS = ("msftx msfty msfux msfuy msfvx msfvy c1h c2h c3h c4h c1f c2f c3f c4f dn dnw rdn rdnw cf1 cf2 cf3 fnm fnp "
           "dzdx dzdy dzdx_u dzdy_v f e sina cosa p_top").split()
TEND = "u v w theta qv p ph mu".split()
WRF = {"noahmp_case.tbot": "TMN (wrfinput; over water = SST at t0)", "noahmp_case.shdfac": "VEGFRA/100 (wrfinput)",
       "lower_boundary.sst": "SST series (wrflowinp aux4, 13 x 6-hourly records)",
       "lower_boundary.seaice": "SEAICE series (wrflowinp)", "lower_boundary.vegfra": "VEGFRA series (wrflowinp)",
       "lower_boundary.albbck": "ALBBCK series (wrflowinp)", "lower_boundary.activation_steps": "aux4 record step indices",
       "radiation_static.xlat_deg": "XLAT", "radiation_static.xlong_deg": "XLONG",
       "radiation_static.terrain_height_m": "HGT", "radiation_static.slope_rad": "slope from HGT",
       "radiation_static.slope_azimuth_rad": "slope azimuth from HGT", "radiation_static.sina": "SINALPHA",
       "radiation_static.cosa": "COSALPHA"}


def pretty(leaf):
    for pre, names in (("metrics", METRICS), ("tendencies", TEND)):
        if leaf.startswith(pre + "[<flat index "):
            i = int(leaf.split("flat index ")[1].split(">")[0])
            return f"{pre}.{names[i]}"
    return leaf


def load(p):
    p = L / p
    return json.loads(p.read_text()) if p.exists() else None


probe = load("probe1/probe_summary.json")
cross = load("probe2/crosscheck.json")
raw = load("rawdiff/rawdiff.json")
exe = load("probe3/exec_swap.json")
census = load("gpu147/census.json")
cases = probe["cases"]
ref = cases["g2_20260707"]
leaves_npz = {k: np.load(L / f"probe1/leaves_{k}.npz") for k in cases}
xrow = {r["leaf"]: r for r in cross["rows"]} if cross else {}
rows = []
for r in ref["leaves"]:
    key = f"{r['dom']}|{r['leaf']}"
    name = pretty(r["leaf"])
    rec = {"domain": r["dom"], "leaf": name, "wrf_source": WRF.get(name, "geometry/static (wrfinput)"),
           "shape": r["shape"], "dtype": r["dtype"],
           "consumed_structured_liveness": r.get("consumed_any_output"),
           "consumed_jax_conservative_dce": (xrow.get(key) or {}).get("jax_dce_used"),
           "reaches_state_nonbdy": (xrow.get(key) or {}).get("live_state_nonbdy")}
    for other in ("wn3_0227", "g2_20260702"):
        o = next(x for x in cases[other]["leaves"] if x["dom"] == r["dom"] and x["leaf"] == r["leaf"])
        if o["sha256"] == r["sha256"]:
            rec[f"vs_{other}"] = "identical"
        else:
            a = leaves_npz[other][key].astype(np.float64)
            b = leaves_npz["g2_20260707"][key].astype(np.float64)
            d = a - b
            rec[f"vs_{other}"] = {"n_diff": int(np.count_nonzero(d)), "n": int(d.size),
                                  "max_abs": float(np.abs(d).max()), "rmse": float(np.sqrt(np.mean(d * d)))}
    rows.append(rec)
differ_consumed = [f"{r['domain']}:{r['leaf']}" for r in rows if r["consumed_structured_liveness"]
                   and (r["vs_wn3_0227"] != "identical" or r["vs_g2_20260702"] != "identical")]
impact = {
    "lane": "w41-v033impact", "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "label_legend": "[M] measured, [I] inferred, [U] unknown; GPU P/F/V comparisons are GPU-vs-GPU impact evidence, not fidelity",
    "program": {"release": "v0.3.3", "commit": "86e63037020392df38c3f600a9d903e258e3e9d4",
                "fused_program": "domain_tree._build_fused_cascade_program fused_jit (parent d02 + child d03), closes over parent/child OperationalNamelist",
                "cheap_key": "v4 fn_identity ignores closures; GPUWRF_AOT_VERIFY default OFF -> a cheap-key hit loads silently"},
    "inputs": {"cases": ["wn3_0227 (issue 20260227_18z, start 2026-02-28_00)", "g2_20260702 (start 2026-07-03_00)",
                         "g2_20260707 (start 2026-07-08_00)"],
               "provenance": "guarded copies made by w41-bundlekey/w41-bdyrec; sha256+origin in CASE_PROVENANCE.json; namelists differ only in dates; no new ALISIOS reads"},
    "q1_baked_leaves": {
        "summary": ("[M] 108 closed-over array leaves (54 per domain d02/d03). Lifted-namelist jaxpr structure hash identical "
                    "0227 vs 0707 and all other constants/nest weights identical -> the date content of a fused blob is these "
                    "leaf values only. Leaves that differ between dates: tbot, shdfac, lower_boundary sst/vegfra/albbck. "
                    "Consumed by the outputs: ONLY tbot (TMN); lower_boundary (incl. SST series) and shdfac are dead "
                    "(structured liveness and JAX's conservative DCE agree), because v0.3.3 also passes the aux4 series "
                    "through the traced clock_base and the step prefers it, and Noah-MP dveg=4 takes FVEG from SHDMAX."),
        "differ_and_consumed": differ_consumed,
        "tbot_cells": "[M] TMN differs only on water cells (d02 29666/31239, d03 8289/10323; 0 land cells); over water TMN == wrfinput SST(t0)",
        "consumption_at_cell_level": ("[M source] noah_mp_step runs on all columns; every atmosphere-facing output (HFX, QFX, "
                                      "ZNT, TSK, T2, Q2, QSFC, land history) is where(is_land, noah, water-bulk) -> the "
                                      "water TMN can only enter the Noah land carry at water columns (deep soil temperature)"),
        "trace": {k: ref["trace"][k] for k in ("n_eqns_total", "n_outputs", "jaxpr_struct_sha", "other_consts_sha")},
        "leaves": rows},
    "raw_inputs": raw,
}
lifted = load("probe4/lifted_lowered.json")
if lifted:
    impact["q1_baked_leaves"]["lifted_program_identity"] = {
        "claim": "[M] cuda-lowered StableHLO (native Triton, E156 recipe) of the fused program with all namelist leaves lifted to arguments is byte-identical for 0227 and 0707 -> no other date-dependent bake (static-aux date scalars dead)",
        **lifted}
impact["q1_baked_leaves"]["independent_confirmation"] = (
    "[M] w41-bundlekey KEY_DIFF.json (2026-10-18T08:59Z): v0.3.3 fused/d02 cheap key EQUAL 0227 vs 0707, lowered HLO differs "
    "in exactly one dense constant per fused domain (117x267 f32 d02, 93x111 f32 d03) whose payload matches TMN")
impact["q1_baked_leaves"]["source_cell_level_chain"] = [
    "physics/noahmp_coupler.py noahmp_surface_adapter: noah_mp_step on ALL columns; HFX/QFX/ZNT/TSK/QSFC/T2/Q2 = where(is_land, noah, water bulk)",
    "coupling/physics_couplers.py _surface_radiation_properties: RRTMG albedo/emissivity from the land carry only where(is_land)",
    "io/land_history.py: wrfout land leaves = np.where(LANDMASK, carry, seed); over water TSLB = 273.16 (all layers) / SST (top, SST_UPDATE)",
    "=> the foreign water-cell TMN can only change the dummy water-column Noah land carry (deep soil temperature), which reaches neither the atmosphere nor wrfout [I from source; GPU #147 tests it]"]
impact["q2_time_behaviour"] = {
    "TMN": "[M] time-constant within a run (tmn_update unset = WRF default 0); consumed only via Noah-MP bottom soil BC; differs between dates only over water where real.exe sets TMN = SST(t0)",
    "aux4 SST/SEAICE/VEGFRA/ALBBCK": "[M] time-varying (13 x 6-hourly records), baked copy DEAD in v0.3.3: the run reads its own series via the traced clock_base",
    "VEGFRA (wrfinput, shdfac)": "[M] monthly field, baked copy DEAD (dveg=4 takes FVEG from SHDMAX; aux4 VEGFRA replaces shdfac anyway)",
    "metrics / radiation_static / tendencies / nest weights": "[M] geometry-only, bit-identical across all three dates",
    "magnitude_cross_season_0227_vs_0707": "[M] water TMN rmse 4.02 K d02 / 3.53 K d03 (max 8.8 / 6.6 K); land 0",
    "expected_effect": "[I] water-column dummy soil temperatures only (bottom BC at 8 m: ~0.03 K/24 h in layer 4); no atmospheric or wrfout effect by source"}
impact["q3_same_month"] = {
    "answer": "[M] yes, the bake still differs within a month (TMN over water = that day's SST): 0702 vs 0707 water TMN rmse 0.68 K d02 / 0.79 K d03 (max 3.6 / 1.9 K); same leaf set; consumption identical (only water TMN)",
    "raw_inputs": "rawdiff 0702-0707: SST series max 7.6 K, VEGFRA max 0.93 %, both dead"}
if exe:
    impact["q4_cpu_same_executable_swap"] = exe
swap = load("gpu147/swap/exec_swap.json")
if swap:
    impact["q4_gpu_same_executable_swap"] = swap
blobs = load("gpu147/blob_compare.json")
if blobs:
    impact["q4_gpu_blob_compare"] = blobs
noise = load("gpu147/noise_table.json")
if noise:
    impact["q4_gpu_noise_table"] = noise
census = load("gpu147/census.json")
if census:
    impact["q4_gpu_ab_census_summary"] = {
        arm: {dom: {"frames": len(fr), "vars_compared_per_frame": sorted({r.get("identical_vars", 0) + len(r.get("differing", {})) for r in fr}),
                    "differing_vars_per_frame": [len(r.get("differing", {})) for r in fr]} for dom, fr in doms.items()}
        for arm, doms in census["arms"].items()}
d6 = sorted((L / "gpu147/d6").glob("*.json")) if (L / "gpu147/d6").exists() else []
if d6:
    def _d6(p):
        d = json.loads(p.read_text())
        s = d["summaries"]
        return {"tolerance_failure_count": s.get("tolerance_failure_count"), "comparable_fields": s.get("comparable_field_count"),
                "max_worst_lead_rmse": max((x.get("worst_lead_rmse") or 0.0) for x in s.get("top_drift_signals", [])),
                "fields": sorted(x["field"] for x in s.get("top_drift_signals", []))}
    impact["q4_gpu_ab_frozen_d6_P_vs_F"] = {p.name: _d6(p) for p in d6}
rec = load("gpu147/record_compare.json")
if rec:
    impact["q4_gpu_record_compare"] = {
        "blobs": {k: {"n_records": v["n_records"]} for k, v in rec["blobs"].items()},
        "pairs": {k: [r for r in pr if not r["identical"]] for k, pr in rec["pairs"].items()},
        "decoded": load("gpu147/record_tmn_match.json"),
        "claim": "[M] P(0227) vs F(0707) fused blobs: identical cubin; 919/922 decompressed records identical; differing: rec 317/780 = constant buffers whose payload == -TMN (d02/d03) of the respective date, rec 921 = module metadata which also differs between F and V (same 0707 HLO, identical outputs) -> programs differ only by TMN"}
tmn = load("TMN_LAND_INVARIANCE.json")
if tmn:
    impact["followup_tmn_land_invariance"] = {"summary": tmn["summary"],
        "w9n_geometry": tmn["parts"]["part2_w9n_9nest"]["namelist_geometry"],
        "sessions": {k: v["session"] for k, v in tmn["parts"].items()},
        "nine_nest_source_answer": ("[M] W9N parent_id 1,1,2,2,2,2,2,2,2, feedback 0 -> v0.3.3 _fusable_parent(d02) = 7 one-way "
            "leaf children -> one fused/d02 program closing over 8 namelists (8 baked TMN constants); each child advanced by the "
            "same _advance_chunk(child, child_namelist) -> identical Noah-MP where(is_land) blend, RRTMG where(is_land) "
            "albedo/emiss and np.where(LANDMASK) land writer as in the 3-nest case"),
        "file": str(L / "TMN_LAND_INVARIANCE.json")}
sess = (L / "gpu147/session.log")
if sess.exists():
    impact["q4_gpu_session"] = {"log": sess.read_text().splitlines(), "script_sha256": (L / "gpu147/session_run.sha256").read_text().split()[0],
                                "aot_lines": {a: (L / f"gpu147/aot_lines_{a}.txt").read_text().splitlines()[:12] for a in ("prime", "P", "F", "V")
                                              if (L / f"gpu147/aot_lines_{a}.txt").exists()}}
impact["answers"] = {
    "q1": "Baked into the v0.3.3 fused d02/d03 program and date-dependent: d02/d03 TMN (consumed), wrfinput VEGFRA/shdfac and the aux4 SST/SEAICE/VEGFRA/ALBBCK series (both DEAD: the run reads its own traced copies). Geometry leaves identical across dates. [M]",
    "q2": "TMN is time-constant within a run; differs between dates ONLY over water (TMN=SST(t0)); land TMN exactly date-invariant across 10 WN3 issues Jan-Aug and 4 W9N issues (max|dTMN| land = 0.0). The time-varying aux4 series are not consumed from the bake. [M]",
    "q3": "Same month still bakes a different program (0702 vs 0707: water TMN rmse 0.68/0.79 K), but with the same zero output effect. [M]",
    "q4": "0707 24 h run with the wrongly loaded 0227 program (P) is whole-file byte-identical to the fresh compile (F) and to the AOT_VERIFY recompile (V) in all 72 wrfout files (d01-d03 x 24 h); frozen D6 P vs F: 0 failures, all RMSE 0; noise floor F-V = 0. [M]",
    "conclusion": "v0.3.x cross-date fused cache hits change NO model output for the ALISIOS WN3 3-nest and (by identical land TMN + same masking code path) the W9N 9-nest geometry: the only consumed baked constant is water-cell TMN, confined to the dummy water-column Noah-MP land carry that never reaches the atmosphere, radiation or wrfout. [M for WN3 0227->0707 end-to-end; M source + M inputs for other dates/W9N]",
    "caveats": "Restart checkpoints (if any) would carry the dummy water-column soil temperatures of the foreign TMN (never written to wrfout, no physical effect); a future case with date-varying LAND TMN (tmn_update=1 or a different real.exe source) would break the land-invariance premise -> GPUWRF_AOT_VERIFY=1 is the safe v0.3.x setting for shared caches (one fused lower per process; recompile ~6 min only on a foreign-date hit). v0.4.0 key v5 already separates dates; w41-bundlekey fix makes the fused program date-blind by construction.",
    "mitigation_tested": "GPUWRF_AOT_VERIFY=1 on v0.3.3: loaded 0227 blob -> 'fused-verify-MISMATCH 90b94d23 != f7dd60cd' -> quarantine + recompile; outputs identical to F. [M]"}
(L / "IMPACT.json").write_text(json.dumps(impact, indent=1, default=str))
print("differ_and_consumed:", differ_consumed)
print("wrote", L / "IMPACT.json")
