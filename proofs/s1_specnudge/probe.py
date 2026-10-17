"""Reuse the established CLI release-identity/CPU smoke probe in this lane.

Invocation is identical to proofs/o2_fdda/probe.py. Changes below only redirect
artifacts, recognize spectral namelist keys, and retain PH/initial fields for
the independent original CPU-WRF state comparison. Numerical code is untouched.
"""
from pathlib import Path

source = Path(__file__).resolve().parents[1] / "o2_fdda/probe.py"
text = source.read_text()
text = text.replace('LANE = Path("<USER_HOME>/wrf_gpu2_lanes/o2-fdda")',
                    'LANE = Path("<USER_HOME>/wrf_gpu2_lanes/s1-specnudge")')
text = text.replace('FDDA_KEYS = {"grid_fdda",', 'FDDA_KEYS = {"gph", "xwavenum", "ywavenum", '
                    '"if_no_pbl_nudging_ph", "if_zfac_ph", "k_zfac_ph", '
                    '"dk_zfac_uv", "dk_zfac_t", "dk_zfac_ph", "dk_zfac_q", "ktrop", "grid_fdda",')
text = text.replace('om._resolve_operational_suite(ns)',
                    'rec["fdda_options"] = repr(getattr(ns.data_assimilation, "options", None))\n'
                    '        rec["spectral_carry_dtypes"] = ({n: str(getattr(carry.fdda, n).dtype) '
                    'for n in ("ph_old", "ph_new", "rphndgdten")} '
                    'if hasattr(carry.fdda, "rphndgdten") else None)\n'
                    '        om._resolve_operational_suite(ns)')
text = text.replace('rec["trace_s"] = round(time.time() - t1, 1)',
                    'rec["trace_s"] = round(time.time() - t1, 1)\n'
                    '        rec["status"] = "LOADED_TRACED_EXECUTION_PENDING"\n'
                    '        out["arms"][arm] = rec\n'
                    '        Path(OUT).write_text(json.dumps(out, indent=1, default=str))')
text = text.replace('qv=np.asarray(s1.qv))',
                    'qv=np.asarray(s1.qv), ph=np.asarray(s1.ph_perturbation), '
                    'w=np.asarray(s1.w), p=np.asarray(s1.p_perturbation), '
                    'theta0=np.asarray(s0.theta), qv0=np.asarray(s0.qv), '
                    'u0=np.asarray(s0.u), v0=np.asarray(s0.v), ph0=np.asarray(s0.ph_perturbation))')
text = text.replace('rec["final_npz"] = str(',
                    'if hasattr(c.fdda, "rphndgdten"):\n'
                    '                np.savez(LANE / "spectral_final_rates.npz", '
                    '**{n: np.asarray(getattr(c.fdda, n)) for n in '
                    '("rundgdten", "rvndgdten", "rthndgdten", "rqvndgdten", "rphndgdten")})\n'
                    '            rec["status"] = "EXECUTED"\n'
                    '            rec["final_npz"] = str(')
exec(compile(text, str(source), "exec"), {"__file__": str(source), "__name__": "__main__"})
