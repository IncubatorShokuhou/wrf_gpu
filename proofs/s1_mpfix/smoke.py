"""Real Swiss CLI loader + bounded CPU steps, with actual held-radius evidence.

Inherits the established O1 CPU Thomas solver shim, not the GPU solver. CPU
wiring/finite evidence only; no original-WRF trajectory or GPU speed claim.
"""
from pathlib import Path
root=Path(__file__).resolve().parents[2]
probe=root/'proofs/o1_nlbind/probe.py'
text=probe.read_text()
text=text.replace('LANE = Path("<USER_HOME>/wrf_gpu2_lanes/o1-nlbind")',
                  'LANE = Path("<USER_HOME>/wrf_gpu2_lanes/s1-mpfix")')
marker='            s0, s1 = carry.state, c.state\n'
assert text.count(marker)==1
extra='''            rec["radius_active"] = bool(om.mp_re_active(ns))
            rec["radius_carry"] = {}
            for name in ("re_cloud", "re_ice", "re_snow"):
                value = getattr(s1, name)
                if value is None:
                    rec["radius_carry"][name] = None
                else:
                    data = np.asarray(value)
                    rec["radius_carry"][name] = dict(shape=list(data.shape), dtype=str(data.dtype),
                        minimum=float(data.min()), maximum=float(data.max()),
                        changed=int(np.count_nonzero(data != np.asarray(getattr(s0,name)))),
                        sha256=hashlib.sha256(data.tobytes()).hexdigest())
            rec["initial_hydrometeor_nonzero"] = {name:int(np.count_nonzero(np.asarray(getattr(s0,name))))
                for name in ("qc", "qi", "qs", "qg")}
            rec["Nc_nonzero"] = None if s1.Nc is None else int(np.count_nonzero(np.asarray(s1.Nc)))
            assert rec["nonfinite_values"] == 0, rec["nonfinite_values"]
            assert rec.get("guard_total", 0) == 0, rec.get("guard_total")
            assert rec["radius_active"] == (int(ns.mp_physics) in (18,24,28))
            assert all((value is not None) == rec["radius_active"] for value in rec["radius_carry"].values())
            assert all(value is None or value["dtype"] == "float32" for value in rec["radius_carry"].values())
'''
text=text.replace(marker,marker+extra)
exec(compile(text,str(probe),'exec'),{'__file__':str(probe),'__name__':'__main__'})
