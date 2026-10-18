"""All-frame original-CPU proof that WRF's SPP history fields FIELD_CONV/FIELD_PBL/FIELD_SF are +0 with
cu_physics=0 (lane w41-stoch st3 = spp, st4 = combined; MYNN, Noah-MP), extending the c46 (cu_physics=1)
class-B proof. Reuses proofs/t_writer_fields/inactive_constants.audit unchanged; WRF source: field_conv is
only written by the cumulus driver (:753 zero, GF), field_pbl never, field_sf only by RUC (module_sf_ruclsm.F).
usage: python inactive_fields.py <cases_root> <out.json>
"""
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "inactive_constants", Path(__file__).resolve().parents[1]/"t_writer_fields/inactive_constants.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.FIELDS = {"st3": ("FIELD_CONV", "FIELD_PBL", "FIELD_SF"), "st4": ("FIELD_CONV", "FIELD_PBL", "FIELD_SF")}
module.audit(Path(sys.argv[1]), Path(sys.argv[2]))
