"""R02: guarded extraction of full model state of selected CONVECTIVE CPU-reference frames (A-148 ADD1, CALIBRATION_READ).
usage: extract_frames.py <label>   -> analysis/data/frames/<dom>_t<lead>.npz (untracked). Values never logged."""
import glob, os, sys
from datetime import datetime
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gread
CPU_REF = "<DATA_ROOT>/alisios/scratch/n148-cpuref-20261018_12z/cpu_reference_a2_a3"
T0 = datetime(2026, 10, 18, 18)
# picked from the CPU side of w43-148fid derived colmax/W (cells with column QRAIN > 3 g/kg, |W| max)
SEL = {"d06": (22, 24, 25, 29, 34), "d04": (58, 62, 66), "d02": (38, 39, 55)}
V3 = ["U", "V", "W", "T", "THM", "PH", "PHB", "P", "PB", "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP", "QNICE",
      "QNRAIN", "QKE", "TSQ", "QSQ", "COV", "EL_PBL", "CLDFRA", "EXCH_H", "EXCH_M", "RTHRATEN", "H_DIABATIC", "QC_BL", "QI_BL",
      "CLDFRA_BL", "TKE_PBL", "RUBLTEN", "RVBLTEN", "RTHBLTEN", "RQVBLTEN", "REFL_10CM", "AL", "ALB", "P_HYD", "TSLB", "SMOIS", "SH2O"]
V2 = ["MU", "MUB", "PSFC", "HGT", "MAPFAC_M", "MAPFAC_U", "MAPFAC_V", "MAPFAC_MX", "MAPFAC_MY", "MAPFAC_UX", "MAPFAC_UY",
      "MAPFAC_VX", "MAPFAC_VY", "MF_VX_INV", "MSFTX", "MSFTY", "MSFUX", "MSFUY", "MSFVX", "MSFVY", "F", "E", "SINALPHA",
      "COSALPHA", "XLAND", "LANDMASK", "TSK", "UST", "HFX", "QFX", "LH", "PBLH", "ZNT", "MOL", "QSFC", "U10", "V10", "T2", "Q2",
      "TH2", "RAINNC", "RAINNCV", "SNOWNC", "GRAUPELNC", "RAINC", "SR", "XLAT", "XLONG", "CHS", "CHS2", "CQS2", "FLHC", "FLQC",
      "WSPD", "BR", "ZOL", "RMOL", "VEGFRA", "SST", "SEAICE", "SNOWH", "SNOW", "ALBEDO", "EMISS", "SWDOWN", "GLW", "HFX_FORCE",
      "KPBL", "UOCE", "VOCE", "CH", "QGH", "GRDFLX", "ACSNOM", "LAI", "IVGTYP", "ISLTYP", "TMN", "XICE", "ZS", "DZS"]
V1 = ["ZNU", "ZNW", "DN", "DNW", "RDN", "RDNW", "FNM", "FNP", "C1H", "C2H", "C1F", "C2F", "C3H", "C4H", "C3F", "C4F",
      "CF1", "CF2", "CF3", "CFN", "CFN1", "P_TOP", "T00", "P00", "TLP", "TISO", "TLP_STRAT", "P_STRAT", "RDX", "RDY", "DT",
      "ITIMESTEP", "XTIME", "DZS", "ZETATOP"]


def lead_of(name):
    return int(round((datetime.strptime(name.split("_", 2)[2], "%Y-%m-%d_%H:%M:%S") - T0).total_seconds() / 3600.0))


def main(label):
    import netCDF4
    cpu = {}
    for dom, leads in SEL.items():
        m = {lead_of(os.path.basename(p)): os.path.realpath(p) for p in glob.glob(f"{CPU_REF}/wrfout_{dom}_*")}
        cpu.update({(dom, L): m[L] for L in leads})
    gread.use_add()
    sess, rdir = gread.session(sorted(set(cpu.values())), label)
    out = HERE / "data" / "frames"; out.mkdir(parents=True, exist_ok=True)
    with sess:
        for (dom, L), p in sorted(cpu.items()):
            d = {}
            with netCDF4.Dataset(p) as nc:
                nc.set_auto_mask(False)
                for v in V3 + V2 + V1:
                    if v in nc.variables:
                        d[v] = np.asarray(nc.variables[v][0])
                for a in ("DX", "DY", "DT", "CEN_LAT", "TRUELAT1", "TRUELAT2", "MAP_PROJ", "GRID_ID", "PARENT_ID",
                          "I_PARENT_START", "J_PARENT_START", "PARENT_GRID_RATIO", "MP_PHYSICS", "BL_PBL_PHYSICS",
                          "SF_SFCLAY_PHYSICS", "HYBRID_OPT", "USE_THETA_M", "ETAC"):
                    if hasattr(nc, a):
                        d["attr_" + a] = np.asarray(getattr(nc, a))
                d["src_path"] = np.asarray(p); d["lead"] = np.asarray(L)
            np.savez(out / f"{dom}_t{L:02d}.npz", **d)
            print(dom, L, len(d), flush=True)
    gread.close_receipt(sess, rdir)


if __name__ == "__main__":
    main(sys.argv[1])
