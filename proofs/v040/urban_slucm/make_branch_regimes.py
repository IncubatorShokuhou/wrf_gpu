"""Branch-coverage SLUCM oracle regimes (rv-urban F2, from the critic's work/mkreg.py).

gcov of the 25 base regimes left three admitted branches of module_sf_urban.F:urban unexecuted:
IMP_SCHEME=2 roof saturation (:1045-1046), road saturation (:1310-1311) and the canopy-wind
``ZR+2 >= ZA`` else-branch (:901-902).  These 12 regimes, derived from base regimes.txt rows, execute
them (long heavy rain, first-level thickness dz8w1 17.0-19.1 m around the UTYPE-2 canopy limit
ZR+2 = 9.5 m = 0.5*dz8w1, plus hot calm, gale, cold rain, dew and a wrfinput FRC_URB2D of 0.35).
60 steps (roof storage saturates after ~9 steps at 0.05 mm/step).

Usage: python make_branch_regimes.py regimes.txt regimes_branch.txt
"""

from __future__ import annotations

import sys

NSTEPS = 60.0


def main(src_path, out_path):
    src = open(src_path).read().splitlines()
    nreg, nv = map(int, src[0].split())
    hdr = src[1]
    cols = hdr.split()
    assert len(cols) == nv == 44
    rows = {src[2 + 2 * i].strip(): [float(x) for x in src[3 + 2 * i].split()] for i in range(nreg)}

    def mk(base, **kw):
        r = list(rows[base])
        for k, v in kw.items():
            r[cols.index(k)] = v
        r[cols.index("nsteps")] = NSTEPS
        return r

    new = {
        "rv_longrain_day": mk("day_j16_i44", rainbl=0.05),
        "rv_longrain_night": mk("night_j24_i37", rainbl=0.05),
        "rv_lowza_17p0": mk("day_j16_i44", dz8w1=17.0),
        "rv_lowza_18p8": mk("night_j56_i38", dz8w1=18.8),
        "rv_lowza_19p0": mk("day_j62_i77", dz8w1=19.0),
        "rv_lowza_19p1": mk("day_j62_i77", dz8w1=19.1),
        "rv_lowza_longrain": mk("day_j16_i44", dz8w1=18.0, rainbl=0.05),
        "rv_hot_unstable": mk("day_j16_i44", tsk=318.0, t3d1=300.0, u1=0.4, v1=0.3, swdown=950.0),
        "rv_gale": mk("day_j16_i44", u1=25.0, v1=-10.0),
        "rv_cold_rain": mk("night_j74_i72", t3d1=268.0, tsk=266.0, rainbl=0.01, tslb1=268.0, tslb2=268.5,
                           tslb3=269.0, tslb4=270.0),
        "rv_dew": mk("night_j16_i44", qv1=0.014, t3d1=285.0, tsk=283.0),
        "rv_frc035_longrain": mk("day_j56_i38", frc_in=0.35, rainbl=0.03),
    }
    with open(out_path, "w") as f:
        f.write(f"{len(new)} {nv}\n{hdr}\n")
        for n, r in new.items():
            f.write(n + "\n" + " ".join(repr(x) for x in r) + "\n")
    print(len(new), "regimes, nsteps", NSTEPS)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
