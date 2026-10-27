"""Kernel source module -> family (PORT_PLAN family table; lane o5-port)."""

FAMILY = [("dyn_acoustic", "acoustic"), ("fused_vertical", "acoustic(legacy)"), ("mynn", "MYNN"), ("edmf", "MYNN"),
          ("dyn_flux", "advection"), ("dyn_momflux", "advection"), ("dyn_pd", "PD"), ("ring_relax", "lateral bdy"),
          ("dyn_boundary", "lateral bdy"), ("boundary_apply", "lateral bdy"), ("dyn_rk", "RK/EOS"),
          ("dyn_uvtend", "RK/EOS"), ("dyn_rhsph", "RK/EOS"), ("dyn_omega", "RK/EOS"), ("thompson", "Thompson"),
          ("dyn_diff", "diffusion"), ("noahmp", "Noah-MP"), ("gwdo", "GWDO"), ("hydrostatic", "coupling"),
          ("sint", "force-down"), ("rad_", "RRTMG"), ("kf_column", "KF")]


def family(src):
    for key, fam in FAMILY:
        if key in src:
            return fam
    return "other"
