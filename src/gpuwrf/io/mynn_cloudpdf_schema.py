"""Host-only admission of held MYNN PDF moments at restart boundaries."""
from gpuwrf.contracts.precision import mynn_cloudpdf_leaves


def selection(namelist):
    if namelist is None:
        return 2
    get = namelist.get if isinstance(namelist, dict) else lambda name, default: getattr(namelist, name, default)
    return int(get("bl_mynn_cloudpdf", 2)) if int(get("bl_pbl_physics", 5)) == 5 else 2


def validate(fields, saved, requested=None):
    option = selection(saved)
    if requested is not None and selection(requested) != option:
        raise ValueError("bl_mynn_cloudpdf restart selection differs (E78)")
    if fields is not None and any(fields.get(name) is None for name in mynn_cloudpdf_leaves(saved)):
        raise ValueError("bl_mynn_cloudpdf restart is missing held TSQ/COV/SH3D (E78)")
