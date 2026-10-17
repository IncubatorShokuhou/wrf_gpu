"""Scheme-specific additions to the retained Thompson history inventory.

Names follow WRF Registry.EM_COMMON (and registry.new3d_wif for QNBCA).
Only real State/held diagnostic sources may populate these slots. Allocation
of a WRF slot alone is not evidence that the port computes its diagnostic.
"""

from gpuwrf.contracts.ice_diagnostics import ICE_DIAGNOSTIC_LEAVES, ice_diagnostic_leaves

# WRF names -> existing State leaves. NSSL calls Nc QNDROP, not QNCLOUD.
STATE_ALIASES = {
    **{name.upper(): (name.upper(), name) for name in ICE_DIAGNOSTIC_LEAVES},
    "QNICE": ("QNICE", "Ni", "qni"), "QNRAIN": ("QNRAIN", "Nr", "qnr"),
    "QICE2": ("QICE2", "qi2"), "QNICE2": ("QNICE2", "Ni2"),
    "QICE3": ("QICE3", "qi3"), "QNICE3": ("QNICE3", "Ni3"),
    **{name.upper(): (name.upper(), name) for name in
       ("qvoli", "qaoli", "qvoli2", "qaoli2", "qvoli3", "qaoli3")},
    "RE_CLOUD": ("RE_CLOUD", "re_cloud"), "RE_ICE": ("RE_ICE", "re_ice"),
    "QNCLOUD": ("QNCLOUD", "Nc", "qnc", "qnc_cloud"),
    "QNCCN": ("QNCCN", "Nn", "qnn", "qccn"),
    "QNSNOW": ("QNSNOW", "Ns", "qnsnow"),
    "QNGRAUPEL": ("QNGRAUPEL", "Ng", "qngraupel"),
    "QHAIL": ("QHAIL", "qh", "qhail"),
    "QNHAIL": ("QNHAIL", "Nh", "qnh", "qnhail"),
    "QVGRAUPEL": ("QVGRAUPEL", "qvolg"),
    "QVHAIL": ("QVHAIL", "qvolh"),
    "QNDROP": ("QNDROP", "Nc", "qndrop"),
    "QNWFA": ("QNWFA", "nwfa", "qnwfa"),
    "QNIFA": ("QNIFA", "nifa", "qnifa"),
}

# Metadata copied from the original CPU-WRF qualification files. The older
# build replaces Registry '#' unit characters with spaces; retain its ABI.
# name: (description, units, is_volume)
DIAGNOSTIC_METADATA = {
    **{(name + suffix).upper(): (description + str(cat), units, True)
       for suffix, cat in (("", 1), ("2", 2), ("3", 3))
       for name, description, units in (
           ("d_ice", "Mass-weighted mean ice size cat ", "m"),
           ("rho_ice", "Mass-weighted mean ice density cat ", "kg m-3"),
           ("v_ice", "Mass-weighted ice fallspeed cat ", "m s-1"),
           ("phi_ice", "Number-weighted mean ice aspect ratio cat ", ""))},
    "RE_CLOUD": ("Effective radius cloud water", "m", True),
    "RE_ICE": ("Effective radius cloud ice", "m", True),
    "QICE3": ("Ice mixing ratio cat 3", "kg kg-1", True),
    "QNICE3": ("Ice Number concentration cat 3", "# kg-1", True),
    **{f"QVOLI{suffix}": (f"Ice volume-{cat} mixing ratio", "m(3) kg(-1)", True)
       for suffix, cat in (("", 1), ("2", 2), ("3", 3))},
    **{f"QAOLI{suffix}": (f"Ice volume-{cat} times aspect ratio mixing ratio", "m(3) kg(-1)", True)
       for suffix, cat in (("", 1), ("2", 2), ("3", 3))},
    **{f"CCN{i}_GS": (f"Grid Scale Cloud Condensation Nuclei at S={s}%", " /cm-3", True)
       for i, s in enumerate(("0.02", "0.05", "0.1", "0.2", "0.3", "0.5", "1.0"), 1)},
    "EFCG": ("Grid Scale Cloud Effective Radius", "um", True),
    "EFIG": ("Grid Scale Ice Effective Radius", "um", True),
    "EFSG": ("Grid Scale Snow Effective Radius", "um", True),
    "WACT": ("Aerosol Activation Updraft", "m s-1", True),
    "MSKF_REFL_10CM": ("Full Radar reflectivity (lamda = 10 cm)", "dBZ", True),
    "QNBCA": ("black carbon aerosol number con", "  kg(-1)", True),
    "QNWFA2D": ("Surface aerosol number conc emission", "kg-1 s-1", False),
    "QNIFA2D": ("Surface dust number conc emission", "kg-1 s-1", False),
    "TAOD5502D": ("2D aerosol optical depth at 550 nm (MP=28)", "", False),
    "QNDROP": ("Droplet number mixing ratio", "  kg-1", True),
}

# WRF's own value in every admitted configuration is +0, written by
# gpuwrf.io.producer_history under WRF's branch conditions (proofs/t_classd):
# mp_gt_driver zeroes QNBCA unless wif_input_opt=2; TAOD5502D is assigned only
# under aer_opt=3 + RRTMG SW; MSKF_REFL_10CM only under do_radar_ref=1.
WRF_CONSTANT_FIELDS = {
    38: ("QNBCA", "TAOD5502D"),
    40: ("MSKF_REFL_10CM",),
    28: ("QNBCA", "TAOD5502D"),
}

# These diagnostics have no held operational source. Explicit caller-supplied
# diagnostics can be written, but the writer never substitutes allocator zeros.
UNPROVIDED_FIELDS = {
    # Jensen: REFL_10CM comes only from the held radar carry (DIAG_DBZ3D of every
    # mp_jensen_ishmael call); REFD_MAX exists only under compute_radar_ref
    # (do_radar_ref=1), where producer_history writes WRF's allocator zero.
    55: ("REFL_10CM",),
    18: ("REFL_10CM",),
    22: ("REFL_10CM",),
    19: ("REFL_10CM",),
    21: ("REFL_10CM",),
}

# Existing uniform State number slots are not prognostic fields of these WRF
# Registry configurations. They must not become fabricated history variables.
_ONE_MOMENT_NUMBERS=("QNDROP","QNCLOUD","QNRAIN","QNICE","QNSNOW","QNGRAUPEL","QNHAIL")
UNREPRESENTED_FIELDS={
    56: ("QNCCN", "QNDROP", "QVGRAUPEL", "QVHAIL", "QNWFA", "QNIFA", "REFL_10CM", "REFD_MAX"),
    55: ("QSNOW", "QGRAUP", "QNCLOUD", "QNDROP", "QNCCN", "QNSNOW", "QNGRAUPEL",
         "QHAIL", "QNHAIL", "QVGRAUPEL", "QVHAIL", "REFD_MAX"),
    19:(*_ONE_MOMENT_NUMBERS,"QVHAIL"),
    21:(*_ONE_MOMENT_NUMBERS,"QNCCN","QVGRAUPEL","QVHAIL","QHAIL"),
}

SCHEME_FIELDS = {
    7: ("QHAIL",),
    38: ("QNCLOUD", "QNGRAUPEL", "QNWFA", "QNIFA", "QVGRAUPEL", *WRF_CONSTANT_FIELDS[38]),
    55: ("QNICE", "QNRAIN", "QICE2", "QNICE2", "QICE3", "QNICE3",
         "QVOLI", "QAOLI", "QVOLI2", "QAOLI2", "QVOLI3", "QAOLI3", "RE_CLOUD", "RE_ICE",
         *UNPROVIDED_FIELDS[55]),
    9: ("QHAIL", "QNCLOUD", "QNSNOW", "QNGRAUPEL", "QNHAIL"),
    10: ("QNSNOW", "QNGRAUPEL"),
    50: ("QIR", "QIB"),
    51: ("QNCLOUD", "QIR", "QIB"),
    52: ("QNCLOUD", "QIR", "QIB", "QICE2", "QNICE2", "QIR2", "QIB2"),
    53: ("QNCLOUD", "QIR", "QIB", "QZI"),
    40: ("QNCLOUD", "QNSNOW", "QNGRAUPEL", *tuple(f"CCN{i}_GS" for i in range(1,8)),
         "EFCG", "EFIG", "EFSG", "WACT", *WRF_CONSTANT_FIELDS[40]),
    24: ("QHAIL",),
    28: ("QNCLOUD", "QNWFA", "QNIFA", "QNWFA2D", "QNIFA2D", *WRF_CONSTANT_FIELDS[28]),
    14: ("QNCCN", "QNCLOUD"),
    16: ("QNCCN", "QNCLOUD"),
    26: ("QHAIL", "QNCCN", "QNCLOUD"),
    18: ("QHAIL", "QNCCN", "QNDROP", "QNGRAUPEL", "QNHAIL", "QNSNOW",
         "QVGRAUPEL", "QVHAIL", "REFD_MAX", *UNPROVIDED_FIELDS[18]),
    22: ("QNCCN", "QNDROP", "QNGRAUPEL", "QNSNOW", "QVGRAUPEL", "REFD_MAX", *UNPROVIDED_FIELDS[22]),
    19: ("QHAIL","QNCCN","QVGRAUPEL","REFD_MAX",*UNPROVIDED_FIELDS[19]),
    21: ("REFD_MAX",*UNPROVIDED_FIELDS[21]),
}

# Registry.EM_COMMON:635–665, NTU-specific scalar moments.
DIAGNOSTIC_METADATA.update({'QDCN': ('Mass mixing ratio for dry CN ', 'kg kg-1', True), 'QTCN': ('Mass mixing ratio for total CN', 'kg kg-1', True), 'QCCN': ('Mass mixing ratio for cloud CN', 'kg kg-1', True), 'QRCN': ('Mass mixing ratio for rain CN', 'kg kg-1', True), 'QNIN': ('Potential ice nuclei number mixing ratio', '# kg-1', True), 'FI': ('Shape moment mixing ratio for pristine ice', 'm^3 kg-1', True), 'FS': ('Shape moment mixing ratio for snow-aggregates', 'm^3 kg-1', True), 'VI': ('Volume moment mixing ratio for pristine ice', 'm^3 kg-1', True), 'VS': ('Volume moment mixing ratio for snow-aggragates', 'm^3 kg-1', True), 'VG': ('Volume moment mixing ratio for rimed ice', 'm^3 kg-1', True), 'AI': ('Second moment mixing ratio for pristine ice', 'm^2 kg-1', True), 'AS': ('Second moment mixing ratio for snow-aggregates', 'm^2 kg-1', True), 'AG': ('Second moment mixing ratio for rimed ice', 'm^2 kg-1', True), 'AH': ('Second moment mixing ratio for hail', 'm^2 kg-1', True), 'I3M': ('Third moment mixing ratio for pristine ice', 'm^3 kg-1', True)})
STATE_ALIASES.update({name: (name, "ntu_" + name.lower()) for name in ('QDCN', 'QTCN', 'QCCN', 'QRCN', 'QNIN', 'FI', 'FS', 'VI', 'VS', 'VG', 'AI', 'AS', 'AG', 'AH', 'I3M')})
# NTU moist package qv,qc,qr,qi,qs,qg,qh (Registry.EM_COMMON:3049): QHAIL is prognostic.
SCHEME_FIELDS[56] = ('QHAIL', 'QNCLOUD', 'QNRAIN', 'QNICE', 'QNSNOW', 'QNGRAUPEL', 'QNHAIL', 'QDCN', 'QTCN', 'QCCN', 'QRCN', 'QNIN', 'FI', 'FS', 'VI', 'VS', 'VG', 'AI', 'AS', 'AG', 'AH', 'I3M')

# CAM-MG's history stream follows Registry/registry.cam:171-177. Old RH/cloud
# histories, WSEDL and turbulent interfaces are restart-only, never allocator
# diagnostics. QNDROP is not activated by the cammgmpscheme scalar package.
from gpuwrf.contracts.cammg_fields import HISTORY_NAMES as CAMMG_HISTORY_NAMES
STATE_ALIASES.update({name:(name,leaf) for leaf,name in CAMMG_HISTORY_NAMES.items()})
DIAGNOSTIC_METADATA.update({name:(("effective radius of "+("liquid" if leaf=="lradius" else "ice")+" condensate") if leaf in ("lradius","iradius") else
    "current time level "+("cldfrai" if leaf=="cldfrai" else "cldfral" if leaf=="cldfral" else "cldfra")+" for CAMMGMP microphysics","",True)
    for leaf,name in CAMMG_HISTORY_NAMES.items()})
SCHEME_FIELDS[11] = ("QNCLOUD","QNSNOW",*CAMMG_HISTORY_NAMES.values())
UNREPRESENTED_FIELDS[11] = ("QNDROP","QGRAUP","QNGRAUPEL","QNCCN","QHAIL","QNHAIL","QVGRAUPEL","QVHAIL","QNWFA","QNIFA")
# Existing scheme outputs retained at the microphysics call time level.
for _mp in (50, 51, 52, 53, 55):
    SCHEME_FIELDS[_mp] = (*SCHEME_FIELDS.get(_mp, ()),
                         *(name.upper() for name in ice_diagnostic_leaves(_mp)))
