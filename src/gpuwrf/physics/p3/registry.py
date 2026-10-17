"""P3 members of WRF's moist/scalar arrays (Registry.EM_COMMON packages).

Old theta/vapor are physics history, never advected. The second ice mass is
moist; rime mass/volume and the sqrt(N*Z) representation are scalars.
"""

SCHEMES = (50, 51, 52, 53)
INPUT_VARIABLES = {
    "qir": "QIR", "qib": "QIB", "qi2": "QICE2", "Ni2": "QNICE2",
    "qir2": "QIR2", "qib2": "QIB2", "qzi": "QZI",
    "mp_th_old": "TH_OLD", "mp_qv_old": "QV_OLD",
}


def scalar_species(mp):
    if mp not in SCHEMES:
        raise ValueError(f"P3 needs mp_physics in {SCHEMES}, got {mp}")
    return ("Ni", "Nr", "qir", "qib") + (() if mp == 50 else ("Nc",)) + (
        ("Ni2", "qir2", "qib2") if mp == 52 else ("qzi",) if mp == 53 else ())


def moist_species(mp):
    scalar_species(mp)  # same fail-closed variant check
    return ("qv", "qc", "qr", "qi") + (("qi2",) if mp == 52 else ())


def flow_only_species(mp):
    return tuple(name for name in scalar_species(mp) if name not in ("Ni", "Nr")) + (
        ("qi2",) if mp == 52 else ())


def input_variables(mp):
    names = scalar_species(mp) + moist_species(mp) + ("mp_th_old", "mp_qv_old")
    return {name: variable for name, variable in INPUT_VARIABLES.items() if name in names}
