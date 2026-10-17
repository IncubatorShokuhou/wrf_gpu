"""Original Registry.EM_COMMON NTU moist and scalar prognostics (mp=56)."""

EXTRA_SCALARS = tuple('ntu_' + n for n in
                      'qdcn qtcn qccn qrcn qnin fi fs vi vs vg ai as ag ah i3m'.split())
SURFACE_LEAVES = ('ntu_rainnc', 'ntu_snownc', 'ntu_sr')
INPUT_VARIABLES = dict(zip(EXTRA_SCALARS,
                           'QDCN QTCN QCCN QRCN QNIN FI FS VI VS VG AI AS AG AH I3M'.split(), strict=True))


def scalar_species(mp=56):
    if int(mp) != 56:
        raise ValueError('NTU inventory requires mp_physics=56')
    return ('Nc', 'Nr', 'Ni', 'Ns', 'Ng', 'Nh', *EXTRA_SCALARS)


def moist_species(mp=56):
    scalar_species(mp)
    return ('qv', 'qc', 'qr', 'qi', 'qs', 'qg', 'qh')


def input_variables(mp=56):
    scalar_species(mp)
    return dict(INPUT_VARIABLES)


def output_mapping():
    return {**{n: n for n in moist_species()},
            **dict(nc='Nc', nr='Nr', ni='Ni', ns='Ns', ng='Ng', nh='Nh'),
            **{n[4:]: n for n in EXTRA_SCALARS}}
