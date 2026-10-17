"""WRF Registry / microphysics_driver1718..1775 Jensen moist/scalar inventory."""

INPUT_VARIABLES={
    'qi2':'QICE2','Ni2':'QNICE2','qi3':'QICE3','Ni3':'QNICE3',
    'qvoli':'QVOLI','qaoli':'QAOLI','qvoli2':'QVOLI2','qaoli2':'QAOLI2',
    'qvoli3':'QVOLI3','qaoli3':'QAOLI3',
}


def scalar_species(mp=55):
    if int(mp)!=55:raise ValueError('Jensen inventory requires mp_physics=55')
    return ('Ni','Nr','qvoli','qaoli','Ni2','qvoli2','qaoli2','Ni3','qvoli3','qaoli3')


def moist_species(mp=55):
    scalar_species(mp)
    return ('qv','qc','qr','qi','qi2','qi3')


def input_variables(mp=55):
    scalar_species(mp)
    return dict(INPUT_VARIABLES)


def output_mapping():
    return dict(qv='qv',qc='qc',qr='qr',qi1='qi',ni1='Ni',nr='Nr',ai1='qvoli',ci1='qaoli',
                qi2='qi2',ni2='Ni2',ai2='qvoli2',ci2='qaoli2',qi3='qi3',ni3='Ni3',ai3='qvoli3',ci3='qaoli3',
                effc='re_cloud',effi='re_ice')
