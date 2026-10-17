"""Backend-dark WRF lake Registry history/restart metadata."""
LAKE_DIMENSIONS = {'soil_levels_or_lake_levels_stag':10,'snow_and_soil_levels_stag':15,'interface_levels_stag':16}
LAKE_FIELDS = (
    ('lakedepth2d','m',0),('savedtke12d','W m-1 K-1',0),('snowdp2d','m',0),
    ('h2osno2d','mm',0),('snl2d','1',0),('t_grnd2d','K',0),
    ('t_lake3d','K',10),('lake_icefrac3d','1',10),('z_lake3d','m',10),('dz_lake3d','m',10),
    ('t_soisno3d','K',15),('h2osoi_ice3d','kg m-2',15),('h2osoi_liq3d','kg m-2',15),
    ('h2osoi_vol3d','m3 m-3',15),('z3d','m',15),('dz3d','m',15),('zi3d','m',16),
    ('watsat3d','m3 m-3',10),('csol3d','J m-3 K-1',10),('tkmg3d','W m-1 K-1',10),
    ('tkdry3d','W m-1 K-1',10),('tksatu3d','W m-1 K-1',10),
)
LAKE_HISTORY_FIELDS=tuple(name.upper() for name,_,_ in LAKE_FIELDS)


def specs(factory):
    dim={extent:name for name,extent in LAKE_DIMENSIONS.items()}
    return {name.upper():factory(name.upper(),('Time',dim[n],'south_north','west_east') if n else ('Time','south_north','west_east'),
        'XYZ' if n else 'XY ',name.replace('_',' '),unit,stagger='Z' if n else '',coordinates='XLONG XLAT XTIME')
        for name,unit,n in LAKE_FIELDS}
