"""Pre-JAX selection checks for OML1D and hurricane surface fluxes."""
import math


def validate_options(cfg, *, sfclay, sst_update=False, noahmp=True, lake=0, lsm=None, urban=0, pbl=5):
    cfg={**{'sf_ocean_physics':0,'isftcflx':0,'oml_hml0':50.,'oml_gamma':.14,'oml_relaxation_time':0.},
         **{k:v for k,v in cfg.items() if v is not None}}
    ocean=int(cfg.get('sf_ocean_physics',0))
    flux=int(cfg.get('isftcflx',0))
    if ocean not in (0,1):
        raise ValueError('sf_ocean_physics=2 (PWP 3-D ocean) is not implemented; use sf_ocean_physics=1 or 0')
    if flux not in (0,1,2):
        raise ValueError('isftcflx supports 0/1/2; options 3/4 are not qualified')
    if flux and int(5 if sfclay is None else sfclay) not in (1,5,91):
        raise ValueError('isftcflx=1/2 requires sf_sfclay_physics=1/5/91 (WRF supported hurricane flux schemes)')
    if flux:
        if lsm == 3:
            raise ValueError('isftcflx with RUC is not qualified: RUC repeats MYNN without hurricane diagnostics/USTM; select isftcflx=0')
        if urban == 1:
            raise ValueError('isftcflx with SLUCM is not qualified: hurricane CHS exchange coefficients are not supplied; select isftcflx=0')
        if urban in (2,3):
            raise ValueError('isftcflx with BEP/BEM is not qualified: urban surface solve discards hurricane diagnostics; select isftcflx=0')
        if pbl in (16,17):
            raise ValueError('isftcflx with epsilon-PBL is not qualified: both paths own the surface diagnostics; select isftcflx=0')
    for key,default in (('oml_hml0',50.),('oml_gamma',.14),('oml_relaxation_time',0.)):
        if not math.isfinite(float(cfg.get(key,default))):
            raise ValueError(f'{key} must be finite')
    if ocean:
        if lake:
            raise ValueError('OML with sf_lake_physics=1 is not qualified: ocean_driver must precede LakeModel; select one')
        if float(cfg.get('oml_gamma',.14))<=0:
            raise ValueError('OML requires oml_gamma>0 (WRF divides by Gamma/BV2)')
        if sst_update:
            raise ValueError('OML prognostic SST and sst_update forcing are not qualified together; set sst_update=0')
        if not noahmp:
            raise ValueError('OML currently requires sf_surface_physics=4 (Noah-MP emissivity/radiation)')
