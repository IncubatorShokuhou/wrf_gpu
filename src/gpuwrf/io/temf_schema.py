"""Backend-dark WRF Registry.EM_COMMON:1098–1115 TEMF history metadata."""

TEMF_HISTORY = (
    ('te_temf','Total energy from TEMF PBL scheme','m2 s-2',True),
    ('kh_temf','Diffusion coefficient for heat from TEMF PBL','',True),
    ('km_temf','Diffusion coefficient for momentum from TEMF PBL','',True),
    ('shf_temf','Sensible heat flux from TEMF PBL','K m s-1',True),
    ('qf_temf','Sensible heat flux from TEMF PBL','kg/kg m s-1',True),
    ('uw_temf','U momentum flux from TEMF PBL','m2 s-2',True),
    ('vw_temf','V momentum flux from TEMF PBL','m2 s-2',True),
    ('wupd_temf','Updraft velocity from TEMF PBL','m s-1',True),
    ('mf_temf','Mass flux from TEMF PBL','m s-1',True),
    ('thup_temf','Updraft thetal from TEMF PBL','K',True),
    ('qtup_temf','Updraft qt from TEMF PBL','1',True),
    ('qlup_temf','Updraft ql (liquid water) from TEMF PBL','1',True),
    ('cf3d_temf','3D Cloud fraction from TEMF PBL','1',True),
    ('hd_temf','Dry thermal top height from TEMF PBL','m',False),
    ('lcl_temf','Lifting condensation level from TEMF PBL','m',False),
    ('hct_temf','Cloud top height from TEMF PBL','m',False),
    ('cfm_temf','Column cloud fraction from TEMF PBL','1',False),
    ('wm_temf','Velocity scale in TEMF surface layer scheme','m s-1',False),
)
