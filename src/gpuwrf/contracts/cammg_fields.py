"""JAX-free CAM-MG Registry metadata; safe before State/physics imports.

Original Registry/registry.cam and Registry.EM_COMMON cammgmpscheme.
"""
MASS_FIELDS = (
    'rh_old_mp', 'lcd_old_mp', 'cldfra_old_mp', 'cldfra_mp',
    'cldfra_mp_all', 'cldfra_conv', 'cldfrai', 'cldfral',
    'wsedl3d', 'lradius', 'iradius',
)
INTERFACE_FIELDS = ('cam_kvh3d', 'cam_tke_pbl', 'cam_turbtype3d', 'cam_smaw3d')
SURFACE_FIELDS = ('cam_rainnc', 'cam_snownc', 'cam_sr', 'cam_fatal', 'cam_snowh')
STATE_FIELDS = MASS_FIELDS + INTERFACE_FIELDS + SURFACE_FIELDS
# Registry/registry.cam:171-177 has h; old RH/cloud/PBL work fields are r only.
HISTORY_NAMES = {name: name.upper() for name in (
    'cldfra_mp','cldfra_mp_all','lradius','iradius','cldfra_conv','cldfrai','cldfral')}
