"""Content identity of tables read outside the traced call operands.

Namelist holders cover land/urban/RUC tables. These remaining modules bind
tables in globals/defaults or host-only cached loaders. Select by the effective
physics options, including the namelists of fused cascades. This runs
only when building a full metadata key; runtime key memos retain that key.
"""
from __future__ import annotations

import importlib
import dataclasses
import hashlib
from collections.abc import Mapping
import types
from pathlib import Path


def _namelists(fn, namelist):
    from gpuwrf.runtime.operational_mode import OperationalNamelist

    found = []
    def visit(value):
        if isinstance(value, OperationalNamelist):
            found.append(value)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)
        elif getattr(value, "table_namelists", None) is not None:
            # Fused-cascade key carrier: its namelists are jit arguments, not closure cells.
            visit(value.table_namelists)
    visit(namelist)
    target = getattr(fn, "__wrapped__", fn)
    for cell in getattr(target, "__closure__", None) or ():
        visit(cell.cell_contents)
    return found


def baked_table_hash(fn, namelist):
    """Hash actual bound values, rather than only their source/file names.

    The host loaders below return NumPy values and never trace device work.
    No global digest memo: a replacement table in the same interpreter must be
    distinguished. Deduplicate shared objects only within this one snapshot.
    """
    from gpuwrf.assets import LazyTable, pinned_sha256
    from gpuwrf.runtime.aot_cheap_key import canonical_digest

    modules = set()
    loaders = set()
    cloud_kinds = set()
    p3_kinds = set()
    nssl = False
    nssl_one_moment = set()
    morrison_radar = False
    qnse = False
    gfs_shallow = False
    legacy_grids=[]
    stock_triton=[]
    legacy_dates=[]
    land_controls = []
    def radiation(sw, lw):
        if 4 in (sw, lw):
            modules.add("rrtmg_tables")
            loaders.add(("rrtmg_mp_re", "_retab"))
        if sw == 4:
            modules.add("rrtmg_sw")
            cloud_kinds.add("sw")
        if lw == 4:
            modules.add("rrtmg_lw")
            cloud_kinds.add("lw")
            loaders.update(("rrtmg_lw", name) for name in
                           ("_native_lw_tables", "_native_lw_cloud_tables"))

    # Initialization has _StaticKwargs, not an OperationalNamelist. Include the
    # solvers' other cached host tables too; their defaults cover only the main
    # RRTMGTableBundle. Both are baked even when the call's columns are traced.
    target = getattr(fn, "__wrapped__", fn)
    if getattr(target, "__module__", "") == "gpuwrf.physics.rrtmg_sw":
        radiation(4, 0)
    if getattr(target, "__module__", "") == "gpuwrf.physics.rrtmg_lw":
        radiation(0, 4)
    if getattr(target,'__module__','') in ('gpuwrf.physics.ra_flg','gpuwrf.physics.flg_transfer'):
        modules.update(('ra_flg','flg_transfer'))
        loaders.add(('ra_flg','_host_tables'))
    if getattr(target,'__module__','') in ('gpuwrf.physics.ra_gfdleta','gpuwrf.physics.gfdleta_column',
                                         'gpuwrf.physics._gfdleta_radtn','gpuwrf.physics.ra_gfdleta_support'):
        modules.update(('ra_gfdleta','ra_gfdleta_support','gfdleta_column'))
        loaders.update((('ra_gfdleta','_host_tables'),('_gfdleta_radtn','_host_tables')))
    for nl in _namelists(fn, namelist):
        if getattr(nl, 'noahmp_options', None) is not None:
            # MPTABLE BATS/RSURF/FSATMX/TIMEAN/DRAIN_LAYER_OPT are resolved
            # scalar controls baked in this frozen auxiliary object. Hash the
            # bound values, never a reread of a possibly changed table file.
            # Soil/runoff/tile tables already ride NoahMPStatic's content-keyed
            # holder; irrigation/carbon/crop maps are traced children (call
            # operands of every step program, fused cascades included).
            land_controls.append(nl.noahmp_options)
        mp, sw, lw = int(nl.mp_physics), int(nl.ra_sw_physics), int(nl.ra_lw_physics)
        if mp in (8, 28):
            modules.update(("thompson_tables", "thompson_column"))
            loaders.add(("thompson_tables", "load_wrf_cold_collection_tables"))
        if mp == 11:
            modules.update(("cammg.constants", "cammg.cloud", "cammg.gamma", "cammg.micro", "cammg.macro",
                            "cammg.aerosol", "cammg.ice_activation", "cammg.error_function",
                            "cammg.tables", "cammg.saturation_table", "cammg.column", "cammg._support", "bl_camuw"))
            loaders.add(("cammg.tables", "_raw"))
            stock_triton.append(True)  # native column PTX comes from stock Triton (kernels/_triton_kernel_call)
        if mp == 28:
            modules.add("thompson_aero_tables")
        if mp in (5, 15):
            modules.update(("ferrier.native", "ferrier.column", "ferrier.saturation"))
            # native_column uses this imported provider, not a fresh lookup
            # through saturation.host_tables after its import-time binding.
            loaders.add(("ferrier.native", "host_tables"))
        if mp == 7:
            modules.update(("goddard4ice.native", "goddard4ice.column",
                            "goddard4ice.equations", "goddard4ice.support"))
        if mp == 38:
            # The DOUBLE tables reach the step as traced operands (thompson_hail.tables);
            # their loader values stay keyed below, so a changed table byte still splits keys.
            modules.update(("thompson_tables", "thompson_column", "thompson_aero_tables",
                            "thompson_hail", "thompson_hail._coefficients", "thompson_hail.tables"))
            loaders.update((
                ("thompson_tables", "load_wrf_cold_collection_tables"),
                ("thompson_hail.native", "_cloud_coefficients"),
                ("thompson_hail.native", "_freezing_tables"),
                ("thompson_hail.context", "collision_tables"),
                ("thompson_hail.aot_tables", "resolved_bindings"),
            ))
        if int(nl.sf_sfclay_physics) in (1, 5):
            modules.add("surface_layer")
        if int(nl.sf_sfclay_physics) == 4 or int(nl.bl_pbl_physics) == 4:
            modules.update(("sfclay_qnse", "pbl_qnse"))
            qnse = True
        if int(nl.sf_sfclay_physics) == 2 or int(nl.bl_pbl_physics) == 2:
            # MYJSFC REAL accumulated-ZETA tables/ZTMAX/DZETA built at import (MYJSFCINIT)
            # and the MYJ/MYJPBL constants (t-cfix-sfc F12).
            modules.update(("sfclay_janjic", "sf_myj", "myj_constants", "bl_myj"))
        if int(nl.sf_sfclay_physics) == 91:
            modules.add("sfclay_old_mm5")  # REAL sfclayinit PSIMTB/PSIHTB (t-cfix-sfc F3)
        if int(nl.sf_sfclay_physics) == 3 or int(nl.bl_pbl_physics) == 3:
            modules.update(("sfclay_gfs", "bl_gfs"))  # SF_GFS/BL_GFS REAL + kind_phys literals (F13)
        if int(nl.sf_sfclay_physics) == 7:
            modules.add("sfclay_pleim_xiu")
        if int(nl.cu_physics) in (1, 99):
            modules.add("cumulus_kf_tables")
        if int(nl.cu_physics) in (10, 11):
            modules.add("cumulus_kfcup_tables")
        if int(nl.cu_physics) in (94, 95):
            modules.add("gfs_sas_shallow")
            gfs_shallow = True
        radiation(sw, lw)
        if 4 in (sw, lw):
            loaders.add(("wrf_cam_ozone", "_load_wrf_cam_ozone_asset"))
        if sw == 4 and int(getattr(nl, "aer_opt", 0) or 0) == 1:
            # aer_opt=1: swaerpr band tables are compile constants; aerodm/pina ride as traced children.
            modules.add("wrf_tegen_aerosol")
            loaders.add(("wrf_tegen_aerosol", "_load_wrf_tegen_aerosol_asset"))
        if sw == 4 and int(getattr(nl, "aer_opt", 0) or 0) == 3:
            # aer_opt=3: gt_aod / SF79 rural spectral tables are compile constants (module-level values).
            modules.update(("wrf_aer_opt3", "wrf_aer_opt3_tables"))
        if lw == 1:
            modules.add("ra_lw_rrtm")
            loaders.add(("ra_lw_rrtm", "_load_tables"))
        if 3 in (sw, lw):
            modules.update(("ra_cam_common", "ra_cam_sw"))
            loaders.add(("ra_cam_common", "load_cam_abs_tables"))
            loaders.add(("wrf_cam_ozone", "_load_wrf_cam_ozone_asset"))
            loaders.update(("ra_cam_sw", name) for name in
                           ("_band_tables", "load_cam_aeropt_tables"))
        if 5 in (sw, lw):
            modules.add("ra_goddard_common")
            loaders.add(("ra_goddard_common", "_host_tables"))
        if 7 in (sw,lw):
            modules.update(("ra_flg","flg_transfer"))
            loaders.add(("ra_flg","_host_tables"))
            legacy_dates.append(str(nl.time_utc))
        if 99 in (sw,lw):
            modules.update(("ra_gfdleta","ra_gfdleta_support","gfdleta_column"))
            loaders.update((("ra_gfdleta","_host_tables"),("_gfdleta_radtn","_host_tables")))
            legacy_dates.append(str(nl.time_utc))
            if nl.gfdl_grid is not None:legacy_grids.append(nl.gfdl_grid)
        if mp in (10, 40) and int(getattr(nl, "do_radar_ref", 0) or 0) == 1:
            # refl10cm_hm init/radar_init constants bound as host globals (t-classd).
            morrison_radar = True
        if mp in (17, 18, 19, 20, 21, 22):
            modules.add("nssl2mom.constants")
            modules.add("nssl2mom.diagnostics")  # REAL Log10 split constants (radar)
            nssl = True
            if mp in (19, 21):
                modules.add("nssl2mom.gs_a")  # Active one-moment _CWR/_GRAD bound tables.
                nssl_one_moment.add(mp)
        if mp in (50, 51, 52, 53):
            modules.add("p3.constants")
            p3_kinds.update(('3mom' if mp == 53 else '2mom', 'warm'))
            if mp in (51, 52):
                p3_kinds.add('ice_ice')
        if mp == 56:
            modules.update("ntu." + name for name in
                           ("constants", "tables", "math", "support", "_native", "column", "initialize", "registry", "adapter"))
            loaders.add(("ntu.aot_tables", "resolved_bindings"))
        if mp == 55:
            # Actual bound IGR/defaults and original-caller argument maps,
            # plus the resolved raw cache used by every typed table promotion.
            modules.update("jensen_ishmael." + name for name in
                           ("math", "level_support", "processes", "final_checks",
                            "shape", "lookup", "vaporgrow", "aggregation",
                            "sedimentation", "finish", "column", "tables", "_native", "registry", "adapter"))
            loaders.add(("jensen_ishmael.tables", "_host_tables"))
            loaders.add(("jensen_ishmael._native", "configuration"))

    memo = {}
    def digest(value):
        if isinstance(value, LazyTable):
            value = value.resolve()  # same digest as a bundle loaded at import
        # IDs are local deduplication only; they never enter the persistent key.
        ident = id(value)
        if ident not in memo:
            if isinstance(value, (tuple, list)):
                item = (type(value).__module__, type(value).__qualname__,
                        tuple(digest(child) for child in value))
            elif isinstance(value, Mapping):
                item = ("dict", tuple((digest(k), digest(v)) for k, v in
                                      sorted(value.items(), key=lambda pair: repr(pair[0]))))
            elif dataclasses.is_dataclass(value) and not isinstance(value, type):
                item = (type(value).__module__, type(value).__qualname__,
                        tuple((field.name, digest(getattr(value, field.name)))
                              for field in dataclasses.fields(value)))
            elif isinstance(value, Path) and value.is_file():
                with value.open('rb') as source:
                    item = ("asset-content", hashlib.file_digest(source, 'sha256').hexdigest())
            elif isinstance(value, Path) and pinned_sha256(value) is not None:
                # Registered table not downloaded yet: same item as after the download.
                item = ("asset-content", pinned_sha256(value))
            else:
                item = value
            memo[ident] = (value, canonical_digest(item))
        return memo[ident][1]

    records = []
    for date in legacy_dates:records.append(('flg_gfdl','initial-date','static-clock',digest(date)))
    for grid in legacy_grids:
        records.append(('gfdleta','actual-bound-provider','host-table',digest(grid.baked_tables())))
    for controls in land_controls:
        records.append(('noahmp_options', 'resolved-MPTABLE-controls', 'bound-values', digest(controls)))
    for name in sorted(modules):
        module = importlib.import_module("gpuwrf.physics." + name)
        for attr, value in sorted(vars(module).items()):
            if attr.startswith("__"):
                continue
            target = getattr(value, "__wrapped__", value)
            if isinstance(target, types.FunctionType) and target.__module__ == module.__name__:
                records.append((name, attr, "defaults", digest((target.__defaults__, target.__kwdefaults__))))
            elif isinstance(value, Path):
                # External binary/source assets used by lazy table loaders.
                if value.is_file() or pinned_sha256(value) is not None:
                    records.append((name, attr, "asset", digest(value)))
            elif attr.isupper() and not isinstance(value, (type, types.ModuleType)) and not callable(value):
                records.append((name, attr, "value", digest(value)))
    if stock_triton:
        from gpuwrf.kernels._triton_kernel_call import stock_compiler_identity
        records.append(("kernels._triton_kernel_call", "stock_compile", "compiler-identity", digest(stock_compiler_identity())))
    for name, attr in sorted(loaders):
        module = importlib.import_module("gpuwrf.physics." + name)
        records.append((name, attr, "host-table", digest(getattr(module, attr)())))
    if cloud_kinds:
        module = importlib.import_module("gpuwrf.physics.rrtmg_mp_re")
        for kind in sorted(cloud_kinds):
            records.append(("rrtmg_mp_re", kind, "raw-cloud-tables", digest(module.raw_cloud_tables(kind))))
    if p3_kinds:
        module = importlib.import_module("gpuwrf.physics.p3.tables")
        for kind in sorted(p3_kinds):
            records.append(("p3.tables", kind, "host-table", digest(module.load_tables(kind))))
    if nssl:
        module = importlib.import_module("gpuwrf.physics.nssl2mom.constants")
        for mode in ('fp32', 'fp64'):
            records.append(("nssl2mom.constants", mode, "resolved-values", digest(vars(module.get_constants(mode)))))
        for variant in sorted(nssl_one_moment):
            for mode in ('fp32', 'fp64'):
                records.append(("nssl2mom.constants", (variant, mode), "resolved-one-moment-values",
                                digest(vars(module.get_one_moment_constants(mode, variant)))))
    if morrison_radar:
        module = importlib.import_module("gpuwrf.diagnostics.morrison_radar")
        records.append(("morrison_radar", "CONSTANTS", "resolved-values", digest(module.CONSTANTS)))
    if qnse:
        surface = importlib.import_module("gpuwrf.physics.sfclay_qnse")
        pbl = importlib.import_module("gpuwrf.physics.pbl_qnse")
        for dtype in ('float32', 'float64'):
            records.append(("sfclay_qnse", dtype, "host-table", digest(surface._tables(dtype))))
            records.append(("sfclay_qnse", dtype, "resolved-values", digest(vars(surface._surface_constants(dtype)))))
            records.append(("pbl_qnse", dtype, "resolved-values", digest(pbl._constants(dtype))))
    if gfs_shallow:
        module = importlib.import_module("gpuwrf.physics.gfs_sas_shallow")
        for mode in ('r4', 'r8'):
            records.append(("gfs_sas_shallow", mode, "host-table", digest(module._tables(mode))))
    return canonical_digest(records)
