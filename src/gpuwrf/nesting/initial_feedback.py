"""WRF med_nest_initial feedback and parent start_domain, initialization only.

No reference history is read. Inputs are the native case and initialized child;
static-grid and land setup complete before the first timestep is compiled.
"""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.coupling.boundary_feedback import (
    _feedback_leaf, apply_carry_feedback, build_state_feedback_weights,
    point_feedback_weights,
)
from gpuwrf.coupling.feedback_registry import FEEDBACK_FIELDS
from gpuwrf.io.netcdf_lock import Dataset


def _input_point_overrides(run, parent, child, weights):
    """WRF initial feedback samples pre-start input land fields, not Noah solves."""
    overrides={}
    from gpuwrf.coupling.feedback_aux import FLUXES
    surface_fields={'T2','Q2','U10','V10','PSFC','TH2',*FLUXES}
    with Dataset(run.path/f"wrfinput_{parent}") as pd, Dataset(run.path/f"wrfinput_{child}") as cd:
        for name in set(pd.variables)&set(cd.variables)&set(FEEDBACK_FIELDS):
            method,smooth,stagger,_=FEEDBACK_FIELDS[name]
            if method not in ("copy_fcnm","copy_fcni") and name not in surface_fields:
                continue
            geometry=weights.u if stagger=='X' else weights.v if stagger=='Y' else weights.mass
            pv,cv=pd[name],cd[name]
            if not pv.dimensions or not cv.dimensions:
                continue
            pa=np.asarray(pv[0] if pv.dimensions[0]=="Time" else pv[:])
            ca=np.asarray(cv[0] if cv.dimensions[0]=="Time" else cv[:])
            if pa.ndim not in (2,3) or ca.ndim!=pa.ndim:
                continue
            if pa.shape[-2:]!=(geometry.psn,geometry.pwe) or ca.shape[-2:]!=(geometry.csn,geometry.cwe):
                continue
            plan=((weights.point if geometry is weights.mass and weights.point is not None
                   else point_feedback_weights(geometry)) if method in ('copy_fcnm','copy_fcni') else geometry)
            overrides[name]=_feedback_leaf(jnp.asarray(pa),jnp.asarray(ca),plan,None,0)
    if "XLAND" in overrides:
        # LANDMASK is the port's internal land-tile adapter, WRF uses XLAND.
        overrides["LANDMASK"]=(overrides["XLAND"]<1.5).astype(jnp.float64)
    return overrides


def initialize_two_way_domains(hierarchy,bundles,carries,run,nesting,*,kernel_call=None):
    """Complete initialization feedback before runtime/cache keys are frozen."""
    from gpuwrf.integration import d02_replay as init
    from gpuwrf.io.noahmp_land_init import build_noahmp_land_state,build_noahmp_params
    from gpuwrf.kernels.dyn_carry_fp32 import real_carry
    from gpuwrf.runtime.operational_mode import _initial_carry_for_run,noahmp_initial_rad,_base_state_from_totals
    from gpuwrf.runtime.operational_state import history_instep_diag_enabled,seed_history_diagnostics

    reports={}
    topo_weights={}
    for edge in hierarchy.nests:
        parent,child=edge.parent,edge.child
        pb,cb=bundles[parent],bundles[child]
        pc,cc=carries[parent],carries[child]
        # The legacy CPU path has no resident base until feedback needs it.
        if pc.base_state is None:
            pc=pc.replace(base_state=_base_state_from_totals(pc.state))
        if cc.base_state is None:
            cc=cc.replace(base_state=_base_state_from_totals(cc.state))
        _,_,_,child_tinit,child_alb=init._wrf_start_domain_base_from_hgt(
            run,child,hgt=cb.grid.terrain_height,metrics=cb.metrics)
        cc=cc.replace(base_state=cc.base_state.replace(alb=child_alb.astype(jnp.float32),
            theta_base=(child_tinit+300.).astype(cc.base_state.theta_base.dtype)))
        weights=build_state_feedback_weights(parent_grid_ratio=edge.parent_grid_ratio,
            i_parent_start=edge.i_parent_start,j_parent_start=edge.j_parent_start,
            parent_grid=pb.grid,child_grid=cb.grid,spec_zone=nesting["spec_zone"],smooth_option=nesting["smooth_option"])
        topo_weights[(parent,child)]=weights.mass
        # med_nest_initial calls feedback before start_domain(child): MU/W
        # still have their input values, while T/QV already passed adjust_tempqv.
        with Dataset(run.path/f"wrfinput_{child}") as ds:
            mu=jnp.asarray(np.asarray(ds["MU"][0]),dtype=cc.state.mu_perturbation.dtype)
            w=jnp.asarray(np.asarray(ds["W"][0]),dtype=cc.state.w.dtype)
        pre=cc.replace(state=cc.state.replace(_cast=False,mu_perturbation=mu,
            mu_total=cc.base_state.mub+mu,w=w))
        fed=apply_carry_feedback(pc,pre,weights)
        option=nesting["smooth_option"]
        hgt=_feedback_leaf(pb.grid.terrain_height.astype(jnp.float32),cb.grid.terrain_height.astype(jnp.float32),
            weights.mass,weights.mass_smooth,option).astype(pb.grid.terrain_height.dtype)
        metrics={}
        for name in ("msftx","msfty","msfux","msfuy","msfvx","msfvy","f","e","sina","cosa"):
            plan=weights.u if name.startswith("msfu") else weights.v if name.startswith("msfv") else weights.mass
            # Every represented map/Coriolis/rotation metric is explicitly
            # u=(copy_fcnm) in Registry.EM_COMMON, including F and E.
            plan=point_feedback_weights(plan)
            metrics[name]=_feedback_leaf(getattr(pb.metrics,name).astype(jnp.float32),
                getattr(cb.metrics,name).astype(jnp.float32),plan,None,0).astype(getattr(pb.metrics,name).dtype)
        m=replace(pb.metrics,**metrics)
        m=init._metrics_with_terrain(m,terrain_height=hgt,grid=pb.grid,provenance_suffix="WRF med_nest_initial feedback")
        terrain=replace(pb.grid.terrain,source_path=pb.grid.terrain.source_path+";initial_feedback",
                        max_elevation_m=float(np.asarray(hgt).max()))
        grid=replace(pb.grid,terrain=terrain,terrain_height=hgt,metrics=m)
        bpressure,bmu,bphi,btinit,balb=init._wrf_start_domain_base_from_hgt(run,parent,hgt=hgt,metrics=m)
        bpressure=bpressure.astype(pc.base_state.pb.dtype)
        bmu=bmu.astype(pc.base_state.mub.dtype)
        bphi=bphi.astype(pc.base_state.phb.dtype)
        theta_base=btinit+300.
        base=pc.base_state.replace(pb=bpressure,phb=bphi,mub=bmu,
            alb=balb.astype(jnp.float32),
            theta_base=theta_base.astype(pc.base_state.theta_base.dtype))
        pressure,mu,w,_=init._wrf_live_nest_start_domain_perturb_init(run,domain=parent,grid=grid,metrics=m,
            ph_perturbation=fed.state.ph_perturbation,mu_perturbation=fed.state.mu_perturbation,
            theta_full=fed.state.theta,w=fed.state.w,u=fed.state.u,v=fed.state.v,
            ht_fine=hgt,base_pb=bpressure,base_mub=bmu,base_phb=bphi,press_adjust=False)
        state=fed.state.replace(_cast=False,p_perturbation=pressure.astype(pc.state.p_perturbation.dtype),
            p_total=bpressure+pressure,ph_total=bphi+fed.state.ph_perturbation,
            mu_perturbation=mu.astype(pc.state.mu_perturbation.dtype),mu_total=bmu+mu,w=w.astype(pc.state.w.dtype))
        overrides=_input_point_overrides(run,parent,child,weights)
        from gpuwrf.io.land_state import load_prescribed_land_state
        surface=load_prescribed_land_state(run,parent,field_overrides=overrides)
        state=state.replace(_cast=False,roughness_m=surface.roughness_m.astype(state.roughness_m.dtype),
            mavail=surface.mavail.astype(state.mavail.dtype),lakemask=surface.lakemask.astype(state.lakemask.dtype))
        from gpuwrf.io.radiation_static import load_radiation_static
        radiation_static,_=load_radiation_static(run,parent,grid=grid,metrics=m)
        from gpuwrf.io.gwdo_static import load_gwdo_statics
        gwdo_static=pb.namelist.gwdo_statics
        if int(pb.namelist.gwd_opt)==1:
            gwdo_static,_=load_gwdo_statics(run,parent,grid=grid,metrics=m,field_overrides=overrides)
        nl=replace(pb.namelist,grid=grid,metrics=m,radiation_static=radiation_static,gwdo_statics=gwdo_static)
        land=None
        if pc.noahmp_land is not None:
            land,static,land_meta=build_noahmp_land_state(run.path,parent,field_overrides=overrides)
            energy,rad,nroot=build_noahmp_params(static)
            nl=replace(nl,noahmp_static=static,noahmp_land=land,
                noahmp_energy_params=energy,noahmp_rad_params=rad,noahmp_nroot=nroot)
        new=_initial_carry_for_run(state,nl).replace(base_state=base,census=pc.census)
        if land is not None:
            surface,full=noahmp_initial_rad(new.state,nl,land_state=land,_with_diagnostics=True,_kernel_call=kernel_call)
            from gpuwrf.coupling.topographic_shadow import preserve_initial_height
            full=preserve_initial_height(new.radiation_diagnostics,full)
            new=new.replace(noahmp_land=land,noahmp_rad=surface,
                radiation_diagnostics=full if new.radiation_diagnostics is not None else None)
            from gpuwrf.diagnostics.census import count_work
            new=new.replace(census=count_work(new.census,"radiation_init_calls"))
            if history_instep_diag_enabled():
                new=new.replace(history_diagnostics=seed_history_diagnostics(new.state))
        new=real_carry(new)
        carries[parent]=new
        carries[child]=cc
        bundles[parent]=replace(pb,state=new.state,namelist=nl,grid=grid,metrics=m,
            writer_static_overrides=overrides)
        if "XLAND" in overrides:
            topo_weights[("xland",parent)]=overrides["XLAND"]
        reports[parent]={"child":child,"source":"mediation_integrate.F med_nest_initial: feedback then start_domain(parent)",
                         "static_override_fields":sorted(overrides),"static_updates_before_steps":True,
                         "radiation_refresh_calls":int(land is not None)}
    _two_way_topo_wind(hierarchy,bundles,carries,run,topo_weights)
    from gpuwrf.coupling.feedback_aux import seed_feedback_history
    for domain,carry in carries.items():
        if carry.history_diagnostics is None:
            raise ValueError('two-way feedback requires GPUWRF_HISTORY_INSTEP_DIAG=1')
        carries[domain]=carry.replace(feedback_history=seed_feedback_history(
            run.path,domain,bundles[domain],carry))
    return reports


def _two_way_topo_wind(hierarchy,bundles,carries,run,topo_weights):
    """Registry ctopo/ctopo2 'u' feedback (io.topo_wind_static.feedback_topo_wind_statics).

    start_domain(parent) after med_nest_feedback recomputes CTOPO/CTOPO2 from the fed-back
    HGT/XLAND (VAR_SSO/VAR are not fed back); each parent step's feedback then writes the
    child CTOPO/CTOPO2 over the nest. No-op unless some domain sets topo_wind."""
    from gpuwrf.integration.nested_pipeline import _domain_physics_int
    from gpuwrf.io.topo_wind_static import domain_ctopo_fields,feedback_topo_wind_statics
    names=list(bundles)
    option={d:_domain_physics_int(run,"topo_wind",d,0) for d in names}
    if not any(option.values()):
        return
    own={d:domain_ctopo_fields(run,d,topo_wind=option[d],ht=bundles[d].grid.terrain_height,
                               xland=topo_weights.get(("xland",d),carries[d].state.xland)) for d in names}
    depth=lambda d:0 if hierarchy.parent(d) is None else 1+depth(hierarchy.parent(d))
    edges=sorted(((e.parent,e.child) for e in hierarchy.nests),key=lambda e:-depth(e[1]))
    statics=feedback_topo_wind_statics(own,{d:int(bundles[d].namelist.bl_pbl_physics) for d in names},
                                       edges,{e:topo_weights[e] for e in edges})
    from gpuwrf.integration.d02_replay import wrf_nest_opens_with_parent_start
    opens_at_start=lambda d:hierarchy.parent(d) is None or (
        wrf_nest_opens_with_parent_start(run,hierarchy.parent(d),d) and opens_at_start(hierarchy.parent(d)))
    for d in names:
        if statics[d] is not None and statics[d].ctopo_first is not None and not opens_at_start(d):
            # operational_mode._topo_wind_ctopo keys the own-value first step on lead 0.
            raise NotImplementedError(f"{d}: topo_wind CTOPO feedback onto a nest opened after the run start")
        bundles[d]=replace(bundles[d],namelist=replace(bundles[d].namelist,topo_wind_static=statics[d]))
