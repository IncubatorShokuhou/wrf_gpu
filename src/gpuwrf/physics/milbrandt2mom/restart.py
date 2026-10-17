"""Required held MY2 radar in the State-checkpoint API (E78)."""


def validate_checkpoint_radar(payload):
    from gpuwrf.io.restart_schema import physics_option
    namelist=payload.get('namelist')
    mp=physics_option(namelist,'mp_physics',8)
    if mp not in (9,50,51,52,53):return
    from gpuwrf.diagnostics.online import configuration,validate_restart
    runtime=payload.get('runtime_state')
    radar=getattr(runtime,'radar_reflectivity',None)
    if radar is None:
        scheme='Milbrandt' if mp==9 else 'P3'
        raise ValueError(f'{scheme} checkpoint requires full runtime_state with held radar (E78)')
    # P3 uses the existing full runtime carry and checkpoint format. Its
    # namelist selection makes this already-defined radar leaf mandatory.
    version=payload.get('milbrandt_radar_schema_version') if mp==9 else 1
    validate_restart(None,radar,configuration(namelist),version,
                     payload['state_fields']['qv'].shape)
