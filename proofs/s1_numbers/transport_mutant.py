"""Pytest plugin deleting the newly transported number species only."""
def pytest_collection_modifyitems(session, config, items):
    from gpuwrf.runtime import operational_mode as op
    original = op._advected_scalar_species
    def missing(namelist):
        species = original(namelist)
        if int(namelist.mp_physics) in (10, 14, 16, 26):
            species = tuple(n for n in species if n not in {'Ns', 'Ng', 'Nc', 'Nr', 'Nn'})
        return species
    op._advected_scalar_species = missing
