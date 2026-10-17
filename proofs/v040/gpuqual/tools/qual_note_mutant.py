"""Deletion-sensitive check: erase the actual catalog qualification at setup."""
def pytest_runtest_setup(item):
    if item.name == 'test_notes_match_recorded_gpu_qualification[cu_physics-5]':
        from gpuwrf.io import scheme_catalog
        key = ('cu_physics', 5)
        before = scheme_catalog._IMPLEMENTED_NOTES[key]
        assert 'GPU-qualified vs CPU-WRF' in before
        scheme_catalog._IMPLEMENTED_NOTES[key] = before.replace(
            'GPU-qualified vs CPU-WRF', 'qualification pending vs CPU-WRF')
