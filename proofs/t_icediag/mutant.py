"""Genuine wrong-category/quantity mutation, without changing oracle truth."""
def pytest_configure(config):
    from gpuwrf.physics.p3 import adapter
    original = adapter.output_mapping

    def swapped(mp):
        result = original(mp)
        result['di'], result['rhoi'] = result['rhoi'], result['di']
        return result

    adapter.output_mapping = swapped
