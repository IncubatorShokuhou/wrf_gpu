"""Genuine E39/E202 mutation: delete the fair denominator correction in memory."""


def pytest_configure(config):
    from gpuwrf.ens import statistics

    original = statistics.crps
    def empirical_only(values, observations, *, fair=False):
        return original(values, observations, fair=False)
    statistics.crps = empirical_only
