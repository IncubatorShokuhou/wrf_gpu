"""archive_ensemble moisture handling (lane a1-da): positive members, exact control mean, linear fields unchanged."""
import numpy as np

from gpuwrf.assimilation.letkf import archive_ensemble as ae


def test_log_members_positive_with_exact_mean_and_scale_zero_identity():
    rng = np.random.default_rng(0)
    centre = 1e-5 * np.exp(rng.normal(size=(4, 5)))
    stack = 1e-5 * np.exp(2.0 * rng.normal(size=(9, 4, 5)))      # huge relative spread (dry levels)
    for s in (0.0, 0.3, 1.0, 2.0):
        m = ae._log_members(centre, stack, s)
        assert (m > 0).all()
        np.testing.assert_allclose(m.mean(axis=0), centre, rtol=1e-12)
        if s == 0.0:
            np.testing.assert_allclose(m, np.broadcast_to(centre, m.shape), rtol=1e-12)
    additive = centre[None] + (stack - stack.mean(0)[None])
    assert (additive < 0).any()                                   # the defect the log route removes


def test_rescale_keeps_mean_and_scales_linear_fields_exactly():
    rng = np.random.default_rng(1)
    frames = {"T": rng.normal(size=(7, 3, 4)), "QVAPOR": np.exp(rng.normal(size=(7, 3, 4))) * 1e-3}
    out = ae.rescale(frames, 0.5)
    np.testing.assert_allclose(out["T"].mean(0), frames["T"].mean(0), atol=1e-6)
    np.testing.assert_allclose(out["T"] - out["T"].mean(0), 0.5 * (frames["T"] - frames["T"].mean(0)), atol=1e-5)
    np.testing.assert_allclose(out["QVAPOR"].mean(0), frames["QVAPOR"].mean(0), rtol=1e-5)
    assert (out["QVAPOR"] > 0).all()


def test_innovation_scale_recovers_known_spread():
    rng = np.random.default_rng(2)
    var = np.full(20000, 4.0)
    d = rng.normal(scale=np.sqrt(1.0 + 0.25 * 4.0), size=20000)    # sigma_o = 1, true s = 0.5
    s, diag = ae.innovation_scale(d, np.ones(20000), var)
    assert abs(s - 0.5) < 0.03 and diag["n"] == 20000
