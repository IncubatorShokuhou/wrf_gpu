"""Fresh-process host EKI scaling probe. Synthetic algebra only, no forecast/skill claims.

Run each shape as a separate process, pinned and memory-capped by the caller; append its JSON.
"""
import argparse
import json
import resource
import time

import numpy as np

from gpuwrf.surface_params import eki


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", type=int, required=True)
    parser.add_argument("--members", type=int, default=64)
    parser.add_argument("--block", type=int, default=0)
    args = parser.parse_args()
    m, n, p = args.observations, args.members, 4
    rng = np.random.default_rng(18)
    theta = rng.normal(size=(n, p))
    predictions = rng.normal(size=(n, m))
    y = rng.normal(size=m)
    covariance = np.linspace(.5, 2., m)
    if args.block:
        blocks = []
        for start in range(0, m, args.block):
            k = min(args.block, m-start)
            blocks.append(.8*np.eye(k) + .2*np.ones((k, k)))
        covariance = eki.BlockDiagonalCovariance(tuple(blocks))
    problem = eki.EKIProblem(tuple(f"p{k}" for k in range(p)), np.zeros(p), np.eye(p), y, covariance)
    times, checksums = [], []
    state = eki.EKIState(theta)
    for _ in range(3):
        start = time.perf_counter()
        new = eki.update(problem, state, predictions, dt=.2, seed=19)
        times.append(time.perf_counter()-start)
        assert np.all(np.isfinite(new.theta))
        checksums.append(float(new.theta.sum()))
    assert len(set(checksums)) == 1
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # Linux KiB
    result = dict(observations=m, members=n, parameters=p, block=args.block,
                  median_update_s=float(np.median(times)), update_s=times,
                  peak_rss_bytes=peak, checksum=checksums[0],
                  perturb=True, tikhonov=True, dt=.2, dtype="float64", cache="NONE",
                  forecast=False, platform="CPU NumPy", limit_bytes=2_000_000_000)
    print(json.dumps(result))
    assert peak <= 2_000_000_000, result


if __name__ == "__main__":
    main()
