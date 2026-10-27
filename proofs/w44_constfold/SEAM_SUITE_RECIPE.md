# w44-constfold release-defaults touched-seam suite (for the w44-release RC tiers)

Critic condition (cr-w44-constfold 22:10Z, E76): the touched-seam subset must show 0 NEW failures with
`GPUWRF_WRF_REAL_CONSTFOLD` ON vs OFF under the release defaults. The manager (22:12Z) moved this run into the
w44-release RC tiers, on the composed candidate.

- **Files.** `SEAM_SUITE_FILES.txt`: 124 test files. They are every test that imports a module changed by this lane,
  plus `tests/test_w44_constfold.py` and `tests/test_fast_defaults.py`.
- **Two arms**, same tree and same env otherwise:
  - OFF: `JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 GPUWRF_FAST_DEFAULTS=1 GPUWRF_WRF_REAL_CONSTFOLD=0 python -m pytest -q
    -p no:cacheprovider --junitxml OFF.xml $(cat SEAM_SUITE_FILES.txt)`
  - ON: the same command with `GPUWRF_WRF_REAL_CONSTFOLD=1` and `--junitxml ON.xml`.
- **Diff.** `python proofs/w44_constfold/tools/junit_diff.py OFF.xml ON.xml DIFF.json` lists every test whose outcome
  changes.
  - Gate: 0 tests passed → failed/error.
  - Pre-existing release-default failures (E144) are equal in both arms and do not count.
- **Cost [M, partial run 2026-10-24].** About 11 % of the files ran in 12 min, so expect ≈ 2 h per arm on 3 niced cores.
- **When the RC switches the flag on**, two edits are needed:
  - add `"GPUWRF_WRF_REAL_CONSTFOLD": "1"` to `FAST_PATH_DEFAULTS`;
  - add the key to `_NOT_IN_LW9_FILE` in `tests/test_fast_defaults.py`.
- **Expected RC identities with the flag ON**, from this lane alone (topo/boundary-clock changes come on top):
  - canonical: Swiss 91d91b50, PROD 5451bf2e, WN3-0227 d01 c37b0069;
  - production raw HLO: 1cfbd158 / 7c223677 / 9ecc2104.
