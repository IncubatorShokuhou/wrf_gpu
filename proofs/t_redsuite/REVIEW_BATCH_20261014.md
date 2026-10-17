# Verified batch for one critic

Frozen range: `d185e9dce..4ee280755169dd74a92976530e3c39373fb81fe5`. Includes `bc62dfc89`.
39 commits; 35 test commits; 51 test files. No production changes.
Selected unique test outcomes: {'passed': 721, 'xfailed': 1, 'skipped': 28}. Original failed attempts and retry chains remain in receipts.

Receipts: `<USER_HOME>/wrf_gpu2_lanes/t-redsuite/RS03`. `REVIEW_BATCH_20261014.json` binds original command/XML SHA256, immutable source commits, exact kept file hashes and selected test nodes.

| Commit | Summary | Receipts |
|---|---|---|
| 8356e0efd | proof: extend verified batch to 174 passes, 5 skips, 3 inherited xfails | proof/notes; see manifest |
| 0b92840d1 | test: keep extended State on selected backend; 2 pass | C004c2-M6-FIFO |
| cd0b99737 | test: use original WRF EOS target for warm bubble; 2 pass | C004c2-M6-FIFO |
| 4db3b5a44 | test: recognize WRF REAL scheme carries; registry retry passes | C004c3-Jensen-precision |
| 34c2f0423 | test: use pristine WRF relaxation stencil; 3 pass, 1 inherited xfail | C004d-M7 |
| d8f7bd35e | proof: bind four more repaired reds; 9 pass and 1 inherited xfail | proof/notes; see manifest |
| bc62dfc89 | test: supply valid WRF acoustic fixtures; 2 caller checks pass | C004d3-pure-eta-callers |
| f4f648012 | proof: preserve M7 fixture diagnosis and 2 passing callers | proof/notes; see manifest |
| 4d9b9646e | test: align graduated admission expectations; 11 checks verified | C004e2-feature-classifier |
| 172f6ab2f | test: recognize qualified GFS SAS beside Tiedtke; 16 pass | C005 |
| 4c33a3143 | test: complete RUC and radar selector mocks; 44 pass | C005 |
| 745328b2f | test: align nesting and KF spies with live interfaces; 34 pass | C005 |
| 82c8bacfa | test: update theta restart and aerosol source pins; 17 pass | C005 |
| f1682f7e9 | test: isolate CUDA sizing metadata on CPU; 46 pass | C006 |
| 547293c60 | test: scope native RK import flags and caches; 49 pass | C006 |
| ab7bb26ae | test: isolate regional stream JIT census; 57 pass | C007 |
| 85f93a30a | test: pin paired FLG rejection in LW wiring; 13 pass | C008 |
| d414501fd | test: report recorded DFI qualification scope; 37 pass | C009 |
| a53dc1da6 | test: recognize declared SBU fp64 adapter island; 3 pass, 19 skip | C009 |
| d8bd77550 | test: give stochastic binding mock real CU default; 27 pass | C009 |
| ae30256a0 | test: seed preceding restart schema in tslist mock; 36 pass | C010 |
| 40e95284c | test: recognize experimental SSiB land selection; 42 pass, 1 skip | C010 |
| 5f3566396 | test: preserve append-only State and edge mock contracts; 35 pass | C010 |
| 4973d4505 | test: mark inactive ice categories in PSFC mock; strict integral passes | C010b-inactive-ice |
| f3fa483fb | test: preserve frozen State prefixes and active leaves; 12 pass | C011 |
| 33f46be4b | test: reflect recorded scheme graduations; 21 pass, 6 historical skips | C011 |
| d438812e1 | test: complete operational and partial NetCDF mocks; 49 pass | C011 |
| ffa3ce383 | test: bind historical ancestry to original private authority; 19 pass, 2 skips | C011 |
| 417cdd9ab | test: pin dynamic scalar selector with Thompson identity; 5 pass | C011 |
| 35a39ce0f | test: locate the unique urban holder by identity; 15 pass | C012 |
| 98666ca65 | test: preserve MP radii prefix and startup context; 13 pass | C012 |
| 060d263a4 | test: complete inactive ice context in PG spy; exact mu check passes | C012b-PG-inactive-ice |
| 74fb3252c | test: narrow albedo deletion to its real seed; 15 oracle checks pass | C013 |
| cd28e2d1c | test: pin original B39 carry independently of later leaves; 7 pass | C014 |
| b5b826f9b | test: complete cloud and radiation cadence mocks; 13 pass | C014 |
| b4af113b2 | test: honor CPU McICA fallback with exact draws; 4 pass | C014 |
| f5f227119 | test: pin held WSPD bytes and one surface producer; 48 pass | C014 |
| a9f7b978f | test: target current typed Thompson mutation sites; 31 oracle checks pass | C014 |
| 4ee280755 | proof: preserve kept repair inventory and open hail counterexample | proof/notes; see manifest |

Full-suite gate remains **INCOMPLETE**. Dirty files and later commits are excluded. New-main composition and remaining chunks are pending. MP28/MP40 require authorized post-Nightly-DONE 12 GiB singleton runs. Supplemental active-hail PSFC counterexample remains an open product red (90 Pa deficit); owner requested via MAILBOX.

No production source changes: no model launch/allocation/traffic/compile changes. Test cache isolation adds CPU fixture compilation, recorded in original receipts; no GPU speed claim.
