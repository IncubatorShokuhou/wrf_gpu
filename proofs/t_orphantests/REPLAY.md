# Restored default-feature oracle tests

`RESULT.json` binds the 23 file results, unchanged-bound proof, exact imported blobs, and 20 in-memory deletion controls.
All measurements use immutable snapshots, JAX CPU, cores 24,25 at nice 19, and the official 6 GiB/no-swap RAM slot.
Run each listed suite file in a fresh pytest process (JAX caches otherwise accumulate); use a lane-owned `--basetemp` and disable the persistent JAX cache.
The four skips require the original P0227 d01 restart. No numerical tolerance or assertion was weakened.
The history AST isolation check is rebound to receiving main `1e9f09c07`; original numerical fixtures stay frozen.

To replay a deletion, prepend this directory and the checkout's `src` to PYTHONPATH, set `ORPHAN_MUTANT` to the recorded family, and pass `-p mutation_plugin` plus its recorded node(s) to pytest inside the same CPU/RAM/QUIET guard.
Expected result is a genuine assertion failure with no setup errors; nonzero status by itself is insufficient.
For source_parity use `final_nodes`: the real mean-floor cohorts do not activate the corrected top-face branch; the existing original footprint witness and GTR gates do.

Production src/scripts/data/package trees are identical to the receiving base. Tests-only scope carries no changed kernel launches, device allocations, transfers, synchronization or executable readiness cost.
These CPU oracle/unit/regression results do not replace coupled GPU or forecast release evidence.
