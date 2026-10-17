The TP25b/TP27 namelist staging incident

The first recipes copied symlinks from a producer CPU directory, including
`namelist.input -> ../input/namelist.input`, then wrote the short-run namelist
through that link. TP25b changed v-prep/c00, and TP27 changed c00/c21/c22.
Only time/history controls changed. This was a worker error, not authorization
to modify producer cases. The manager and v-prep received the exact command and
paths in MAIN MAILBOX at 2026-10-14T22:24:10Z.

v-prep restored the three authoritative tracked namelists at 22:25:47Z and
preserved the rewritten versions with provenance receipts. This worker did not
restore producer files. All offending links in the lane were removed and
replaced by regular owned files. The exact short-run namelist bytes used by
TP27 were reconstructed from the sealed originals and match all three hashes
in the original TP27 receipt. The TP25b used namelist hash matches too. The
numerical results and original receipts were not edited or rerun.

`stage_case.py` now excludes mutable filenames before linking static assets
and creates namelists with exclusive creation, which refuses any existing
symlink. `test_stage_case.py` checks the former foreign-symlink layout, six
mutable output aliases, and refusal of a pre-existing hostile namelist link.
The checked-in recipes have this guard. Earlier revisions retain the failed
staging code as the historical record. Full-reference runs additionally bind
their private namelist to the original CPU_RUN sealed hash.

External receipts: `<USER_HOME>/wrf_gpu2_lanes/t-p3fix/NML_INCIDENT.json`,
`NML_USED_BYTE_RECOVERY.json`, and `STAGE_GUARD.log`.
