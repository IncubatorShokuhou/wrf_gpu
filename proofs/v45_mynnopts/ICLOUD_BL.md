# icloud_bl=0: existing implementation audit

Base: `3b93841d70943d79417c936696b884c70c832ef4`. Audit commit `2e701b5b8b`
adds no production source changes. The separate PDF implementation is in REPORT.md.

The scalar `&physics icloud_bl=0` is bound by `io/namelist_binding.py`, validated as 0/1,
stored in the static OperationalNamelist tree, and passed by `_rrtmg_pbl_cloud_kwargs` to
all radiation callers. It skips only the WRF radiation-driver MYNN cloud merge. MYNN
still computes its SGS clouds. WRF icloud=0/2/3 bypasses this block independently.

Reuse: `tests/v044/test_icloud_bl.py` extracts the literal pristine WRF merge block;
both 0/1 selections and first/later calls are checked bytewise, with active liquid/ice
operands and unchanged MYNN state. Existing critic deletion mutant kills the merge
wiring (EXPERIMENTS CR03_12_mut_p_icloudbl, 2026-10-25). The existing fixture is
Swiss/Canary winter columns plus active structured BL condensate; **this is not a
new day/night PROD-column claim**. Prior GPU combined run and limitations are in
`docs/radiation_pbl_clouds.md` and `proofs/w44_raddiag/RADDIAG.md`.

Default identity: this commit changes no src bytes, including constants and native
kernel bodies; E222/E224 therefore carry unchanged. It adds no launches, state,
allocation, traffic, synchronization or readiness cost [I, source identity].

The manager's 2026-10-27 23:08Z amendment permits the labelled gate-case WN3/Swiss
evidence. V045 rechecked all four literal merge combinations on the composed source.
Separately timed native OFF/ON arms and final coupled qualification remain pending;
do not infer cloud skill or isolated speed from combined measurements.
