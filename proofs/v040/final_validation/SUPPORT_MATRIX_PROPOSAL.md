# Support-matrix updates proposed by v-prep

Authority: explicit Owner decisions in the lane chat2026-10-14/15. Manager owns
the release support matrix/catalog integration; no model/admission code changed
in this lane.

| Option | Qualification label | Final GPU validation scope |
|---|---|---|
| `km_opt=3` / `diff_opt=2` | Experimental, column-oracle-qualified; no coupled reference: original CPU-WRF crashes (SIGSEGV ~00:04) on Swiss and Madrid with km_opt=3 | Owner-excluded from the full-coverage GPU gate; no GPU coupled qualification claim |
| API-only options, including LSM1/2/7 | API/CPU-oracle-qualified | Owner-approved CLI-only matrix: no GPU runs for these options |
| MP15 | Experimental, column-oracle-qualified; no coupled reference: original CPU-WRF4.7.1 SIGSEGV at first output in canonical and compiler-zero profiles | Owner-excluded from the 3-D coverage gate; see UPSTREAM_WRF_BUGS.md |
| Noah crop1, tile drainage1/2, geographically selected irrigation maps | Experimental: needs geographically valid NCAR maps, no 3-D validation | Column-oracle-qualified; coupled map-driven qualification pending; Owner-excluded unavailable map branches |
| Noah soil2/3 and explicit irrigation1/2/3 | Genuine Madrid maps and matched original-WRF24h references prepared | Owner-accepted coupled targets c71/c72/c73; GPU/D6 qualification recorded separately |
| DFL/DFI1 | Original24h reference produces24 natural frames,00:09 through23:09 | Owner-accepted count24 at the exact original Times; no padding or time-warp; GPU must match |
| MYNN/PBL5 references | Manager22:51 reference policy: preserve raw originals; numerical D6 failures use zero-init original build9a67 with original-vs-zero spread disclosed | c00/c17 original repeat spotchecks passed18075fieldframes; c21–24 separate zero refs complete. Frozen D6 bounds unchanged |
| Jensen55 / NTU56 | OriginalCPU25finite references complete; c87/c88 compiler qualification pending | t-ntucompile owns the native compiler fix. Earlier GPU compiler caps and c88 exact-cache attempt remain INCOMPLETE; no unchanged cold reruns or GPU qualification claim |
| CAM-MG11 | OriginalCPU25finite reference complete; native precipitation defect remains open | c89 actualGPU15partial frames has zeroRAINNC/SNOWNC versus activeoriginal precipitation (h14max17.128628mm), causing4bounded RAINNC failures. t-rainnc/t-cammgws own the fix; CPU-only fallback awaits manager integration decision |
| Separate-hail cases c07/c12/c13/c14/c17/c18/c64/c88 | Final qh-mass fix science verified on all listed cases except compiler-blocked c88 | Per-case D6/E41 disclosures retained. c18 final WDM7 radius/caller source de374 has full25 D6+E41 PASS; c07 final Shin-Hong source44b7 has full25 D6PASS/E41FAIL. These are source-bound results, not blanket latest-main qualification |
| Mixed composites c05/c06 | c05 ACM2 product 5131ddffb and c06 WSM caller/radius de374: each full25 original-CPU D6PASS, zero bounded misses/nonfinite. c05 RAINNC h24 RMSE 0.129037 mm vs frozen1.0 (old1.108272 FAIL SHA archived). | Full CPU inventory; actual E41FAIL both disclosed/F2OPEN. c05 singleton cold wall671.913s, sampled5380MiB; not S1 or a throughput comparison. Source-bound paid evidence. |
| c10 MP14 composite | Final WDM5 radius/caller source de374: full25 D6PASS/0bounded/NF0 | Actual E41FAIL disclosed; full24h numerical qualification is complete on the measured source |
| c04 WSM3 | Final cold-phase/radius source061d53020: full25 D6PASS/0bounded/NF0 | Actual E41FAIL disclosed. Earlier de374 result is archived and does not qualify the later WSM3 phase fix |
| MP16 WDM6, c11 | Full25-frame originalCPU D6PASS/0bounded/NF0; actual E41FAIL disclosed. [M]whole warm run2634.240s/24=109.760s/fc-h, sampledpeakVRAM7050MiB; roughly4× slower than peer composites [I] | Completed one continuous manager-approved3300s/55min arm on exactde374 and original verified normal cache. Original30min timeout monitor exit124 is retained beside actual GPU modelrc0/full25 and the approved-cap receipt. Speed is not a release blocker; manager logs optimization for v0.4.1. Not an S1 benchmark |
| Goddard radiation5/5, c45 | Experimental, validated (6h full +24h minus last frame); tile2048 fix removed the historical slowdown. Manager06:33 accepts this scope for v0.4.0 |7frames/346CPUfields/frozenD6PASS/0bounded/NF0/E41PASS; capped24h attempt has24passingframes(h0–23), withh24missing. Full24h clock remains INCOMPLETE; isolatedpost-fixforecast/S1 speed unmeasured. No secondGod24h GPU arm |
| Active K1 TR17, c90 | Matched pristine-real initialization and original24h reference, paidc46GPU initialcarry/constants identical except2kernel-unusedprovenance paths | FullD6PASS25/401fields/0bounded/NF0; TR17REPORT_ONLY overallRMSE6.00146e-6/maxabs9.93975e-4. This adds no tracer tolerance or independent tracer-fidelity PASS |
| Surface FDDA1/2, c77/c78 | Host admission fix merged; both full25 D6+E41 PASS/0bounded/NF0 | Earlier startup TypeError attempts preserved; source-bound full24h qualification complete |

The official NCAR archives total949368423B (<5GB). Original Madrid WPS/real
populates SOILCL/SOILCOMP/IRFRACT. Its crop, drainage and irrigation-method maps
are outside the datasets' geographic coverage; WPS calendar fill values are not
observed crop inputs. Preserve the one authentic Madrid case and its NCAR index
hashes. Future map-driven qualification needs an authentic covered geography.
Explicit sprinkler and flood references exercise171 cells each; the micro-
irrigation reference has zero events in this weather, disclosed as inactive.

KM3 evidence: preserved historical `proofs/v040/gpuqual/km3` Swiss and Madrid
failures; fresh compiler-zero reference
`<USER_HOME>/wrf_gpu2_lanes/v-prep/c48/cpu_run.json` (rc139,1frame,29.6s;
SIGSEGV near00:04:12). The zero-build receipt binds400unchanged source bodies
and records the inactive canonical Morrison-object exception. No field
tolerance, scientific threshold or numerical PASS was changed.

`aer_init_opt=1` is a recognized programmatic control, but WRF declares this
key **derived**, and the native CLI honour check rejects it as an explicit
namelist key. Keep it API-only in this CLI coverage inventory. MP38 additionally
requires derived `aer_init_opt=0`. Its refused preparation attempt is retained;
no GPU run is scheduled for an unadmitted explicit request.
