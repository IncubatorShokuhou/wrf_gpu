# Original NSSL compiler initialization

WRF4.7.1 module_mp_nssl_2mom.F wet-growth iteration around19971 reads local graupel axx/bxx where no graupel is present (existing o1-nssl finding). The port carries zeros at those sites; pristine bodies remain unchanged. Oracle builds define local REAL/integer/logical initialization as zero/0/false, with production-style FP traps off.

For legacy22, MPM22S compiled the unchanged original REAL4 module with `-finit-real=inf` instead of zero, keeping all other flags and all17 inputs fixed. All returned columns and scalars were BYTE-identical in17/17 cases. This is a finite corpus sensitivity check, not a general proof about undefined original memory or an R8 sensitivity claim. The independent initializer receipt is source-sealed in nssl22/STATUS22.json.

For legacy19/21, MPM1R repeats that REAL4 sensitivity control: all returned
columns/scalars are BYTE-identical in34/34 cases. Original source remains
unchanged. The one-moment cleanup does not use undefined wet-growth scratch
values as prognostic fields. See the original input and stage manifests under
nssl1mom/ and the lane receipt MPM1R-init-sensitivity.json.

MPM1V independently compiles the unchanged original REAL4 module with FMA
contraction enabled and disabled. The caller reads identical frozen raw input
bytes in both arms; the no-FMA output is BYTE-identical to the reference corpus.
For case16 cloud water at the first level, the original spread is
3.070793809456518e-6 to3.104321422142675e-6 for both19 and21. The compiled port's
3.0880706e-6 lies inside it. Only that value uses this independently derived
spread; every other value retains the frozen NSSL gate. The tracked
nssl1mom/fma/manifest.json seals inputs, original source, compiled objects and
controls; scripts reproduce the experiment. This is compiler-rounding
qualification under the standing policy, not a full-forecast D6 result.
