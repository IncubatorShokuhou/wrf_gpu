# Pristine initialization sensitivity

No uninitialized-output dependency was observed in the active Milbrandt fixture.
MY32 compiled byte-verbatim `my2_mod` from pristine source SHA256
`76d11c00976f3c7dac30ff6ebb3efac1698bf43c13131b3aae43e464f22d23e5` twice,
using the same `-O2 -ftree-vectorize -funroll-loops -fcheck=bounds` flags.
The compiler initialization modes were REAL zero / INTEGER 0 / LOGICAL false
and REAL signaling NaN / INTEGER 42 / LOGICAL true, with traps disabled.
All 3168 output values were finite and byte-identical: 14 prognostics,
11 precipitation rates, six discarded diameter profiles, ZET and ZEC,
over 12 active cold/hail columns of 12 levels at DT=18 s.

Reproduce with `oracle/init_sensitivity.py BUILD tests/v040/milbrandt/literal_real4.npz OUT.json`.
The generated port initializes masked temporaries to zero so every vector branch
has defined operands. This probe found no scientific output dependence on those
initial values; it does not prove absence of undefined reads in other regimes.
No pristine body was edited and no upstream bug is claimed from this probe.
