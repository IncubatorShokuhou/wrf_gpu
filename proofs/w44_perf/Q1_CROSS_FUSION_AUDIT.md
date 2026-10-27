# Q1: Cross-Fusion-/Liveness-Audit

**Entscheidung: kein qualifizierter Prototyp.** CPU-only, vorhandene exakt
gebundene GPU-HLOs; 2026-10-24 07:01:09–07:07:57Z, innerhalb 45 Minuten.
Kein Modell-Lower, kein Compile, kein GPU-Zugriff. Der eine erlaubte
CPU-Lower-/Prototyp-Arm ist unverbraucht.

Die bindenden Grenzen aus Astras Planreview gelten: ≥0,5 ms/root netto und
jenseits A/A, keine neuen Vollfeld-Temporaries/Peak-VRAM-Zunahme, kompilierte
BYTE-/E41-/WRF-Gates. Die +7,100 ms der nichtnativen Dycore-Kohorte pro drei
Roots sind ein **beobachtetes Mehrbudget**, keine mathematische Rückholgrenze
und kein kausal belegter Q1-Gewinn.

| Bestehende HLO-Region | untersuchte gebundene Dycore-Aufrufstellen | ausgegebene Array-/Skalarwerte | davon ohne SSA-Nutzer | exakt doppelt ausgegebene Fusionsausdrücke |
|---|---:|---:|---:|---:|
| d01 Hauptbody | 248 | 374 | 0 | 0 |
| d01 akustische Unterregionen, zusammen | 22 | 22 | 0 | 0 |
| d02 Hauptbody | 93 | 167 | 0 | 0 |
| d02 akustische Unterregion | 3 | 3 | 0 | 0 |
| Summe | 366 | 566 | 0 | 0 |

Der Screen expandiert Fusionsparameter auf ihre tatsächlichen Operanden in
derselben Region. Er erhält Operandenreihenfolge, dtype, Shape, Layout und
nichtdiagnostische Attribute; native Aufrufe und Kontrollfluss bleiben opaque.
Kein Gleichsetzen verschiedener RK-Massen oder Domains. Ein positiver
Kontrollfall wird erkannt; Änderungen an Operand, Layout, dtype oder
Operandenreihenfolge werden jeweils getrennt gehalten (5/5 Methodenkontrollen).
SSA-Liveness ist keine Messung physischer Peak-Puffer oder dynamischer Bytes.

## Was tatsächlich wiederholt wird

Im großen d02-Aufruf `loop_add_multiply_fusion.3` wiederholen sich über
Fusionsgrenzen vier Broadcasts, fünf Slices, fünf Multiplikationen und eine
Division. Die Arithmetik ist konkret gebunden:

- `1 / get-tuple-element.13558` erscheint in drei Fusionen. Quelle:
  `_apply_moisture_large_step`, Zeile 5714. Das Roh-Massenfeld kommt bereits
  aus `loop_add_divide_slice_fusion.1`, Ausgabe 2. Acht native Aufrufe und drei
  Fusionen brauchen es; ein Ersetzen durch den Kehrwert wäre kein sicherer
  Wegfall eines ungenutzten Puffers.
- Die fünf Produkte aus `get-tuple-element.13909` und den RK-Eingangsfeldern
  `get-tuple-element.15361` bis `.15365` erscheinen in vier Fusionen. Quelle:
  `_apply_moisture_large_step`, Zeile 5725. Die Operanden sind tatsächlich
  identisch; die Produkte sind **keine bestehenden Fusionsausgaben**.
- Ein f32-Vollfeld d02 hat 5.498.064 B. Gemeinsame neue Ausgaben für fünf
  Produkte benötigen 27.490.320 B; mit dem Kehrwert 32.988.384 B und längere
  Lebensdauern. Keine tote gleichwertige Ausgabe wurde gefunden. Diese
  Variante erfüllt das aktuelle Verbot neuer Vollfeld-Temporaries nicht.
  Ein Speedup oder netto weniger Traffic ist damit nicht nachgewiesen.

In d01 gibt es zwei wiederholt berechnete Vollfelder, die bereits einmal
ausgegeben werden: `input_minimum_multiply_transpose_fusion`, Ausgaben 2 und 5,
je 1.478.400 B. Sämtliche möglicherweise zugehörigen Kernelzeilen zusammen
kosten nur **136.609 ns pro drei Roots = 0,045536 ms/root**. Das ist eine bewusst
überinklusive aktuelle Gesamtkostenschranke dieser konkreten Zielkohorte;
selbst deren vollständiger Wegfall unterschritte 0,5 ms/root deutlich.
Zwei weitere geteilte ausgegebene Werte sind lediglich 4-Byte-Slices.

Die zwölf Ausgaben der teuersten d02-Fusion sind alle live; unter anderem
werden die aktualisierten Spezies von späteren RK-/Rand-/Transpose-Fusionen
und den akustischen Pfaden konsumiert. Der Screen findet keine abzuschaltende
tote Ausgabe. Das beweist keine universelle Unmöglichkeit einer größeren
Dycore-Optimierung; es liefert **keinen zulässigen schnellen Kandidaten** in
diesem Budget. Kein Prototyp und kein GPU-Arm werden aus Q1 angemeldet.

[Kompakter Beleg](receipts/Q1_CROSS_FUSION_DECISION.json) ·
[reproduzierbarer Screen](q1_cross_fusion_audit.py).
Vollständige Def-use-Daten privat unter dem hashgebundenen Receipt-Pfad;
keine neuen Caches/Snapshots oder numerischen Validierungsbehauptungen.
