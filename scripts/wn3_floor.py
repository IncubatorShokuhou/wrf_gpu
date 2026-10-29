"""Frozen original CPU-WRF twin spread for the WN3 0227 RW3 72-hour gate.

Generate the complete archive, or score an existing wn3_score summary without
editing it. D6 raw verdict and output integrity remain separate from A3.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/release/evidence/v033/w2/W2bR/metrics.json"
REFERENCE = ROOT / "docs/release/evidence/rw3/wn3_0227_full_floor.json"
SOURCE_SHA256 = "d3577c22d5800265589e6b6734dc42201194fa5f12320acf15ce4bc5e832ee02"
# Original six frozen C values (4491e479), independently confirmed cr-w44-rc.
ANCHORS = {
    ("d03", "RAINNC", 68): "3ff1301cddf34d3b",
    ("d03", "RAINNC", 69): "3ff8268cb9431a2f",
    ("d03", "RAINNC", 70): "3fffc8fef332fde5",
    ("d03", "RAINNC", 71): "4000ca6a497d00f2",
    ("d03", "RAINNC", 72): "4000e515bf3487b1",
    ("d03", "W", 69): "3fd51e1647aae7f0",
}
DOMAINS = ("d01", "d02", "d03")
START = datetime(2026, 2, 28, tzinfo=timezone.utc)


def generate(source: Path = SOURCE) -> dict:
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("original CPU pair archive SHA256 mismatch")
    archive = json.loads(raw)
    # Check the historical values BEFORE extracting any new leads.
    for (domain, field, lead), expected in ANCHORS.items():
        rows = archive["domains"][domain]["fields"][field]["by_lead"]
        value = next(row["pair_rmse"] for row in rows if row["lead_h"] == lead)
        if struct.pack(">d", value).hex() != expected:
            raise ValueError(f"frozen C mismatch: {domain}/{field}/{lead}")
    domains = {}
    count = 0
    for domain in DOMAINS:
        fields = {}
        for field, values in archive["domains"][domain]["fields"].items():
            rows = values["by_lead"]
            if [row["lead_h"] for row in rows] != list(range(73)):
                raise ValueError(f"incomplete original series: {domain}/{field}")
            # Hourly RAINNC has no pair statistic at lead 0; retain that absence.
            series = [row.get("pair_rmse") for row in rows]
            if any(v is not None and (not math.isfinite(v) or v < 0) for v in series):
                raise ValueError("invalid CPU pair RMSE")
            count += sum(v is not None for v in series)
            fields[field] = dict(unit=values["unit"], scale=values["scale"], C=series)
        domains[domain] = fields
    return dict(schema="wn3-0227-full-floor-v1", case="20260227_18z_a1",
                init_time_utc=START.isoformat(), hours=72, metric="rmse",
                source=dict(path=str(SOURCE.relative_to(ROOT)), sha256=SOURCE_SHA256,
                            pair_kind=archive["pair_kind"], pair_reference=archive["pair_reference"]),
                original_pair_source_sha256="fcd8b975ca1178248bdc34fd9a33904a99843874b5bcc21337e8f721cdfe2380",
                frozen_six_sha256="4491e479ce5e201b5071c735b0842ed88564cc028adc0ece8ec44d574a255be2",
                anchors_reproduced=6, pair_values=count, domains=domains)


def encode(reference: dict) -> str:
    """Compact arrays keep the entire series small and reviewable by field."""
    return json.dumps(reference, separators=(",", ":"), allow_nan=False) + "\n"


def load_reference() -> dict:
    reference = json.loads(REFERENCE.read_text())
    # Exact reproduction also detects truncated/deleted/modified C series.
    if reference != generate():
        raise ValueError("frozen full floor differs from original CPU pair archive")
    return reference


def score_floor(summary: dict, reference: dict | None = None) -> dict:
    reference = load_reference() if reference is None else reference
    rows, other = [], []
    complete = set(summary["domains"]) == set(DOMAINS)
    for domain in DOMAINS:
        d = summary["domains"].get(domain)
        if d is None:
            continue
        p = d["pairing"]
        # The reference is case-specific. Never transfer it to another date or
        # a subset of the 72-hour gate, even when filenames/counts look similar.
        if datetime.fromisoformat(p["init_time_utc"].replace("Z", "+00:00")) != START:
            raise ValueError("0227 floor used for a different case initialization")
        complete &= (p["paired_file_count"] == 73 and p["common_leads_h"] == list(range(73))
                     and not p["unmatched_cpu_count"] and not p["unmatched_gpu_count"]
                     and not d["missing_required_fields"] and not d["gpu_only_nonfinite"])
        failures = d["policy_failures"]
        if not failures and (d["pass"] is not True or d["summaries"]["verdict"] != "PASS"):
            other.append(dict(domain=domain, kind="unexplained_raw_failure"))
        for failure in failures:
            if failure["kind"] != "frame_tolerance" or not failure["detail"]:
                other.append(dict(domain=domain, **failure))
                continue
            for check in failure["detail"]:
                x, limit = check["value"], check["limit"]
                lead, field, metric = failure["lead_h"], failure["field"], check["metric"]
                archived = reference["domains"][domain].get(field)
                c = None
                # C is RMSE, never a bound for max_abs/relative/other metrics.
                # QVAPOR in the archive is g/kg; scorer RMSE is native kg/kg.
                if archived and metric == "rmse" and isinstance(lead, int) and 0 <= lead <= 72:
                    c = archived["C"][lead]
                    if c is not None:
                        c /= archived["scale"]
                valid = (math.isfinite(x) and math.isfinite(limit) and x > limit >= 0)
                ratio = c / x if valid and c is not None else None
                floor = valid and c is not None and (c > limit or ratio >= 0.75)
                rows.append(dict(domain=domain, field=field, lead_h=lead, metric=metric,
                                 X=x, limit=limit, C=c, C_over_X=ratio,
                                 status="FLOOR-LIMITED" if floor else "REAL FAIL"))
    real = len(other) + sum(row["status"] == "REAL FAIL" for row in rows)
    return dict(rule="original A3: C>limit OR C/X>=0.75 (unchanged)",
                C_reference=dict(path=str(REFERENCE.relative_to(ROOT)),
                                 sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
                                 source=reference["source"], pair_values=reference["pair_values"]),
                rows=rows, other_policy_failures=other,
                counts={"FLOOR-LIMITED": sum(r["status"] == "FLOOR-LIMITED" for r in rows), "REAL FAIL": real},
                coverage_complete=bool(complete), raw72="PASS" if summary["pass"] else "FAIL",
                binding72_A3="PASS" if complete and real == 0 else "FAIL",
                output_integrity_pass=summary.get("output_integrity_pass"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--out", type=Path, default=REFERENCE)
    score = sub.add_parser("score")
    score.add_argument("--summary", type=Path, required=True)
    score.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "generate":
        result = generate()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encode(result))
    else:
        result = score_floor(json.loads(args.summary.read_text()))
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k in
                     ("pair_values", "anchors_reproduced", "counts", "binding72_A3", "output_integrity_pass")}))


if __name__ == "__main__":
    main()
