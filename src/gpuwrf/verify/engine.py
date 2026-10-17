"""Offline extraction, interval alignment, scoring and portable reports."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

from .observations import read_forecast_csv, read_observations, read_stations
from .scores import aggregate_intervals, score_sources
from .wrfout import extract_wrfout


def _identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def markdown_report(report):
    lines = ["# Station verification", "", "Descriptive scores against station observations; no promotion or release verdict.", "",
             "All sources use identical station/time/variable/precipitation-interval support per metric.", "",
             "| Source | Variable | Units | N | Stations | Bias | MAE | RMSE |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    fmt = lambda value: "unavailable" if value is None else f"{value:.6g}"
    safe = lambda value: str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    for row in report["metrics"]:
        lines.append("| " + " | ".join([safe(row["source"]), row["variable"], row["units"], str(row["n"]), str(row["n_stations"]),
                                       *[fmt(row[key]) for key in ("bias", "mae", "rmse")]]) + " |")
    for row in report["metrics"]:
        if row["variable"] == "wind":
            lines += ["", f"{safe(row['source'])}: vector wind RMSE {fmt(row['vector_rmse'])} m/s (N={row['vector_n']}).", ""]
        if row["variable"] == "wind_direction":
            lines += ["", f"{safe(row['source'])}: median absolute circular direction error {fmt(row['median_abs_error'])} degrees; both observed and every forecast speed >4 m/s.", ""]
        if row["variable"] == "precip_rate":
            lines += ["", f"{safe(row['source'])}: precipitation events (inclusive threshold, mm/h):", "",
                      "| Threshold | H | M | F | C | ETS | FBI |", "|---:|---:|---:|---:|---:|---:|---:|"]
            for event in row["contingency"]:
                lines.append("| " + " | ".join(fmt(event[key]) for key in ("threshold_mmh", "hits", "misses", "false_alarms", "correct_negatives", "ets", "fbi")) + " |")
            lines += [""]
    if report["comparisons"]:
        lines += ["", "Candidate minus baseline on common support:", "",
                  "| Variable | N | RMSE delta | RMSE ratio |", "|---|---:|---:|---:|"]
        for row in report["comparisons"]:
            lines.append(f"| {row['variable']} | {row['n']} | {fmt(row['rmse_delta'])} | {fmt(row['rmse_ratio'])} |")
    lines += ["", "Hourly ALISIOS exports use scalar means, circular direction means and gust maxima; canonical CSV keeps its declared sampling.",
              "Forecast history is sampled at output times, not every model step. Gust must be an explicitly identified gust field with the same window as truth.",
              "No confidence interval or superiority claim: station-hours are correlated. Canonical ALISIOS publication eligibility is outside this offline tool.",
              "", "Extraction and provenance (includes station metadata, corrections and support counts):", "", "```json",
              json.dumps({key: value for key, value in report.items() if key not in ("metrics", "comparisons")}, indent=2, allow_nan=False), "```", ""]
    return "\n".join(lines)


def write_report(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Reject non-finite report values before publishing any file.
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    markdown = markdown_report(report)
    (output/"verification.json").write_text(payload, encoding="utf-8")
    (output/"verification.md").write_text(markdown, encoding="utf-8")
    # Standalone, safe offline HTML; no network scripts or markdown dependency.
    esc = lambda value: html.escape(str(value))
    fmt = lambda value: "unavailable" if value is None else f"{value:.6g}"
    document = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Station verification</title>'
    document += '<style>body{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:1rem;color:#213547}pre{white-space:pre-wrap;overflow-wrap:anywhere}table{border-collapse:collapse;width:100%}th,td{padding:.6rem;border-bottom:1px solid #ccd;text-align:right}th:first-child,td:first-child{text-align:left}summary{cursor:pointer;margin:1rem 0}</style>'
    document += '<body><h1>Station verification</h1><p>Descriptive scores against station observations on identical support. No promotion or release verdict.</p>'
    document += '<table><thead><tr>'+''.join('<th>'+label+'</th>' for label in ('Source','Variable','Units','N','Stations','Bias','MAE','RMSE'))+'</tr></thead><tbody>'
    for row in report['metrics']:
        values = [row['source'],row['variable'],row['units'],row['n'],row['n_stations'],*[fmt(row[key]) for key in ('bias','mae','rmse')]]
        document += '<tr>'+''.join('<td>'+esc(value)+'</td>' for value in values)+'</tr>'
    document += '</tbody></table><details><summary>Wind, rainfall events, comparisons and assumptions</summary><pre>'+html.escape(markdown)+'</pre></details>'
    document += '<details><summary>Complete machine-readable scores and provenance</summary><pre>'+html.escape(payload)+'</pre></details></body></html>'
    (output/"verification.html").write_text(document, encoding="utf-8")


def verify(station_path, observation_path, model_paths, *, comparison_paths=None, output=None,
           observation_format="csv", precip_hours=None, thresholds=(0.1, 1, 5),
           candidate_name="candidate", baseline_name="baseline", **extraction):
    from gpuwrf import __version__

    if candidate_name == baseline_name and comparison_paths:
        raise ValueError("candidate and baseline names must differ")
    stations = read_stations(station_path)
    observations, obs_audit = read_observations(observation_path, observation_format, precip_hours=precip_hours)
    unknown = set(observations.station_uid)-set(stations.station_uid)
    if unknown:
        raise ValueError(f"observations contain unregistered stations: {sorted(unknown)}")
    sources, audits, inputs = {}, {}, [station_path, observation_path]
    for name, paths in ((candidate_name, model_paths), (baseline_name, comparison_paths)):
        if paths is None:
            continue
        paths = list(paths)
        inputs.extend(paths)
        if len(paths) == 1 and Path(paths[0]).suffix.lower() == ".csv":
            data = read_forecast_csv(paths[0])
            audit = {"format": "station CSV", "corrections": "already applied by producer; no re-correction", "files": [str(paths[0])]}
        else:
            data, audit = extract_wrfout(paths, stations, **extraction)
        if set(data.station_uid)-set(stations.station_uid):
            raise ValueError("forecast contains unregistered stations")
        sources[name], audits[name] = data, audit
    # Baseline rain windows govern when comparing, otherwise candidate windows.
    target = sources[baseline_name if comparison_paths else candidate_name]
    target = target[target.variable.eq("precip_rate")]
    observations = aggregate_intervals(observations, target)
    sources = {name: aggregate_intervals(data, target) for name, data in sources.items()}
    report = score_sources(observations, sources, thresholds=thresholds)
    report.update(gpuwrf_version=__version__, observations=obs_audit, extraction=audits,
                  stations=stations[["station_uid", "lat", "lon", "elevation_m", "landmask", "wind_height_m", "roughness_m"]].to_dict("records"),
                  precipitation_target=baseline_name if comparison_paths else candidate_name,
                  inputs=[_identity(path) for path in dict.fromkeys(inputs)])
    if not any(row["n"] for row in report["metrics"]):
        raise ValueError("no common station/time/variable support to score")
    if output is not None:
        write_report(report, output)
    return report
