"""Sanitized per-root device counts; never read Nsight environment tables."""
import argparse
import json
from pathlib import Path
import sqlite3
import statistics


def counts(db):
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_KERNEL"} <= tables
    names = ("LSMPARAMS:off", "LSMPARAMS:on")
    accepted = {alias: name for name in names for alias in (name, "TSL:" + name)}
    keys = tuple(accepted)
    labels = dict(db.execute("SELECT id,value FROM StringIds WHERE length(value)<64 AND value IN (?,?,?,?)", keys))
    ranges = [(start, end, accepted[text or labels.get(text_id)]) for start, end, text, text_id in db.execute(
        "SELECT start,end,text,textId FROM NVTX_EVENTS WHERE text IN (?,?,?,?) OR textId IN "
        "(SELECT id FROM StringIds WHERE length(value)<64 AND value IN (?,?,?,?))", keys + keys)]
    out = {name: [] for name in names}
    for start, end, label in ranges:
        assert label in out and end > start
        row = {}
        for key, table in (("kernel", "CUPTI_ACTIVITY_KIND_KERNEL"),
                           ("memcpy", "CUPTI_ACTIVITY_KIND_MEMCPY"),
                           ("memset", "CUPTI_ACTIVITY_KIND_MEMSET")):
            if table not in tables:
                row[key + "_rows"], row[key + "_ns"] = 0, 0
                continue
            n, duration = db.execute(f"SELECT count(*),coalesce(sum(end-start),0) FROM {table} "
                                     "WHERE start>=? AND end<=?", (start, end)).fetchone()
            row[key + "_rows"], row[key + "_ns"] = n, duration
        assert row["kernel_rows"] > 0
        row["device_ops"] = sum(row[key + "_rows"] for key in ("kernel", "memcpy", "memset"))
        if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
            row["copies"] = [{"kind": kind, "rows": n, "bytes": size} for kind, n, size in db.execute(
                "SELECT copyKind,count(*),sum(bytes) FROM CUPTI_ACTIVITY_KIND_MEMCPY "
                "WHERE start>=? AND end<=? GROUP BY copyKind", (start, end))]
        out[label].append(row)
    assert len(out[names[0]]) == len(out[names[1]]) > 0, "missing paired root ranges"
    summary = {name: {"roots": len(rows), **{key: statistics.median(row[key] for row in rows)
                for key in ("kernel_rows", "kernel_ns", "memcpy_rows", "memcpy_ns", "memset_rows", "device_ops")}}
               for name, rows in out.items()}
    return {"ranges": out, "median_per_root": summary,
            "on_over_off_kernel_time": summary[names[1]]["kernel_ns"] / summary[names[0]]["kernel_ns"],
            "scope": "ordinary PROD nested-root replay; kernel duration sum, not end-to-end S1 or stream-union busy time"}


def selftest():
    with sqlite3.connect(":memory:") as db:
        db.executescript("CREATE TABLE StringIds(id INTEGER,value TEXT);"
                         "CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,text TEXT,textId INTEGER);"
                         "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER);"
                         "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER,copyKind INTEGER,bytes INTEGER);")
        db.executemany("INSERT INTO StringIds VALUES(?,?)", [(1, "TSL:LSMPARAMS:on"), (2, "ignored")])
        db.executemany("INSERT INTO NVTX_EVENTS VALUES(?,?,?,?)", [(0, 100, "LSMPARAMS:off", None), (100, 200, None, 1)])
        db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?)", [(10, 30), (110, 120), (150, 160), (210, 260)])
        db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(?,?,?,?)", [(35, 40, 8, 32), (160, 165, 8, 32)])
        result = counts(db)
        assert result["on_over_off_kernel_time"] == 1
        assert result["median_per_root"]["LSMPARAMS:off"]["device_ops"] == 2
        assert result["median_per_root"]["LSMPARAMS:on"]["device_ops"] == 3
        db.execute("DELETE FROM NVTX_EVENTS WHERE textId=1")
        try:
            counts(db)
        except AssertionError:
            return
        raise AssertionError("missing arm was accepted")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path, nargs="?")
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        print("profile parser selftest PASS")
    else:
        assert args.database and args.output
        with sqlite3.connect("file:" + str(args.database.resolve()) + "?mode=ro", uri=True) as connection:
            result = counts(connection)
        args.output.write_text(json.dumps(result, indent=2))
