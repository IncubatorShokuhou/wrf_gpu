"""Tabulate frame_probe.json / replay.json (markdown to stdout)."""
import json
import sys


def main():
    d = json.load(open(sys.argv[1]))
    items = d.get("frames") or d.get("steps")
    key_fin = "classify_fin" if "frames" in d else "classify_fin_chain"
    print("| case | sp | port neg (int) | WRF neg | port min | WRF min | FAITH (wrf>=0) | LARGER | SMALLER | WRF_ONLY |"
          " max diff/(eps S) | born pre-add | min pre-add-born / adv-born, dominant sc |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for it in items:
        label = it.get("file", "").split("/")[-1].replace("wrfout_d01_", "")
        if "step" in it:
            label = f"step {it['step']}"
        for sp, v in it["species"].items():
            c = v[key_fin]
            b = v["birth"]
            dm = c.get("diff_over_eps_scale_at_port_neg", {}).get("max")
            f = lambda x: "-" if x is None else f"{x:.4g}"  # noqa: E731
            print(f"| {label} | {sp} | {c['port_negative']} ({c['port_negative_interior']}) | {c['wrf_negative']} | "
                  f"{f(c['port_min'])} | {f(c['wrf_min'])} | {c['FAITHFUL']} ({c['FAITHFUL_wrf_nonnegative_at_rounding']}) | "
                  f"{c['LARGER_IN_PORT']} | {c['SMALLER_IN_PORT']} | {c['WRF_ONLY']} | {f(dm)} | "
                  f"{b['born_in_preadd']}/{b['negative']} | {f(b.get('min_born_preadd'))} / {f(b.get('min_born_advection'))} {b.get('preadd_dominant_negative_sc_part', {})} |")
    print()
    print("Isolated operator comparison (port vs pristine on the port's own inputs): max rel diff / FMA envelope max")
    stages = ("hdiff", "diff6", "sc", "pre", "adv", "upd", "fin")
    print("| case | sp | " + " | ".join(stages) + " |")
    print("|---|---|" + "---|" * len(stages))
    for it in items:
        label = it.get("file", "").split("/")[-1].replace("wrfout_d01_", "")
        if "step" in it:
            label = f"step {it['step']}"
        for sp, v in it["species"].items():
            cells = []
            for s in stages:
                r = v["isolated"][s]
                cells.append(f"{r['rel_max']:.1e} (int {r.get('rel_max_interior', float('nan')):.1e}) / env {r.get('fma_envelope_max', 0):.1e}")
            print(f"| {label} | {sp} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
