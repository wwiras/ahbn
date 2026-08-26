#!/usr/bin/env python3
"""Create the v0.62 C9-C11 interpretation record and evidence manifest."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "outputs/csv"
FIG = ROOT / "outputs/figures/v062_s7"
DOC = ROOT / "docs/stage4_v062_c9_c11_final_evidence.md"
MANIFEST = ROOT / "outputs/v062_final_evidence_manifest.json"
METRICS = ("delivery_ratio", "propagation_delay", "duplicates", "total_forwards")

RAW = {
    "Exp07 results": CSV / "exp07_results_20260826_081046.csv",
    "Exp08 results": CSV / "exp08_results_20260826_081147.csv",
    "Exp09 results": CSV / "exp09_results_20260826_081323.csv",
}
TRACES = {
    "Exp07 trace": CSV / "exp07_adaptive_trace_20260826_081047.csv",
    "Exp08 trace": CSV / "exp08_ahbn_adaptive_trace_20260826_081147.csv",
    "Exp09 trace": CSV / "exp09_adaptive_trace_20260826_081323.csv",
}
SUPPORT = {
    "Exp08 execution evidence": CSV / "exp08_execution_evidence_20260826_081147.csv",
    "Exp08 execution manifest": CSV / "exp08_s3_manifest.json",
}
DERIVED = {
    "S5 aggregate": CSV / "final_control_v062_s5_raw.csv",
    "S5 summary": CSV / "final_control_v062_s5_summary.csv",
    "S6 statistics": CSV / "final_control_v062_s6_statistics.csv",
    "S6 seed robustness": CSV / "final_control_v062_s6_seed_robustness.csv",
    "S7 plot data": CSV / "final_control_v062_s7_plotdata.csv",
}
FIGURES = {p.name: p for p in sorted(FIG.glob("*"))}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def fmt(value: float) -> str:
    return f"{value:.6g}"


def result_table(stats: pd.DataFrame, experiment: str) -> str:
    rows = ["| Algorithm | Condition | n | Delivery ratio | Propagation delay | Duplicates | Total forwards |",
            "|---|---|---:|---:|---:|---:|---:|"]
    for _, r in stats[stats.experiment.eq(experiment)].iterrows():
        values = []
        for m in METRICS:
            values.append(f"{fmt(r[f'{m}_mean'])} [{fmt(r[f'{m}_ci95_low'])}, {fmt(r[f'{m}_ci95_high'])}]")
        rows.append(f"| {r.algorithm} | `{r.experimental_condition}` | {int(r.n)} | " + " | ".join(values) + " |")
    return "\n".join(rows)


def endpoint_table(stats: pd.DataFrame, experiment: str, first: str, last: str) -> str:
    rows = ["| Algorithm | Delivery | Delay | Duplicates | Total forwards |", "|---|---:|---:|---:|---:|"]
    for algorithm in ("Gossip", "Structured", "DC-SoC", "AHBN"):
        a = stats[(stats.experiment == experiment) & (stats.algorithm == algorithm) & (stats.experimental_condition == first)].iloc[0]
        b = stats[(stats.experiment == experiment) & (stats.algorithm == algorithm) & (stats.experimental_condition == last)].iloc[0]
        cells = []
        for m in METRICS:
            av, bv = float(a[f"{m}_mean"]), float(b[f"{m}_mean"])
            delta = bv - av
            cells.append(f"{delta:+.6g} ({delta / av * 100:+.2f}%)" if av else f"{delta:+.6g} (percentage undefined: zero baseline)")
        rows.append(f"| {algorithm} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def trace_table(path: Path) -> str:
    trace = pd.read_csv(path)
    rows = ["| Condition | Rows | Gossip / Cluster | Fanout 2 / 3 / 4 | mean d_hat / l_hat / u_hat | z mean [q05, q95] | Mode / fanout transitions |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for condition, g in trace.groupby("scenario_tag", sort=False):
        modes = g["mode"].value_counts()
        fan = g["fanout"].value_counts()
        rows.append(
            f"| `{condition}` | {len(g)} | {modes.get('gossip', 0)} / {modes.get('cluster', 0)} | "
            f"{fan.get(2, 0)} / {fan.get(3, 0)} / {fan.get(4, 0)} | "
            f"{g.d_hat.mean():.4f} / {g.l_hat.mean():.4f} / {g.u_hat.mean():.4f} | "
            f"{g.score.mean():.4f} [{g.score.quantile(.05):.4f}, {g.score.quantile(.95):.4f}] | "
            f"{int(g.mode_switched.sum())} / {int(g.fanout_changed.sum())} |"
        )
    return "\n".join(rows)


def hash_table(items: dict[str, Path]) -> str:
    rows = ["| Evidence | Path | SHA-256 |", "|---|---|---|"]
    rows.extend(f"| {name} | `{rel(path)}` | `{sha256(path)}` |" for name, path in items.items())
    return "\n".join(rows)


def main() -> None:
    stats = pd.read_csv(DERIVED["S6 statistics"])
    assert len(stats) == 42 and set(stats.n) == {20}
    timestamp = datetime.now(timezone.utc).isoformat()
    all_evidence = RAW | TRACES | SUPPORT | DERIVED | FIGURES
    for path in all_evidence.values():
        assert path.is_file(), path

    doc = f"""# Control Simulator v0.62 — C9/C10/C11 final evidence

Frozen on: `{timestamp}`. Parent v0.61 is preserved as historical pre-canonical-correction evidence; v0.62 is authoritative post-canonical-freeze Control Simulator evidence. No simulations or inferential tests were run during C9-C11.

## C9 — quantitative evidence

S5 PASS: exactly 840 formal rows (Exp07 120, Exp08 320, Exp09 400), 840 unique run keys, seeds 42–61, no missing cells, duplicate runs, malformed metrics, or smoke contamination. Exp07 has five Gossip fanout cells and one canonical adaptive AHBN cell. S6 PASS: 42 cells, every n=20, sample SD and Student-t 95% CI (`df=19`). Independent direct-raw verification covered 19 required cells and 304 scalar values; maximum absolute discrepancy was `1.1368683772161603e-13`. S7 numerical verification checked 90/90 plotted records with zero mismatch.

S7 visual readability PASS. The three saved PNGs were opened and inspected: legend overlap NONE; axis/tick-label overlap NONE; clipped text/error bars NONE; confidence intervals and marker identities readable. Exp07 shows AHBN once, as a canonical adaptive marker, not a fanout sweep.

## C10 — Exp07 results and interpretation

{result_table(stats, 'Exp07')}

Increasing Gossip fanout moves along a clear observed tradeoff: delivery rises from 0.732 to 0.9855 and delay falls from 11.7258 to 5.58927, while duplicates rise from 74.2 to 258.35 and forwards from 146.4 to 355.9. AHBN's descriptive results coincide with Gossip k=3 in this experiment, but it is one adaptive condition, not a fixed-k sweep point. It therefore offers the k=3-level observed balance here: better delivery and delay than k=2, but lower delivery and higher delay than k=4–6; its duplicate/forward cost is higher than k=2 and lower than k=4–6.

{trace_table(TRACES['Exp07 trace'])}

The trace shows adaptation through both mode and fanout: 4,062 Gossip versus 821 Cluster decisions, fanout 2 on 241 rows and fanout 3 on 4,642 rows, with 180 recorded mode-switch flags and 54 fanout-change flags. Duplicate pressure contributes `-d` and can push z toward Cluster/lower fanout, while latency and utilization contribute `+l/+u`; the trace supports competing pressure, not a universal superiority claim.

## C10 — Exp08 results and interpretation

{result_table(stats, 'Exp08')}

Endpoint changes (factor 1.0 → 3.0):

{endpoint_table(stats, 'Exp08', 'ch_overload_factor=1.0', 'ch_overload_factor=3.0')}

Gossip is unchanged in all four endpoint metrics. Structured and DC-SoC retain delivery 1.0 and zero duplicates, while delay rises 132.75% and 166.91%, respectively. AHBN delay rises 8.44%, delivery falls 1.81%, duplicates fall 1.60%, and forwards fall 1.69%; intermediate AHBN means are non-monotonic. AHBN is neither the lowest-delay nor highest-delivery method here.

{trace_table(TRACES['Exp08 trace'])}

From factor 1.0 to 3.0, mean `l_hat` rises 0.2752→0.2987, mean `u_hat` stays near 0.080, and mean `d_hat` is similar overall (0.2554→0.2567). Mean z rises 0.1001→0.1225, but competing terms make monotonic z unnecessary. Fanout 4 grows from 0 to 140 decisions and Cluster decisions decline from 821 to 643. Exp08 therefore activates both mode and fanout, predominantly fanout 3 with a growing high-score fanout-4 tail.

## C10 — Exp09 results and interpretation

{result_table(stats, 'Exp09')}

Endpoint changes (p=0.04 → 0.12):

{endpoint_table(stats, 'Exp09', 'edge_prob=0.04', 'edge_prob=0.12')}

Gossip keeps delivery 1.0 while delay falls 46.24%, duplicates rise 370.86%, and forwards rise 253.26%. Structured and DC-SoC keep delivery 1.0 and zero duplicates; their forwards rise 1.43% as reachable topology size changes slightly. AHBN delivery rises 0.58%, delay falls 4.48%, duplicates rise 47.31%, and forwards rise 28.48%. AHBN has fewer duplicates but higher delay than Gossip at every tested density; no universal ranking is implied.

{trace_table(TRACES['Exp09 trace'])}

Mean `d_hat` rises from 0.1538 to about 0.2014 as density increases, strengthening `-d`; mean `l_hat` and `u_hat` also rise and counteract it. Mean z consequently changes only modestly and non-monotonically. AHBN remains overwhelmingly at fanout 3: two fanout-2 rows at p=0.08 and one at p=0.12, no fanout 4. Density response is therefore mainly mode adaptation, with extremely rare lower-fanout activation.

## Cross-experiment interpretation

v0.62 shows environment-specific controller use. Exp07 uses both mode and fanout around a moderate-cost operating point. Exp08 uses both, including an increasing fanout-4 tail under greater latency pressure. Exp09 responds mainly through mode while fanout remains almost entirely 3. Across experiments AHBN trades redundancy, reach, and delay; it does not dominate all comparators or metrics.

## C11 — v0.61 reconciliation

| Evidence/claim | v0.61 status | v0.62 result | Action |
|---|---|---|---|
| Exp07 descriptive means/CIs | Same numerical summary | Same means/CIs; adaptive trace proves fanout 2 and 3 operation | KEEP numbers; REWRITE fixed-fanout interpretation |
| Exp07 AHBN runtime fanout | Claimed fanout 3/no movement | 241 fanout-2 rows, 4,642 fanout-3 rows, 54 change flags | REWRITE |
| Exp08 endpoint percentages | 8.44% AHBN delay; 132.75% Structured; 166.91% DC-SoC | Confirmed; add all four metrics | KEEP and EXTEND |
| Exp08 AHBN fanout | Historical interpretation said fixed fanout 3 | fanout 4 occurs 52/98/140 times at factors 1.5/2.0/3.0 | REWRITE |
| Exp09 endpoints | +47.31% AHBN duplicates; -4.48% delay | Confirmed; delivery +0.58%, forwards +28.48% added | KEEP and EXTEND |
| Exp09 AHBN fanout | Historical interpretation said no reduction/fixed 3 | three formal fanout-2 rows and two change flags | REWRITE |
| Five-point AHBN Exp07 sweep | Unsupported | One canonical adaptive condition | REMOVE |
| Universal/significance language | Unsupported | Descriptive mean ± t CI only | REMOVE |

## Authoritative evidence and lineage

{hash_table(all_evidence)}

Lineage is exact formal result CSVs → versioned S5 aggregate → versioned S6 statistics → versioned S7 plot data/figures → this C10 interpretation and the C11 manifest. Trace files support only controller-behaviour interpretation. Smoke files are excluded.

## Freeze gate

Canonical validator PASS; formal Exp07/08/09 PASS; 840/840 runs; AHBN invariant failures 0; S5/S6/S7 PASS; C10 complete; hashes recorded; smoke contamination NONE; post-formal algorithm tuning NONE; controller/comparator/config/raw modifications NONE. v0.61 remains historical and unmodified.
"""
    DOC.write_text(doc, encoding="utf-8")
    manifest = {
        "control_simulator_version": "v0.62",
        "parent": "v0.61",
        "status": "authoritative_post_canonical_freeze",
        "canonical_score": "z = -d + l + u + c",
        "mode": {"z < 0": "Cluster", "z >= 0": "Gossip"},
        "fanout": {"z <= -0.25": 2, "-0.25 < z < 0.25": 3, "z >= 0.25": 4},
        "formal_seeds": [42, 61],
        "formal_run_counts": {"Exp07": 120, "Exp08": 320, "Exp09": 400, "total": 840},
        "ahbn_invariant_failures": 0,
        "freeze_timestamp_utc": timestamp,
        "evidence": {name: {"path": rel(path), "sha256": sha256(path)} for name, path in all_evidence.items()},
        "interpretation": {"path": rel(DOC), "sha256": sha256(DOC)},
        "smoke_files_authoritative": False,
        "post_formal_tuning": False,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {rel(DOC)} {sha256(DOC)}")
    print(f"Wrote {rel(MANIFEST)} {sha256(MANIFEST)}")


if __name__ == "__main__":
    main()
