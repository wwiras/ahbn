# Stage 4 Final Rerun — Exp08 CH Overload

## Execution record

- Date/time: 2026-08-21 16:45:41–16:45:42 MYT (Asia/Kuala_Lumpur)
- Project root: `/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6`
- Python interpreter: `/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python`
- Frozen configuration: `configs/exp08_ch_bottleneck.yaml`
- Exact batch command:

```bash
cd /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6
set -o pipefail
/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python run_batch.py --config configs/exp08_ch_bottleneck.yaml 2>&1 | tee outputs/logs/stage4rerun_exp08_20260821.log
```

## Pre-run verification

The repository's production configuration and runner were inspected before execution. The existing inspection command was also run:

```bash
/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python scripts/inspect_exp08_e0.py
```

Relevant terminal evidence:

```text
Python:
  interpreter           : /Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python
Runs per setting        : 20
Exact seeds             : [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61]
Configured              : ['gossip', 'cluster', 'dcsoc', 'ahbn']
Required Stage 4 set    : ['gossip', 'cluster', 'dcsoc', 'ahbn']
Result                  : PASS
Overload levels         : [1.0, 1.5, 2.0, 3.0]

DC-SoC:
  DBSCAN eps            : 2.0
  DBSCAN min_samples    : 3
  forwarding            : intra-cluster physical-neighbour fanout 3; CH gateway reserve 1
  runtime AHBN control  : NO
  explicit in Exp08     : YES
  matches Stage 3.5     : YES
  status                : PASS

AHBN:
  alpha                 : 0.30
  d0/l0/u0/c0           : 0.50/0.50/0.50/0.50
  w_d/w_l/w_u/w_c       : -1.0/+1.0/-1.0/+1.0
  kappa                 : 1.0
  beta                  : 1.0
  tau_mode              : 0.50 (config key: mode_threshold)
  fanout bounds         : [2, 4]
  default fanout        : 3
  EWMA                  : YES
  adaptive score/fanout : YES/YES
  status                : PASS

Overload config consumed          : PASS
Timing config consumed            : PASS
Workload config consumed          : PASS
Seed config consumed              : PASS
AHBN frozen config consumed       : PASS
DC-SoC frozen config consumed     : PASS
Algorithm parameters frozen       : PASS
Only overload level externally varies: PASS
E0 RESULT: PASS
```

Additional code-path checks confirmed:

- The Exp08 strategy loop consumes exactly `gossip`, `cluster` (Structured), `dcsoc`, and `ahbn` from the YAML.
- The DC-SoC branch directly constructs `assign_dcsoc_clusters(...)` and `DCSOCStrategy(...)`; no Exp08 fallback or AHBN-controller substitution is present.
- The canonical AHBN branch directly constructs `AHBNController(build_ahbn_params(cfg))` and `AHBNStrategy(adaptive_fanout=True)`.
- `ResultRow` contains the required `delivery_ratio`, `propagation_delay`, `duplicates`, and `total_forwards` metrics.
- `save_results_csv` and `save_adaptive_trace_csv` add a current timestamp, so prior raw CSV files are not overwritten.
- Expected grid: 4 comparators x 4 overload factors x 20 seeds = 320 runs.

Pre-run verdict: **PASS**. No algorithm or comparator parameter was modified.

## Batch terminal output

The complete batch terminal log is preserved at `outputs/logs/stage4rerun_exp08_20260821.log`. Its complete output is:

```text
Saved outputs/csv/exp08_results_20260821_164541.csv
Saved outputs/csv/exp08_adaptive_trace_20260821_164542.csv
```

The runner emits only the two save confirmations; it emits no per-run progress. Exit status was 0. No warnings or errors were emitted.

## Generated files

- Raw results: `/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/outputs/csv/exp08_results_20260821_164541.csv`
- AHBN adaptive trace: `/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/outputs/csv/exp08_adaptive_trace_20260821_164542.csv`
- Terminal log: `/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/outputs/logs/stage4rerun_exp08_20260821.log`

SHA-256 checksums:

```text
ea960877a33b8ae755e9319392f91c8b03b1bf4dc24bea3b868edcee387d4a30  outputs/csv/exp08_results_20260821_164541.csv
2e7ab084cf1abe8bcdde28dfe4806940055146eb54011709535a4e972ccb3362  outputs/csv/exp08_adaptive_trace_20260821_164542.csv
```

## Post-run structural validation

Validation was restricted to the newly generated files.

```text
RESULT_ROWS 320
STRATEGY_COUNTS {'ahbn': 80, 'cluster': 80, 'dcsoc': 80, 'gossip': 80}
OVERLOAD_COUNTS {1.0: 80, 1.5: 80, 2.0: 80, 3.0: 80}

strategy  ch_overload_factor
ahbn      1.0                   20
          1.5                   20
          2.0                   20
          3.0                   20
cluster   1.0                   20
          1.5                   20
          2.0                   20
          3.0                   20
dcsoc     1.0                   20
          1.5                   20
          2.0                   20
          3.0                   20
gossip    1.0                   20
          1.5                   20
          2.0                   20
          3.0                   20

SEEDS [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61]
SEED_COUNTS_PER_CELL_MINMAX (20, 20)
DUPLICATE_IDENTITIES 0
METRIC_MISSING {'delivery_ratio': 0, 'propagation_delay': 0, 'duplicates': 0, 'total_forwards': 0}
NEGATIVE_COUNTS {'duplicates': 0, 'total_forwards': 0}
DELIVERY_OUT_OF_RANGE 0
NONFINITE_METRICS 0
```

The CSV has 320 blank values only in the intentionally unused `fanout` metadata column. No required metric is blank. There are no malformed or duplicated run identities, missing comparator/condition combinations, negative count metrics, non-finite required metrics, or delivery ratios outside [0,1].

Completed runs: **320**.

Runs per comparator:

- Gossip (`gossip`): 80
- Structured (`cluster`): 80
- DC-SoC (`dcsoc`): 80
- AHBN (`ahbn`): 80

Runs per overload factor:

- 1.0: 80
- 1.5: 80
- 2.0: 80
- 3.0: 80

## AHBN adaptive-trace integrity

```text
TRACE_ROWS 19985
TRACE_STRATEGIES {'ahbn': 19985}
TRACE_SEEDS [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61]
TRACE_SCENARIOS ['ch_overload_factor=1.0', 'ch_overload_factor=1.5', 'ch_overload_factor=2.0', 'ch_overload_factor=3.0']
TRACE_CONTROLLER_MISSING {'d_hat': 0, 'l_hat': 0, 'u_hat': 0, 'c_hat': 0, 'score': 0, 'weight': 0, 'mode': 0, 'fanout': 0}
TRACE_FANOUT_MINMAX (3, 3)
TRACE_NONFINITE_NUMERIC 0
TRACE_DUPLICATE_ROWS 0
```

The trace covers all 80 AHBN seed/overload cells, contains populated controller fields, and keeps runtime fanout at 3, inside the frozen [2,4] bounds. No scientific interpretation was performed.

## Final verdict

**STAGE 4 FINAL EXP08 RERUN: PASS**

- S4 freeze intact: YES
- S5 comparator reconciliation intact: YES
- Algorithms modified: NO
- Comparator parameters modified: NO
- Structural validation: PASS
- Warnings/errors: NONE

No aggregation, confidence intervals, plots, scientific interpretation, or later experiment was started.

## E5 — Final aggregation

The existing `scripts/aggregate_exp08_e7.py` was minimally adapted on the analysis side to require the exact final rerun CSV and to validate the complete run grid before aggregation. It calculates the sample SD (`ddof=1`), SE, and the two-sided Student-t 95% CI margin (`t(0.975,19) * SE`). Adaptive-trace rows were not used as samples.

Exact command:

```bash
/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python scripts/aggregate_exp08_e7.py --input /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/outputs/csv/exp08_results_20260821_164541.csv --timestamp 20260821_170910
```

Terminal output:

```text
E5 Exp08 final aggregation
Input: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/outputs/csv/exp08_results_20260821_164541.csv
Input SHA-256: ea960877a33b8ae755e9319392f91c8b03b1bf4dc24bea3b868edcee387d4a30
Raw rows: 320
Comparators: 4
Overload factors: 4
Conditions: 16
Runs per condition: 20
Seeds: 42..61 per condition
Duplicate identities: 0
Invalid required metrics: 0
95% CI: two-sided Student t; df=19
Saved: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/outputs/csv/exp08_final_summary_20260821_170910.csv
E5 RESULT: PASS
```

Validation confirmed 320 raw rows, 16 conditions, 20 unique seeds (42–61) in every condition, four comparators, four overload factors, no missing cells, no duplicated run identities, and no malformed, NaN, or non-finite required metric values. The raw input checksum was unchanged after aggregation. Output: `outputs/csv/exp08_final_summary_20260821_170910.csv`. E5: **PASS**.

## E6 — Final plotting

The existing `scripts/plot_exp08_e8.py` was minimally adapted to accept only a named `exp08_final_summary_*` CSV, validate its 16-cell grid and `n=20`, and produce one timestamped PNG per metric. It reads the mean and Student-t CI margin directly from the E5 summary; it does not read raw results or the adaptive trace.

Exact command:

```bash
MPLCONFIGDIR=/private/tmp/exp08-mpl /Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python scripts/plot_exp08_e8.py --summary outputs/csv/exp08_final_summary_20260821_170910.csv --timestamp 20260821_170910
```

Terminal output:

```text
E6 Exp08 final plotting
Summary input only: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/outputs/csv/exp08_final_summary_20260821_170910.csv
Validation: 16 conditions; n=20; 4 comparators x 4 overloads
Error bars: mean +/- Student-t 95% CI
Saved: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/outputs/figures/exp08_final_delivery_ratio_20260821_170910.png
Saved: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/outputs/figures/exp08_final_propagation_delay_20260821_170910.png
Saved: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/outputs/figures/exp08_final_duplicates_20260821_170910.png
Saved: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/outputs/figures/exp08_final_total_forwards_20260821_170910.png
E6 RESULT: PASS
```

E6: **PASS**.

## E7 — Scientific interpretation

Values below are run-level mean ± two-sided Student-t 95% CI (`n=20`, `df=19`). No hypothesis test was performed; differences are not described as statistically significant.

| Overload | Comparator | Delivery ratio | Delay (s) | Duplicates | Total forwards |
|---:|---|---:|---:|---:|---:|
| 1.0 | Gossip | 0.831 ± 0.021 | 10.015 ± 0.459 | 167.1 ± 4.2 | 249.2 ± 6.3 |
| 1.0 | Structured | 1.000 ± 0.000 | 4.498 ± 0.044 | 99.0 ± 0.0 | 198.0 ± 0.0 |
| 1.0 | DC-SoC | 0.040 ± 0.000 | 1.799 ± 0.254 | 3.0 ± 0.0 | 6.0 ± 0.0 |
| 1.0 | AHBN | 0.831 ± 0.021 | 10.015 ± 0.459 | 167.1 ± 4.2 | 249.2 ± 6.3 |
| 1.5 | Gossip | 0.831 ± 0.021 | 10.015 ± 0.459 | 167.1 ± 4.2 | 249.2 ± 6.3 |
| 1.5 | Structured | 1.000 ± 0.000 | 6.023 ± 0.043 | 99.0 ± 0.0 | 198.0 ± 0.0 |
| 1.5 | DC-SoC | 0.040 ± 0.000 | 2.099 ± 0.371 | 3.0 ± 0.0 | 6.0 ± 0.0 |
| 1.5 | AHBN | 0.830 ± 0.023 | 9.671 ± 0.627 | 166.9 ± 4.7 | 248.8 ± 7.0 |
| 2.0 | Gossip | 0.831 ± 0.021 | 10.015 ± 0.459 | 167.1 ± 4.2 | 249.2 ± 6.3 |
| 2.0 | Structured | 1.000 ± 0.000 | 7.523 ± 0.043 | 99.0 ± 0.0 | 198.0 ± 0.0 |
| 2.0 | DC-SoC | 0.040 ± 0.000 | 2.399 ± 0.489 | 3.0 ± 0.0 | 6.0 ± 0.0 |
| 2.0 | AHBN | 0.837 ± 0.023 | 10.533 ± 1.048 | 168.4 ± 4.6 | 251.1 ± 6.9 |
| 3.0 | Gossip | 0.831 ± 0.021 | 10.015 ± 0.459 | 167.1 ± 4.2 | 249.2 ± 6.3 |
| 3.0 | Structured | 1.000 ± 0.000 | 10.523 ± 0.043 | 99.0 ± 0.0 | 198.0 ± 0.0 |
| 3.0 | DC-SoC | 0.040 ± 0.000 | 2.999 ± 0.723 | 3.0 ± 0.0 | 6.0 ± 0.0 |
| 3.0 | AHBN | 0.820 ± 0.024 | 9.929 ± 0.577 | 165.1 ± 4.9 | 246.2 ± 7.3 |

### Delivery ratio

Structured is the most resilient comparator by delivery, remaining at 1.000 at every overload. Gossip is invariant at 0.831. AHBN stays close to Gossip (0.820–0.837); at overload 3.0 its mean is 0.011 below Gossip, but their intervals overlap substantially. DC-SoC remains at 0.040, demonstrating that its very low dissemination cost does not provide broad delivery. There is no evidence that Structured's delivery degrades, although its latency does. AHBN does not dominate delivery and its small mean changes should be interpreted cautiously.

### Propagation delay

Structured exhibits the clearest CH bottleneck: mean delay grows from 4.498 s to 10.523 s (+133.9%). DC-SoC rises from 1.799 s to 2.999 s (+66.7%) but must be read alongside its 0.040 delivery. Gossip remains at 10.015 s, consistent with resilience obtained outside a CH-sensitive forwarding path but at high dissemination cost. AHBN remains near 10 s (9.671–10.533 s), with overlapping intervals across levels and with Gossip. At overload 3.0, AHBN's 9.929 ± 0.577 s is slightly below Structured's 10.523 ± 0.043 s, while its delivery is much lower. Thus AHBN does not provide a superior reliability/latency combination here; it supplies a Gossip-like balance.

### Duplicates

Gossip generates 167.1 duplicates, and AHBN is similarly high (165.1–168.4). Structured produces 99.0 duplicates while achieving complete delivery. DC-SoC produces only 3.0, but that efficiency coincides with almost no delivery. AHBN's duplicate means vary slightly with its dissemination outcomes and are lowest at overload 3.0, where delivery and forwarding are also lowest. The overlapping AHBN/Gossip intervals provide no basis for a strong separation claim.

### Total forwards

Forwarding effort mirrors duplication: Gossip uses 249.2 forwards and AHBN 246.2–251.1; Structured uses 198.0; DC-SoC uses 6.0. Low forwarding is not intrinsically favourable: DC-SoC's six forwards coincide with 0.040 delivery. In this experiment, AHBN spends Gossip-like forwarding effort for Gossip-like delivery, rather than showing a clear cost reduction. Structured offers the strongest delivery/cost combination, but its overload-dependent delay exposes its trade-off.

### Cross-metric conclusion and AHBN behavioural evidence

The strategies occupy distinct trade-off points. Structured maximizes delivery with moderate fixed traffic but becomes latency-bottlenecked as CH overload increases. Gossip is overload-insensitive in all four aggregates but has high duplicate and forwarding cost. DC-SoC minimizes delay and traffic at the cost of extremely poor reach. AHBN produces a Gossip-like operating point and does not universally dominate any comparator.

The trace is supporting mechanism evidence only (19,985 events, not independent statistical samples). Both controller modes occur at every overload. Cluster-mode events decline proportionally from 3,268/5,003 (65.3%) at overload 1.0 to 3,152/4,943 (63.8%) at 3.0, while gossip-mode events rise from 34.7% to 36.2%. Mean controller weight rises slightly from 0.4848 to 0.4899, and the maximum rises from 0.5390 to 0.5556. Approximately 1,205–1,212 mode switches occur per overload level. Fanout remains 3 for every trace row and no fanout-change event occurs. Accordingly, Exp08 supports runtime controller-state observation and mode switching, with a modest overload-related shift toward gossip mode; it does **not** demonstrate fanout adaptation in this scenario, nor does it establish that the modest shift caused superior aggregate performance.

Limitations: the CIs quantify uncertainty around condition means but are not formal pairwise tests; zero-width intervals reflect identical run-level values for some conditions; extensive CI overlap warrants caution; and the trace is descriptive mechanism evidence rather than an additional sample population. E7: **PASS**.

## E8 — Final sign-off

The final rerun is structurally valid; E5 aggregation and Student-t CIs are valid; E6 figures were generated directly from the final E5 summary; and the interpretation reports the observed trade-offs without tuning or unsupported significance claims.

```text
E8 FINAL EXP08 SIGN-OFF: PASS
S4 freeze intact: YES
S5 comparator reconciliation intact: YES
Algorithms modified: NO
Comparator/controller parameters modified: NO
Raw data modified: NO
Results scientifically defensible: YES
EXP08: CLOSED
NEXT PERMITTED EXPERIMENT: EXP09
```
