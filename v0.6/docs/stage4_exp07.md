# Stage 4 — EXP07 FINAL ANALYSIS AND PLOT CORRECTION

Exp07 — Forwarding Fanout Latency–duplication boundaries

## Smoke Test

```bash
% cp configs/exp07_fanout.yaml /tmp/exp07_smoke.yaml
(venv0.6) wwiras@wwirass-MacBook-Air v0.6 % vi /tmp/exp07_smoke.yaml 
(venv0.6) wwiras@wwirass-MacBook-Air v0.6 % mkdir -p outputs/logs
(venv0.6) wwiras@wwirass-MacBook-Air v0.6 % python run_batch.py \ 
  --config /tmp/exp07_smoke.yaml \
  2>&1 | tee outputs/logs/exp07_smoke.log
Saved outputs/csv/exp07_results_20260820_074018.csv
Saved outputs/csv/exp07_adaptive_trace_20260820_074018.csv
(venv0.6) wwiras@wwirass-MacBook-Air v0.6 % head -20 outputs/csv/exp07_results_20260820_074018.csv 
experiment,strategy,seed,num_nodes,topology_type,topology_param,fanout,num_clusters,ch_overload_factor,delivery_ratio,propagation_delay,duplicates,total_forwards
exp07,gossip,42,100,ba,3,2.0,4,,0.47,15.545292527324056,48,94
exp07,gossip,42,100,ba,3,3.0,4,,0.76,10.003912158391428,153,228
exp07,gossip,42,100,ba,3,4.0,4,,0.95,8.674713478983895,249,343
exp07,gossip,42,100,ba,3,5.0,4,,0.94,6.388058785517448,287,380
exp07,gossip,42,100,ba,3,6.0,4,,0.98,5.4551568859505855,329,426
exp07,ahbn,42,100,ba,3,,4,,0.76,10.003912158391428,153,228
(venv0.6) wwiras@wwirass-MacBook-Air v0.6 % python run_batch.py \ 
  --config configs/exp07_fanout.yaml \
  2>&1 | tee outputs/logs/exp07_stage4_final.log
Saved outputs/csv/exp07_results_20260820_074418.csv
Saved outputs/csv/exp07_adaptive_trace_20260820_074418.csv
```

## STAGE 4 — EXP07 FINAL ANALYSIS AND PLOT CORRECTION

Date: 2026-08-20

Status: **PASS**

This analysis used the existing final Exp07 evidence only. The experiment was not
rerun, and neither raw CSV was modified.

### Files used

- `outputs/csv/exp07_results_20260820_074418.csv`
- `outputs/csv/exp07_adaptive_trace_20260820_074418.csv`
- `outputs/logs/exp07_stage4_final.log`
- `scripts/plot_results.py`
- `scripts/summarize_results.py`
- `scripts/plot_exp07_publication.py` (inspected only)
- `scripts/plot_exp07_side_by_side.py` (inspected only)

Exact interpreter:

```text
/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python
```

### Implementation decision

Option A was selected: narrowly scoped Exp07 handling in the existing generic
plotting and summary scripts. Exp08/Exp09 and all simulation/controller code were
left unchanged. The 95% confidence intervals are two-sided Student-t intervals
over the independent simulation runs:

```text
mean ± t(0.975, n - 1) × sample_std(ddof=1) / sqrt(n)
```

The adaptive trace is used only to describe observed controller behavior, never
as the statistical replication unit.

### Command log and relevant terminal output

Inspection:

```bash
pwd
rg --files -g 'AGENTS.md' -g '!venv*' -g '!outputs/**' . .. 2>/dev/null | head -50
git status --short
sed -n '1,260p' scripts/plot_results.py
sed -n '1,260p' scripts/summarize_results.py
sed -n '260,620p' scripts/plot_results.py
sed -n '1,260p' docs/stage4_exp07.md
sed -n '620,980p' scripts/plot_results.py
tail -80 outputs/logs/exp07_stage4_final.log
sed -n '1,240p' scripts/plot_exp07_publication.py
sed -n '1,240p' scripts/plot_exp07_side_by_side.py
rg -n "ci95|confidence|sem\\(|t\\.ppf|1\\.96|errorbar|fill_between" scripts ahbn tests docs -g '*.py' -g '*.md' 2>/dev/null
rg -n "scipy" requirements*.txt pyproject.toml setup.cfg setup.py 2>/dev/null
```

Relevant inspection result: the original `plot_exp07` grouped by
`[strategy, fanout]`, which excluded AHBN because its fanout is intentionally
null. The two older dedicated Exp07 scripts made the same grouping assumption
and used standard deviation rather than 95% confidence intervals. No established
project CI implementation was found.

Initial shape/column inspection:

```bash
'/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python' - <<'PY'
import pandas as pd
from pathlib import Path
r=Path('outputs/csv/exp07_results_20260820_074418.csv')
t=Path('outputs/csv/exp07_adaptive_trace_20260820_074418.csv')
df=pd.read_csv(r)
tr=pd.read_csv(t)
print('RESULT_COLUMNS', df.columns.tolist())
print('TRACE_COLUMNS', tr.columns.tolist())
print('RESULT_SHAPE', df.shape)
print('GROUP_COUNTS')
print(df.groupby(['strategy','fanout'],dropna=False).size().to_string())
print('AHBN_FANOUT_VALUES', df.loc[df.strategy.eq('ahbn'),'fanout'].unique().tolist())
print('TRACE_SHAPE', tr.shape)
print('TRACE_HEAD')
print(tr.head().to_string(index=False))
PY
```

```text
RESULT_SHAPE (120, 13)
GROUP_COUNTS
strategy  fanout
ahbn      NaN       20
gossip    2.0       20
          3.0       20
          4.0       20
          5.0       20
          6.0       20
AHBN_FANOUT_VALUES [nan]
TRACE_SHAPE (5003, 29)
```

Dependency and pre-change checksum check:

```bash
'/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python' -c 'import scipy; print(scipy.__version__)'
git diff -- scripts/plot_results.py scripts/summarize_results.py docs/stage4_exp07.md
shasum -a 256 outputs/csv/exp07_results_20260820_074418.csv outputs/csv/exp07_adaptive_trace_20260820_074418.csv
```

```text
1.18.0
72bd18236776bab009a70d832d84b4382f3034812340e76215dc2371585e2522  outputs/csv/exp07_results_20260820_074418.csv
4da401398b06d5b130e7a2fc331be5f50d9fce23080165c4ec742f4456f43fa5  outputs/csv/exp07_adaptive_trace_20260820_074418.csv
```

Summary-statistics and initial plotting command:

```bash
'/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python' -m py_compile scripts/plot_results.py scripts/summarize_results.py
'/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python' scripts/summarize_results.py outputs/csv/exp07_results_20260820_074418.csv outputs/csv/exp07_summary_20260820_074418.csv
'/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python' scripts/plot_results.py outputs/csv/exp07_results_20260820_074418.csv
```

```text
Saved outputs/csv/exp07_summary_20260820_074418.csv
strategy  fanout            metric  n       mean       std   ci95_low  ci95_high
    ahbn     NaN    delivery_ratio 20   0.830500  0.045128   0.809379   0.851621
    ahbn     NaN        duplicates 20 167.100000  9.025694 162.875845 171.324155
    ahbn     NaN propagation_delay 20  10.015010  0.980442   9.556149  10.473871
    ahbn     NaN    total_forwards 20 249.150000 13.538541 242.813768 255.486232
  gossip     2.0    delivery_ratio 20   0.562500  0.089318   0.520698   0.604302
  gossip     2.0        duplicates 20  57.250000  8.931759  53.069808  61.430192
  gossip     2.0 propagation_delay 20  13.443694  2.058793  12.480149  14.407239
  gossip     2.0    total_forwards 20 112.500000 17.863518 104.139616 120.860384
  gossip     3.0    delivery_ratio 20   0.830500  0.045128   0.809379   0.851621
  gossip     3.0        duplicates 20 167.100000  9.025694 162.875845 171.324155
  gossip     3.0 propagation_delay 20  10.015010  0.980442   9.556149  10.473871
  gossip     3.0    total_forwards 20 249.150000 13.538541 242.813768 255.486232
  gossip     4.0    delivery_ratio 20   0.929000  0.036114   0.912098   0.945902
  gossip     4.0        duplicates 20 247.850000 10.974972 242.713555 252.986445
  gossip     4.0 propagation_delay 20   8.023286  1.437209   7.350652   8.695921
  gossip     4.0    total_forwards 20 339.750000 14.425398 332.998706 346.501294
  gossip     5.0    delivery_ratio 20   0.970500  0.018771   0.961715   0.979285
  gossip     5.0        duplicates 20 299.150000  8.904611 294.982514 303.317486
  gossip     5.0 propagation_delay 20   6.656570  0.876010   6.246585   7.066555
  gossip     5.0    total_forwards 20 395.200000 10.430724 390.318271 400.081729
  gossip     6.0    delivery_ratio 20   0.984000  0.012312   0.978238   0.989762
  gossip     6.0        duplicates 20 333.200000  8.003946 329.454038 336.945962
  gossip     6.0 propagation_delay 20   6.023180  0.597739   5.743430   6.302931
  gossip     6.0    total_forwards 20 430.600000  8.720212 426.518815 434.681185
Unable to revert mtime: /Library/Fonts
Matplotlib is building the font cache; this may take a moment.
```

The first plotting attempt did not create a plot because Matplotlib attempted to
initialize its cache in the system font location. A second diagnostic attempt
reported the same cache initialization and confirmed that the plot was absent:

```bash
ls -l outputs/csv/exp07_summary_20260820_074418.csv outputs/plots/exp07_3panel_20260820_074418.png 2>&1
'/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python' scripts/plot_results.py outputs/csv/exp07_results_20260820_074418.csv
```

```text
ls: outputs/plots/exp07_3panel_20260820_074418.png: No such file or directory
-rw-r--r--@ 1 wwiras staff 2334 Aug 20 08:29 outputs/csv/exp07_summary_20260820_074418.csv
Matplotlib is building the font cache; this may take a moment.
```

Plotting succeeded after directing only Matplotlib's cache to a temporary
directory; this did not alter the experiment or input data:

```bash
mkdir -p /tmp/exp07_mplconfig
MPLCONFIGDIR=/tmp/exp07_mplconfig '/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python' scripts/plot_results.py outputs/csv/exp07_results_20260820_074418.csv
ls -l outputs/plots/exp07_3panel_20260820_074418.png
```

```text
Matplotlib is building the font cache; this may take a moment.
Saved outputs/plots/exp07_3panel_20260820_074418.png
Plots saved (offset=False) with timestamp: 20260820_074418
-rw-r--r--@ 1 wwiras staff 159644 Aug 20 08:30 outputs/plots/exp07_3panel_20260820_074418.png
```

Final validation:

```bash
'/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python' - <<'PY'
import pandas as pd
from pathlib import Path
result_path = Path('outputs/csv/exp07_results_20260820_074418.csv')
trace_path = Path('outputs/csv/exp07_adaptive_trace_20260820_074418.csv')
df = pd.read_csv(result_path)
trace = pd.read_csv(trace_path)
metrics = ['delivery_ratio','propagation_delay','duplicates','total_forwards']
print('total_rows:', len(df))
print('invalid_metric_cells:', int(df[metrics].isna().sum().sum()))
for fanout in [2,3,4,5,6]:
    print(f'gossip_f{fanout}_n:', len(df[(df.strategy == 'gossip') & (df.fanout == fanout)]))
ahbn = df[df.strategy == 'ahbn']
print('ahbn_n:', len(ahbn))
print('ahbn_fanout_all_blank:', bool(ahbn.fanout.isna().all()))
print('trace_min_fanout:', int(trace.fanout.min()))
print('trace_max_fanout:', int(trace.fanout.max()))
print('trace_fanout_changes:', int(trace.fanout_changed.fillna(False).astype(bool).sum()))
print('trace_within_2_4:', bool(trace.fanout.between(2,4).all()))
g3 = df[(df.strategy == 'gossip') & (df.fanout == 3)]
merged = ahbn[['seed'] + metrics].merge(g3[['seed'] + metrics], on='seed', suffixes=('_ahbn','_gossip'), validate='one_to_one')
checks = {metric: bool(merged[f'{metric}_ahbn'].eq(merged[f'{metric}_gossip']).all()) for metric in metrics}
print('matched_seed_count:', len(merged))
print('seed_by_seed_metric_checks:', checks)
print('seed_by_seed_identical:', all(checks.values()) and len(merged) == 20)
PY
shasum -a 256 outputs/csv/exp07_results_20260820_074418.csv outputs/csv/exp07_adaptive_trace_20260820_074418.csv
git diff --check
git status --short
```

```text
total_rows: 120
invalid_metric_cells: 0
gossip_f2_n: 20
gossip_f3_n: 20
gossip_f4_n: 20
gossip_f5_n: 20
gossip_f6_n: 20
ahbn_n: 20
ahbn_fanout_all_blank: True
trace_min_fanout: 3
trace_max_fanout: 3
trace_fanout_changes: 0
trace_within_2_4: True
matched_seed_count: 20
seed_by_seed_metric_checks: {'delivery_ratio': True, 'propagation_delay': True, 'duplicates': True, 'total_forwards': True}
seed_by_seed_identical: True
72bd18236776bab009a70d832d84b4382f3034812340e76215dc2371585e2522  outputs/csv/exp07_results_20260820_074418.csv
4da401398b06d5b130e7a2fc331be5f50d9fce23080165c4ec742f4456f43fa5  outputs/csv/exp07_adaptive_trace_20260820_074418.csv
 M scripts/plot_results.py
 M scripts/summarize_results.py
?? docs/stage4_exp07.md
```

The before/after SHA-256 values are identical, confirming that neither raw CSV
was modified.

### Results and interpretation

- Result rows: 120.
- Gossip fixed fanout groups 2, 3, 4, 5, and 6: n = 20 each.
- AHBN: n = 20; result-row fanout is blank for all rows.
- Observed AHBN runtime fanout: minimum 3, maximum 3, changes 0; all observations
  are within the frozen controller bounds [2,4].
- AHBN and Gossip f=3 are identical seed-by-seed for delivery ratio, propagation
  delay, duplicates, and total forwards across all 20 seeds.

The frozen adaptive AHBN controller remained at its moderate operating fanout of
3 throughout Exp07. Under nominal Exp07 conditions, this yielded the same
dissemination behavior as fixed Gossip f=3. This is observed runtime behavior,
not an experiment-specific fixed-AHBN configuration.

The plot shows exactly five fixed-Gossip x-points (fanout 2–6), each with a 95%
CI error bar. AHBN is shown once per metric as a horizontal adaptive mean
reference with a 95% CI band. No AHBN fanout value or curve was fabricated.

### Generated files

- `outputs/csv/exp07_summary_20260820_074418.csv`
- `outputs/plots/exp07_3panel_20260820_074418.png`

### Final checklist

```text
Gossip x-points = exactly 5: PASS
AHBN fabricated x-points = 0: PASS
AHBN represented as adaptive reference: PASS
95% CI uses independent runs: PASS
Raw result CSV modified: NO
Adaptive trace modified: NO
Experiment rerun: NO
Controller/strategy/config/orchestration changes: NO
EXP07 FINAL ANALYSIS STATUS: PASS
```

Final artifact/diff review command:

```bash
git diff --check
git diff --stat
git status --short
sed -n '35,180p' scripts/plot_results.py
sed -n '1,180p' scripts/summarize_results.py
head -12 outputs/csv/exp07_summary_20260820_074418.csv
tail -35 docs/stage4_exp07.md
```

```text
v0.6/scripts/plot_results.py      | 150 +++++++++++---------------------------
v0.6/scripts/summarize_results.py |  52 ++++++++++++-
2 files changed, 89 insertions(+), 113 deletions(-)
 M scripts/plot_results.py
 M scripts/summarize_results.py
?? docs/stage4_exp07.md
```

`git diff --check` produced no output (PASS). The review also confirmed that the
AHBN summary fanout field is empty in the generated CSV.
