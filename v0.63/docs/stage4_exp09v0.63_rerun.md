# Stage 4 — Exp09 ControlSim v0.63 Rerun

## Environment

Pinned project and Python are enforced by `scripts/run_stage4_v063.sh`. Smoke uses seed 42 and one run per ER density × strategy cell; formal preserves 20 runs per setting.

## GKE Canonical Source

Live GKE final S5 mapping inspected and matched.

## v0.62 -> v0.63 Migration

Only AHBN actuator range/mapping changed; Exp09 factors and all baselines remain frozen.

## Regression Validation

See Exp07 gate documentation.

## Smoke Command

```bash
cd /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63
bash scripts/run_stage4_exp09_v063_smoke.sh
```

## Smoke Terminal Output

Not run. The timestamped output directory will contain `terminal.log` with stdout, stderr, exit code, and output path.

## Smoke Validation

Pending manual execution and only after Exp07 review.

## Formal Command

```bash
cd /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63
bash scripts/run_stage4_exp09_v063_formal.sh
```

## Formal Terminal Output

Not run.

## Aggregation

The prepared analyzer reports each strategy × ER probability and AHBN trace occupancy.

## Statistical Analysis

Pending manual formal run.

## Scientific Interpretation

Do not assume density raises fanout: increased duplicate pressure can lower `z = -d_hat + l_hat + u_hat + c_hat`.

## Final Status

Prepared; not executed.
