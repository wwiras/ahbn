## STAGE 1 — CANONICAL SANITY VALIDATION

This stage validates AHBN canonical from theory to code. 

- Verify d̂, l̂, û, ĉ
- Verify adaptive score / probability
- Verify Gossip ↔ Structured decisions
- Verify fanout movement
- Verify mode_switched / fanout_changed
- Verify expected behaviour under Exp07–Exp12 conditions

### Results

```bash
(venv0.5) wwiras@wwirass-MacBook-Air AHBNProj % cd ahbn/v0.6 
(venv0.5) wwiras@wwirass-MacBook-Air v0.6 % >....                                                                         
)

ctl = AHBNController(p)

cases = {
    "neutral":            (0.5, 0.5, 0.5, 0.5),
    "high_duplicate":     (0.9, 0.5, 0.5, 0.5),
    "high_utilization":   (0.5, 0.5, 0.9, 0.5),
    "high_latency":       (0.5, 0.9, 0.5, 0.5),
    "high_churn":         (0.5, 0.5, 0.5, 0.9),
    "dup_util_max":       (1.0, 0.5, 1.0, 0.5),
    "lat_churn_max":      (0.5, 1.0, 0.5, 1.0),
    "gossip_extreme":     (0.0, 1.0, 0.0, 1.0),
    "structured_extreme": (1.0, 0.0, 1.0, 0.0),
}

print(
    f"{'CASE':22s} {'SCORE':>9s} {'WEIGHT':>9s} "
    f"{'MODE':>10s} {'FANOUT':>7s}"
)

for name, (d, l, u, c) in cases.items():
    state = NodeControlState(
        d_hat=d,
        l_hat=l,
        u_hat=u,
        c_hat=c,
    )

    ctl.decide_mode_and_fanout(state)

    print(
        f"{name:22s} "
        f"{state.score:9.6f} "
        f"{state.weight:9.6f} "
        f"{state.mode:>10s} "
        f"{state.fanout:7d}"
    )
PY
CASE                       SCORE    WEIGHT       MODE  FANOUT
neutral                 0.000000  0.500000     gossip       3
high_duplicate         -0.400000  0.401312    cluster       3
high_utilization       -0.400000  0.401312    cluster       3
high_latency            0.400000  0.598688     gossip       3
high_churn              0.400000  0.598688     gossip       3
dup_util_max           -1.000000  0.268941    cluster       3
lat_churn_max           1.000000  0.731059     gossip       3
gossip_extreme          2.000000  0.880797     gossip       4
structured_extreme     -2.000000  0.119203    cluster       2
(venv0.5) wwiras@wwirass-MacBook-Air v0.6 % python3 - <<'PY'
from ahbn.control import AHBNController, AHBNParams, NodeControlState

ctl = AHBNController(AHBNParams(alpha=0.3))
s = NodeControlState()

print("Initial:", s.d_hat)

for obs in [1.0, 1.0, 1.0, 0.0]:
    ctl.update_metrics(s, duplicate_obs=obs)
    print(s.d_hat)
PY
Initial: 0.0
0.3
0.51
0.657
0.4599
(venv0.5) wwiras@wwirass-MacBook-Air v0.6 % python3 - <<'PY'
from ahbn.control import AHBNController, AHBNParams, NodeControlState

ctl = AHBNController(AHBNParams(alpha=0.3))

s = NodeControlState(
    d_hat=0.40,
    l_hat=0.30,
    u_hat=0.20,
    c_hat=0.10,
)

ctl.update_metrics(
    s,
    duplicate_obs=None,
    latency_obs=None,
    utilization_obs=None,
    churn_obs=0.50,
)

print("d_hat =", s.d_hat)
print("l_hat =", s.l_hat)
print("u_hat =", s.u_hat)
print("c_hat =", s.c_hat)
PY
d_hat = 0.4
l_hat = 0.3
u_hat = 0.2
c_hat = 0.21999999999999997
(venv0.5) wwiras@wwirass-MacBook-Air v0.6 % python3 - <<'PY'
import yaml

src = "configs/exp12_mixed_resources.yaml"
dst = "configs/sanity_neutral.yaml"

with open(src) as f:
    cfg = yaml.safe_load(f)

cfg["runs_per_setting"] = 1
cfg["num_nodes"] = 20
cfg["use_topology_cache"] = False
cfg["strategies"] = ["ahbn"]
cfg["resource_scenarios"] = ["balanced"]

with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print("Created:", dst)
PY
Created: configs/sanity_neutral.yaml
(venv0.5) wwiras@wwirass-MacBook-Air v0.6 % python3 run_batch.py --config configs/sanity_neutral.yaml
Saved outputs/csv/exp12_results_20260815_205140.csv
Saved outputs/csv/exp12_adaptive_trace_20260815_205140.csv
(venv0.5) wwiras@wwirass-MacBook-Air v0.6 % python3 - <<'PY'
import glob
import pandas as pd

path = sorted(
    glob.glob("outputs/csv/exp12_adaptive_trace_*.csv")
)[-1]

df = pd.read_csv(path)

cols = [
    "time",
    "node_id",
    "event_type",
    "duplicate_obs",
    "latency_obs",
    "utilization_obs",
    "churn_obs",
    "d_hat",
    "l_hat",
    "u_hat",
    "c_hat",
    "score",
    "weight",
    "mode",
    "fanout",
    "mode_switched",
    "fanout_changed",
]

print("FILE:", path)
print(df[cols].head(20).to_string(index=False))

print("\nMODE COUNTS")
print(df["mode"].value_counts())

print("\nFANOUT COUNTS")
print(df["fanout"].value_counts())
PY
FILE: outputs/csv/exp12_adaptive_trace_20260815_205140.csv
    time  node_id        event_type  duplicate_obs  latency_obs  utilization_obs  churn_obs  d_hat    l_hat    u_hat  c_hat     score   weight    mode  fanout  mode_switched  fanout_changed
0.000000        0       new_receive       0.000000     0.000000         0.000000        NaN  0.000 0.000000 0.000000    0.0  0.000000 0.500000  gossip       3          False           False
2.048978        4       new_receive       0.000000     0.650680         0.000000        NaN  0.000 0.195204 0.000000    0.0  0.195204 0.548647  gossip       3          False           False
2.148310        3       new_receive       0.000000     0.661362         0.000000        NaN  0.000 0.198409 0.000000    0.0  0.198409 0.549440  gossip       3          False           False
3.805335       15       new_receive       0.000000     0.614894         0.000000        NaN  0.000 0.184468 0.000000    0.0  0.184468 0.545987  gossip       3          False           False
3.817717       13       new_receive       0.000000     0.616556         0.000000        NaN  0.000 0.184967 0.000000    0.0  0.184967 0.546110  gossip       3          False           False
3.835559        6       new_receive       0.000000     0.605346         0.000000        NaN  0.000 0.181604 0.000000    0.0  0.181604 0.545277  gossip       3          False           False
4.041514        9       new_receive       0.000000     0.632501         0.000000        NaN  0.000 0.189750 0.000000    0.0  0.189750 0.547296  gossip       3          False           False
4.167077        0 duplicate_receive       0.500000     0.658183         0.454545        NaN  0.150 0.197455 0.136364    0.0 -0.088909 0.477787 cluster       3           True           False
4.288575        0 duplicate_receive       0.666667     0.660522         0.303030        NaN  0.305 0.336375 0.186364    0.0 -0.154989 0.461330 cluster       3          False           False
5.306635       12       new_receive       0.000000     0.577134         0.000000        NaN  0.000 0.173140 0.000000    0.0  0.173140 0.543177  gossip       3          False           False
5.348813       14       new_receive       0.000000     0.581923         0.000000        NaN  0.000 0.174577 0.000000    0.0  0.174577 0.543534  gossip       3          False           False
5.431001        3 duplicate_receive       0.500000     0.591904         0.681818        NaN  0.150 0.316457 0.204545    0.0 -0.038088 0.490479 cluster       3           True           False
5.467221        2       new_receive       0.000000     0.601721         0.000000        NaN  0.000 0.180516 0.000000    0.0  0.180516 0.545007  gossip       3          False           False
5.635768        4 duplicate_receive       0.500000     0.623036         0.681818        NaN  0.150 0.323554 0.204545    0.0 -0.030992 0.492253 cluster       3           True           False
5.673188        4 duplicate_receive       0.666667     0.629362         0.454545        NaN  0.305 0.415296 0.279545    0.0 -0.169249 0.457788 cluster       3          False           False
5.912259        3 duplicate_receive       0.666667     0.629723         0.454545        NaN  0.305 0.410437 0.279545    0.0 -0.174109 0.456582 cluster       3          False           False
5.952940       11       new_receive       0.000000     0.634725         0.000000        NaN  0.000 0.190417 0.000000    0.0  0.190417 0.547461  gossip       3          False           False
5.961013       16       new_receive       0.000000     0.635701         0.000000        NaN  0.000 0.190710 0.000000    0.0  0.190710 0.547534  gossip       3          False           False
6.857978        9 duplicate_receive       0.500000     0.578409         0.375000        NaN  0.150 0.306348 0.112500    0.0  0.043848 0.510960  gossip       3          False           False
6.882342       14 duplicate_receive       0.500000     0.588894         0.375000        NaN  0.150 0.298872 0.112500    0.0  0.036372 0.509092  gossip       3          False           False

MODE COUNTS
mode
gossip     23
cluster    11
Name: count, dtype: int64

FANOUT COUNTS
fanout
3    34
Name: count, dtype: int64
(venv0.5) wwiras@wwirass-MacBook-Air v0.6 % >....                                                                         
import pandas as pd

path = sorted(
    glob.glob("outputs/csv/exp12_adaptive_trace_*.csv")
)[-1]

df = pd.read_csv(path)

checks = {}

for col in [
    "duplicate_obs",
    "latency_obs",
    "utilization_obs",
    "churn_obs",
]:
    s = df[col].dropna()
    checks[f"{col}_bounded"] = bool(((s >= 0) & (s <= 1)).all())

for col in ["d_hat", "l_hat", "u_hat", "c_hat", "weight"]:
    checks[f"{col}_bounded"] = bool(
        ((df[col] >= 0) & (df[col] <= 1)).all()
    )

checks["fanout_bounded"] = bool(
    ((df["fanout"] >= 2) & (df["fanout"] <= 4)).all()
)

expected_mode = df["weight"].apply(
    lambda w: "gossip" if w >= 0.5 else "cluster"
)

checks["mode_matches_weight"] = bool(
    (df["mode"] == expected_mode).all()
)

for name, ok in checks.items():
    print(f"{name:30s}: {'PASS' if ok else 'FAIL'}")
PY
duplicate_obs_bounded         : PASS
latency_obs_bounded           : PASS
utilization_obs_bounded       : PASS
churn_obs_bounded             : PASS
d_hat_bounded                 : PASS
l_hat_bounded                 : PASS
u_hat_bounded                 : PASS
c_hat_bounded                 : PASS
weight_bounded                : PASS
fanout_bounded                : PASS
mode_matches_weight           : PASS
(venv0.5) wwiras@wwirass-MacBook-Air v0.6 % python3 - <<'PY'
import yaml

src = "configs/exp10_failure.yaml"
dst = "configs/sanity_overload.yaml"

with open(src) as f:
    cfg = yaml.safe_load(f)

cfg["runs_per_setting"] = 1
cfg["num_nodes"] = 20
cfg["use_topology_cache"] = False
cfg["strategies"] = ["ahbn"]
cfg["failure_modes"] = ["overload"]

with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print("Created:", dst)
PY
Created: configs/sanity_overload.yaml
(venv0.5) wwiras@wwirass-MacBook-Air v0.6 % python3 run_batch.py --config configs/sanity_overload.yaml
Saved outputs/csv/exp10_results_20260815_212033.csv
Saved outputs/csv/exp10_adaptive_trace_20260815_212033.csv
(venv0.5) wwiras@wwirass-MacBook-Air v0.6 % python3 - <<'PY'
import yaml

src = "configs/exp11_churn.yaml"
dst = "configs/sanity_churn.yaml"

with open(src) as f:
    cfg = yaml.safe_load(f)

cfg["runs_per_setting"] = 1
cfg["num_nodes"] = 20
cfg["use_topology_cache"] = False
cfg["strategies"] = ["ahbn"]
cfg["churn_rates"] = [0.30]

with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print("Created:", dst)
PY
Created: configs/sanity_churn.yaml
(venv0.5) wwiras@wwirass-MacBook-Air v0.6 % python3 run_batch.py --config configs/sanity_churn.yaml
Saved outputs/csv/exp11_results_20260815_215238.csv
Saved outputs/csv/exp11_adaptive_trace_20260815_215238.csv
(venv0.5) wwiras@wwirass-MacBook-Air v0.6 % >....                                                                         
    expected_score,
    atol=1e-10,
)

expected_weight = 1.0 / (
    1.0 + np.exp(-df["score"])
)

weight_ok = np.allclose(
    df["weight"],
    expected_weight,
    atol=1e-10,
)

expected_mode = np.where(
    df["weight"] >= 0.5,
    "gossip",
    "cluster",
)

mode_ok = (df["mode"] == expected_mode).all()

expected_fanout = np.rint(
    np.clip(
        2 + df["weight"] * 2,
        2,
        4,
    )
).astype(int)

fanout_ok = (
    df["fanout"].astype(int) == expected_fanout
).all()

print("score equation :", "PASS" if score_ok else "FAIL")
print("sigmoid weight :", "PASS" if weight_ok else "FAIL")
print("mode decision  :", "PASS" if mode_ok else "FAIL")
print("fanout equation:", "PASS" if fanout_ok else "FAIL")
PY
score equation : PASS
sigmoid weight : PASS
mode decision  : PASS
fanout equation: PASS
```