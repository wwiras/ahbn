# AHBN Experimental Roadmap

## Stage A: Highest-Value Baseline Analysis


### 1. Experiment 1: Fanout vs. Duplication
* **Why first:** Directly motivates AHBN adaptive fanout.
* **Purpose:** Prove duplicate explosion in gossip.
* **Metrics:** Duplicate ratio, transmissions, propagation time, delivery ratio.
* **Anticipation:** Duplicates grow sharply; delay improves only modestly.
* **Output:** Strongest early graph.

### 2. Experiment 2: CH Count vs. Node Count
* **Why second:** Proves static clustering is sensitive and brittle.
* **Purpose:** Show bottleneck/imbalance of CH-based dissemination.
* **Metrics:** Propagation time, max CH load, load variance, duplicates.
* **Anticipation:** Too few CHs overload; too many add inefficiency.
* **Output:** U-shaped or bottleneck curve.

### 3. Experiment 3: Topology Density vs. Performance
* **Why third:** Shows dissemination is topology-sensitive.
* **Purpose:** Prove dense connectivity amplifies duplicates.
* **Metrics:** Duplicates, useful transmissions, delay.
* **Anticipation:** Denser graphs increase duplicate opportunities.
* **Output:** Strong justification for adaptive behavior.

### 4. Experiment 4: CH Overload / Failure
* **Why fourth:** Proves structural fragility.
* **Purpose:** Show static cluster degrades under CH stress.
* **Metrics:** Delay, delivery ratio, recovery time.
* **Anticipation:** CH failure harms performance much more than member failure.
* **Output:** Resilience argument.

---

## Stage B: First AHBN Validation
*Once baseline is clear, validate AHBN against the same conditions.*

### 5. Experiment 7: AHBN vs. Gossip (Fanout Pressure)
* **Why fifth:** Directly proves AHBN suppresses duplicates where gossip explodes.
* **Purpose:** Show AHBN reduces redundancy without large delay penalty.
* **Metrics:** Duplicates, transmissions, delay, adaptive fanout.
* **Anticipation:** AHBN lower duplicates, similar-ish delay.

### 6. Experiment 8: AHBN vs. Cluster (CH Bottleneck)
* **Why sixth:** Proves AHBN reduces rigid dependence on CHs.
* **Purpose:** Compare load concentration and delay.
* **Metrics:** Max load, delay, delivery ratio, fallback count.
* **Anticipation:** AHBN distributes load better.

### 7. Experiment 9: AHBN under Dense Topologies
* **Why seventh:** Shows AHBN adapts better as density grows.
* **Purpose:** Validate graceful scaling.
* **Metrics:** Duplicates, delay, efficiency.
* **Anticipation:** AHBN sits between gossip and cluster, with better trade-off.

### 8. Experiment 10: AHBN under Failure / Overload
* **Why eighth:** Shows graceful degradation.
* **Purpose:** Compare resilience.
* **Metrics:** Delivery ratio, delay, recovery time.
* **Anticipation:** AHBN degrades less severely than static cluster.

---

## Stage C: Extended Realism
*Do these after the first 8 are stable.*

| Exp # | Title | Purpose | Anticipation |
| :--- | :--- | :--- | :--- |
| 5 | Churn Sensitivity | Analyze dynamic membership effect. | All degrade; AHBN more balanced later. |
| 6 | Heterogeneous Resources | Analyze mixed capability nodes. | Weak nodes hurt performance; adaptation helps. |
| 11 | AHBN under Churn | Validate AHBN under instability. | AHBN improves balance, not necessarily every metric. |
| 12 | AHBN in Mixed Resources | Validate practical usefulness in realistic systems. | AHBN behaves sensibly under uneven resources. |

---

## One-Line Role Summary

| Exp | Role |
| :--- | :--- |
| 1 | Prove duplicate explosion in gossip |
| 2 | Prove CH bottleneck and imbalance |
| 3 | Prove topology sensitivity |
| 4 | Prove CH fragility under stress |
| 5 | Prove performance drop under churn |
| 6 | Prove impact of heterogeneous resources |
| 7 | Show AHBN suppresses gossip duplicates |
| 8 | Show AHBN reduces CH bottleneck |
| 9 | Show AHBN scales better in dense topologies |
| 10 | Show AHBN degrades more gracefully under failure |
| 11 | Show AHBN adapts under churn |
| 12 | Show AHBN is practical in heterogeneous environments |