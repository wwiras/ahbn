## STAGE 3 — EXTERNAL HYBRID BASELINE - DC-SoC

Implement + validate comparator DC-SoC in simulator

### 3.4 Sanity Validation

#### S1  Cluster assignment correct

```bash
% python -m scripts.validate_dcsoc_s1
========================================================================
STAGE 3.4 — DC-SoC SANITY VALIDATION
S1 — Cluster assignment correct
========================================================================

Test configuration:
  Topology type      : BA
  Topology nodes     : 30
  Topology edges     : 81
  BA m               : 3
  Seed               : 42
  DBSCAN eps         : 2.0
  DBSCAN min_samples : 3

Node construction:
  Nodes constructed  : 30
  Node dictionary    : PASS
  Physical overlay   : PASS

Node -> cluster assignments:
  Node  0 -> Cluster 0
  Node  1 -> Cluster 0
  Node  2 -> Cluster 0
  Node  3 -> Cluster 0
  Node  4 -> Cluster 0
  Node  5 -> Cluster 0
  Node  6 -> Cluster 0
  Node  7 -> Cluster 0
  Node  8 -> Cluster 0
  Node  9 -> Cluster 0
  Node 10 -> Cluster 0
  Node 11 -> Cluster 0
  Node 12 -> Cluster 0
  Node 13 -> Cluster 0
  Node 14 -> Cluster 0
  Node 15 -> Cluster 0
  Node 16 -> Cluster 0
  Node 17 -> Cluster 0
  Node 18 -> Cluster 0
  Node 19 -> Cluster 0
  Node 20 -> Cluster 0
  Node 21 -> Cluster 0
  Node 22 -> Cluster 0
  Node 23 -> Cluster 0
  Node 24 -> Cluster 0
  Node 25 -> Cluster 0
  Node 26 -> Cluster 0
  Node 27 -> Cluster 0
  Node 28 -> Cluster 0
  Node 29 -> Cluster 0

Cluster membership:
  Cluster 0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29] [n=30]

S1 invariant summary:
  Topology nodes             : 30
  Node objects               : 30
  Assigned nodes             : 30
  Unassigned nodes           : 0
  ClusterManager nodes       : 30
  Missing manager nodes      : 0
  Unknown manager nodes      : 0
  Duplicate memberships      : 0
  Node/manager inconsistencies: 0
  Final clusters             : 1

========================================================================
S1 PASS — Every node has exactly one valid and internally consistent DC-SoC cluster assignment.
========================================================================
```
S1 passes. The implementation satisfies the assignment-integrity checks we defined: all 30 topology nodes became real simulator Node objects, all 30 received exactly one cluster ID, the ClusterManager covers the same 30 nodes, and there are no missing, unknown, duplicate, or inconsistent memberships.
So we can mark:
[x] S1  Cluster assignment correct
There is one result worth recording for later, though it is not an S1 failure: with the frozen settings
BA, N=30, m=3, seed=42
eps=2.0
min_samples=3
the clustering collapsed the whole topology into:
Cluster 0 = all 30 nodes
For S1, that is perfectly valid because we were testing correctness of assignment, not whether the partition is useful or multi-clustered.

#### S2  Cluster-head identification correct

```bash
% /Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python -m scripts.validate_dcsoc_s2
========================================================================
STAGE 3.4 — DC-SoC SANITY VALIDATION
S2 — Cluster-head identification correct
========================================================================

Test configuration:
  Topology type      : BA
  Topology nodes     : 30
  Topology edges     : 81
  BA m               : 3
  Seed               : 42
  DBSCAN eps         : 2.0
  DBSCAN min_samples : 3

Cluster-head validation:

  Cluster 0:
    Members          : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    Member degrees   : [(0, 12), (1, 4), (2, 3), (3, 8), (4, 13), (5, 15), (6, 11), (7, 11), (8, 5), (9, 4), (10, 7), (11, 3), (12, 6), (13, 3), (14, 6), (15, 3), (16, 5), (17, 3), (18, 3), (19, 3), (20, 4), (21, 3), (22, 4), (23, 3), (24, 3), (25, 4), (26, 4), (27, 3), (28, 3), (29, 3)]
    Maximum degree   : 15
    Expected CH      : Node 5
    Actual CH        : Node 5
    Membership check : PASS
    Selection check  : PASS

Global checks:
  Non-noise clusters             : 1
  Cluster heads identified       : 1
  One CH per cluster             : PASS
  Every CH belongs to cluster    : PASS
  Highest-degree selection       : PASS
  Deterministic tie-breaking     : NOT OBSERVED (no natural maximum-degree tie)
  Noise excluded from CHs        : PASS
  Same-seed reproducibility      : PASS

========================================================================
S2 RESULT: PASS
Cluster-head identification is correct.
========================================================================
```

S2 passes cleanly. For Cluster 0, the independent physical-overlay-degree oracle identified Node 5 as the expected cluster head with the maximum degree of 15. The DC-SoC implementation also selected Node 5. Membership, exactly one cluster head per cluster, highest-degree selection, noise exclusion, and same-seed reproducibility all passed.

Deterministic lowest-node-ID tie-breaking was not naturally exercised because this topology contained no tie for the maximum physical-overlay degree. This is recorded as **not exercised in this topology**, not as a failure. The topology and frozen DBSCAN parameters are not changed merely to manufacture a tie.

This configuration produced one DBSCAN cluster containing all 30 nodes. That does not invalidate S2: S2 verifies cluster-head selection for the clustering actually produced. No DBSCAN tuning is required merely to obtain a more varied partition.

Therefore, freeze the S2 result as:

> **S2 — PASS.** DC-SoC correctly identifies one valid cluster head per non-noise cluster using the highest physical-overlay degree. The selected cluster head belongs to its assigned cluster and is reproducible under the same seed. Deterministic tie-breaking was not naturally exercised because no maximum-degree tie occurred.

Stage 3.4 status:

- [x] S1 — Cluster assignment correct
- [x] S2 — Cluster-head identification correct
- [ ] S3 — Next

Stage 3.4 can now proceed directly to S3 without changing the DC-SoC implementation or parameters.
