## STAGE 3 — EXTERNAL HYBRID BASELINE - DC-SoC

Implement + validate comparator DC-SoC in simulator

### 3.4 Sanity Validation

STAGE 3.4 — DC-SoC SANITY VALIDATION status:

- [X] S1  Cluster assignment correct
- [X] S2  Cluster heads correctly identified
- [X] S3  Intra-cluster dissemination observed
- [ ] S4  Inter-cluster dissemination observed
- [ ] S5  Duplicate behaviour plausible
- [ ] S6  Structural update works when triggered
- [ ] S7  No AHBN runtime controller used
- [ ] S8  Forwarding remains structurally determined
- [ ] S9  End-to-end propagation works
- [ ] S10 Same seed/topology is reproducible
- [ ] S11 AHBN-controller isolation confirmed

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

#### S3 — Intra-cluster dissemination observed

```bash
python -m scripts.validate_dcsoc_s3
========================================================================
STAGE 3.4 — DC-SoC SANITY VALIDATION
S3 — Intra-cluster dissemination observed
========================================================================

Test configuration:
  Topology type       : BA
  Topology nodes      : 30
  Topology edges      : 81
  BA m                : 3
  Seed                : 42
  DBSCAN eps          : 2.0
  DBSCAN min_samples  : 3

Cluster summary:
  Cluster 0:
    Members           : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    Cluster head      : 5

Transaction:
  Source node         : 0
  Source cluster      : 0
  Source is CH        : NO

Observed intra-cluster forwarding:
  sender -> receiver   sender_cluster  receiver_cluster  CH status
       0 -> 21             0                 0  sender=NO, receiver=NO
       0 -> 2              0                 0  sender=NO, receiver=NO
       0 -> 1              0                 0  sender=NO, receiver=NO
       1 -> 0              0                 0  sender=NO, receiver=NO
       1 -> 8              0                 0  sender=NO, receiver=NO
       1 -> 12             0                 0  sender=NO, receiver=NO
       2 -> 0              0                 0  sender=NO, receiver=NO
       2 -> 4              0                 0  sender=NO, receiver=NO
       2 -> 6              0                 0  sender=NO, receiver=NO
      21 -> 0              0                 0  sender=NO, receiver=NO
      21 -> 4              0                 0  sender=NO, receiver=NO
      21 -> 7              0                 0  sender=NO, receiver=NO
       8 -> 14             0                 0  sender=NO, receiver=NO
       8 -> 6              0                 0  sender=NO, receiver=NO
       8 -> 1              0                 0  sender=NO, receiver=NO
      12 -> 6              0                 0  sender=NO, receiver=NO
      12 -> 4              0                 0  sender=NO, receiver=NO
      12 -> 22             0                 0  sender=NO, receiver=NO
       6 -> 2              0                 0  sender=NO, receiver=NO
       6 -> 11             0                 0  sender=NO, receiver=NO
       6 -> 26             0                 0  sender=NO, receiver=NO
       4 -> 16             0                 0  sender=NO, receiver=NO
       4 -> 2              0                 0  sender=NO, receiver=NO
       4 -> 11             0                 0  sender=NO, receiver=NO
       7 -> 14             0                 0  sender=NO, receiver=NO
       7 -> 21             0                 0  sender=NO, receiver=NO
       7 -> 9              0                 0  sender=NO, receiver=NO
      22 -> 16             0                 0  sender=NO, receiver=NO
      22 -> 8              0                 0  sender=NO, receiver=NO
      22 -> 27             0                 0  sender=NO, receiver=NO
      16 -> 14             0                 0  sender=NO, receiver=NO
      16 -> 5              0                 0  sender=NO, receiver=YES
      16 -> 22             0                 0  sender=NO, receiver=NO
      14 -> 27             0                 0  sender=NO, receiver=NO
      14 -> 3              0                 0  sender=NO, receiver=NO
      14 -> 7              0                 0  sender=NO, receiver=NO
       9 -> 7              0                 0  sender=NO, receiver=NO
       9 -> 26             0                 0  sender=NO, receiver=NO
       9 -> 4              0                 0  sender=NO, receiver=NO
      11 -> 4              0                 0  sender=NO, receiver=NO
      11 -> 5              0                 0  sender=NO, receiver=YES
      11 -> 6              0                 0  sender=NO, receiver=NO
      26 -> 25             0                 0  sender=NO, receiver=NO
      26 -> 6              0                 0  sender=NO, receiver=NO
      26 -> 28             0                 0  sender=NO, receiver=NO
      27 -> 3              0                 0  sender=NO, receiver=NO
      27 -> 14             0                 0  sender=NO, receiver=NO
      27 -> 22             0                 0  sender=NO, receiver=NO
       3 -> 29             0                 0  sender=NO, receiver=NO
       3 -> 4              0                 0  sender=NO, receiver=NO
       3 -> 13             0                 0  sender=NO, receiver=NO
       5 -> 18             0                 0  sender=YES, receiver=NO
       5 -> 16             0                 0  sender=YES, receiver=NO
       5 -> 9              0                 0  sender=YES, receiver=NO
      28 -> 6              0                 0  sender=NO, receiver=NO
      28 -> 16             0                 0  sender=NO, receiver=NO
      28 -> 26             0                 0  sender=NO, receiver=NO
      25 -> 0              0                 0  sender=NO, receiver=NO
      25 -> 26             0                 0  sender=NO, receiver=NO
      25 -> 7              0                 0  sender=NO, receiver=NO
      29 -> 3              0                 0  sender=NO, receiver=NO
      29 -> 4              0                 0  sender=NO, receiver=NO
      29 -> 12             0                 0  sender=NO, receiver=NO
      13 -> 0              0                 0  sender=NO, receiver=NO
      13 -> 3              0                 0  sender=NO, receiver=NO
      13 -> 6              0                 0  sender=NO, receiver=NO
      18 -> 0              0                 0  sender=NO, receiver=NO
      18 -> 3              0                 0  sender=NO, receiver=NO
      18 -> 5              0                 0  sender=NO, receiver=YES

Validation checks:
  Actual forwarding events observed            : PASS
  Intra-cluster forwarding observed            : PASS
  Sender/receiver cluster membership valid     : PASS
  Non-noise cluster used                       : PASS

========================================================================
S3 RESULT: PASS
========================================================================
```

S3 executes the production `DCSOCStrategy` through the production event-driven
`Simulator`. A validation-only simulator subclass observes the non-self
point-to-point receive events scheduled by the existing transport path; it does
not select targets, create forwarding events, or modify dissemination.

With the frozen deterministic configuration (BA, 30 nodes, `m=3`, seed 42,
DBSCAN `eps=2.0`, `min_samples=3`), source Node 0 belongs to non-noise Cluster
0. The run observed 69 actual forwarding events for which the independently
read sender and receiver cluster IDs were both 0. For example, the first
observed transmission was Node 0 -> Node 21, and both nodes are members of
Cluster 0. Initial source self-delivery was explicitly excluded from the
forwarding observations.

Validation checks passed for actual forwarding, intra-cluster forwarding,
sender/receiver cluster membership, and use of a non-noise cluster.

> **S3 — PASS.** Intra-cluster dissemination is exercised and observable in
> the frozen DC-SoC implementation.
