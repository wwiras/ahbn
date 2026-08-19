## STAGE 3 — EXTERNAL HYBRID BASELINE - DC-SoC

Implement + validate comparator DC-SoC in simulator

### 3.4 Sanity Validation

STAGE 3.4 — DC-SoC SANITY VALIDATION status:

- [X] S1  Cluster assignment correct
- [X] S2  Cluster heads correctly identified
- [X] S3  Intra-cluster dissemination observed
- [X] S4  Cluster-head relay behaviour correct
- [X] S5  Duplicate behaviour plausible
- [X] S6  Structural update works when triggered
- [X] S7  No AHBN runtime controller used
- [X] S8  Forwarding remains structurally determined
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
% python -m scripts.validate_dcsoc_s2
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
$ python -m scripts.validate_dcsoc_s3
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

#### S4 — Cluster-head relay behaviour correct

```bash
$ python -m scripts.validate_dcsoc_s4
========================================================================
STAGE 3.4 — DC-SoC SANITY VALIDATION
S4 — Cluster-head relay behaviour correct
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
    Expected CH       : 5
    Actual CH         : 5

Transaction:
  Source node         : 0
  Source selection    : lowest-ID non-CH member of Cluster 0
  Source cluster      : 0
  Cluster head        : 5

CH relay validation:
  CH reached          : PASS
  CH relay path       : PASS
  Other clusters      : 0
  Valid relay targets : []
  Selected relay      : []
  Selected local      : [9, 16, 18]
  Invalid targets     : 0
  Fanout budget       : 3/3

Inter-cluster dissemination : NOT EXERCISED

S4 result:
  Cluster-head relay behaviour : PASS

========================================================================
S4 RESULT: PASS
========================================================================
```

S4 runs the production `DCSOCStrategy` through the production event-driven
`Simulator`. Validation-only subclasses observe the target list returned by
the real strategy and the non-self receive events scheduled by the real
transport path; they do not choose targets, create forwarding events, or
change dissemination behaviour.

The expected cluster head is derived independently from physical-overlay
degree using highest degree with lowest node ID as the deterministic
tie-break. Under the frozen deterministic configuration (BA, 30 nodes,
`m=3`, seed 42, DBSCAN `eps=2.0`, `min_samples=3`), expected CH Node 5 matched
observed CH Node 5. The deterministic source was Node 0, selected as the
lowest-ID non-CH member of non-noise Cluster 0. The transaction reached Node 5
and caused exactly one first-receive execution of the cluster-head strategy
path.

The frozen cluster-head path found no valid structured relay targets and used
the remaining bounded fanout for local targets Nodes 9, 16, and 18. Its three
scheduled transmissions matched the production strategy output, stayed within
the configured fanout of 3, and contained zero invalid or noise targets.

This deterministic configuration produces one non-noise cluster, so there are
zero other clusters and the independently derived valid inter-cluster relay
target set is empty. S4 confirms that the selected cluster head is reached and
that cluster-head relay behaviour executes correctly under the frozen
configuration. No inter-cluster target was fabricated. Therefore,
**inter-cluster dissemination was NOT EXERCISED**, and no claim of successful
inter-cluster dissemination validation is made.

> **S4 — PASS.** Cluster-head relay behaviour is correct for the clustering
> actually produced by the frozen deterministic sanity configuration.

#### S5 — Duplicate behaviour plausible

S5 asks whether duplicate receptions produced by the frozen DC-SoC forwarding
policy occur only where the observed forwarding structure permits them, and
whether the simulator counts them consistently. It reconstructs receiver-based
duplicates independently from the actual processed reception-event order, using
its own seen-node set rather than the simulator's duplicate-counting helper.

Exact command:

```bash
$ python -m scripts.validate_dcsoc_s5
```

Actual output:

```text
========================================================================
STAGE 3.4 — DC-SoC SANITY VALIDATION
S5 — Duplicate behaviour plausible
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
  Source selection    : lowest-ID non-CH member of Cluster 0

Duplicate-accounting semantics:
  Unit                : processed reception at an active receiver
  First reception     : marks receiver seen; not a duplicate
  Later reception     : increments message duplicate count and does not forward
  Source self-receive : included as the source's first reception

Reception / duplicate trace:
  #07 t=2.095511: receiver 0 first from 0 (#01), duplicate from 2; cluster=0; roles=source->source, later member->source; overlay=physical; edge=YES
  #08 t=2.146006: receiver 0 first from 0 (#01), duplicate from 1; cluster=0; roles=source->source, later member->source; overlay=physical; edge=YES
  #12 t=2.288575: receiver 4 first from 2 (#10), duplicate from 21; cluster=0; roles=member->member, later member->member; overlay=physical; edge=YES
  #13 t=2.291514: receiver 0 first from 0 (#01), duplicate from 21; cluster=0; roles=source->source, later member->source; overlay=physical; edge=YES
  #15 t=3.066196: receiver 6 first from 2 (#09), duplicate from 8; cluster=0; roles=member->member, later member->member; overlay=physical; edge=YES
  #16 t=3.113966: receiver 4 first from 2 (#10), duplicate from 12; cluster=0; roles=member->member, later member->member; overlay=physical; edge=YES
  #17 t=3.118787: receiver 1 first from 0 (#02), duplicate from 8; cluster=0; roles=source->member, later member->member; overlay=physical; edge=YES
  #20 t=3.228018: receiver 2 first from 0 (#03), duplicate from 4; cluster=0; roles=source->member, later member->member; overlay=physical; edge=YES
  #21 t=3.238089: receiver 6 first from 2 (#09), duplicate from 12; cluster=0; roles=member->member, later member->member; overlay=physical; edge=YES
  #22 t=3.241379: receiver 21 first from 0 (#04), duplicate from 7; cluster=0; roles=source->member, later member->member; overlay=physical; edge=YES
  #24 t=3.281973: receiver 2 first from 0 (#03), duplicate from 6; cluster=0; roles=source->member, later member->member; overlay=physical; edge=YES
  #27 t=3.322653: receiver 11 first from 4 (#25), duplicate from 6; cluster=0; roles=member->member, later member->member; overlay=physical; edge=YES
  #28 t=3.373128: receiver 14 first from 8 (#19), duplicate from 7; cluster=0; roles=member->member, later member->member; overlay=physical; edge=YES
  #29 t=4.141221: receiver 8 first from 1 (#05), duplicate from 22; cluster=0; roles=member->member, later member->member; overlay=physical; edge=YES
  #32 t=4.238493: receiver 16 first from 4 (#18), duplicate from 22; cluster=0; roles=member->member, later member->member; overlay=physical; edge=YES
  #33 t=4.256196: receiver 14 first from 8 (#19), duplicate from 16; cluster=0; roles=member->member, later member->member; overlay=physical; edge=YES
  #34 t=4.278478: receiver 7 first from 21 (#11), duplicate from 14; cluster=0; roles=member->member, later member->member; overlay=physical; edge=YES
  #35 t=4.292854: receiver 27 first from 22 (#30), duplicate from 14; cluster=0; roles=member->member, later member->member; overlay=physical; edge=YES
  #37 t=4.319177: receiver 5 first from 11 (#36), duplicate from 16; cluster=0; roles=member->CH, later member->CH; overlay=physical; edge=YES
  #38 t=4.325506: receiver 22 first from 12 (#14), duplicate from 16; cluster=0; roles=member->member, later member->member; overlay=physical; edge=YES
  #39 t=4.340920: receiver 4 first from 2 (#10), duplicate from 11; cluster=0; roles=member->member, later member->member; overlay=physical; edge=YES
  #40 t=4.358201: receiver 6 first from 2 (#09), duplicate from 11; cluster=0; roles=member->member, later member->member; overlay=physical; edge=YES
  #41 t=4.414717: receiver 7 first from 21 (#11), duplicate from 9; cluster=0; roles=member->member, later member->member; overlay=physical; edge=YES
  #42 t=4.420610: receiver 6 first from 2 (#09), duplicate from 26; cluster=0; roles=member->member, later member->member; overlay=physical; edge=YES
  #43 t=4.432994: receiver 4 first from 2 (#10), duplicate from 9; cluster=0; roles=member->member, later member->member; overlay=physical; edge=YES
  #44 t=4.446364: receiver 26 first from 6 (#26), duplicate from 9; cluster=0; roles=member->member, later member->member; overlay=physical; edge=YES
  #47 t=5.198403: receiver 3 first from 14 (#31), duplicate from 27; cluster=0; roles=member->member, later member->member; overlay=physical; edge=YES
  #48 t=5.255724: receiver 14 first from 8 (#19), duplicate from 27; cluster=0; roles=member->member, later member->member; overlay=physical; edge=YES
  #50 t=5.332814: receiver 22 first from 12 (#14), duplicate from 27; cluster=0; roles=member->member, later member->member; overlay=physical; edge=YES
  #52 t=5.345394: receiver 9 first from 7 (#23), duplicate from 5; cluster=0; roles=member->member, later CH->member; overlay=physical; edge=YES
  #53 t=5.381410: receiver 16 first from 4 (#18), duplicate from 5; cluster=0; roles=member->member, later CH->member; overlay=physical; edge=YES
  #54 t=5.383986: receiver 4 first from 2 (#10), duplicate from 3; cluster=0; roles=member->member, later member->member; overlay=physical; edge=YES
  #55 t=5.468938: receiver 26 first from 6 (#26), duplicate from 28; cluster=0; roles=member->member, later member->member; overlay=physical; edge=YES
  #57 t=5.552662: receiver 16 first from 4 (#18), duplicate from 28; cluster=0; roles=member->member, later member->member; overlay=physical; edge=YES
  #58 t=5.574233: receiver 7 first from 21 (#11), duplicate from 25; cluster=0; roles=member->member, later member->member; overlay=physical; edge=YES
  #59 t=5.615290: receiver 0 first from 0 (#01), duplicate from 25; cluster=0; roles=source->source, later member->source; overlay=physical; edge=YES
  #60 t=5.648217: receiver 26 first from 6 (#26), duplicate from 25; cluster=0; roles=member->member, later member->member; overlay=physical; edge=YES
  #61 t=5.650264: receiver 6 first from 2 (#09), duplicate from 28; cluster=0; roles=member->member, later member->member; overlay=physical; edge=YES
  #62 t=6.275619: receiver 3 first from 14 (#31), duplicate from 29; cluster=0; roles=member->member, later member->member; overlay=physical; edge=YES
  #63 t=6.339238: receiver 4 first from 2 (#10), duplicate from 29; cluster=0; roles=member->member, later member->member; overlay=physical; edge=YES
  #64 t=6.448606: receiver 0 first from 0 (#01), duplicate from 13; cluster=0; roles=source->source, later member->source; overlay=physical; edge=YES
  #65 t=6.462138: receiver 12 first from 1 (#06), duplicate from 29; cluster=0; roles=member->member, later member->member; overlay=physical; edge=YES
  #66 t=6.483391: receiver 0 first from 0 (#01), duplicate from 18; cluster=0; roles=source->source, later member->source; overlay=physical; edge=YES
  #67 t=6.514939: receiver 6 first from 2 (#09), duplicate from 13; cluster=0; roles=member->member, later member->member; overlay=physical; edge=YES
  #68 t=6.536999: receiver 3 first from 14 (#31), duplicate from 13; cluster=0; roles=member->member, later member->member; overlay=physical; edge=YES
  #69 t=6.617437: receiver 5 first from 11 (#36), duplicate from 18; cluster=0; roles=member->CH, later member->CH; overlay=physical; edge=YES
  #70 t=6.625239: receiver 3 first from 14 (#31), duplicate from 18; cluster=0; roles=member->member, later member->member; overlay=physical; edge=YES

Accounting summary:
  Total transmissions              : 69
  Unique receivers                 : 23
  Independent duplicate count      : 47
  Simulator-reported duplicates    : 47
  Accounting match                 : PASS

Structural plausibility:
  Duplicate receptions observed    : 47
  Valid forwarding/overlay edges   : PASS

Checks:
  [PASS] Duplicate accounting semantics identified
  [PASS] First receptions are not counted as duplicates
  [PASS] Observed duplicates are structurally plausible
  [PASS] Independent duplicate count matches simulator accounting
  [N/A] Zero-duplicate case structurally justified (duplicates observed)

Final result:
  S5 duplicate behaviour plausible: PASS
```

> **S5 — PASS.** Under the tested frozen DC-SoC forwarding policy, the
> observed duplicate behaviour is structurally plausible and the receiver-based
> duplicate accounting is internally consistent. This is a sanity result only;
> it makes no comparative or duplicate-rate claim.

#### S6 — Structural update works when triggered

The S6 validator exercises the production churn path rather than calling the
repair routine directly:

```bash
$ python scripts/validate_dcsoc_s6.py
```

Observed output:

```text
========================================================================
STAGE 3.4 — DC-SoC SANITY VALIDATION
S6 — Structural update works when triggered
========================================================================

Test configuration:
  Topology type       : BA
  Topology nodes      : 30
  Topology edges      : 81
  BA m                : 3
  Seed                : 42
  DBSCAN eps          : 2.0
  DBSCAN min_samples  : 3

Frozen structural-update mechanism:
  Trigger type        : active node availability transition (churn leave/join)
  Trigger location    : Simulator.run() -> handle_churn_leave()/handle_churn_join()
  Update function     : repair_topology_after_churn()

Before structural change:
  Nodes               : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
  Edges               : 81
  Clusters            : {0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]}
  Cluster heads       : {0: 5}
  Trigger condition   : FALSE

Deterministic trigger event:
  Change applied      : production churn_leave event
  Affected node/edge  : node 5; 15 physical edges inactive
  Reason              : node 5 is the initial CH of Cluster 0

Trigger validation:
  Expected trigger    : TRUE
  Actual trigger      : TRUE
  Trigger check       : PASS
  Repair counter      : 0 -> 1
  Update executed     : PASS

After structural update:
  Nodes               : [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
  Edges               : 66
  Clusters            : {0: [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]}
  Cluster heads       : {0: 0}
  Structure changed   : PASS

Independent reconstruction:
  Expected clusters   : {0: [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]}
  Actual clusters     : {0: [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]}
  Cluster check       : PASS
  Expected CHs        : {0: 0}
  Actual CHs          : {0: 0}
  CH check            : PASS

Structural integrity:
  One cluster/node    : PASS
  One CH/cluster      : PASS
  CH membership       : PASS
  Noise handling      : PASS
  Valid references    : PASS

Forwarding-policy isolation:
  Runtime forwarding adaptation introduced : NO
  Check                                 : PASS

------------------------------------------------------------------------
S6 RESULT: PASS
------------------------------------------------------------------------
The deterministic node-availability change genuinely satisfied the frozen
DC-SoC structural-update trigger. The resulting active memberships and
cluster heads matched the independent post-change reconstruction. No
unrelated runtime forwarding adaptation was introduced.
```

Using the frozen BA(30, 3), seed 42, DBSCAN `eps=2.0`, `min_samples=3`
configuration, it schedules a deterministic `churn_leave` event for the
initial cluster head (node 5). The validator independently checks the
availability transition (`FALSE -> TRUE`), observes the simulator's cluster
repair counter increment, and reconstructs the resulting active memberships
and heads from the frozen repair semantics (active members, lowest active ID
as head). The expected and actual structures match, all references remain
valid, and no runtime forwarding adaptation is introduced.

> **S6 — PASS.** A genuine frozen DC-SoC structural-update trigger caused the
> normal repair path to run, and the resulting cluster membership and cluster
> head structure matched an independent reconstruction.

#### S7 — No AHBN runtime controller used

Status: **PASS**

Exact command:

```bash
$ python -m scripts.validate_dcsoc_s7
```

Actual output:

```text
========================================================================
STAGE 3.4 — DC-SoC SANITY VALIDATION
S7 — No AHBN runtime controller used
========================================================================

Test configuration:
  Topology type       : BA
  Topology nodes      : 30
  Topology edges      : 81
  BA m                : 3
  Seed                : 42
  DBSCAN eps          : 2.0
  DBSCAN min_samples  : 3

Static implementation inspection:
  DC-SoC construction : run_one.build_simulation_from_config('dcsoc')
                        -> assign_dcsoc_clusters() -> DCSOCStrategy(...)
                        -> Simulator(..., controller=None)
  DC-SoC forwarding   : Simulator.handle_receive()
                        -> DCSOCStrategy.select_targets()
                        -> same-cluster physical neighbours; CH gateway neighbours
                        -> seeded bounded sampling -> Simulator.send_message()
  DC-SoC repair       : churn handler -> repair_topology_after_churn()
                        -> active neighbours / cluster overlay / CH gateway refresh
  AHBN control path   : Simulator.update_ahbn_state()
                        -> normalized observations -> AHBNController.update_metrics()
                        -> EWMA -> compute_score()/sigmoid()
                        -> decide_mode_and_fanout() -> node.control
                        -> AHBNStrategy.select_targets()
  AHBNStrategy instantiated by DC-SoC : NO
  AHBN controller dependency          : NONE

DC-SoC forwarding inputs:
  Cluster membership                  : USED (node.cluster_id)
  Cluster-head information            : USED (node.is_cluster_head)
  Physical topology                   : USED (node.neighbors)
  CH gateway overlay                  : USED (node.gateway_neighbors)
  Fixed fanout/inter-fanout limits    : USED
  Simulator seeded RNG                : USED for bounded sampling
  AHBN EWMA observations              : NOT USED
  AHBN adaptive score / sigmoid       : NOT USED
  AHBN runtime mode                   : NOT USED
  AHBN adaptive fanout                : NOT USED
  NodeControlState forwarding input   : NOT USED
  Shared AHBN control fields on Node  : YES

Runtime instrumentation (raising sentinels):
  Guarded update_ahbn_state dispatches: 70 (returns at controller=None)
  AHBN sensing calls                  : 0
  AHBN EWMA/controller update calls   : 0
  AHBN score/sigmoid calls            : 0
  AHBN mode/fanout decision calls     : 0
  AHBNStrategy construction calls     : 0
  AHBNStrategy forwarding calls       : 0
  Sentinel triggered                  : NO

DC-SoC transaction:
  Source node                         : 0
  Source selection                    : lowest-ID non-CH cluster member
  Transaction ID                      : dcsoc-s7-transaction
  Dissemination completed             : YES
  Delivered nodes                     : 23/30
  Transmission count                  : 69
  Duplicate count                     : 47

Structural-maintenance distinction:
  Structural update capability        : PRESENT
  AHBN forwarding adaptation          : ABSENT

Validation:
  DC-SoC dissemination completed      : PASS
  No AHBN runtime mechanism invoked   : PASS
  No AHBN-driven forwarding decision  : PASS

------------------------------------------------------------------------
S7 RESULT: PASS
------------------------------------------------------------------------
Conclusion:
  DC-SoC uses its predefined dissemination policy and structural
  maintenance mechanism without AHBN runtime forwarding adaptation.

  DC-SoC : structure-adaptive, forwarding-fixed
  AHBN   : runtime forwarding-adaptive
```

The validator combines semantic static inspection of the production DC-SoC
strategy with runtime raising sentinels installed temporarily through
`unittest.mock.patch`. The deterministic BA(30, 3), seed 42, DBSCAN
`eps=2.0`, `min_samples=3` transaction completed normally: 23 of 30 nodes
received the transaction through 69 transmissions, with 47 duplicates.

Observed runtime evidence:

```text
AHBN sensing calls                  : 0
AHBN EWMA/controller update calls   : 0
AHBN score/sigmoid calls            : 0
AHBN mode/fanout decision calls     : 0
AHBNStrategy construction calls     : 0
AHBNStrategy forwarding calls       : 0
Sentinel triggered                  : NO
```

All simulator nodes contain the shared `NodeControlState` field, but
`DCSOCStrategy.select_targets()` does not read it. DC-SoC instead uses cluster
membership, cluster-head status, physical neighbours, the cluster-head gateway
overlay, fixed fanout/inter-fanout limits, and the simulator's seeded RNG.

The shared receive path called the guarded `Simulator.update_ahbn_state()` hook
70 times. This is generic simulator dispatch, not controller execution: the
hook returned immediately on every call because the DC-SoC simulator has
`controller=None`. No sensing, EWMA, score, sigmoid, mode, adaptive-fanout, or
AHBN strategy sentinel was reached.

Scientific interpretation:

> DC-SoC uses its predefined dissemination policy and structural-maintenance
> mechanism without AHBN runtime forwarding adaptation.

```text
DC-SoC : structure-adaptive, forwarding-fixed
AHBN   : runtime forwarding-adaptive
```

#### S8 — Forwarding remains structurally determined

Status: **PASS**

Purpose: demonstrate positively that, with RNG state and fixed policy held
constant, the production DC-SoC target decision is invariant to irrelevant
`NodeControlState` changes and sensitive to a relevant structural change.

Validation method: the validator calls the real
`DCSOCStrategy.select_targets()` on the canonical deterministic BA(30, 3),
seed 42, DBSCAN `eps=2.0`, `min_samples=3` setup. It saves and restores the
simulator RNG state around comparisons, mutates only the source node's real
AHBN control fields for the invariance check, then removes a symmetric physical
link containing a baseline-selected target for the structural-sensitivity
check. It also checks the production fixed-fanout bound at fanouts 1 and 3.

Exact command:

```bash
/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python -m scripts.validate_dcsoc_s8
```

Actual output:

```text
========================================================================
STAGE 3.4 — DC-SoC SANITY VALIDATION
S8 — Forwarding remains structurally determined
========================================================================

Test configuration:
  Topology type       : BA
  Topology nodes      : 30
  Topology edges      : 81
  BA m                : 3
  Seed                : 42
  DBSCAN eps          : 2.0
  DBSCAN min_samples  : 3

Static implementation inspection:
  DC-SoC forwarding   : Simulator.handle_receive()
                        -> DCSOCStrategy.select_targets()
                        -> same-cluster active physical neighbours
                        -> CH gateway neighbours when source is a CH
                        -> fixed fanout/inter-fanout + simulator.rng sampling
                        -> Simulator.send_message()
  AHBN state consulted: NO (node.control is not read by select_targets())

Baseline forwarding case:
  Source node         : 4
  Cluster             : 0
  Is cluster head     : NO
  Physical neighbours : [0, 2, 3, 5, 9, 10, 11, 12, 16, 20, 21, 25, 29]
  Eligible neighbours : [0, 2, 3, 5, 9, 10, 11, 12, 16, 20, 21, 25, 29]
  Gateway neighbours  : []
  Fixed fanout        : 3
  Fixed inter-fanout  : 1 (not exercised by non-CH source)
  RNG seed/state      : seed=42; state saved before selection
  Forwarding targets  : [21, 2, 0]

AHBN control-state invariance:
  AHBN state before   : {'mode': 'gossip', 'fanout': 3, 'score': 0.0, 'weight': 0.5, 'd_hat': 0.0, 'l_hat': 0.0, 'u_hat': 0.0, 'c_hat': 0.0}
  AHBN state after    : {'mode': 'cluster', 'fanout': 4, 'score': -1000.0, 'weight': 0.0, 'd_hat': 1.0, 'l_hat': 1.0, 'u_hat': 1.0, 'c_hat': 1.0}
  Control state changed: PASS
  Structure unchanged : PASS
  Fixed policy unchanged: PASS
  RNG unchanged/reset : PASS
  Targets before      : [21, 2, 0]
  Targets after       : [21, 2, 0]
  Targets identical   : PASS
  AHBN control-state invariance : PASS

Structural sensitivity:
  Structural field    : symmetric physical-neighbour link membership
  Change applied      : remove link 4 <-> 21
  Original value      : [0, 2, 3, 5, 9, 10, 11, 12, 16, 20, 21, 25, 29]
  Modified value      : [0, 2, 3, 5, 9, 10, 11, 12, 16, 20, 25, 29]
  Original targets    : [21, 2, 0]
  Modified targets    : [25, 2, 0]
  Removed target absent: PASS
  Targets changed     : PASS
  Structural sensitivity : PASS

Fixed-policy sensitivity:
  Fanout before       : 1
  Fanout after        : 3
  Targets before      : [21]
  Targets after       : [21, 2, 0]
  Policy effect       : PASS

Required assertions:
  Valid deterministic case           : PASS
  Baseline targets structurally valid: PASS
  AHBN state independently changed   : PASS
  AHBN-state target invariance       : PASS
  Valid structural change applied    : PASS
  Structural target effect observed  : PASS
  AHBN controller/strategy required  : NO

------------------------------------------------------------------------
S8 RESULT: PASS
Forwarding remains structurally determined.
------------------------------------------------------------------------
DC-SoC forwarding was invariant under irrelevant AHBN runtime state
changes with structure, fixed policy, and RNG state held constant.
A relevant structural change altered the forwarding decision.
```

> **S8 — PASS.** DC-SoC forwarding remained invariant under changes to
> irrelevant AHBN runtime control state when structural state, fixed forwarding
> parameters, and RNG conditions were held constant. Removing a valid physical
> neighbour link containing a selected target changed `[21, 2, 0]` to
> `[25, 2, 0]`, and changing fixed fanout from 1 to 3 changed the bounded target
> count from 1 to 3. Forwarding is therefore determined by DC-SoC structural
> state, fixed forwarding parameters, and its seeded sampling process rather
> than AHBN runtime adaptation.
