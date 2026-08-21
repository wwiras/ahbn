# S5 — pre-rerun comparator reconciliation

## 1. Purpose and boundary

S5 reconciles only the final Stage 4 configuration comparator lists with the
S4 freeze. It does not modify an algorithm, controller, fanout rule, clustering
rule, runner, schedule, resource scenario, or frozen comparator parameter. No
final Stage 4 batch was run.

Prerequisite records `docs/S4_finalfreeze_dcsoc.md` and
`docs/S2_faithful_dcsoc.md` were read completely before changes. S4 was PASS;
its implementation manifest is the authority for this gate.

## 2. Pre-change integrity snapshot

Commands and relevant output before editing:

```text
$ pwd
/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6

$ git rev-parse --show-toplevel
/Users/wwiras/Documents/src/AHBNProj/ahbn

$ git status --short
[no output]

$ git diff --stat
[no output]

$ git diff
[no output]
```

Pre-change configuration SHA-256 values:

```text
4d18537eb8c02b5b1208c0554b8544d88465367827dac2dde32c23fc6479966f  configs/exp08_ch_bottleneck.yaml
e50b78929ff7079e6cb494a5b71792564f7e2922ec25527ea95db0f6bfe3da61  configs/exp09_dense_topology.yaml
deef57bdfae2c0799e70fa0cc91b6a2b1544e5eb348e4a83700b6c33da2809e5  configs/exp10_failure.yaml
05a32cf9189919b8b1b439ac69968245774962ba234803e16cb732efb9e95135  configs/exp11_churn.yaml
763faf75a63baa52054f88b60056f0eec2e15adddff8ef06277cfb1955a03e9e  configs/exp12_mixed_resources.yaml
```

## 3. S4 implementation-hash verification

`sha256sum` was run on every required frozen file. Each current value exactly
matched the S4 final freeze manifest:

```text
114a6311ef17b10f2fda12ac696402f2b383c849d663c3a66861a26bb6fb1664  ahbn/strategies/gossip.py
0c5a421434806099e43fbc3f118f6737920fc155e750c2100492e6dbe93c027c  ahbn/strategies/cluster.py
e0c1b109427cdd5e6a0055da350a2b9a7f2003af2b9f3da58e56dff73a3ac58e  ahbn/strategies/dcsoc.py
50ed8c10408bb5601ccd6f441b2aed834a3a427b00d434aea10a4222b72441db  ahbn/strategies/ahbn.py
9a19ae2c9766ea36fe873d4d643cf51d9e8df555b42d2de11d946d30fb60f75f  ahbn/control.py
0da5a733a01909e67591773ffdee1939ed7d79c833fab6afa57a31527502ab35  run_one.py
c9adcbdb20e6d8ae052b6de712fb11dea2e3364bdc50d33211c0e2d79f06c853  run_batch.py
```

S4 implementation freeze intact: **YES**.

## 4. Scientific applicability decisions

### Exp10 — failure: applicable

DC-SoC is applicable. The frozen S2 comparator implements core failure,
deterministic core replacement, local structural repair, return-as-leaf,
recovery, and explicit periodic `du` regeneration. These lifecycle semantics
are directly relevant to Exp10. The failure mechanism and schedule were not
changed.

### Exp11 — churn: applicable

DC-SoC is applicable. Frozen inactive/leave handling, structural repair,
return-as-leaf, recovery, and regeneration are directly relevant to churn.
Churn rates and schedules were not changed.

### Exp12 — mixed resources: applicable

DC-SoC is applicable as a frozen structure-adaptive dissemination comparator.
Inspection found no configuration or execution-path incompatibility. It was
not made resource-aware: no utilization-aware clustering, resource-aware core
selection, adaptive fanout, AHBN-like behavior, or heterogeneity tuning was
added. The existing resource scenario remains an environmental condition.

## 5. Exact configuration reconciliation

Exp08 and Exp09 already had the required comparator list and explicit frozen
DC-SoC block, so they were unchanged. Exp10, Exp11, and Exp12 each received
only `dcsoc` in the existing comparator list and this exact block:

```yaml
dcsoc:
  eps: 2.0
  min_samples: 3
  fanout: 3
  inter_fanout: 1
```

No setting was reordered or tuned. The end-of-file newline normalization in
each changed YAML is non-semantic and occurred while applying the scoped edit.

## 6. Comparator and fallback verification

For Exp08–Exp12, AHBN remains canonical: `alpha=0.3`, `kappa=1.0`,
`beta=1.0`, `mode_threshold=0.5`, `min_fanout=2`, `max_fanout=4`, and
`default_fanout=3`. No final configuration contains AHBN max fanout 5 or 6.

Gossip retains each experiment's static fanout condition. In particular,
Exp09's global `fanout=4` remains the intentional Gossip dense-topology
condition. Structured remains the frozen static structure with no runtime
adaptation or DC-SoC lifecycle behavior.

Both runners were inspected without modification. Their DC-SoC construction
paths consult the comparator-specific `dcsoc` block first; the explicit
`fanout=3` therefore wins over experiment-level fanout, including Exp09's 4.
The config-only validator mirrors the runner resolution and confirms all four
resolved DC-SoC fields rather than relying only on block presence.

## 7. Config-only validator

Added `scripts/validate_stage4_prerun_comparators.py`. It checks exact ordered
comparator lists, duplicates, explicit frozen DC-SoC values, canonical AHBN
values, runner strategy support, parsed required blocks, and resolved fallback
values. It performs no simulation and writes no result.

Command used exactly the required interpreter:

```text
$ PYTHONDONTWRITEBYTECODE=1 /Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python -m scripts.validate_stage4_prerun_comparators
EXP08: strategies=['gossip', 'cluster', 'dcsoc', 'ahbn']; dcsoc={'eps': 2.0, 'min_samples': 3, 'fanout': 3, 'inter_fanout': 1}; ahbn.max_fanout=4
EXP09: strategies=['gossip', 'cluster', 'dcsoc', 'ahbn']; dcsoc={'eps': 2.0, 'min_samples': 3, 'fanout': 3, 'inter_fanout': 1}; ahbn.max_fanout=4
EXP10: strategies=['gossip', 'cluster', 'dcsoc', 'ahbn']; dcsoc={'eps': 2.0, 'min_samples': 3, 'fanout': 3, 'inter_fanout': 1}; ahbn.max_fanout=4
EXP11: strategies=['gossip', 'cluster', 'dcsoc', 'ahbn']; dcsoc={'eps': 2.0, 'min_samples': 3, 'fanout': 3, 'inter_fanout': 1}; ahbn.max_fanout=4
EXP12: strategies=['gossip', 'cluster', 'dcsoc', 'ahbn']; dcsoc={'eps': 2.0, 'min_samples': 3, 'fanout': 3, 'inter_fanout': 1}; ahbn.max_fanout=4
STAGE 4 PRE-RUN COMPARATOR VALIDATION: PASS
No simulations were run and no result files were written.
```

The optional construction smoke test was unnecessary and was skipped.

## 8. Post-change diff and file boundary

Immediately after config reconciliation and validator creation:

```text
$ git status --short
 M configs/exp10_failure.yaml
 M configs/exp11_churn.yaml
 M configs/exp12_mixed_resources.yaml
?? scripts/validate_stage4_prerun_comparators.py

$ git diff --stat
 v0.6/configs/exp10_failure.yaml         | 9 ++++++++−
 v0.6/configs/exp11_churn.yaml           | 9 ++++++++−
 v0.6/configs/exp12_mixed_resources.yaml | 9 ++++++++−
 3 files changed, 24 insertions(+), 3 deletions(−)
```

The config diff consists only of one `dcsoc` list entry and the exact frozen
four-key block in each of Exp10–Exp12 (plus EOF newline normalization). The
dedicated diff of all seven frozen implementation/runner files was empty.
This documentation file is the only additional record created.

## 9. Old and new configuration hashes

| Configuration | Old S4 SHA-256 | New S5 SHA-256 | Changed | Exact reason |
|---|---|---|---:|---|
| Exp08 | `4d18537eb8c02b5b1208c0554b8544d88465367827dac2dde32c23fc6479966f` | `4d18537eb8c02b5b1208c0554b8544d88465367827dac2dde32c23fc6479966f` | No | Already reconciled |
| Exp09 | `e50b78929ff7079e6cb494a5b71792564f7e2922ec25527ea95db0f6bfe3da61` | `e50b78929ff7079e6cb494a5b71792564f7e2922ec25527ea95db0f6bfe3da61` | No | Already reconciled |
| Exp10 | `deef57bdfae2c0799e70fa0cc91b6a2b1544e5eb348e4a83700b6c33da2809e5` | `b3d4915078ea84c969d0d0c7161543ffcb4bc33774e5d8db1bd9f60de368d1ca` | Yes | Comparator-list reconciliation and explicit copy of already-frozen DC-SoC parameters. |
| Exp11 | `05a32cf9189919b8b1b439ac69968245774962ba234803e16cb732efb9e95135` | `93d930ff67b0d7554c0b53f4ded705859d2d185c4df113f660d9cb734033f7f9` | Yes | Comparator-list reconciliation and explicit copy of already-frozen DC-SoC parameters. |
| Exp12 | `763faf75a63baa52054f88b60056f0eec2e15adddff8ef06277cfb1955a03e9e` | `5910f3d4b02fd0b85ba5367d6a21ba1bd5d9efa8dec6808fe0650b5a5c892138` | Yes | Comparator-list reconciliation and explicit copy of already-frozen DC-SoC parameters. |

## 10. Final pre-rerun matrix

| Experiment | Gossip | Structured | DC-SoC | AHBN | DC-SoC frozen params explicit? | AHBN max=4? | Ready |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exp08 | YES | YES | YES | YES | YES | YES | READY |
| Exp09 | YES | YES | YES | YES | YES | YES | READY |
| Exp10 | YES | YES | YES | YES | YES | YES | READY |
| Exp11 | YES | YES | YES | YES | YES | YES | READY |
| Exp12 | YES | YES | YES | YES | YES | YES | READY |

## 11. Decision

**S5 PASS.** The S4 implementation freeze is intact. Only Exp10–Exp12
comparator-list/config blocks, this validator, and this record changed. No
algorithm or runner file changed, no comparator parameter was tuned, no smoke
performance was evaluated, and no final Stage 4 batch or result was produced.

The next permitted step is the Stage 4 final rerun, beginning with Exp08.
