# Stage 4 Exp08/Exp09 dissemination correction — rerun 2

## Stage 1 — Initial inspection

Date/time: 2026-08-21 18:31:01 +08

Project root: `/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6`

Required interpreter: `/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python`

### Command

```sh
git status --short --branch && git rev-parse HEAD && git diff --stat && git diff --name-status
```

### Complete terminal output

```text
## main...origin/main
?? docs/stage4_exp08_rerun2.md
76499abdf8e869739535b9511cdb70b3a6c3ccab

```

### Command

```sh
find .. -name AGENTS.md -print && rg -n "Exp0[789]|exp0[789]|fanout|Gossip|DC-SoC|DCSOC|dc_soc" --glob '*.py' --glob '*.yaml' --glob '*.yml' --glob '*.json' --glob '*.toml' .
```

### Complete terminal output

```text
./ahbn/strategies/dcsoc.py:10:class DCSOCStrategy(ForwardingStrategy):
./ahbn/strategies/dcsoc.py:12:    DC-SoC-inspired density-clustered hybrid dissemination baseline.
./ahbn/strategies/dcsoc.py:36:        - adaptive fanout
./ahbn/strategies/dcsoc.py:43:        fanout: int = 3,
./ahbn/strategies/dcsoc.py:44:        inter_fanout: int = 1,
./ahbn/strategies/dcsoc.py:47:        if fanout < 1:
./ahbn/strategies/dcsoc.py:49:                "DC-SoC fanout must be >= 1"
./ahbn/strategies/dcsoc.py:52:        if inter_fanout < 0:
./ahbn/strategies/dcsoc.py:54:                "DC-SoC inter_fanout must be >= 0"
./ahbn/strategies/dcsoc.py:57:        self.fanout = int(
./ahbn/strategies/dcsoc.py:58:            fanout
./ahbn/strategies/dcsoc.py:61:        self.inter_fanout = int(
./ahbn/strategies/dcsoc.py:62:            inter_fanout
./ahbn/strategies/dcsoc.py:79:        DC-SoC follows the same random-seed discipline as Gossip.
./ahbn/strategies/dcsoc.py:127:        # existing fixed fanout remains a total resource bound.
./ahbn/strategies/dcsoc.py:134:            return structural_children[: self.fanout]
./ahbn/strategies/dcsoc.py:137:        # Intra-cluster Gossip candidates.
./ahbn/strategies/dcsoc.py:176:                    self.fanout,
./ahbn/strategies/dcsoc.py:211:        # This is important for fairness: DC-SoC does not get
./ahbn/strategies/dcsoc.py:212:        # fanout + extra unlimited gateway transmissions.
./ahbn/strategies/dcsoc.py:216:            self.inter_fanout,
./ahbn/strategies/dcsoc.py:217:            self.fanout,
./ahbn/strategies/dcsoc.py:238:            self.fanout
./ahbn/node.py:92:    # DC-SoC dissemination overlay (independent of AHBN control state).
./ahbn/topology.py:406:    Build the DC-SoC-inspired density-clustered dissemination overlay.
./ahbn/topology.py:415:        raise ValueError("DC-SoC eps must be > 0")
./ahbn/topology.py:418:        raise ValueError("DC-SoC min_samples must be > 0")
./ahbn/control.py:26:    fanout: int = 3
./ahbn/control.py:58:    # Positive score => stronger preference toward Gossip.
./ahbn/control.py:60:    # latency ↑      => Gossip preference ↑
./ahbn/control.py:61:    # churn ↑        => Gossip preference ↑
./ahbn/control.py:62:    # duplicates ↑   => Gossip preference ↓
./ahbn/control.py:63:    # utilization ↑  => Gossip preference ↓
./ahbn/control.py:78:    # Stage 3: fanout response sensitivity
./ahbn/control.py:84:    min_fanout: int = 2
./ahbn/control.py:85:    max_fanout: int = 4
./ahbn/control.py:87:    # weight >= threshold => Gossip
./ahbn/control.py:191:            preference shifts toward Gossip.
./ahbn/control.py:207:    # Score -> Gossip preference
./ahbn/control.py:215:        Map the controller score to a bounded Gossip preference:
./ahbn/control.py:220:            stronger Gossip preference.
./ahbn/control.py:235:    # Mode + fanout decision
./ahbn/control.py:238:    def decide_mode_and_fanout(
./ahbn/control.py:255:            weight -> bounded fanout
./ahbn/control.py:266:        # Gossip preference
./ahbn/control.py:287:        # Adaptive forwarding fanout
./ahbn/control.py:290:        # beta controls how strongly Gossip
./ahbn/control.py:291:        # preference influences forwarding fanout.
./ahbn/control.py:294:        #     minimum fanout
./ahbn/control.py:303:        fanout_span = p.max_fanout - p.min_fanout
./ahbn/control.py:305:        raw_fanout = (
./ahbn/control.py:306:            p.min_fanout
./ahbn/control.py:307:            + p.beta * state.weight * fanout_span
./ahbn/control.py:310:        state.fanout = int(
./ahbn/control.py:313:                    raw_fanout,
./ahbn/control.py:314:                    p.min_fanout,
./ahbn/control.py:315:                    p.max_fanout,
./ahbn/control.py:343:            "fanout": state.fanout,
./ahbn/utils.py:24:    fanout: int | None
./ahbn/utils.py:53:        mode + fanout
./ahbn/utils.py:106:    fanout: int
./ahbn/utils.py:112:    fanout_changed: bool
./run_one.py:10:from ahbn.strategies.dcsoc import DCSOCStrategy
./run_one.py:11:from ahbn.strategies.gossip import GossipStrategy
./run_one.py:35:        min_fanout=ahbn_cfg.get("min_fanout", 2),
./run_one.py:36:        max_fanout=ahbn_cfg.get("max_fanout", 4),
./run_one.py:41:def build_ahbn_strategy(cfg: dict, fanout_override: int | None = None) -> AHBNStrategy:
./run_one.py:44:    default_fanout = (
./run_one.py:45:        fanout_override
./run_one.py:46:        if fanout_override is not None
./run_one.py:47:        else ahbn_cfg.get("default_fanout", 3)
./run_one.py:51:        default_fanout=default_fanout,
./run_one.py:52:        adaptive_fanout=True,
./run_one.py:81:        fanout = cfg.get("fanout", 3)
./run_one.py:82:        strategy = GossipStrategy(fanout=fanout)
./run_one.py:93:    #     strategy = build_ahbn_strategy(cfg, fanout_override=cfg.get("fanout"))
./run_one.py:112:            fanout_override=cfg.get("fanout"),
./run_one.py:138:        strategy = DCSOCStrategy(
./run_one.py:139:            fanout=int(
./run_one.py:141:                    "fanout",
./run_one.py:143:                        "fanout",
./run_one.py:148:            inter_fanout=int(
./run_one.py:150:                    "inter_fanout",
./ahbn/strategies/hybrid_fixed.py:12:    Fixed-hybrid forwarding for Exp07.
./ahbn/strategies/hybrid_fixed.py:15:    def __init__(self, fanout: int = 3, external_leakage: int = 1) -> None:
./ahbn/strategies/hybrid_fixed.py:16:        self.fanout = fanout
./ahbn/strategies/hybrid_fixed.py:28:        if cluster_mgr is None or self.fanout <= 0:
./ahbn/strategies/hybrid_fixed.py:43:        remaining_budget = self.fanout
./ahbn/strategies/hybrid_fixed.py:104:        if len(unique_targets) > self.fanout:
./ahbn/strategies/hybrid_fixed.py:105:            unique_targets = unique_targets[:self.fanout]
./ahbn/metrics.py:32:    fanout_change_count: int = 0
./ahbn/metrics.py:78:    def record_adaptation(self, mode_switched: bool, fanout_changed: bool) -> None:
./ahbn/metrics.py:81:        if fanout_changed:
./ahbn/metrics.py:82:            self.fanout_change_count += 1
./ahbn/metrics.py:83:        if mode_switched or fanout_changed:
./ahbn/metrics.py:123:            "fanout_change_count": self.fanout_change_count,
./configs/sanity_churn.yaml:11:fanout: 3
./configs/sanity_churn.yaml:41:  min_fanout: 2
./configs/sanity_churn.yaml:42:  max_fanout: 4
./configs/sanity_churn.yaml:44:  default_fanout: 3
./configs/exp08_ch_bottleneck.yaml:1:experiment: exp08
./configs/exp08_ch_bottleneck.yaml:24:# Frozen Stage 3.5 DC-SoC baseline values, made explicit for Exp08.
./configs/exp08_ch_bottleneck.yaml:28:  fanout: 3
./configs/exp08_ch_bottleneck.yaml:29:  inter_fanout: 1
./configs/exp08_ch_bottleneck.yaml:47:  min_fanout: 2
./configs/exp08_ch_bottleneck.yaml:48:  max_fanout: 4
./configs/exp08_ch_bottleneck.yaml:51:  default_fanout: 3
./ahbn/simulator.py:295:                (avg_forwarding / max_fanout)
./ahbn/simulator.py:312:            max_fanout = max(
./ahbn/simulator.py:314:                int(self.controller.params.max_fanout),
./ahbn/simulator.py:317:            max_fanout = 1
./ahbn/simulator.py:321:            / float(max_fanout)
./ahbn/simulator.py:367:        fanout_changed: bool = False,
./ahbn/simulator.py:427:                fanout=snap["fanout"],
./ahbn/simulator.py:433:                fanout_changed=fanout_changed,
./ahbn/simulator.py:503:        prev_fanout = node.control.fanout
./ahbn/simulator.py:519:        self.controller.decide_mode_and_fanout(
./ahbn/simulator.py:527:        fanout_changed = (
./ahbn/simulator.py:528:            node.control.fanout != prev_fanout
./ahbn/simulator.py:533:            fanout_changed,
./ahbn/simulator.py:552:            fanout_changed=fanout_changed,
./ahbn/simulator.py:585:            prev_fanout = node.control.fanout
./ahbn/simulator.py:592:            self.controller.decide_mode_and_fanout(
./ahbn/simulator.py:600:            fanout_changed = (
./ahbn/simulator.py:601:                node.control.fanout != prev_fanout
./ahbn/simulator.py:606:                fanout_changed,
./ahbn/simulator.py:621:                fanout_changed=fanout_changed,
./ahbn/simulator.py:754:        if self.strategy.__class__.__name__ == "DCSOCStrategy":
./ahbn/simulator.py:783:        if self.strategy.__class__.__name__ == "DCSOCStrategy":
./configs/exp09_dense_topology.yaml:1:experiment: exp09
./configs/exp09_dense_topology.yaml:9:fanout: 4
./configs/exp09_dense_topology.yaml:23:# Frozen S2 DC-SoC comparator parameters.  These are explicit because the
./configs/exp09_dense_topology.yaml:24:# Exp09 global fanout=4 is the fixed Gossip condition, not a DC-SoC override.
./configs/exp09_dense_topology.yaml:28:  fanout: 3
./configs/exp09_dense_topology.yaml:29:  inter_fanout: 1
./configs/exp09_dense_topology.yaml:47:  min_fanout: 2
./configs/exp09_dense_topology.yaml:48:  max_fanout: 4
./configs/exp09_dense_topology.yaml:51:  default_fanout: 3
./configs/stage3_dcsoc.yaml:2:# Stage 3.3 — DC-SoC Baseline Smoke Test
./configs/stage3_dcsoc.yaml:38:# Existing experiment-level fanout retained for compatibility.
./configs/stage3_dcsoc.yaml:39:fanout: 3
./configs/stage3_dcsoc.yaml:42:# DC-SoC determines its actual number of clusters using DBSCAN.
./configs/stage3_dcsoc.yaml:48:# DC-SoC-inspired baseline
./configs/stage3_dcsoc.yaml:67:  fanout: 3
./configs/stage3_dcsoc.yaml:69:  # Maximum portion of a cluster head's fanout reserved for
./configs/stage3_dcsoc.yaml:71:  inter_fanout: 1
./ahbn/strategies/ahbn.py:9:from ahbn.strategies.gossip import GossipStrategy
./ahbn/strategies/ahbn.py:17:        Execute the dissemination mode and fanout selected by
./ahbn/strategies/ahbn.py:22:        - fanout: bounded forwarding budget
./ahbn/strategies/ahbn.py:25:        - mix Gossip and Structured targets
./ahbn/strategies/ahbn.py:34:        default_fanout: int = 3,
./ahbn/strategies/ahbn.py:35:        adaptive_fanout: bool = True,
./ahbn/strategies/ahbn.py:37:        self.default_fanout = default_fanout
./ahbn/strategies/ahbn.py:38:        self.adaptive_fanout = adaptive_fanout
./ahbn/strategies/ahbn.py:40:        self._gossip = GossipStrategy(
./ahbn/strategies/ahbn.py:41:            fanout=default_fanout
./ahbn/strategies/ahbn.py:50:    def _get_effective_fanout(
./ahbn/strategies/ahbn.py:57:        If adaptive fanout is enabled:
./ahbn/strategies/ahbn.py:58:            use node.control.fanout.
./ahbn/strategies/ahbn.py:61:            use the configured default fanout.
./ahbn/strategies/ahbn.py:64:        if self.adaptive_fanout:
./ahbn/strategies/ahbn.py:67:                int(node.control.fanout),
./ahbn/strategies/ahbn.py:72:            int(self.default_fanout),
./ahbn/strategies/ahbn.py:106:        fanout = self._get_effective_fanout(node)
./ahbn/strategies/ahbn.py:116:            self._gossip.fanout = fanout
./ahbn/strategies/ahbn.py:126:            self._cluster.fanout = fanout
./ahbn/strategies/gossip.py:10:class GossipStrategy(ForwardingStrategy):
./ahbn/strategies/gossip.py:12:    Pure Gossip dissemination strategy.
./ahbn/strategies/gossip.py:17:        Randomly select up to `fanout` active physical neighbors.
./ahbn/strategies/gossip.py:19:    AHBN may update `fanout` before calling this strategy.
./ahbn/strategies/gossip.py:22:    def __init__(self, fanout: int = 3) -> None:
./ahbn/strategies/gossip.py:23:        if fanout < 1:
./ahbn/strategies/gossip.py:24:            raise ValueError("fanout must be >= 1")
./ahbn/strategies/gossip.py:26:        self.fanout = fanout
./ahbn/strategies/gossip.py:35:        Select up to `fanout` active neighbors uniformly at random.
./ahbn/strategies/gossip.py:50:            int(self.fanout),
./scripts/validate_dcsoc_s4.py:2:Stage 3.4 — DC-SoC Sanity Validation
./scripts/validate_dcsoc_s4.py:5:Runs the production DC-SoC strategy and simulator while observing the
./scripts/validate_dcsoc_s4.py:16:from ahbn.strategies.dcsoc import DCSOCStrategy
./scripts/validate_dcsoc_s4.py:34:class ObservingDCSOCStrategy(DCSOCStrategy):
./scripts/validate_dcsoc_s4.py:75:    print("STAGE 3.4 — DC-SoC SANITY VALIDATION")
./scripts/validate_dcsoc_s4.py:132:    strategy = ObservingDCSOCStrategy(
./scripts/validate_dcsoc_s4.py:133:        fanout=FANOUT,
./scripts/validate_dcsoc_s4.py:134:        inter_fanout=INTER_FANOUT,
./configs/stage2_parameter_sensitivity.yaml:45:  min_fanout: 2
./configs/stage2_parameter_sensitivity.yaml:46:  max_fanout: 4
./configs/stage2_parameter_sensitivity.yaml:49:  default_fanout: 3
./configs/stage2_parameter_sensitivity.yaml:65:# modifying Exp07–Exp12 or the validated controller implementation.
./configs/stage2_parameter_sensitivity.yaml:70:    source_experiment: exp09
./configs/stage2_parameter_sensitivity.yaml:76:    fanout: 4
./configs/stage2_parameter_sensitivity.yaml:93:    source_experiment: exp08
./configs/stage2_parameter_sensitivity.yaml:121:    fanout: 3
./configs/exp07_fanout.yaml:1:experiment: exp07
./configs/exp07_fanout.yaml:8:fanouts: [2, 3, 4, 5, 6]
./configs/exp07_fanout.yaml:20:adaptive_fanout: false
./run_batch.py:12:from ahbn.strategies.dcsoc import DCSOCStrategy
./run_batch.py:13:from ahbn.strategies.gossip import GossipStrategy
./run_batch.py:40:        min_fanout=ahbn_cfg.get("min_fanout", 2),
./run_batch.py:41:        max_fanout=ahbn_cfg.get("max_fanout", 4),
./run_batch.py:46:def build_ahbn_strategy(cfg: dict, fanout: int | None = None) -> AHBNStrategy:
./run_batch.py:48:    default_fanout = (
./run_batch.py:49:        fanout
./run_batch.py:50:        if fanout is not None
./run_batch.py:51:        else ahbn_cfg.get("default_fanout", 3)
./run_batch.py:55:        default_fanout=default_fanout,
./run_batch.py:56:        adaptive_fanout=True,
./run_batch.py:70:    fanout: int | None = None,
./run_batch.py:99:        strategy = GossipStrategy(fanout=fanout if fanout is not None else 3)
./run_batch.py:116:        strategy = build_ahbn_strategy(cfg, fanout=fanout)
./run_batch.py:126:            fanout=(
./run_batch.py:127:                fanout
./run_batch.py:128:                if fanout is not None
./run_batch.py:156:        strategy = DCSOCStrategy(
./run_batch.py:157:            fanout=int(
./run_batch.py:159:                    "fanout",
./run_batch.py:161:                        fanout
./run_batch.py:162:                        if fanout is not None
./run_batch.py:167:            inter_fanout=int(
./run_batch.py:169:                    "inter_fanout",
./run_batch.py:230:def exp07(cfg: dict) -> tuple[list[ResultRow], list]:
./run_batch.py:236:    fanouts = cfg["fanouts"]
./run_batch.py:252:        for fanout in fanouts:
./run_batch.py:266:                    fanout=fanout,
./run_batch.py:271:                    scenario_tag=f"fanout={fanout}",
./run_batch.py:275:                        experiment="exp07",
./run_batch.py:281:                        fanout=fanout,
./run_batch.py:305:                fanout=None,
./run_batch.py:314:                    experiment="exp07",
./run_batch.py:320:                    fanout=None,
./run_batch.py:336:def exp08(cfg: dict) -> tuple[list[ResultRow], list]:
./run_batch.py:381:                        experiment="exp08",
./run_batch.py:387:                        fanout=None,
./run_batch.py:403:def exp09(cfg: dict) -> tuple[list[ResultRow], list]:
./run_batch.py:416:    fanout = cfg.get("fanout", 3)
./run_batch.py:420:        raise ValueError("Exp09 density sweep is intended for ER topology.")
./run_batch.py:440:                    fanout=fanout,
./run_batch.py:448:                        experiment="exp09",
./run_batch.py:454:                        fanout=fanout if strategy_name != "cluster" else None,
./run_batch.py:483:    fanout = cfg.get("fanout", 3)
./run_batch.py:507:                    fanout=fanout,
./run_batch.py:523:                        "fanout": fanout if strategy_name != "cluster" else None,
./run_batch.py:555:    fanout = cfg.get("fanout", 3)
./run_batch.py:579:                    fanout=fanout,
./run_batch.py:595:                        "fanout": fanout if strategy_name != "cluster" else None,
./run_batch.py:607:                        "fanout_change_count": summary["fanout_change_count"],
./run_batch.py:632:    fanout = cfg.get("fanout", 3)
./run_batch.py:656:                    fanout=fanout,
./run_batch.py:672:                        "fanout": fanout if strategy_name != "cluster" else None,
./run_batch.py:701:    if experiment == "exp07":
./run_batch.py:702:        rows, trace_rows = exp07(cfg)
./run_batch.py:703:        path = save_results_csv(rows, "outputs/csv/exp07_results.csv")
./run_batch.py:709:                "outputs/csv/exp07_adaptive_trace.csv",
./run_batch.py:714:    elif experiment == "exp08":
./run_batch.py:715:        rows, trace_rows = exp08(cfg)
./run_batch.py:716:        path = save_results_csv(rows, "outputs/csv/exp08_results.csv")
./run_batch.py:722:                "outputs/csv/exp08_adaptive_trace.csv",
./run_batch.py:727:    elif experiment == "exp09":
./run_batch.py:728:        rows, trace_rows = exp09(cfg)
./run_batch.py:729:        path = save_results_csv(rows, "outputs/csv/exp09_results.csv")
./run_batch.py:735:                "outputs/csv/exp09_adaptive_trace.csv",
./configs/sanity_neutral.yaml:11:fanout: 3
./configs/sanity_neutral.yaml:57:  min_fanout: 2
./configs/sanity_neutral.yaml:58:  max_fanout: 4
./configs/sanity_neutral.yaml:60:  default_fanout: 3
./configs/exp12_mixed_resources.yaml:15:fanout: 3
./configs/exp12_mixed_resources.yaml:27:  fanout: 3
./configs/exp12_mixed_resources.yaml:28:  inter_fanout: 1
./configs/exp12_mixed_resources.yaml:85:  min_fanout: 2
./configs/exp12_mixed_resources.yaml:86:  max_fanout: 4
./configs/exp12_mixed_resources.yaml:89:  default_fanout: 3
./scripts/validate_exp08_e6.py:2:"""Validate Exp08 E6 using only the frozen E5 AHBN CSV evidence."""
./scripts/validate_exp08_e6.py:15:TRACE_PATH = ROOT / "outputs/csv/exp08_ahbn_adaptive_trace_20260820_115817.csv"
./scripts/validate_exp08_e6.py:16:RESULTS_PATH = ROOT / "outputs/csv/exp08_ahbn_results_20260820_115817.csv"
./scripts/validate_exp08_e6.py:40:    "seed", "node_id", "fanout", "received_new", "received_duplicate", "forwarded",
./scripts/validate_exp08_e6.py:46:    "score", "weight", "mode", "fanout", "mode_switched",
./scripts/validate_exp08_e6.py:47:    "fanout_changed", "duplicate_ratio_raw", "resource_class",
./scripts/validate_exp08_e6.py:97:                if row["experiment"] != "exp08" or row["strategy"] != "ahbn":
./scripts/validate_exp08_e6.py:98:                    raise ValueError("non-Exp08/AHBN result row")
./scripts/validate_exp08_e6.py:127:                row["fanout_changed"] = parse_bool(source["fanout_changed"])
./scripts/validate_exp08_e6.py:147:        if row["experiment"] != "exp08" or row["strategy"] != "ahbn":
./scripts/validate_exp08_e6.py:153:        if not MIN_FANOUT <= row["fanout"] <= MAX_FANOUT:
./scripts/validate_exp08_e6.py:154:            fail(f"trace row {line}: fanout outside [{MIN_FANOUT}, {MAX_FANOUT}]", errors)
./scripts/validate_exp08_e6.py:181:        expected_fanout = round(MIN_FANOUT + expected_weight * (MAX_FANOUT - MIN_FANOUT))
./scripts/validate_exp08_e6.py:186:            and row["fanout"] == expected_fanout
./scripts/validate_exp08_e6.py:223:    fanout_transition_counts = {
./scripts/validate_exp08_e6.py:224:        run: sum(row["fanout_changed"] for row in run_rows) for run, run_rows in by_run.items()
./scripts/validate_exp08_e6.py:228:    print("E6 — Exp08 AHBN Adaptive Trace Validation")
./scripts/validate_exp08_e6.py:265:        counts = Counter(row["fanout"] for row in by_factor[factor])
./scripts/validate_exp08_e6.py:267:        transitions = sum(fanout_transition_counts.get((factor, seed), 0) for seed in EXPECTED_SEEDS)
./scripts/validate_exp08_e6.py:268:        details = ", ".join(f"fanout={value}: {counts[value]:,} ({pct(counts[value], total)})" for value in range(MIN_FANOUT, MAX_FANOUT + 1))
./scripts/validate_exp08_e6.py:270:    fanout_total = sum(fanout_transition_counts.values())
./scripts/validate_exp08_e6.py:271:    fanout_runs = sum(value > 0 for value in fanout_transition_counts.values())
./scripts/validate_exp08_e6.py:272:    observed = [row["fanout"] for row in rows]
./scripts/validate_exp08_e6.py:274:    print(f"  total transitions: {fanout_total:,}")
./scripts/validate_exp08_e6.py:275:    print(f"  runs with transitions: {fanout_runs}; runs with zero transitions: {len(by_run) - fanout_runs}")
./scripts/validate_exp08_e6.py:276:    fanout_ok = bool(rows) and all(MIN_FANOUT <= value <= MAX_FANOUT for value in observed)
./scripts/validate_exp08_e6.py:277:    print(f"  Assessment: {'PASS' if fanout_ok else 'FAIL'}")
./scripts/validate_exp08_e6.py:280:            "  SCIENTIFIC OBSERVATION: fanout remained at "
./scripts/validate_exp08_e6.py:295:    overall = bool(rows) and not errors and observation_pass and fanout_ok and mismatches == 0
./ahbn/strategies/cluster.py:34:        fanout: Optional[int] = None,
./ahbn/strategies/cluster.py:36:        self.fanout = fanout
./ahbn/strategies/cluster.py:128:        if self.fanout is None:
./ahbn/strategies/cluster.py:140:            int(self.fanout),
./scripts/validate_dcsoc_faithful_structure.py:1:"""S2 pre/post validator for explicit DC-SoC propagation structure."""
./scripts/validate_dcsoc_faithful_structure.py:50:    print("PASS — explicit DC-SoC propagation structure is valid")
./configs/exp11_churn.yaml:15:fanout: 3
./configs/exp11_churn.yaml:27:  fanout: 3
./configs/exp11_churn.yaml:28:  inter_fanout: 1
./configs/exp11_churn.yaml:67:  min_fanout: 2
./configs/exp11_churn.yaml:68:  max_fanout: 4
./configs/exp11_churn.yaml:71:  default_fanout: 3
./scripts/validate_dcsoc_core_driven_push.py:11:from ahbn.strategies.dcsoc import DCSOCStrategy
./scripts/validate_dcsoc_core_driven_push.py:21:    targets = DCSOCStrategy(fanout=3, inter_fanout=1).select_targets(
./configs/sanity_overload.yaml:11:fanout: 3
./configs/sanity_overload.yaml:37:  min_fanout: 2
./configs/sanity_overload.yaml:38:  max_fanout: 4
./configs/sanity_overload.yaml:40:  default_fanout: 3
./scripts/validate_dcsoc_s9.py:1:"""Stage 3.4 S9: validate end-to-end DC-SoC dissemination."""
./scripts/validate_dcsoc_s9.py:9:from ahbn.strategies.dcsoc import DCSOCStrategy
./scripts/validate_dcsoc_s9.py:54:        fanout=4,
./scripts/validate_dcsoc_s9.py:67:        strategy=DCSOCStrategy(fanout=FANOUT, inter_fanout=INTER_FANOUT),
./scripts/validate_dcsoc_s9.py:119:    print("STAGE 3.4 — DC-SoC SANITY VALIDATION")
./scripts/validate_dcsoc_s9.py:180:    assert passed, "S9 end-to-end DC-SoC dissemination validation failed."
./scripts/validate_dcsoc_s7.py:1:"""Stage 3.4 S7: validate DC-SoC independence from AHBN runtime control."""
./scripts/validate_dcsoc_s7.py:19:from ahbn.strategies.dcsoc import DCSOCStrategy
./scripts/validate_dcsoc_s7.py:35:    tree = ast.parse(inspect.getsource(DCSOCStrategy))
./scripts/validate_dcsoc_s7.py:40:        "compute_weight", "sigmoid", "decide_mode_and_fanout", "mode",
./scripts/validate_dcsoc_s7.py:54:            f"AHBN runtime controller was invoked during DC-SoC dissemination: {label}"
./scripts/validate_dcsoc_s7.py:61:    print("STAGE 3.4 — DC-SoC SANITY VALIDATION")
./scripts/validate_dcsoc_s7.py:68:    strategy = DCSOCStrategy(fanout=FANOUT, inter_fanout=INTER_FANOUT)
./scripts/validate_dcsoc_s7.py:86:    correct_strategy = type(simulator.strategy) is DCSOCStrategy
./scripts/validate_dcsoc_s7.py:93:        "AHBNController.decide_mode_and_fanout": 0,
./scripts/validate_dcsoc_s7.py:113:        patch.object(AHBNController, "decide_mode_and_fanout", forbidden_call("AHBNController.decide_mode_and_fanout", calls)),
./scripts/validate_dcsoc_s7.py:153:    print("  DC-SoC construction : run_one.build_simulation_from_config('dcsoc')")
./scripts/validate_dcsoc_s7.py:154:    print("                        -> assign_dcsoc_clusters() -> DCSOCStrategy(...)")
./scripts/validate_dcsoc_s7.py:156:    print("  DC-SoC forwarding   : Simulator.handle_receive()")
./scripts/validate_dcsoc_s7.py:157:    print("                        -> DCSOCStrategy.select_targets()")
./scripts/validate_dcsoc_s7.py:160:    print("  DC-SoC repair       : churn handler -> repair_topology_after_churn()")
./scripts/validate_dcsoc_s7.py:165:    print("                        -> decide_mode_and_fanout() -> node.control")
./scripts/validate_dcsoc_s7.py:167:    print("  AHBNStrategy instantiated by DC-SoC : NO" if no_ahbn_strategy_dependency else "  AHBNStrategy instantiated by DC-SoC : YES")
./scripts/validate_dcsoc_s7.py:170:    print("\nDC-SoC forwarding inputs:")
./scripts/validate_dcsoc_s7.py:175:    print("  Fixed fanout/inter-fanout limits    : USED")
./scripts/validate_dcsoc_s7.py:180:    print("  AHBN adaptive fanout                : NOT USED")
./scripts/validate_dcsoc_s7.py:189:    print(f"  AHBN mode/fanout decision calls     : {calls['AHBNController.decide_mode_and_fanout']}")
./scripts/validate_dcsoc_s7.py:194:    print("\nDC-SoC transaction:")
./scripts/validate_dcsoc_s7.py:213:    print(f"  DC-SoC dissemination completed      : {pass_fail(dissemination_completed)}")
./scripts/validate_dcsoc_s7.py:223:    print("  DC-SoC uses its predefined dissemination policy and structural")
./scripts/validate_dcsoc_s7.py:225:    print("\n  DC-SoC : structure-adaptive, forwarding-fixed")
./scripts/run_stage2_sensitivity.py:74:            "mean_fanout": None,
./scripts/run_stage2_sensitivity.py:75:            "min_observed_fanout": None,
./scripts/run_stage2_sensitivity.py:76:            "max_observed_fanout": None,
./scripts/run_stage2_sensitivity.py:88:        "mean_fanout": float(df["fanout"].mean()),
./scripts/run_stage2_sensitivity.py:89:        "min_observed_fanout": int(df["fanout"].min()),
./scripts/run_stage2_sensitivity.py:90:        "max_observed_fanout": int(df["fanout"].max()),
./scripts/run_stage2_sensitivity.py:199:                    fanout = scenario.get("fanout")
./scripts/run_stage2_sensitivity.py:220:                        fanout=fanout,
./scripts/run_stage2_sensitivity.py:249:                            "configured_fanout": fanout,
./scripts/run_stage2_sensitivity.py:258:                            "fanout_change_count": summary["fanout_change_count"],
./scripts/validate_dcsoc_s11.py:1:"""Stage 3.4 S11: validate DC-SoC runtime isolation from AHBN."""
./scripts/validate_dcsoc_s11.py:10:from ahbn.strategies.dcsoc import DCSOCStrategy
./scripts/validate_dcsoc_s11.py:22:    """Observe the production DC-SoC call path without changing it."""
./scripts/validate_dcsoc_s11.py:58:    print("STAGE 3.4 — DC-SoC SANITY VALIDATION")
./scripts/validate_dcsoc_s11.py:65:    strategy = DCSOCStrategy(fanout=FANOUT, inter_fanout=INTER_FANOUT)
./scripts/validate_dcsoc_s11.py:85:    original_adaptive_decision = AHBNController.decide_mode_and_fanout
./scripts/validate_dcsoc_s11.py:86:    original_select_targets = DCSOCStrategy.select_targets
./scripts/validate_dcsoc_s11.py:109:            "decide_mode_and_fanout",
./scripts/validate_dcsoc_s11.py:112:        patch.object(DCSOCStrategy, "select_targets", count_select_targets),
./scripts/validate_dcsoc_s11.py:129:        and isinstance(simulator.strategy, DCSOCStrategy)
./scripts/validate_dcsoc_s11.py:169:    print("DCSOCStrategy.select_targets()")
./scripts/validate_dcsoc_s11.py:187:    assert passed, "S11 DC-SoC runtime isolation validation failed."
./configs/exp10_failure.yaml:15:fanout: 3
./configs/exp10_failure.yaml:27:  fanout: 3
./configs/exp10_failure.yaml:28:  inter_fanout: 1
./configs/exp10_failure.yaml:60:  min_fanout: 2
./configs/exp10_failure.yaml:61:  max_fanout: 4
./configs/exp10_failure.yaml:64:  default_fanout: 3
./scripts/summarize_results.py:18:def summarize_exp07(df: pd.DataFrame) -> pd.DataFrame:
./scripts/summarize_results.py:20:    for (strategy, fanout), group in df.groupby(["strategy", "fanout"], dropna=False):
./scripts/summarize_results.py:29:                "fanout": fanout,
./scripts/summarize_results.py:37:    return pd.DataFrame(rows).sort_values(["strategy", "fanout", "metric"], na_position="first")
./scripts/summarize_results.py:43:    if experiments == {"exp07"}:
./scripts/summarize_results.py:44:        summary = summarize_exp07(df)
./scripts/summarize_results.py:53:    group_cols = [c for c in ["experiment", "strategy", "fanout", "edge_prob", "ch_overload_factor"] if c in df.columns]
./scripts/validate_stage4_prerun_comparators.py:1:"""Configuration-only gate for the final Stage 4 Exp08--Exp12 rerun."""
./scripts/validate_stage4_prerun_comparators.py:12:    "exp08_ch_bottleneck.yaml",
./scripts/validate_stage4_prerun_comparators.py:13:    "exp09_dense_topology.yaml",
./scripts/validate_stage4_prerun_comparators.py:19:EXPECTED_DCSOC = {
./scripts/validate_stage4_prerun_comparators.py:22:    "fanout": 3,
./scripts/validate_stage4_prerun_comparators.py:23:    "inter_fanout": 1,
./scripts/validate_stage4_prerun_comparators.py:49:        if dcsoc != EXPECTED_DCSOC:
./scripts/validate_stage4_prerun_comparators.py:59:            "min_fanout": 2,
./scripts/validate_stage4_prerun_comparators.py:60:            "max_fanout": 4,
./scripts/validate_stage4_prerun_comparators.py:61:            "default_fanout": 3,
./scripts/validate_stage4_prerun_comparators.py:67:        # any experiment-level fanout (notably Exp09's Gossip condition of 4).
./scripts/validate_stage4_prerun_comparators.py:73:            "fanout": dcsoc.get("fanout", cfg.get("fanout", 3)) if dcsoc else None,
./scripts/validate_stage4_prerun_comparators.py:74:            "inter_fanout": dcsoc.get(
./scripts/validate_stage4_prerun_comparators.py:75:                "inter_fanout", cfg.get("dcsoc_inter_fanout", 1)
./scripts/validate_stage4_prerun_comparators.py:78:        if resolved != EXPECTED_DCSOC:
./scripts/validate_stage4_prerun_comparators.py:83:            f"ahbn.max_fanout={ahbn.get('max_fanout')}"
./scripts/validate_fanout6_amendment.py:22:RAW_PATH = ROOT / "outputs" / "csv" / "fanout6_validation_raw.csv"
./scripts/validate_fanout6_amendment.py:23:TRACE_PATH = ROOT / "outputs" / "csv" / "fanout6_validation_trace.csv"
./scripts/validate_fanout6_amendment.py:24:SUMMARY_PATH = ROOT / "outputs" / "csv" / "fanout6_validation_summary.csv"
./scripts/validate_fanout6_amendment.py:36:    requested = self._get_effective_fanout(node)
./scripts/validate_fanout6_amendment.py:70:        values = node_rows["fanout"].astype(int).tolist()
./scripts/validate_fanout6_amendment.py:84:    assert int(stage2["ahbn"]["min_fanout"]) == 2
./scripts/validate_fanout6_amendment.py:85:    assert int(stage2["ahbn"]["max_fanout"]) == 4
./scripts/validate_fanout6_amendment.py:98:            for max_fanout in MAX_FANOUTS:
./scripts/validate_fanout6_amendment.py:102:                        "experiment": "fanout6_amendment_validation",
./scripts/validate_fanout6_amendment.py:107:                    run_cfg["ahbn"]["max_fanout"] = max_fanout
./scripts/validate_fanout6_amendment.py:126:                        fanout=scenario.get("fanout"),
./scripts/validate_fanout6_amendment.py:133:                        scenario_tag=f"scenario={scenario_name};max_fanout={max_fanout}",
./scripts/validate_fanout6_amendment.py:140:                    adaptive.insert(0, "validation_max_fanout", max_fanout)
./scripts/validate_fanout6_amendment.py:144:                    fanouts = adaptive["fanout"].astype(int)
./scripts/validate_fanout6_amendment.py:148:                            "max_fanout": max_fanout,
./scripts/validate_fanout6_amendment.py:153:                            "mean_requested_fanout": float(fanouts.mean()),
./scripts/validate_fanout6_amendment.py:154:                            "min_observed_fanout": int(fanouts.min()),
./scripts/validate_fanout6_amendment.py:155:                            "max_observed_fanout": int(fanouts.max()),
./scripts/validate_fanout6_amendment.py:156:                            "fanout_transition_count": int(adaptive["fanout_changed"].sum()),
./scripts/validate_fanout6_amendment.py:158:                            "upper_bound_pct": float((fanouts == max_fanout).mean() * 100.0),
./scripts/validate_fanout6_amendment.py:159:                            "fanout_above_four": bool((fanouts > 4).any()),
./scripts/validate_fanout6_amendment.py:167:                        f"max_fanout={max_fanout} seed={seed} "
./scripts/validate_fanout6_amendment.py:170:                        f"requested_range={fanouts.min()}-{fanouts.max()} "
./scripts/validate_fanout6_amendment.py:188:        "mean_requested_fanout",
./scripts/validate_fanout6_amendment.py:189:        "min_observed_fanout",
./scripts/validate_fanout6_amendment.py:190:        "max_observed_fanout",
./scripts/validate_fanout6_amendment.py:191:        "fanout_transition_count",
./scripts/validate_fanout6_amendment.py:198:    for (scenario, max_fanout), group in raw.groupby(["scenario", "max_fanout"]):
./scripts/validate_fanout6_amendment.py:204:                    "max_fanout": max_fanout,
./scripts/validate_fanout6_amendment.py:214:    above_four = bool(raw.loc[raw["max_fanout"] == 6, "fanout_above_four"].any())
./scripts/validate_fanout6_amendment.py:215:    decreased = bool(raw.loc[raw["max_fanout"] == 6, "decreased_after_above_four"].any())
./scripts/validate_fanout6_amendment.py:217:        (raw.loc[raw["max_fanout"] == 6, "upper_bound_pct"] == 100.0).all()
./scripts/validate_fanout6_amendment.py:224:    print(f"Fanout >4 reached with max_fanout=6: {above_four}")
./scripts/validate_fanout6_amendment.py:226:    print(f"Permanently at fanout 6: {permanently_six}")
./scripts/validate_fanout6_amendment.py:238:        raise RuntimeError("Required max_fanout=6 adaptive behaviour was not demonstrated")
./scripts/inspect_exp08_e0.py:1:"""Stage 4 Exp08 E0: inspect and freeze the current configuration only."""
./scripts/inspect_exp08_e0.py:12:CONFIG_PATH = PROJECT_ROOT / "configs" / "exp08_ch_bottleneck.yaml"
./scripts/inspect_exp08_e0.py:13:FROZEN_DCSOC_PATH = PROJECT_ROOT / "configs" / "stage3_dcsoc.yaml"
./scripts/inspect_exp08_e0.py:21:from ahbn.strategies.dcsoc import DCSOCStrategy  # noqa: E402
./scripts/inspect_exp08_e0.py:28:    "kappa": 1.0, "beta": 1.0, "min_fanout": 2,
./scripts/inspect_exp08_e0.py:29:    "max_fanout": 4, "mode_threshold": 0.5,
./scripts/inspect_exp08_e0.py:31:EXPECTED_DCSOC = {"eps": 2.0, "min_samples": 3, "fanout": 3, "inter_fanout": 1}
./scripts/inspect_exp08_e0.py:43:    # Construct through the production Exp08 path, but do not inject/run a message.
./scripts/inspect_exp08_e0.py:66:    frozen_dcsoc = load_yaml_config(FROZEN_DCSOC_PATH)["dcsoc"]
./scripts/inspect_exp08_e0.py:82:    ahbn_ok = ahbn_ok and cfg.get("ahbn", {}).get("default_fanout", 3) == 3
./scripts/inspect_exp08_e0.py:84:    ahbn_ok = ahbn_ok and isinstance(ahbn_strategy, AHBNStrategy) and ahbn_strategy.adaptive_fanout
./scripts/inspect_exp08_e0.py:85:    dcsoc_explicit = cfg.get("dcsoc") == EXPECTED_DCSOC
./scripts/inspect_exp08_e0.py:87:    dcsoc_ok = dcsoc_explicit and dcsoc_effective == frozen_dcsoc == EXPECTED_DCSOC
./scripts/inspect_exp08_e0.py:96:    only_overload_varies = True  # exp08 loops overload -> run -> strategy; other arguments are invariant.
./scripts/inspect_exp08_e0.py:103:    print("Exp08 — CH Overload")
./scripts/inspect_exp08_e0.py:142:    print("  configs/exp08_ch_bottleneck.yaml: ch_overload_factor")
./scripts/inspect_exp08_e0.py:144:    print("    -> run_batch.exp08 overload loop")
./scripts/inspect_exp08_e0.py:157:    print("  Gossip                : no CH role; no CH-specific target")
./scripts/inspect_exp08_e0.py:159:    print("  DC-SoC                : own DBSCAN-derived cluster heads")
./scripts/inspect_exp08_e0.py:165:    print("\nGossip:")
./scripts/inspect_exp08_e0.py:166:    print("  fanout                : 3 (runner default; Exp08 has no fanout key)")
./scripts/inspect_exp08_e0.py:169:    print(f"  present in Exp08      : {'YES' if 'gossip' in configured else 'NO'}")
./scripts/inspect_exp08_e0.py:176:    print("\nDC-SoC:")
./scripts/inspect_exp08_e0.py:180:    print(f"  forwarding            : intra-cluster physical-neighbour fanout {dcsoc_effective['fanout']}; CH gateway reserve {dcsoc_effective['inter_fanout']}")
./scripts/inspect_exp08_e0.py:182:    print("  Exp08-specific tuning : NO")
./scripts/inspect_exp08_e0.py:183:    print(f"  explicit in Exp08     : {'YES' if dcsoc_explicit else 'NO'}")
./scripts/inspect_exp08_e0.py:193:    print(f"  fanout bounds         : [{params.min_fanout}, {params.max_fanout}]")
./scripts/inspect_exp08_e0.py:194:    print("  default fanout        : 3")
./scripts/inspect_exp08_e0.py:196:    print("  adaptive score/fanout : YES/YES")
./scripts/inspect_exp08_e0.py:198:    print("  Exp08-specific tuning : NO")
./scripts/inspect_exp08_e0.py:206:    print(f"DC-SoC frozen config consumed     : {status(dcsoc_ok)}")
./scripts/inspect_exp08_e0.py:208:    print("  None identified. Gossip has no CH role, so the consumed CH-overload")
./scripts/inspect_exp08_e0.py:209:    print("  mechanism intentionally has no directly targeted Gossip nodes.")
./scripts/inspect_exp08_e0.py:226:        print("Stage 4 Exp08 comparator configuration is now explicit and frozen.\n")
./scripts/inspect_exp08_e0.py:228:        print("  Gossip / Structured / DC-SoC / AHBN\n")
./scripts/validate_dcsoc_s2.py:2:Stage 3.4 — DC-SoC Sanity Validation
./scripts/validate_dcsoc_s2.py:5:Independently verifies the frozen Stage 3.3 DC-SoC cluster-head rule.
./scripts/validate_dcsoc_s2.py:24:    """Construct the deterministic S1 topology through the real DC-SoC path."""
./scripts/validate_dcsoc_s2.py:37:    print("STAGE 3.4 — DC-SoC SANITY VALIDATION")
./scripts/validate_dcsoc_s2.py:78:        # NetworkX physical overlay, not a DC-SoC selection helper.
./scripts/plot_exp07_side_by_side.py:13:    exp07_results_20260407_140852.csv
./scripts/plot_exp07_side_by_side.py:28:    parser.add_argument("csv_path", help="Path to exp07 results CSV")
./scripts/plot_exp07_side_by_side.py:45:        "fanout",
./scripts/plot_exp07_side_by_side.py:53:    # Keep only Exp07 and only the two strategies we want
./scripts/plot_exp07_side_by_side.py:54:    df = df[df["experiment"] == "exp07"].copy()
./scripts/plot_exp07_side_by_side.py:58:        raise ValueError("No Exp07 rows found for strategies 'gossip' and 'ahbn'.")
./scripts/plot_exp07_side_by_side.py:60:    # Aggregate mean and std across runs for each strategy/fanout
./scripts/plot_exp07_side_by_side.py:62:        df.groupby(["strategy", "fanout"], as_index=False)
./scripts/plot_exp07_side_by_side.py:69:        .sort_values(["strategy", "fanout"])
./scripts/plot_exp07_side_by_side.py:72:    gossip = grouped[grouped["strategy"] == "gossip"].sort_values("fanout")
./scripts/plot_exp07_side_by_side.py:73:    ahbn = grouped[grouped["strategy"] == "ahbn"].sort_values("fanout")
./scripts/plot_exp07_side_by_side.py:79:    output_path = output_dir / f"exp07_gossip_vs_ahbn_side_by_side_{timestamp}.png"
./scripts/plot_exp07_side_by_side.py:88:        gossip["fanout"],
./scripts/plot_exp07_side_by_side.py:91:        label="Gossip",
./scripts/plot_exp07_side_by_side.py:94:        ahbn["fanout"],
./scripts/plot_exp07_side_by_side.py:103:            gossip["fanout"],
./scripts/plot_exp07_side_by_side.py:111:            ahbn["fanout"],
./scripts/plot_exp07_side_by_side.py:118:    plt.title("Exp07: Delay vs Fanout")
./scripts/plot_exp07_side_by_side.py:121:    plt.xticks(sorted(df["fanout"].dropna().unique()))
./scripts/plot_exp07_side_by_side.py:130:        gossip["fanout"],
./scripts/plot_exp07_side_by_side.py:133:        label="Gossip",
./scripts/plot_exp07_side_by_side.py:136:        ahbn["fanout"],
./scripts/plot_exp07_side_by_side.py:145:            gossip["fanout"],
./scripts/plot_exp07_side_by_side.py:153:            ahbn["fanout"],
./scripts/plot_exp07_side_by_side.py:160:    plt.title("Exp07: Duplicates vs Fanout")
./scripts/plot_exp07_side_by_side.py:163:    plt.xticks(sorted(df["fanout"].dropna().unique()))
./scripts/plot_exp07_side_by_side.py:167:    plt.suptitle("Experiment 07: Gossip vs AHBN", fontsize=12)
./scripts/aggregate_exp08_e7.py:2:"""Validate and aggregate the final Stage 4 Exp08 rerun for E5."""
./scripts/aggregate_exp08_e7.py:10:FINAL=(ROOT/'outputs/csv/exp08_results_20260821_164541.csv').resolve()
./scripts/aggregate_exp08_e7.py:12:NAMES={'gossip':'Gossip','cluster':'Structured','dcsoc':'DC-SoC','ahbn':'AHBN'}
./scripts/aggregate_exp08_e7.py:23:    print('E5 Exp08 final aggregation'); print(f'Input: {source}'); print(f'Input SHA-256: {before}')
./scripts/aggregate_exp08_e7.py:44:    dest=ROOT/'outputs/csv'/f'exp08_final_summary_{a.timestamp}.csv'; out.to_csv(dest,index=False)
./scripts/plot_exp07_publication.py:27:        "fanout",
./scripts/plot_exp07_publication.py:35:    df = df[df["experiment"] == "exp07"].copy()
./scripts/plot_exp07_publication.py:39:        raise ValueError("No Exp07 rows found for strategies 'gossip' and 'ahbn'.")
./scripts/plot_exp07_publication.py:42:        df.groupby(["strategy", "fanout"], as_index=False)
./scripts/plot_exp07_publication.py:49:        .sort_values(["strategy", "fanout"])
./scripts/plot_exp07_publication.py:52:    gossip = grouped[grouped["strategy"] == "gossip"].sort_values("fanout")
./scripts/plot_exp07_publication.py:53:    ahbn = grouped[grouped["strategy"] == "ahbn"].sort_values("fanout")
./scripts/plot_exp07_publication.py:63:    parser.add_argument("csv_path", help="Path to exp07 results CSV")
./scripts/plot_exp07_publication.py:78:    output_png = output_dir / f"exp07_publication_{timestamp}.png"
./scripts/plot_exp07_publication.py:79:    output_pdf = output_dir / f"exp07_publication_{timestamp}.pdf"
./scripts/plot_exp07_publication.py:98:    x_ticks = sorted(gossip["fanout"].dropna().unique())
./scripts/plot_exp07_publication.py:106:        gossip["fanout"],
./scripts/plot_exp07_publication.py:113:        label="Gossip",
./scripts/plot_exp07_publication.py:117:        ahbn["fanout"],
./scripts/plot_exp07_publication.py:140:        gossip["fanout"],
./scripts/plot_exp07_publication.py:147:        label="Gossip",
./scripts/plot_exp07_publication.py:151:        ahbn["fanout"],
./scripts/plot_exp07_publication.py:168:    fig.suptitle("Experiment 07: Gossip and AHBN under Fanout Variation", y=1.02, fontsize=13)
./scripts/plot_exp08_e8.py:2:"""Generate four final Exp08 figures from the E5 summary only."""
./scripts/plot_exp08_e8.py:10:COMPS=['Gossip','Structured','DC-SoC','AHBN']; LEVELS=[1.0,1.5,2.0,3.0]
./scripts/plot_exp08_e8.py:18:    source=a.summary.resolve(); req(source.parent==(ROOT/'outputs/csv').resolve() and source.name.startswith('exp08_final_summary_'),'not a final summary')
./scripts/plot_exp08_e8.py:33:        dest=ROOT/'outputs/figures'/f'exp08_final_{metric}_{a.timestamp}.png'; dest.parent.mkdir(parents=True,exist_ok=True); fig.savefig(dest,dpi=300,bbox_inches='tight'); plt.close(fig)
./scripts/plot_exp08_e8.py:35:    print('E6 Exp08 final plotting'); print(f'Summary input only: {source}'); print('Validation: 16 conditions; n=20; 4 comparators x 4 overloads'); print('Error bars: mean +/- Student-t 95% CI')
./scripts/validate_dcsoc_s3.py:2:Stage 3.4 — DC-SoC Sanity Validation
./scripts/validate_dcsoc_s3.py:5:Runs the production DC-SoC strategy and simulator, while observing the
./scripts/validate_dcsoc_s3.py:15:from ahbn.strategies.dcsoc import DCSOCStrategy
./scripts/validate_dcsoc_s3.py:60:    print("STAGE 3.4 — DC-SoC SANITY VALIDATION")
./scripts/validate_dcsoc_s3.py:72:    strategy = DCSOCStrategy(
./scripts/validate_dcsoc_s3.py:73:        fanout=FANOUT,
./scripts/validate_dcsoc_s3.py:74:        inter_fanout=INTER_FANOUT,
./scripts/validate_dcsoc_s6.py:1:"""Stage 3.4 S6: validate triggered DC-SoC structural maintenance."""
./scripts/validate_dcsoc_s6.py:15:from ahbn.strategies.dcsoc import DCSOCStrategy
./scripts/validate_dcsoc_s6.py:46:        # Frozen DC-SoC repair uses _select_cluster_head with
./scripts/validate_dcsoc_s6.py:56:    print("STAGE 3.4 — DC-SoC SANITY VALIDATION")
./scripts/validate_dcsoc_s6.py:65:        strategy=DCSOCStrategy(fanout=FANOUT, inter_fanout=INTER_FANOUT),
./scripts/validate_dcsoc_s6.py:151:        isinstance(simulator.strategy, DCSOCStrategy)
./scripts/validate_dcsoc_s6.py:154:        and simulator.metrics.fanout_change_count == 0
./scripts/validate_dcsoc_s6.py:228:        print("DC-SoC structural-update trigger. The resulting active memberships and")
./scripts/validate_exp08_e1.py:1:"""Stage 4 Exp08 E1: independently validate CH-overload injection."""
./scripts/validate_exp08_e1.py:12:CONFIG_PATH = PROJECT_ROOT / "configs" / "exp08_ch_bottleneck.yaml"
./scripts/validate_exp08_e1.py:23:from ahbn.strategies.dcsoc import DCSOCStrategy  # noqa: E402
./scripts/validate_exp08_e1.py:24:from ahbn.strategies.gossip import GossipStrategy  # noqa: E402
./scripts/validate_exp08_e1.py:33:    """Construct through the production Exp08 path without running events."""
./scripts/validate_exp08_e1.py:134:        return isinstance(sim.strategy, GossipStrategy) and sim.strategy.fanout == 3 and sim.controller is None
./scripts/validate_exp08_e1.py:136:        return isinstance(sim.strategy, ClusterStrategy) and sim.strategy.fanout is None and sim.controller is None
./scripts/validate_exp08_e1.py:140:            isinstance(sim.strategy, DCSOCStrategy)
./scripts/validate_exp08_e1.py:141:            and sim.strategy.fanout == frozen.get("fanout") == 3
./scripts/validate_exp08_e1.py:142:            and sim.strategy.inter_fanout == frozen.get("inter_fanout") == 1
./scripts/validate_exp08_e1.py:148:        return isinstance(sim.strategy, AHBNStrategy) and isinstance(sim.controller, AHBNController) and sim.strategy.adaptive_fanout
./scripts/validate_exp08_e1.py:202:        # Gossip is the intentionally untargeted CH-independent reference.
./scripts/validate_exp08_e1.py:218:            "gossip": "CH-independent static Gossip reference (no CH target)",
./scripts/validate_exp08_e1.py:271:        and ".fanout" not in source
./scripts/validate_stage4_exp07_execution.py:6:from run_batch import build_ahbn_params, build_ahbn_strategy, exp07
./scripts/validate_stage4_exp07_execution.py:22:    cfg = load_yaml_config("configs/exp07_fanout.yaml")
./scripts/validate_stage4_exp07_execution.py:25:        rows, _ = exp07(cfg)
./scripts/validate_stage4_exp07_execution.py:33:    fanouts = list(cfg["fanouts"])
./scripts/validate_stage4_exp07_execution.py:36:    strategy = build_ahbn_strategy(cfg, fanout=None)
./scripts/validate_stage4_exp07_execution.py:39:        "gossip_sweep": sorted({call["fanout"] for call in gossip_calls}) == fanouts,
./scripts/validate_stage4_exp07_execution.py:40:        "gossip_runs": len(gossip_calls) == len(fanouts) * runs,
./scripts/validate_stage4_exp07_execution.py:42:        "ahbn_no_sweep": all(call["fanout"] is None for call in ahbn_calls),
./scripts/validate_stage4_exp07_execution.py:43:        "ahbn_bounds": (params.min_fanout, params.max_fanout) == (2, 4),
./scripts/validate_stage4_exp07_execution.py:44:        "ahbn_default": strategy.default_fanout == 3,
./scripts/validate_stage4_exp07_execution.py:45:        "ahbn_adaptive": strategy.adaptive_fanout is True,
./scripts/validate_stage4_exp07_execution.py:46:        "ahbn_label": all(row.fanout is None for row in ahbn_rows),
./scripts/validate_stage4_exp07_execution.py:47:        "gossip_labels": sorted({row.fanout for row in gossip_rows}) == fanouts,
./scripts/validate_stage4_exp07_execution.py:54:    print(f"Gossip fixed-fanout sweep: {fanouts}")
./scripts/validate_stage4_exp07_execution.py:55:    print(f"Gossip scheduled runs    : {len(gossip_calls)}")
./scripts/validate_stage4_exp07_execution.py:58:    print(f"AHBN min_fanout          : {params.min_fanout}")
./scripts/validate_stage4_exp07_execution.py:59:    print(f"AHBN max_fanout          : {params.max_fanout}")
./scripts/validate_stage4_exp07_execution.py:60:    print(f"AHBN default_fanout      : {strategy.default_fanout}")
./scripts/validate_stage4_exp07_execution.py:61:    print(f"AHBN result fanout       : {ahbn_rows[0].fanout if ahbn_rows else 'missing'}")
./scripts/validate_dcsoc_lifecycle_post_s2.py:9:from ahbn.strategies.dcsoc import DCSOCStrategy
./scripts/validate_dcsoc_lifecycle_post_s2.py:21:    sim = Simulator(nodes, DCSOCStrategy(3, 1), seed=42, base_delay=1.0, jitter=0.0,
./scripts/validate_dcsoc_s10.py:1:"""Stage 3.4 S10: validate deterministic DC-SoC reproducibility."""
./scripts/validate_dcsoc_s10.py:8:from ahbn.strategies.dcsoc import DCSOCStrategy
./scripts/validate_dcsoc_s10.py:80:    """Build and execute one fresh DC-SoC experiment."""
./scripts/validate_dcsoc_s10.py:86:        strategy=DCSOCStrategy(fanout=FANOUT, inter_fanout=INTER_FANOUT),
./scripts/validate_dcsoc_s10.py:117:    print("STAGE 3.4 — DC-SoC SANITY VALIDATION")
./scripts/validate_dcsoc_s10.py:186:    assert passed, "S10 DC-SoC reproducibility validation failed."
./scripts/validate_dcsoc_s8.py:1:"""Stage 3.4 S8: validate that DC-SoC forwarding is structurally determined."""
./scripts/validate_dcsoc_s8.py:10:from ahbn.strategies.dcsoc import DCSOCStrategy
./scripts/validate_dcsoc_s8.py:38:        "fanout": control.fanout,
./scripts/validate_dcsoc_s8.py:61:    print("STAGE 3.4 — DC-SoC SANITY VALIDATION")
./scripts/validate_dcsoc_s8.py:68:    strategy = DCSOCStrategy(fanout=FANOUT, inter_fanout=INTER_FANOUT)
./scripts/validate_dcsoc_s8.py:84:    assert cases, "FAIL: no non-CH DC-SoC forwarding case has more candidates than fanout."
./scripts/validate_dcsoc_s8.py:99:    print("  DC-SoC forwarding   : Simulator.handle_receive()")
./scripts/validate_dcsoc_s8.py:100:    print("                        -> DCSOCStrategy.select_targets()")
./scripts/validate_dcsoc_s8.py:103:    print("                        -> fixed fanout/inter-fanout + simulator.rng sampling")
./scripts/validate_dcsoc_s8.py:112:    assert targets_before, "FAIL: baseline DC-SoC forwarding selected no targets."
./scripts/validate_dcsoc_s8.py:113:    assert len(targets_before) <= FANOUT, "FAIL: baseline selection exceeds fixed fanout."
./scripts/validate_dcsoc_s8.py:126:    print(f"  Fixed fanout        : {strategy.fanout}")
./scripts/validate_dcsoc_s8.py:127:    print(f"  Fixed inter-fanout  : {strategy.inter_fanout} (not exercised by non-CH source)")
./scripts/validate_dcsoc_s8.py:139:        fanout=4,
./scripts/validate_dcsoc_s8.py:143:    fixed_policy_unchanged = strategy.fanout == FANOUT and strategy.inter_fanout == INTER_FANOUT
./scripts/validate_dcsoc_s8.py:151:    assert structure_unchanged, "FAIL: AHBN mutation changed DC-SoC structural state."
./scripts/validate_dcsoc_s8.py:152:    assert fixed_policy_unchanged, "FAIL: AHBN mutation changed fixed DC-SoC policy."
./scripts/validate_dcsoc_s8.py:153:    assert targets_identical, "FAIL: AHBN NodeControlState changed DC-SoC forwarding targets."
./scripts/validate_dcsoc_s8.py:179:    assert strategy.fanout == FANOUT and strategy.inter_fanout == INTER_FANOUT, (
./scripts/validate_dcsoc_s8.py:180:        "FAIL: fixed DC-SoC forwarding policy changed during structural test."
./scripts/validate_dcsoc_s8.py:203:    policy_strategy = DCSOCStrategy(fanout=1, inter_fanout=INTER_FANOUT)
./scripts/validate_dcsoc_s8.py:205:    targets_fanout_one = policy_strategy.select_targets(source, message, simulator)
./scripts/validate_dcsoc_s8.py:207:    targets_fanout_three = strategy.select_targets(source, message, simulator)
./scripts/validate_dcsoc_s8.py:208:    policy_effect = len(targets_fanout_one) == 1 and len(targets_fanout_three) == FANOUT
./scripts/validate_dcsoc_s8.py:209:    assert policy_effect, "FAIL: fixed fanout did not bound target count as expected."
./scripts/validate_dcsoc_s8.py:214:    print(f"  Targets before      : {targets_fanout_one}")
./scripts/validate_dcsoc_s8.py:215:    print(f"  Targets after       : {targets_fanout_three}")
./scripts/validate_dcsoc_s8.py:231:    print("DC-SoC forwarding was invariant under irrelevant AHBN runtime state")
./scripts/validate_dcsoc_s35_freeze.py:1:"""Stage 3.5: record and verify the frozen DC-SoC comparison baseline."""
./scripts/validate_dcsoc_s35_freeze.py:12:from ahbn.strategies.dcsoc import DCSOCStrategy
./scripts/validate_dcsoc_s35_freeze.py:21:# experiment dimensions, not tunable DC-SoC algorithm parameters.
./scripts/validate_dcsoc_s35_freeze.py:27:    """Return only strategy-name branches whose condition selects DC-SoC."""
./scripts/validate_dcsoc_s35_freeze.py:72:    signature = inspect.signature(DCSOCStrategy.__init__)
./scripts/validate_dcsoc_s35_freeze.py:74:        "fanout": signature.parameters["fanout"].default,
./scripts/validate_dcsoc_s35_freeze.py:75:        "inter_fanout": signature.parameters["inter_fanout"].default,
./scripts/validate_dcsoc_s35_freeze.py:90:        "adaptive fanout": {
./scripts/validate_dcsoc_s35_freeze.py:91:            "adaptive_fanout", "min_fanout", "max_fanout",
./scripts/validate_dcsoc_s35_freeze.py:92:            "decide_mode_and_fanout",
./scripts/validate_dcsoc_s35_freeze.py:112:    required = ("eps", "min_samples", "fanout", "inter_fanout")
./scripts/validate_dcsoc_s35_freeze.py:119:        if int(dcsoc_cfg["fanout"]) != defaults["fanout"]:
./scripts/validate_dcsoc_s35_freeze.py:120:            errors.append("configured fanout does not match DCSOCStrategy default")
./scripts/validate_dcsoc_s35_freeze.py:121:        if int(dcsoc_cfg["inter_fanout"]) != defaults["inter_fanout"]:
./scripts/validate_dcsoc_s35_freeze.py:122:            errors.append("configured inter_fanout does not match DCSOCStrategy default")
./scripts/validate_dcsoc_s35_freeze.py:124:    strategy_tree = _source_tree(DCSOCStrategy)
./scripts/validate_dcsoc_s35_freeze.py:132:        errors.append("DC-SoC construction path could not be identified")
./scripts/validate_dcsoc_s35_freeze.py:139:        errors.append("DC-SoC imports or constructs AHBN controller state")
./scripts/validate_dcsoc_s35_freeze.py:140:    if consumption["adaptive fanout"]:
./scripts/validate_dcsoc_s35_freeze.py:141:        errors.append("DC-SoC reads adaptive fanout")
./scripts/validate_dcsoc_s35_freeze.py:143:        errors.append("DC-SoC reads EWMA metrics")
./scripts/validate_dcsoc_s35_freeze.py:145:        errors.append("DC-SoC reads AHBN node control state")
./scripts/validate_dcsoc_s35_freeze.py:155:        for token in ("neighbors", "cluster_id", "gateway_neighbors", "fanout", "inter_fanout")
./scripts/validate_dcsoc_s35_freeze.py:159:        and _contains_call(ast.Module(body=construction_nodes, type_ignores=[]), "DCSOCStrategy")
./scripts/validate_dcsoc_s35_freeze.py:164:        errors.append("fixed DC-SoC forwarding path could not be identified")
./scripts/validate_dcsoc_s35_freeze.py:166:        errors.append("DC-SoC construction parameters could not be traced")
./scripts/validate_dcsoc_s35_freeze.py:172:    print("STAGE 3.5 — DC-SoC MINIMAL PARAMETER SANITY / FREEZE")
./scripts/validate_dcsoc_s35_freeze.py:174:    print("\nDC-SoC baseline parameter snapshot:\n")
./scripts/validate_dcsoc_s35_freeze.py:182:    _print_field("Fanout", f"fixed (total={dcsoc_cfg.get('fanout', 'UNIDENTIFIED')}, CH gateway reserve={dcsoc_cfg.get('inter_fanout', 'UNIDENTIFIED')})")
./scripts/validate_dcsoc_s35_freeze.py:186:    for label in ("alpha", "beta", "gamma", "EWMA", "thresholds", "mode score", "adaptive fanout"):
./scripts/validate_dcsoc_s35_freeze.py:192:    _print_field("DC-SoC baseline", "FROZEN" if passed else "NOT_FROZEN")
./scripts/validate_dcsoc_s35_freeze.py:209:            "fanout": dcsoc_cfg.get("fanout"),
./scripts/validate_dcsoc_s35_freeze.py:210:            "inter_fanout": dcsoc_cfg.get("inter_fanout"),
./scripts/validate_dcsoc_s1.py:2:Stage 3.4 — DC-SoC Sanity Validation
./scripts/validate_dcsoc_s1.py:6:    Verify that the frozen Stage 3.3 DC-SoC implementation assigns
./scripts/validate_dcsoc_s1.py:11:    - Uses the frozen Stage 3.3 DC-SoC parameters.
./scripts/validate_dcsoc_s1.py:36:# Frozen Stage 3.3 DC-SoC parameters
./scripts/validate_dcsoc_s1.py:56:    print("STAGE 3.4 — DC-SoC SANITY VALIDATION")
./scripts/validate_dcsoc_s1.py:149:    # 4. Run the real Stage 3.3 DC-SoC cluster construction
./scripts/validate_dcsoc_s1.py:198:        "FAIL: Nodes without a DC-SoC cluster assignment: "
./scripts/validate_dcsoc_s1.py:437:        "internally consistent DC-SoC cluster assignment."
./scripts/validate_dcsoc_s5.py:1:"""Stage 3.4 S5: independently sanity-check DC-SoC duplicates."""
./scripts/validate_dcsoc_s5.py:8:from ahbn.strategies.dcsoc import DCSOCStrategy
./scripts/validate_dcsoc_s5.py:60:    print("STAGE 3.4 — DC-SoC SANITY VALIDATION")
./scripts/validate_dcsoc_s5.py:83:        strategy=DCSOCStrategy(fanout=FANOUT, inter_fanout=INTER_FANOUT),
./scripts/plot_results.py:36:def get_exp07_3panel_output_path(timestamp: str) -> str:
./scripts/plot_results.py:37:    return f"outputs/plots/exp07_3panel_{timestamp}.png"
./scripts/plot_results.py:72:# Exp07
./scripts/plot_results.py:74:def plot_exp07(df: pd.DataFrame, ts: str, use_offset: bool) -> None:
./scripts/plot_results.py:80:        "fanout",
./scripts/plot_results.py:90:    if exp_values != {"exp07"} and "exp07" not in exp_values:
./scripts/plot_results.py:93:            f"This plotting script is intended for exp07."
./scripts/plot_results.py:101:        raise ValueError("Exp07 requires both Gossip and AHBN result rows.")
./scripts/plot_results.py:102:    if ahbn["fanout"].notna().any():
./scripts/plot_results.py:103:        raise ValueError("Exp07 AHBN result fanout must be blank (adaptive reference).")
./scripts/plot_results.py:110:    x_ticks = sorted(gossip["fanout"].dropna().unique())
./scripts/plot_results.py:112:        raise ValueError(f"Expected Gossip fanouts [2, 3, 4, 5, 6], got {x_ticks}")
./scripts/plot_results.py:116:        grouped = gossip.groupby("fanout")[metric].agg(["count", "mean", "std"]).reset_index()
./scripts/plot_results.py:124:            grouped["fanout"], grouped["mean"], yerr=half_width,
./scripts/plot_results.py:125:            marker="o", capsize=4, linewidth=1.8, label="Gossip (fixed fanout)",
./scripts/plot_results.py:136:        ax.set_title(f"{label} vs Gossip Fanout")
./scripts/plot_results.py:137:        ax.set_xlabel("Fixed Gossip Fanout")
./scripts/plot_results.py:144:    out_3panel = get_exp07_3panel_output_path(ts)
./scripts/plot_results.py:152:# Exp08
./scripts/plot_results.py:154:def plot_exp08(df: pd.DataFrame, ts: str, use_offset: bool) -> None:
./scripts/plot_results.py:200:    out = get_plot_output_path("exp08", ts)
./scripts/plot_results.py:208:# Exp09
./scripts/plot_results.py:219:def plot_exp09(df: pd.DataFrame, ts: str, use_offset: bool) -> None:
./scripts/plot_results.py:269:    out = get_plot_output_path("exp09", ts)
./scripts/plot_results.py:370:        "fanout",
./scripts/plot_results.py:396:    fanout_df = (
./scripts/plot_results.py:399:            mean_fanout=("fanout", "mean"),
./scripts/plot_results.py:400:            fanout_min=("fanout", "min"),
./scripts/plot_results.py:401:            fanout_max=("fanout", "max"),
./scripts/plot_results.py:431:        fanout_df["time_bin"],
./scripts/plot_results.py:432:        fanout_df["mean_fanout"],
./scripts/plot_results.py:435:        label="mean fanout",
./scripts/plot_results.py:438:        fanout_df["time_bin"],
./scripts/plot_results.py:439:        fanout_df["fanout_min"],
./scripts/plot_results.py:440:        fanout_df["fanout_max"],
./scripts/plot_results.py:709:            .agg(mean_fanout=("fanout", "mean"), mean_weight=("weight", "mean"), mean_chat=("c_hat", "mean"))
./scripts/plot_results.py:713:        axes[0].plot(part["time_bin"], part["mean_fanout"], marker="o", label=cls_name)
./scripts/plot_results.py:735:    axes[1].set_title("Class-wise Gossip Preference Over Time")
./scripts/plot_results.py:767:    adaptive_trace_cols = {"time", "node_id", "fanout", "mode", "d_hat"}
./scripts/plot_results.py:780:    if experiment == "exp07":
./scripts/plot_results.py:781:        plot_exp07(df, ts, use_offset)
./scripts/plot_results.py:782:    elif experiment == "exp08":
./scripts/plot_results.py:783:        plot_exp08(df, ts, use_offset)
./scripts/plot_results.py:784:    elif experiment == "exp09":
./scripts/plot_results.py:785:        plot_exp09(df, ts, use_offset)

```

### Command

```sh
sed -n '1,280p' ahbn/strategies/dcsoc.py && sed -n '1,180p' ahbn/strategies/gossip.py && sed -n '1,220p' run_batch.py && sed -n '220,480p' run_batch.py && sed -n '1,190p' run_one.py && ls -1 docs | rg 'stage4|exp08|exp09' && ls -1 scripts | rg 'exp0[789]|stage4'
```

### Complete terminal output

```text
from __future__ import annotations

from typing import List

from ahbn.message import Message
from ahbn.node import Node
from ahbn.strategies.base import ForwardingStrategy


class DCSOCStrategy(ForwardingStrategy):
    """
    DC-SoC-inspired density-clustered hybrid dissemination baseline.

    Behaviour:

        Ordinary node:
            gossip-like dissemination among physical neighbours
            belonging to the same density cluster.

        Cluster head:
            preserves the same bounded forwarding budget while
            reserving limited capacity for structured forwarding
            toward neighbouring cluster heads.

    The baseline therefore combines:

        intra-cluster gossip-like dissemination
        +
        inter-cluster structured dissemination

    This strategy deliberately contains no:

        - AHBN controller
        - EWMA
        - runtime mode switching
        - adaptive fanout
        - AHBN score/weight
        - runtime observation processing
    """

    def __init__(
        self,
        fanout: int = 3,
        inter_fanout: int = 1,
    ) -> None:

        if fanout < 1:
            raise ValueError(
                "DC-SoC fanout must be >= 1"
            )

        if inter_fanout < 0:
            raise ValueError(
                "DC-SoC inter_fanout must be >= 0"
            )

        self.fanout = int(
            fanout
        )

        self.inter_fanout = int(
            inter_fanout
        )

    # --------------------------------------------------------
    # Utility
    # --------------------------------------------------------

    @staticmethod
    def _sample(
        simulator,
        candidates: List[int],
        k: int,
    ) -> List[int]:
        """
        Sample up to k candidates using the simulator's seeded RNG.

        Reusing simulator.rng preserves reproducibility and ensures
        DC-SoC follows the same random-seed discipline as Gossip.
        """

        if k <= 0:
            return []

        if not candidates:
            return []

        if len(candidates) <= k:
            return candidates[:]

        return simulator.rng.sample(
            candidates,
            k,
        )

    # --------------------------------------------------------
    # Public strategy interface
    # --------------------------------------------------------

    def select_targets(
        self,
        node: Node,
        message: Message,
        simulator,
    ) -> List[int]:

        cluster_mgr = (
            simulator.cluster_manager
        )

        if cluster_mgr is None:
            return []

        if node.cluster_id is None:
            return []

        # Faithful S2 structural push: an ordinary member may only uplink
        # toward its assigned core.  It never expands the payload to an
        # independently sampled set of physical neighbours.
        if getattr(node, "dcsoc_role", "leaf") == "leaf":
            parent = getattr(node, "dcsoc_parent", None)
            if parent is None or parent not in simulator.nodes:
                return []
            return [parent] if simulator.nodes[parent].is_active else []

        # Core/routing nodes drive propagation down the explicit DAG.  The
        # existing fixed fanout remains a total resource bound.
        structural_children = [
            child_id
            for child_id in getattr(node, "dcsoc_children", [])
            if child_id in simulator.nodes and simulator.nodes[child_id].is_active
        ]
        if structural_children:
            return structural_children[: self.fanout]

        # ----------------------------------------------------
        # Intra-cluster Gossip candidates.
        #
        # Only active PHYSICAL neighbours in the same density
        # cluster participate in the gossip-like component.
        # ----------------------------------------------------

        local_candidates = [
            nbr_id
            for nbr_id
            in node.neighbors

            if nbr_id
            != node.node_id

            and nbr_id
            in simulator.nodes

            and simulator.nodes[
                nbr_id
            ].is_active

            and simulator.nodes[
                nbr_id
            ].cluster_id
            == node.cluster_id
        ]

        # ----------------------------------------------------
        # Ordinary density-cluster member.
        #
        # Pure intra-cluster gossip-like dissemination.
        # ----------------------------------------------------

        if not node.is_cluster_head:

            return self._sample(
                simulator,
                local_candidates,
                min(
                    self.fanout,
                    len(
                        local_candidates
                    ),
                ),
            )

        # ----------------------------------------------------
        # Density-cluster core/head.
        #
        # In addition to local gossip-like dissemination,
        # the core/head provides the structured path between
        # density clusters.
        # ----------------------------------------------------

        gateway_candidates = [
            gateway_id
            for gateway_id
            in node.gateway_neighbors

            if gateway_id
            != node.node_id

            and gateway_id
            in simulator.nodes

            and simulator.nodes[
                gateway_id
            ].is_active
        ]

        # ----------------------------------------------------
        # Reserve a bounded portion of the SAME forwarding
        # budget for inter-cluster propagation.
        #
        # This is important for fairness: DC-SoC does not get
        # fanout + extra unlimited gateway transmissions.
        # ----------------------------------------------------

        gateway_budget = min(
            self.inter_fanout,
            self.fanout,
            len(
                gateway_candidates
            ),
        )

        selected_gateways = (
            self._sample(
                simulator,
                gateway_candidates,
                gateway_budget,
            )
        )

        # ----------------------------------------------------
        # Remaining budget performs local gossip-like
        # dissemination.
        # ----------------------------------------------------

        local_budget = max(
            0,
            self.fanout
            - len(
                selected_gateways
            ),
        )

        selected_local = (
            self._sample(
                simulator,
                local_candidates,
                min(
                    local_budget,
                    len(
                        local_candidates
                    ),
                ),
            )
        )

        # ----------------------------------------------------
        # Gateway first preserves an outward structured path.
        # Remove accidental duplicates defensively.
        # ----------------------------------------------------

        return [
            target_id

            for target_id
            in dict.fromkeys(
                selected_gateways
                + selected_local
            )

            if target_id
            != node.node_id
        ]
from __future__ import annotations

from typing import List

from ahbn.message import Message
from ahbn.node import Node
from ahbn.strategies.base import ForwardingStrategy


class GossipStrategy(ForwardingStrategy):
    """
    Pure Gossip dissemination strategy.

    The strategy performs no adaptive decision-making.

    Responsibility:
        Randomly select up to `fanout` active physical neighbors.

    AHBN may update `fanout` before calling this strategy.
    """

    def __init__(self, fanout: int = 3) -> None:
        if fanout < 1:
            raise ValueError("fanout must be >= 1")

        self.fanout = fanout

    def select_targets(
        self,
        node: Node,
        message: Message,
        simulator,
    ) -> List[int]:
        """
        Select up to `fanout` active neighbors uniformly at random.
        """

        candidates = [
            nbr_id
            for nbr_id in node.neighbors
            if nbr_id != node.node_id
            and nbr_id in simulator.nodes
            and simulator.nodes[nbr_id].is_active
        ]

        if not candidates:
            return []

        k = min(
            int(self.fanout),
            len(candidates),
        )

        return simulator.rng.sample(
            candidates,
            k,
        )from __future__ import annotations

import argparse

from ahbn.config import load_yaml_config
from ahbn.control import AHBNController, AHBNParams
from ahbn.churn_manager import ChurnManager
from ahbn.failure_injector import FailureInjector
from ahbn.simulator import Simulator
from ahbn.strategies.ahbn import AHBNStrategy
from ahbn.strategies.cluster import ClusterStrategy
from ahbn.strategies.dcsoc import DCSOCStrategy
from ahbn.strategies.gossip import GossipStrategy
from ahbn.strategies.hybrid_fixed import HybridFixedStrategy
from ahbn.topology import (
    assign_dcsoc_clusters,
    assign_mixed_resources,
    assign_static_clusters,
    build_nodes_from_graph,
    get_or_build_topology,
)
from ahbn.utils import ResultRow, save_results_csv, save_adaptive_trace_csv


def build_ahbn_params(cfg: dict) -> AHBNParams:
    ahbn_cfg = cfg.get("ahbn", {})

    return AHBNParams(
        alpha=ahbn_cfg.get("alpha", 0.3),
        d0=ahbn_cfg.get("d0", 0.5),
        l0=ahbn_cfg.get("l0", 0.5),
        u0=ahbn_cfg.get("u0", 0.5),
        c0=ahbn_cfg.get("c0", 0.5),
        w_d=ahbn_cfg.get("w_d", -1.0),
        w_l=ahbn_cfg.get("w_l", 1.0),
        w_u=ahbn_cfg.get("w_u", -1.0),
        w_c=ahbn_cfg.get("w_c", 1.0),
        kappa=ahbn_cfg.get("kappa", 1.0),
        beta=ahbn_cfg.get("beta", 1.0),
        min_fanout=ahbn_cfg.get("min_fanout", 2),
        max_fanout=ahbn_cfg.get("max_fanout", 4),
        mode_threshold=ahbn_cfg.get("mode_threshold", 0.5),
    )


def build_ahbn_strategy(cfg: dict, fanout: int | None = None) -> AHBNStrategy:
    ahbn_cfg = cfg.get("ahbn", {})
    default_fanout = (
        fanout
        if fanout is not None
        else ahbn_cfg.get("default_fanout", 3)
    )

    return AHBNStrategy(
        default_fanout=default_fanout,
        adaptive_fanout=True,
    )


def run_single(
    cfg: dict,
    strategy_name: str,
    seed: int,
    topology_type: str,
    num_nodes: int,
    use_topology_cache: bool,
    base_delay: float,
    jitter: float,
    message_source: int,
    fanout: int | None = None,
    num_clusters: int | None = None,
    ch_overload_factor: float | None = None,
    edge_prob: float | None = None,
    ba_m: int | None = None,
    failure_mode: str | None = None,
    enable_adaptive_trace: bool = False,
    churn_rate: float | None = None,
    resource_scenario: str | None = None,
    scenario_tag: str | None = None,
) -> dict:
    graph = get_or_build_topology(
        topology_type=topology_type,
        num_nodes=num_nodes,
        seed=seed,
        use_cache=use_topology_cache,
        edge_prob=edge_prob,
        ba_m=ba_m,
    )
    nodes = build_nodes_from_graph(graph)

    experiment_name = cfg.get("experiment", "")
    if experiment_name == "exp12":
        assign_mixed_resources(nodes, cfg, seed=seed, scenario_name=resource_scenario)

    cluster_manager = None
    controller = None

    if strategy_name == "gossip":
        strategy = GossipStrategy(fanout=fanout if fanout is not None else 3)

    elif strategy_name == "cluster":
        cluster_manager = assign_static_clusters(
            nodes,
            num_clusters=num_clusters or 4,
            resource_aware_heads=False,
        )
        strategy = ClusterStrategy()

    elif strategy_name == "ahbn":
        cluster_manager = assign_static_clusters(
            nodes,
            num_clusters=num_clusters or 4,
            resource_aware_heads=False,
        )
        controller = AHBNController(build_ahbn_params(cfg))
        strategy = build_ahbn_strategy(cfg, fanout=fanout)

    elif strategy_name == "hybrid_fixed":

        cluster_manager = assign_static_clusters(
            nodes,
            num_clusters=num_clusters or 4,
        )

        strategy = HybridFixedStrategy(
            fanout=(
                fanout
                if fanout is not None
                else 3
            )
        )

    elif strategy_name == "dcsoc":

        dcsoc_cfg = cfg.get(
            "dcsoc",
            {},
        )

        cluster_manager = assign_dcsoc_clusters(
            nodes,
            eps=float(
                dcsoc_cfg.get(
                    "eps",
                    2.0,
                )
            ),
            min_samples=int(
                dcsoc_cfg.get(
                    "min_samples",
                    3,
                )
            ),
        )

        strategy = DCSOCStrategy(
            fanout=int(
                dcsoc_cfg.get(
                    "fanout",
                    (
                        fanout
                        if fanout is not None
                        else 3
                    ),
                )
            ),
            inter_fanout=int(
                dcsoc_cfg.get(
                    "inter_fanout",
                    1,
                )
            ),
        )

    else:
        raise ValueError(
            f"Unknown strategy: {strategy_name}"
        )

    local_cfg = dict(cfg)
    if failure_mode is not None:
        local_failure = dict(cfg.get("failure", {}))
        local_failure["mode"] = failure_mode
        local_cfg["failure"] = local_failure

    if churn_rate is not None:
        local_churn = dict(cfg.get("churn", {}))
        local_churn["target_fraction"] = churn_rate
        local_cfg["churn"] = local_churn

    failure_injector = FailureInjector(local_cfg, seed=seed)
    churn_manager = ChurnManager(local_cfg, seed=seed)

    sim = Simulator(
        nodes=nodes,
        strategy=strategy,
        seed=seed,
        base_delay=base_delay,
        jitter=jitter,
        cluster_manager=cluster_manager,
        controller=controller,
        ch_overload_factor=ch_overload_factor if ch_overload_factor is not None else 1.0,
        failure_injector=failure_injector,
        churn_manager=churn_manager,
        experiment_name=cfg.get("experiment", "unknown"),
        strategy_name=strategy_name,
        scenario_tag=(
            scenario_tag
            if scenario_tag is not None
            else (
                resource_scenario
                if resource_scenario is not None
                else (failure_mode if failure_mode is not None else topology_type)
            )
        ),
        enable_adaptive_trace=enable_adaptive_trace,
        resource_aware_heads=False,
    )

    sim.inject_message(source_id=message_source, message_id="m1")
    sim.inject_message(source_id=message_source, message_id="m1")
    sim.run()

    summary = sim.metrics.summarize_message("m1", total_nodes=len(sim.nodes))
    summary.update(sim.get_resource_metrics())
    if enable_adaptive_trace:
        summary["adaptive_trace_rows"] = sim.adaptive_trace_rows
    return summary


def exp07(cfg: dict) -> tuple[list[ResultRow], list]:
    rows: list[ResultRow] = []
    trace_rows: list = []

    base_seed = cfg["seed"]
    runs_per_setting = cfg["runs_per_setting"]
    fanouts = cfg["fanouts"]
    num_nodes = cfg["num_nodes"]
    topology_type = cfg["topology_type"]
    use_topology_cache = cfg.get("use_topology_cache", True)

    base_delay = cfg.get("base_delay", 1.0)
    jitter = cfg.get("jitter", 0.2)
    source_id = cfg.get("message_source", 0)
    num_clusters = cfg.get("num_clusters", 4)

    edge_prob = cfg.get("edge_prob")
    ba_m = cfg.get("ba_m")

    strategies = cfg.get("strategies", ["gossip", "ahbn"])

    if "gossip" in strategies:
        for fanout in fanouts:
            for run_idx in range(runs_per_setting):
                seed = base_seed + run_idx

                summary = run_single(
                    cfg=cfg,
                    strategy_name="gossip",
                    seed=seed,
                    topology_type=topology_type,
                    num_nodes=num_nodes,
                    use_topology_cache=use_topology_cache,
                    base_delay=base_delay,
                    jitter=jitter,
                    message_source=source_id,
                    fanout=fanout,
                    num_clusters=num_clusters,
                    edge_prob=edge_prob,
                    ba_m=ba_m,
                    enable_adaptive_trace=False,
                    scenario_tag=f"fanout={fanout}",
                )
                rows.append(
                    ResultRow(
                        experiment="exp07",
                        strategy="gossip",
                        seed=seed,
                        num_nodes=num_nodes,
                        topology_type=topology_type,
                        topology_param=edge_prob if topology_type == "er" else ba_m,
                        fanout=fanout,
                        num_clusters=num_clusters,
                        ch_overload_factor=None,
                        delivery_ratio=summary["delivery_ratio"],
                        propagation_delay=summary["propagation_delay"],
                        duplicates=summary["duplicates"],
                        total_forwards=summary["total_forwards"],
                    )
                )

    if "ahbn" in strategies:
        for run_idx in range(runs_per_setting):
            seed = base_seed + run_idx

            summary = run_single(
                cfg=cfg,
                strategy_name="ahbn",
                seed=seed,
                topology_type=topology_type,
                num_nodes=num_nodes,
                use_topology_cache=use_topology_cache,
                base_delay=base_delay,
                jitter=jitter,
                message_source=source_id,
                fanout=None,
                num_clusters=num_clusters,
                edge_prob=edge_prob,
                ba_m=ba_m,
                enable_adaptive_trace=True,
                scenario_tag="adaptive",
            )
            rows.append(
                ResultRow(
                    experiment="exp07",
                    strategy="ahbn",
                    seed=seed,
                    num_nodes=num_nodes,
                    topology_type=topology_type,
                    topology_param=edge_prob if topology_type == "er" else ba_m,
                    fanout=None,
                    num_clusters=num_clusters,
                    ch_overload_factor=None,
                    delivery_ratio=summary["delivery_ratio"],
                    propagation_delay=summary["propagation_delay"],
                    duplicates=summary["duplicates"],
                    total_forwards=summary["total_forwards"],
                )
            )

            if "adaptive_trace_rows" in summary:
                trace_rows.extend(summary["adaptive_trace_rows"])

    return rows, trace_rows


def exp08(cfg: dict) -> tuple[list[ResultRow], list]:
    rows: list[ResultRow] = []
    trace_rows: list = []

    base_seed = cfg["seed"]
    runs_per_setting = cfg["runs_per_setting"]
    overload_values = cfg["ch_overload_factor"]
    num_nodes = cfg["num_nodes"]
    topology_type = cfg["topology_type"]
    use_topology_cache = cfg.get("use_topology_cache", True)

    base_delay = cfg.get("base_delay", 1.0)
    jitter = cfg.get("jitter", 0.2)
    source_id = cfg.get("message_source", 0)
    num_clusters = cfg["num_clusters"]

    edge_prob = cfg.get("edge_prob")
    ba_m = cfg.get("ba_m")

    strategies = cfg.get("strategies", ["cluster", "ahbn"])

    for overload in overload_values:
        for run_idx in range(runs_per_setting):
            seed = base_seed + run_idx

            for strategy_name in strategies:
                summary = run_single(
                    cfg=cfg,
                    strategy_name=strategy_name,
                    seed=seed,
                    topology_type=topology_type,
                    num_nodes=num_nodes,
                    use_topology_cache=use_topology_cache,
                    base_delay=base_delay,
                    jitter=jitter,
                    message_source=source_id,
                    num_clusters=num_clusters,
                    ch_overload_factor=overload,
                    edge_prob=edge_prob,
                    ba_m=ba_m,
                    enable_adaptive_trace=(strategy_name == "ahbn"),
                    scenario_tag=f"ch_overload_factor={overload}",
                )
                rows.append(
                    ResultRow(
                        experiment="exp08",
                        strategy=strategy_name,
                        seed=seed,
                        num_nodes=num_nodes,
                        topology_type=topology_type,
                        topology_param=edge_prob if topology_type == "er" else ba_m,
                        fanout=None,
                        num_clusters=num_clusters,
                        ch_overload_factor=overload,
                        delivery_ratio=summary["delivery_ratio"],
                        propagation_delay=summary["propagation_delay"],
                        duplicates=summary["duplicates"],
                        total_forwards=summary["total_forwards"],
                    )
                )

                if "adaptive_trace_rows" in summary:
                    trace_rows.extend(summary["adaptive_trace_rows"])

    return rows, trace_rows


def exp09(cfg: dict) -> tuple[list[ResultRow], list]:
    rows: list[ResultRow] = []
    trace_rows: list = []

    base_seed = cfg["seed"]
    runs_per_setting = cfg["runs_per_setting"]
    num_nodes = cfg["num_nodes"]
    topology_type = cfg["topology_type"]
    use_topology_cache = cfg.get("use_topology_cache", True)

    base_delay = cfg.get("base_delay", 1.0)
    jitter = cfg.get("jitter", 0.2)
    source_id = cfg.get("message_source", 0)
    fanout = cfg.get("fanout", 3)
    num_clusters = cfg.get("num_clusters", 4)

    if topology_type != "er":
        raise ValueError("Exp09 density sweep is intended for ER topology.")

    edge_probs = cfg["edge_probs"]
    strategies = cfg.get("strategies", ["gossip", "cluster", "ahbn"])

    for edge_prob in edge_probs:
        for run_idx in range(runs_per_setting):
            seed = base_seed + run_idx

            for strategy_name in strategies:
                summary = run_single(
                    cfg=cfg,
                    strategy_name=strategy_name,
                    seed=seed,
                    topology_type="er",
                    num_nodes=num_nodes,
                    use_topology_cache=use_topology_cache,
                    base_delay=base_delay,
                    jitter=jitter,
                    message_source=source_id,
                    fanout=fanout,
                    num_clusters=num_clusters,
                    edge_prob=edge_prob,
                    enable_adaptive_trace=(strategy_name == "ahbn"),
                    scenario_tag=f"edge_prob={edge_prob}",
                )
                rows.append(
                    ResultRow(
                        experiment="exp09",
                        strategy=strategy_name,
                        seed=seed,
                        num_nodes=num_nodes,
                        topology_type="er",
                        topology_param=edge_prob,
                        fanout=fanout if strategy_name != "cluster" else None,
                        num_clusters=num_clusters,
                        ch_overload_factor=None,
                        delivery_ratio=summary["delivery_ratio"],
                        propagation_delay=summary["propagation_delay"],
                        duplicates=summary["duplicates"],
                        total_forwards=summary["total_forwards"],
                    )
                )

                if "adaptive_trace_rows" in summary:
                    trace_rows.extend(summary["adaptive_trace_rows"])

    return rows, trace_rows


def exp10(cfg: dict) -> tuple[list[dict], list]:
    rows: list[dict] = []
    trace_rows: list = []

    base_seed = cfg["seed"]
    runs_per_setting = cfg["runs_per_setting"]
    num_nodes = cfg["num_nodes"]
    topology_type = cfg["topology_type"]
    use_topology_cache = cfg.get("use_topology_cache", True)

    base_delay = cfg.get("base_delay", 1.0)
from __future__ import annotations

import argparse

from ahbn.config import load_yaml_config
from ahbn.control import AHBNController, AHBNParams
from ahbn.simulator import Simulator
from ahbn.strategies.ahbn import AHBNStrategy
from ahbn.strategies.cluster import ClusterStrategy
from ahbn.strategies.dcsoc import DCSOCStrategy
from ahbn.strategies.gossip import GossipStrategy
from ahbn.topology import (
    assign_dcsoc_clusters,
    assign_static_clusters,
    build_nodes_from_graph,
    get_or_build_topology,
)


def build_ahbn_params(cfg: dict) -> AHBNParams:
    ahbn_cfg = cfg.get("ahbn", {})

    return AHBNParams(
        alpha=ahbn_cfg.get("alpha", 0.3),
        d0=ahbn_cfg.get("d0", 0.5),
        l0=ahbn_cfg.get("l0", 0.5),
        u0=ahbn_cfg.get("u0", 0.5),
        c0=ahbn_cfg.get("c0", 0.5),
        w_d=ahbn_cfg.get("w_d", -1.0),
        w_l=ahbn_cfg.get("w_l", 1.0),
        w_u=ahbn_cfg.get("w_u", -1.0),
        w_c=ahbn_cfg.get("w_c", 1.0),
        kappa=ahbn_cfg.get("kappa", 1.0),
        beta=ahbn_cfg.get("beta", 1.0),
        min_fanout=ahbn_cfg.get("min_fanout", 2),
        max_fanout=ahbn_cfg.get("max_fanout", 4),
        mode_threshold=ahbn_cfg.get("mode_threshold", 0.5),
    )


def build_ahbn_strategy(cfg: dict, fanout_override: int | None = None) -> AHBNStrategy:
    ahbn_cfg = cfg.get("ahbn", {})

    default_fanout = (
        fanout_override
        if fanout_override is not None
        else ahbn_cfg.get("default_fanout", 3)
    )

    return AHBNStrategy(
        default_fanout=default_fanout,
        adaptive_fanout=True,
    )


def build_simulation_from_config(cfg: dict, strategy_name: str):
    seed = cfg["seed"]
    num_nodes = cfg["num_nodes"]
    topology_type = cfg["topology_type"]
    use_cache = cfg.get("use_topology_cache", True)

    base_delay = cfg.get("base_delay", 1.0)
    jitter = cfg.get("jitter", 0.2)
    message_source = cfg.get("message_source", 0)

    graph = get_or_build_topology(
        topology_type=topology_type,
        num_nodes=num_nodes,
        seed=seed,
        use_cache=use_cache,
        edge_prob=cfg.get("edge_prob"),
        ba_m=cfg.get("ba_m"),
    )
    nodes = build_nodes_from_graph(graph)

    cluster_manager = None
    controller = None
    ch_overload_factor = cfg.get("ch_overload_factor", 1.0)

    if strategy_name == "gossip":
        fanout = cfg.get("fanout", 3)
        strategy = GossipStrategy(fanout=fanout)

    elif strategy_name == "cluster":
        num_clusters = cfg.get("num_clusters", 4)
        cluster_manager = assign_static_clusters(nodes, num_clusters=num_clusters)
        strategy = ClusterStrategy()

    # elif strategy_name == "ahbn":
    #     num_clusters = cfg.get("num_clusters", 4)
    #     cluster_manager = assign_static_clusters(nodes, num_clusters=num_clusters)
    #     controller = AHBNController(build_ahbn_params(cfg))
    #     strategy = build_ahbn_strategy(cfg, fanout_override=cfg.get("fanout"))

    # else:
    #     raise ValueError(f"Unknown strategy: {strategy_name}")
    
    elif strategy_name == "ahbn":
        num_clusters = cfg.get("num_clusters", 4)

        cluster_manager = assign_static_clusters(
            nodes,
            num_clusters=num_clusters,
        )

        controller = AHBNController(
            build_ahbn_params(cfg)
        )

        strategy = build_ahbn_strategy(
            cfg,
            fanout_override=cfg.get("fanout"),
        )

    elif strategy_name == "dcsoc":

        dcsoc_cfg = cfg.get(
            "dcsoc",
            {},
        )

        cluster_manager = assign_dcsoc_clusters(
            nodes,
            eps=float(
                dcsoc_cfg.get(
                    "eps",
                    2.0,
                )
            ),
            min_samples=int(
                dcsoc_cfg.get(
                    "min_samples",
                    3,
                )
            ),
        )

        strategy = DCSOCStrategy(
            fanout=int(
                dcsoc_cfg.get(
                    "fanout",
                    cfg.get(
                        "fanout",
                        3,
                    ),
                )
            ),
            inter_fanout=int(
                dcsoc_cfg.get(
                    "inter_fanout",
                    1,
                )
            ),
        )

    else:
        raise ValueError(
            f"Unknown strategy: {strategy_name}"
        )

    sim = Simulator(
        nodes=nodes,
        strategy=strategy,
        seed=seed,
        base_delay=base_delay,
        jitter=jitter,
        cluster_manager=cluster_manager,
        controller=controller,
        ch_overload_factor=ch_overload_factor,
    )
    return sim, message_source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    # parser.add_argument("--strategy", required=True, choices=["gossip", "cluster", "ahbn"])
    parser.add_argument(
        "--strategy",
        required=True,
        choices=[
            "gossip",
            "cluster",
            "dcsoc",
            "ahbn",
        ],
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
exp08_e9_scientific_interpretation.md
stage4_exp07.md
stage4_exp08.md
stage4_exp08_rerun2.md
stage4rerun_exp08.md
aggregate_exp08_e7.py
inspect_exp08_e0.py
plot_exp07_publication.py
plot_exp07_side_by_side.py
plot_exp08_e8.py
validate_exp08_e1.py
validate_exp08_e6.py
validate_stage4_exp07_execution.py
validate_stage4_prerun_comparators.py

```

### Command

```sh
sed -n '1,260p' docs/stage4rerun_exp08.md && sed -n '1,260p' docs/stage4_exp08.md && sed -n '1,280p' scripts/validate_stage4_prerun_comparators.py && sed -n '680,760p' run_batch.py && sed -n '1,120p' configs/exp08_ch_bottleneck.yaml && sed -n '1,120p' configs/exp09_dense_topology.yaml
```

### Complete terminal output

```text
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
# Stage 4 — Exp08 CH Overload

## Experiment checklist

- [x] **E0 — Configuration inspection/freeze:** PASS after the documented E0.1 minimal configuration reconciliation
- [x] **E1 — Validate CH-overload injection:** PASS
- [x] **E2 — Run Gossip:** PASS
- [x] **E3 — Run Structured:** PASS
- [x] **E4 — Run DC-SoC:** PASS; the pre-execution diagnostic deviation is documented and excluded from the official dataset
- [x] **E5 — Run AHBN:** PASS
- [x] **E6 — Validate AHBN adaptive traces:** PASS
- [x] **E7 — Aggregate 20 runs (mean and 95% CI):** PASS
- [x] **E8 — Plot four algorithms:** PASS
- [x] **E9 — Scientific interpretation:** PASS

**Exp08 final status: COMPLETE — E0 through E9 PASS.** Exp09 was not started.

## E0 — Configuration Inspection / Freeze

### Purpose

Inspect the existing Exp08 contract, trace its effective runtime configuration, verify frozen comparator parameters, and stop before E1 if any scientific ambiguity remains. No algorithm or experiment behavior was changed.

### Command

```bash
cd /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6

/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python \
  scripts/inspect_exp08_e0.py
```

Exact interpreter: `/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python`

### Repository state inspected

- Root: `/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6`
- Exp08 configuration: `configs/exp08_ch_bottleneck.yaml`
- Frozen DC-SoC reference: `configs/stage3_dcsoc.yaml`
- Construction/runner paths: `run_one.py`, `run_batch.py`
- Runtime: `ahbn/config.py`, `ahbn/topology.py`, `ahbn/simulator.py`, `ahbn/node.py`, `ahbn/control.py`, `ahbn/metrics.py`
- Strategies: `ahbn/strategies/gossip.py`, `cluster.py`, `dcsoc.py`, `ahbn.py`
- Existing overload/failure support and prior Stage 3.5 freeze validation were also inspected.

### Experimental contract

Exp08 currently defines BA topology with 100 nodes, `m=3`, four static clusters, topology caching, base seed 42, 20 runs per setting (seeds 42–61), one message `m1` from fixed source node 0 at time 0, and event-queue exhaustion as termination. The timing convention is seconds: base delay 1.0 and per-send uniform jitter `[0.0, 0.2]`; default node processing delay is 0.0.

The independent-variable sweep is `ch_overload_factor = [1.0, 1.5, 2.0, 3.0]`. Other production-run arguments remain invariant within the Exp08 loops.

### Overload mechanism

`configs/exp08_ch_bottleneck.yaml:ch_overload_factor` is loaded by `ahbn.config.load_yaml_config`, iterated by `run_batch.exp08`, passed through `run_batch.run_single` into `Simulator.ch_overload_factor`, and consumed by `Simulator.send_message`. For a destination with `is_cluster_head=true`, it adds `base_delay * max(0, factor - 1)` to arrival delay. Thus the effective base component is multiplied by the factor. It does not change capacity, queue pressure, drops, availability, or `Node.is_overloaded`.

Targets are algorithm-specific. At seed 42: Gossip `[]`, Structured `[0,1,2,3]`, DC-SoC `[4]`, AHBN `[0,1,2,3]`. The physical overloaded set is not identical across algorithms.

### Comparator freeze

- Gossip is fixed fanout 3, non-adaptive, and controller-independent, but is absent from Exp08's configured strategy list.
- Structured uses round-robin static clusters, lowest-ID CHs, member-to-CH forwarding, and CH-to-members plus adjacent-CH gateways.
- DC-SoC uses frozen `eps=2.0`, `min_samples=3`, `fanout=3`, and `inter_fanout=1`; however, Exp08 does not state these values and reaches them through runner fallbacks.
- AHBN matches the frozen values (`alpha=.30`, centers `.50`, weights `-1,+1,-1,+1`, `kappa=1`, `beta=1`, threshold `.50`, fanout 2–4, default 3), with EWMA, adaptive score, and adaptive fanout active. No emergency failure/bottleneck override exists in the canonical controller/strategy; adding one was not attempted.

### Configuration-consumption checks

Overload, timing, workload, seed, canonical AHBN, and frozen DC-SoC effective values are consumed. The gate fails because the required four-comparator set is not configured, target fairness is not satisfied, DC-SoC values are implicit, and the requested emergency-override expectation differs from the frozen implementation.

### Controlled-variable checks

Topology is fixed per seed, configured strategies share workload/timing/seed pairing, parameters are frozen, and only the overload factor externally varies. No Exp08-specific tuning was found. The target semantics and comparator omission remain scientifically meaningful ambiguities.

### Terminal output

```text
========================================================================
STAGE 4 — FINAL COMPARATIVE EVALUATION
Exp08 — CH Overload
E0 — Configuration Inspection / Freeze
========================================================================

Repository:
  root                  : /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6

Python:
  interpreter           : /Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python

-----------------------------------------------------------------------
TOPOLOGY
-----------------------------------------------------------------------
Topology type           : BA
Node count              : 100
BA m / equivalent       : 3
Topology fixed params   : num_clusters=4; use_topology_cache=true
Seed policy             : base seed + run index; same seed reused across configured algorithms
Runs per setting        : 20
Exact seeds             : [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61]

-----------------------------------------------------------------------
WORKLOAD
-----------------------------------------------------------------------
Messages/run            : 1 (message_id=m1)
Source selection        : fixed node 0
Message timing          : injected at simulation clock 0.0
Termination             : event queue exhaustion; no duration/time limit
Same workload           : YES for every configured algorithm

-----------------------------------------------------------------------
TIMING
-----------------------------------------------------------------------
Base delay              : 1.0 seconds
Jitter                  : uniform [0.0, 0.2] seconds per send
Processing delay        : 0.0 seconds for default medium nodes
Units                   : seconds (current simulator convention)
Other timing fields     : CH extra = base_delay * max(0, factor - 1)

-----------------------------------------------------------------------
CH OVERLOAD
-----------------------------------------------------------------------
Config source           : /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/configs/exp08_ch_bottleneck.yaml
Configuration key       : ch_overload_factor
Overload levels         : [1.0, 1.5, 2.0, 3.0]

Physical meaning:
  Multiplicative arrival-delay factor for sends whose destination node is
  marked is_cluster_head; effective one-hop base component becomes
  base_delay * factor, before jitter and resource delays. It does not alter
  service capacity, queues, drops, availability, or Node.is_overloaded.

Runtime trace:
  configs/exp08_ch_bottleneck.yaml: ch_overload_factor
    -> ahbn.config.load_yaml_config
    -> run_batch.exp08 overload loop
    -> run_batch.run_single(ch_overload_factor=overload)
    -> Simulator.ch_overload_factor
    -> Simulator.send_message: if dst.is_cluster_head
    -> extra += base_delay * max(0, factor - 1)
Target-selection rule   : each algorithm's constructed nodes with is_cluster_head=true
Target node IDs (seed 42, representative first paired run):
  gossip                : []
  cluster               : [0, 1, 2, 3]
  dcsoc                 : [4]
  ahbn                  : [0, 1, 2, 3]

Same physical targets across algorithms:
  gossip                : []
  cluster               : [0, 1, 2, 3]
  dcsoc                 : [4]
  ahbn                  : [0, 1, 2, 3]
Overloaded physical node set is identical across algorithms: NO

-----------------------------------------------------------------------
COMPARATOR FREEZE
-----------------------------------------------------------------------

Gossip:
  fanout                : 3 (runner default; Exp08 has no fanout key)
  adaptive              : NO
  AHBN controller used  : NO
  present in Exp08      : NO
  status                : FAIL

Structured:
  cluster rule          : sorted node IDs assigned round-robin modulo 4
  CH rule               : lowest node ID in each cluster
  forwarding            : member->CH; CH->all local members + adjacent CH gateways
  status                : PASS

DC-SoC:
  DBSCAN eps            : 2.0 (fallback; section absent in Exp08)
  DBSCAN min_samples    : 3 (fallback; section absent in Exp08)
  CH rule               : highest physical degree; tie -> lowest node ID
  forwarding            : intra-cluster physical-neighbour fanout 3; CH gateway reserve 1
  runtime AHBN control  : NO
  Exp08-specific tuning : NO
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
  emergency override    : NO (none present in canonical controller/strategy)
  Exp08-specific tuning : NO
  status                : PASS

-----------------------------------------------------------------------
CONFIGURATION CONSUMPTION
-----------------------------------------------------------------------
Overload config consumed          : PASS
Timing config consumed            : PASS
Workload config consumed          : PASS
Seed config consumed              : PASS
AHBN frozen config consumed       : PASS
DC-SoC frozen config consumed     : PASS

Unused / overridden fields:
  Exp08 has no dcsoc section; frozen values are reached only through hard-coded
  run_batch fallbacks. Gossip is absent from strategies. ch_overload_factor
  is consumed, but has no effect for Gossip because no CH is constructed.

-----------------------------------------------------------------------
CONTROLLED VARIABLES
-----------------------------------------------------------------------
Topology fixed per seed           : PASS
Same workload                     : PASS (for configured strategies)
Same timing model                 : PASS
Same seed pairing                 : PASS (for configured strategies)
Algorithm parameters frozen       : PASS
Only overload level externally varies: PASS

-----------------------------------------------------------------------
SCIENTIFIC FREEZE CHECK
-----------------------------------------------------------------------
Algorithm-specific tuning detected:
  NO
Unresolved configuration ambiguity:
  YES
Scientific-design modification required:
  YES

Discrepancies:
  1. Comparator set required=[gossip, cluster, dcsoc, ahbn], configured=['cluster', 'dcsoc', 'ahbn'].
  2. Overload targets are algorithm-specific: Gossip none; Structured/AHBN
     static heads; DC-SoC density-cluster heads.
  3. Frozen DC-SoC parameters are not explicit in Exp08 and rely on fallbacks.
  4. The requested emergency failure/bottleneck override is not present in
     the canonical AHBN controller/strategy; adding one would change design.

Smallest likely cause:
  Exp08 predates the four-comparator Stage 4 freeze and retained its original
  algorithm-specific CH-overload semantics and implicit DC-SoC defaults.

Issue classification:
  stale configuration; overload-target fairness; scientific-design ambiguity

========================================================================
E0 RESULT: FAIL
========================================================================

STOPPED BEFORE E1.

Please review the terminal output before any correction is applied.
```

### Result

E0 RESULT: FAIL

STOPPED BEFORE E1.

### Notes

No correction was applied. The inspection script is non-experimental: it constructs effective strategy state through the production `run_single` path but suppresses event-queue execution, so it does not generate comparative paper results. The frozen AHBN controller and all baseline implementations/configurations remain unchanged.

"""Configuration-only gate for the final Stage 4 Exp08--Exp12 rerun."""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = (
    "exp08_ch_bottleneck.yaml",
    "exp09_dense_topology.yaml",
    "exp10_failure.yaml",
    "exp11_churn.yaml",
    "exp12_mixed_resources.yaml",
)
EXPECTED_STRATEGIES = ["gossip", "cluster", "dcsoc", "ahbn"]
EXPECTED_DCSOC = {
    "eps": 2.0,
    "min_samples": 3,
    "fanout": 3,
    "inter_fanout": 1,
}


def main() -> int:
    runner_text = (ROOT / "run_batch.py").read_text(encoding="utf-8")
    failures: list[str] = []

    for strategy in EXPECTED_STRATEGIES:
        if f'elif strategy_name == "{strategy}"' not in runner_text and not (
            strategy == "gossip" and 'if strategy_name == "gossip"' in runner_text
        ):
            failures.append(f"run_batch.py does not support strategy {strategy!r}")

    for filename in CONFIGS:
        path = ROOT / "configs" / filename
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        label = str(cfg.get("experiment", filename)).upper()
        strategies = cfg.get("strategies")
        dcsoc = cfg.get("dcsoc")
        ahbn = cfg.get("ahbn")

        if strategies != EXPECTED_STRATEGIES:
            failures.append(f"{label}: strategies={strategies!r}")
        if strategies and len(strategies) != len(set(strategies)):
            failures.append(f"{label}: duplicate comparator entry")
        if dcsoc != EXPECTED_DCSOC:
            failures.append(f"{label}: explicit dcsoc block={dcsoc!r}")
        if not isinstance(ahbn, dict):
            failures.append(f"{label}: missing ahbn block")
            continue
        for key, expected in {
            "alpha": 0.3,
            "kappa": 1.0,
            "beta": 1.0,
            "mode_threshold": 0.5,
            "min_fanout": 2,
            "max_fanout": 4,
            "default_fanout": 3,
        }.items():
            if ahbn.get(key) != expected:
                failures.append(f"{label}: ahbn.{key}={ahbn.get(key)!r}")

        # Mirrors both runner fallbacks. An explicit complete block must win over
        # any experiment-level fanout (notably Exp09's Gossip condition of 4).
        resolved = {
            "eps": dcsoc.get("eps", cfg.get("dcsoc_eps", 2.0)) if dcsoc else None,
            "min_samples": dcsoc.get(
                "min_samples", cfg.get("dcsoc_min_samples", 3)
            ) if dcsoc else None,
            "fanout": dcsoc.get("fanout", cfg.get("fanout", 3)) if dcsoc else None,
            "inter_fanout": dcsoc.get(
                "inter_fanout", cfg.get("dcsoc_inter_fanout", 1)
            ) if dcsoc else None,
        }
        if resolved != EXPECTED_DCSOC:
            failures.append(f"{label}: resolved dcsoc={resolved!r}")

        print(
            f"{label}: strategies={strategies}; dcsoc={resolved}; "
            f"ahbn.max_fanout={ahbn.get('max_fanout')}"
        )

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        print("STAGE 4 PRE-RUN COMPARATOR VALIDATION: FAIL")
        return 1

    print("STAGE 4 PRE-RUN COMPARATOR VALIDATION: PASS")
    print("No simulations were run and no result files were written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
                        "load_balance_cv": summary["load_balance_cv"],
                        "strong_forward_share": summary["strong_forward_share"],
                        "medium_forward_share": summary["medium_forward_share"],
                        "weak_forward_share": summary["weak_forward_share"],
                    }
                )

                if "adaptive_trace_rows" in summary:
                    trace_rows.extend(summary["adaptive_trace_rows"])

    return rows, trace_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    experiment = cfg["experiment"]

    if experiment == "exp07":
        rows, trace_rows = exp07(cfg)
        path = save_results_csv(rows, "outputs/csv/exp07_results.csv")
        print(f"Saved {path}")

        if trace_rows:
            trace_path = save_adaptive_trace_csv(
                trace_rows,
                "outputs/csv/exp07_adaptive_trace.csv",
                add_timestamp=True,
            )
            print(f"Saved {trace_path}")

    elif experiment == "exp08":
        rows, trace_rows = exp08(cfg)
        path = save_results_csv(rows, "outputs/csv/exp08_results.csv")
        print(f"Saved {path}")

        if trace_rows:
            trace_path = save_adaptive_trace_csv(
                trace_rows,
                "outputs/csv/exp08_adaptive_trace.csv",
                add_timestamp=True,
            )
            print(f"Saved {trace_path}")

    elif experiment == "exp09":
        rows, trace_rows = exp09(cfg)
        path = save_results_csv(rows, "outputs/csv/exp09_results.csv")
        print(f"Saved {path}")

        if trace_rows:
            trace_path = save_adaptive_trace_csv(
                trace_rows,
                "outputs/csv/exp09_adaptive_trace.csv",
                add_timestamp=True,
            )
            print(f"Saved {trace_path}")

    elif experiment == "exp10":
        import pandas as pd
        from pathlib import Path
        from ahbn.utils import current_timestamp

        rows, trace_rows = exp10(cfg)
        out = Path("outputs/csv")
        out.mkdir(parents=True, exist_ok=True)

        ts = current_timestamp()
        path = out / f"exp10_results_{ts}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"Saved {path}")

        if trace_rows:
            trace_path = save_adaptive_trace_csv(
                trace_rows,
                "outputs/csv/exp10_adaptive_trace.csv",
                add_timestamp=True,
            )
            print(f"Saved {trace_path}")
experiment: exp08
topology_type: ba
num_nodes: 100
ba_m: 3
num_clusters: 4
seed: 42
message_source: 0

runs_per_setting: 20

base_delay: 1.0
jitter: 0.2

ch_overload_factor: [1.0, 1.5, 2.0, 3.0]

use_topology_cache: true

strategies:
  - gossip
  - cluster
  - dcsoc
  - ahbn

# Frozen Stage 3.5 DC-SoC baseline values, made explicit for Exp08.
dcsoc:
  eps: 2.0
  min_samples: 3
  fanout: 3
  inter_fanout: 1

ahbn:
  alpha: 0.3

  d0: 0.5
  l0: 0.5
  u0: 0.5
  c0: 0.5

  w_d: -1.0
  w_l: 1.0
  w_u: -1.0
  w_c: 1.0

  kappa: 1.0
  beta: 1.0

  min_fanout: 2
  max_fanout: 4

  mode_threshold: 0.5
  default_fanout: 3
experiment: exp09
seed: 42
runs_per_setting: 30

num_nodes: 100
topology_type: er
edge_probs: [0.04, 0.06, 0.08, 0.10, 0.12]

fanout: 4
num_clusters: 4

base_delay: 1.0
jitter: 0.2
message_source: 0
use_topology_cache: true

strategies:
  - gossip
  - cluster
  - dcsoc
  - ahbn

# Frozen S2 DC-SoC comparator parameters.  These are explicit because the
# Exp09 global fanout=4 is the fixed Gossip condition, not a DC-SoC override.
dcsoc:
  eps: 2.0
  min_samples: 3
  fanout: 3
  inter_fanout: 1

ahbn:
  alpha: 0.3

  d0: 0.5
  l0: 0.5
  u0: 0.5
  c0: 0.5

  w_d: -1.0
  w_l: 1.0
  w_u: -1.0
  w_c: 1.0

  kappa: 1.0
  beta: 1.0

  min_fanout: 2
  max_fanout: 4

  mode_threshold: 0.5
  default_fanout: 3

```

### Command

```sh
rg -n "eligible active|physical-neighbor obligations|structural children|dissemination correction|artificial fanout|rerun2|rerun 2" docs scripts tests . --glob '!outputs/**' --glob '!docs/stage4_exp08_rerun2.md'
```

### Complete terminal output

```text
rg: tests: No such file or directory (os error 2)
./docs/S2_faithful_dcsoc.md:36:Leaves may send only to their assigned parent. Cores drive downstream propagation. The existing total fixed budget remains three targets per forwarding call; inter-cluster relationships are structural children within that same budget. Duplicate suppression is unchanged.
docs/S2_faithful_dcsoc.md:36:Leaves may send only to their assigned parent. Cores drive downstream propagation. The existing total fixed budget remains three targets per forwarding call; inter-cluster relationships are structural children within that same budget. Duplicate suppression is unchanged.
./scripts/validate_dcsoc_s8.py:116:        "FAIL: baseline target is not an eligible active same-cluster physical neighbour."
scripts/validate_dcsoc_s8.py:116:        "FAIL: baseline target is not an eligible active same-cluster physical neighbour."
### Prior command result

FAIL — inspection command referenced absent `tests/` and its logging wrapper assigned zsh reserved variable `status`. User approved correction on 2026-08-21 18:40:48 +08. No implementation or experiment files were changed.

### Corrected command

```sh
rg -n "eligible active|physical-neighbor obligations|structural children|dissemination correction|artificial fanout|rerun2|rerun 2" docs scripts . --glob '!outputs/**' --glob '!docs/stage4_exp08_rerun2.md'
```

### Complete terminal output

```text
scripts/validate_dcsoc_s8.py:116:        "FAIL: baseline target is not an eligible active same-cluster physical neighbour."
docs/S2_faithful_dcsoc.md:36:Leaves may send only to their assigned parent. Cores drive downstream propagation. The existing total fixed budget remains three targets per forwarding call; inter-cluster relationships are structural children within that same budget. Duplicate suppression is unchanged.
./scripts/validate_dcsoc_s8.py:116:        "FAIL: baseline target is not an eligible active same-cluster physical neighbour."
./docs/S2_faithful_dcsoc.md:36:Leaves may send only to their assigned parent. Cores drive downstream propagation. The existing total fixed budget remains three targets per forwarding call; inter-cluster relationships are structural children within that same budget. Duplicate suppression is unchanged.

```

Corrected inspection result: PASS

## Stage 2 — Narrow dissemination correction

Files changed: Gossip supports an explicit uncapped mode; DC-SoC supports explicit fulfillment of all active structural children; runners enable these modes only for Exp08/Exp09; one narrow behavioral validator was added. No configuration values were changed.

### Command

```sh
git diff -- ahbn/strategies/gossip.py ahbn/strategies/dcsoc.py run_batch.py run_one.py scripts/validate_exp08_exp09_dissemination.py && git status --short
```

### Complete terminal output

```text
diff --git a/v0.6/ahbn/strategies/dcsoc.py b/v0.6/ahbn/strategies/dcsoc.py
index b0eaf1e..e246ba3 100644
--- a/v0.6/ahbn/strategies/dcsoc.py
+++ b/v0.6/ahbn/strategies/dcsoc.py
@@ -42,6 +42,7 @@ class DCSOCStrategy(ForwardingStrategy):
         self,
         fanout: int = 3,
         inter_fanout: int = 1,
+        fulfill_all_structural_children: bool = False,
     ) -> None:
 
         if fanout < 1:
@@ -62,6 +63,10 @@ class DCSOCStrategy(ForwardingStrategy):
             inter_fanout
         )
 
+        self.fulfill_all_structural_children = bool(
+            fulfill_all_structural_children
+        )
+
     # --------------------------------------------------------
     # Utility
     # --------------------------------------------------------
@@ -131,6 +136,8 @@ class DCSOCStrategy(ForwardingStrategy):
             if child_id in simulator.nodes and simulator.nodes[child_id].is_active
         ]
         if structural_children:
+            if self.fulfill_all_structural_children:
+                return structural_children
             return structural_children[: self.fanout]
 
         # ----------------------------------------------------
diff --git a/v0.6/ahbn/strategies/gossip.py b/v0.6/ahbn/strategies/gossip.py
index c592181..b6391bd 100644
--- a/v0.6/ahbn/strategies/gossip.py
+++ b/v0.6/ahbn/strategies/gossip.py
@@ -19,8 +19,8 @@ class GossipStrategy(ForwardingStrategy):
     AHBN may update `fanout` before calling this strategy.
     """
 
-    def __init__(self, fanout: int = 3) -> None:
-        if fanout < 1:
+    def __init__(self, fanout: int | None = 3) -> None:
+        if fanout is not None and fanout < 1:
             raise ValueError("fanout must be >= 1")
 
         self.fanout = fanout
@@ -46,6 +46,9 @@ class GossipStrategy(ForwardingStrategy):
         if not candidates:
             return []
 
+        if self.fanout is None:
+            return candidates
+
         k = min(
             int(self.fanout),
             len(candidates),
@@ -54,4 +57,4 @@ class GossipStrategy(ForwardingStrategy):
         return simulator.rng.sample(
             candidates,
             k,
-        )
\ No newline at end of file
+        )
diff --git a/v0.6/run_batch.py b/v0.6/run_batch.py
index 79ef362..68d6aa2 100644
--- a/v0.6/run_batch.py
+++ b/v0.6/run_batch.py
@@ -89,6 +89,7 @@ def run_single(
     nodes = build_nodes_from_graph(graph)
 
     experiment_name = cfg.get("experiment", "")
+    fulfill_all_obligations = experiment_name in {"exp08", "exp09"}
     if experiment_name == "exp12":
         assign_mixed_resources(nodes, cfg, seed=seed, scenario_name=resource_scenario)
 
@@ -96,7 +97,13 @@ def run_single(
     controller = None
 
     if strategy_name == "gossip":
-        strategy = GossipStrategy(fanout=fanout if fanout is not None else 3)
+        strategy = GossipStrategy(
+            fanout=(
+                None
+                if fulfill_all_obligations
+                else (fanout if fanout is not None else 3)
+            )
+        )
 
     elif strategy_name == "cluster":
         cluster_manager = assign_static_clusters(
@@ -170,6 +177,7 @@ def run_single(
                     1,
                 )
             ),
+            fulfill_all_structural_children=fulfill_all_obligations,
         )
 
     else:
diff --git a/v0.6/run_one.py b/v0.6/run_one.py
index 05490c0..b0e3902 100644
--- a/v0.6/run_one.py
+++ b/v0.6/run_one.py
@@ -76,10 +76,13 @@ def build_simulation_from_config(cfg: dict, strategy_name: str):
     cluster_manager = None
     controller = None
     ch_overload_factor = cfg.get("ch_overload_factor", 1.0)
+    fulfill_all_obligations = cfg.get("experiment", "") in {"exp08", "exp09"}
 
     if strategy_name == "gossip":
         fanout = cfg.get("fanout", 3)
-        strategy = GossipStrategy(fanout=fanout)
+        strategy = GossipStrategy(
+            fanout=None if fulfill_all_obligations else fanout
+        )
 
     elif strategy_name == "cluster":
         num_clusters = cfg.get("num_clusters", 4)
@@ -151,6 +154,7 @@ def build_simulation_from_config(cfg: dict, strategy_name: str):
                     1,
                 )
             ),
+            fulfill_all_structural_children=fulfill_all_obligations,
         )
 
     else:
@@ -199,4 +203,4 @@ def main() -> None:
 
 
 if __name__ == "__main__":
-    main()
\ No newline at end of file
+    main()
 M ahbn/strategies/dcsoc.py
 M ahbn/strategies/gossip.py
 M run_batch.py
 M run_one.py
?? docs/stage4_exp08_rerun2.md
?? scripts/validate_exp08_exp09_dissemination.py

```

Stage 2 inspection result: PASS

## Stage 3 — Focused unit/behavioral validations

### Command

```sh
/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python -m py_compile ahbn/strategies/gossip.py ahbn/strategies/dcsoc.py run_batch.py run_one.py scripts/validate_exp08_exp09_dissemination.py && /Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python scripts/validate_exp08_exp09_dissemination.py
```

### Complete terminal output

```text
Traceback (most recent call last):
  File "/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/scripts/validate_exp08_exp09_dissemination.py", line 7, in <module>
    from ahbn.message import Message
ModuleNotFoundError: No module named 'ahbn'

```

Focused dissemination gate: FAIL

### Approved import-path correction

Added only the standard project-root bootstrap to
`scripts/validate_exp08_exp09_dissemination.py` before the `ahbn` imports.

### Focused gate rerun command

```sh
/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python -m py_compile ahbn/strategies/gossip.py ahbn/strategies/dcsoc.py run_batch.py run_one.py scripts/validate_exp08_exp09_dissemination.py && /Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python scripts/validate_exp08_exp09_dissemination.py
```

### Complete terminal output

```text
[Errno 1] Operation not permitted: 'ahbn/strategies/__pycache__/gossip.cpython-314.pyc.4390307472'
```

Focused dissemination gate rerun: FAIL — execution stopped before behavioral
validation because the managed environment denied bytecode-cache creation in
the external project directory. No official experiment was started and no
output data was generated.

### Approved writable bytecode-cache retry

The preceding retry failed only because Python attempted to create compiled
bytecode under the externally located project directory. The focused gate was
therefore rerun without code changes, with `PYTHONPYCACHEPREFIX` redirecting
bytecode-cache files to the writable temporary directory
`/private/tmp/ahbn_stage4_pycache`.

### Exact command

```sh
PYTHONPYCACHEPREFIX=/private/tmp/ahbn_stage4_pycache /Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python -m py_compile ahbn/strategies/gossip.py ahbn/strategies/dcsoc.py run_batch.py run_one.py scripts/validate_exp08_exp09_dissemination.py && PYTHONPYCACHEPREFIX=/private/tmp/ahbn_stage4_pycache /Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python scripts/validate_exp08_exp09_dissemination.py
```

### Complete terminal output

```text
gossip_all_active_physical_neighbors: PASS
gossip_default_cap_preserved: PASS
dcsoc_all_active_structural_children: PASS
dcsoc_default_cap_preserved: PASS
EXP08/EXP09 DISSEMINATION VALIDATION: PASS
```

Focused dissemination gate final status: PASS
### Approved diagnostic after failed bootstrap patch

The bootstrap patch failed context verification and made no changes. User approved inspecting the first 30 lines.

### Command

```sh
sed -n '1,30p' scripts/validate_exp08_exp09_dissemination.py
```

### Complete terminal output

```text
"""Validate the narrowly scoped Exp08/Exp09 dissemination correction."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ahbn.message import Message
from ahbn.node import Node
from ahbn.strategies.dcsoc import DCSOCStrategy
from ahbn.strategies.gossip import GossipStrategy


def active_node(node_id: int) -> Node:
    node = Node(node_id=node_id)
    node.is_active = True
    return node


def main() -> int:
    nodes = {node_id: active_node(node_id) for node_id in range(7)}
    source = nodes[0]
    source.neighbors = [1, 2, 3, 4]
    nodes[4].is_active = False
    simulator = SimpleNamespace(nodes=nodes, rng=__import__("random").Random(42))
    message = Message(message_id="m1", source_id=0, created_at=0.0)

```

Validator import-block inspection: PASS

### Corrected focused-gate rerun

### Command

```sh
/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python -m py_compile ahbn/strategies/gossip.py ahbn/strategies/dcsoc.py run_batch.py run_one.py scripts/validate_exp08_exp09_dissemination.py && /Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python scripts/validate_exp08_exp09_dissemination.py
```

### Complete terminal output

```text
gossip_all_active_physical_neighbors: PASS
gossip_default_cap_preserved: PASS
dcsoc_all_active_structural_children: PASS
dcsoc_default_cap_preserved: PASS
EXP08/EXP09 DISSEMINATION VALIDATION: PASS

```

Corrected focused dissemination gate: PASS

## Stage 4 — Exp07 regression gate

### Command

```sh
/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python scripts/validate_stage4_exp07_execution.py
```

### Complete terminal output

```text
Traceback (most recent call last):
  File "/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/scripts/validate_stage4_exp07_execution.py", line 5, in <module>
    from ahbn.config import load_yaml_config
ModuleNotFoundError: No module named 'ahbn'

```

Exp07 regression gate: FAIL

### Approved Exp07 validator bootstrap correction

Added only the standard project-root import bootstrap to `scripts/validate_stage4_exp07_execution.py`.

### Command

```sh
/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python scripts/validate_stage4_exp07_execution.py
```

### Complete terminal output

```text
========================================================================
STAGE 4 — EXP07 EXECUTION VALIDATION
========================================================================
Gossip fixed-fanout sweep: [2, 3, 4, 5, 6]
Gossip scheduled runs    : 100
AHBN scheduled runs      : 20
AHBN receives sweep value: NO
AHBN min_fanout          : 2
AHBN max_fanout          : 4
AHBN default_fanout      : 3
AHBN result fanout       : None
Expected total runs      : 120
========================================================================
EXP07 EXECUTION VALIDATION: PASS
========================================================================

```

Corrected Exp07 regression gate: PASS

## Stage 5 — One-seed Exp08/Exp09 smoke tests

Effective smoke configurations: production YAML unchanged except in-memory `runs_per_setting=1`; seed 42; every configured scenario; all four comparators. No CSV is written.

### Command

```sh
/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python -m py_compile scripts/validate_exp08_exp09_smoke.py && /Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python scripts/validate_exp08_exp09_smoke.py
```

### Complete terminal output

```text
Traceback (most recent call last):
  File "/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/scripts/validate_exp08_exp09_smoke.py", line 64, in <module>
    raise SystemExit(main())
                     ~~~~^^
  File "/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/scripts/validate_exp08_exp09_smoke.py", line 52, in main
    exp08_pass = validate(
        "EXP08", "configs/exp08_ch_bottleneck.yaml", exp08, 16
    )
  File "/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/scripts/validate_exp08_exp09_smoke.py", line 43, in validate
    and {row["strategy"] for row in traces} == {"ahbn"},
         ~~~^^^^^^^^^^^^
TypeError: 'AdaptiveTraceRow' object is not subscriptable

```

One-seed Exp08/Exp09 smoke gate: FAIL

### Approved smoke-validator attribute correction

Changed only `row["strategy"]` to `row.strategy` for `AdaptiveTraceRow`.

### Command

```sh
/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python -m py_compile scripts/validate_exp08_exp09_smoke.py && /Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python scripts/validate_exp08_exp09_smoke.py
```

### Complete terminal output

```text
EXP08: rows=16 strategies={'ahbn': 4, 'cluster': 4, 'dcsoc': 4, 'gossip': 4}
EXP08 row_count: PASS
EXP08 all_comparators: PASS
EXP08 one_seed: PASS
EXP08 unique_identities: PASS
EXP08 finite_metrics: PASS
EXP08 delivery_range: PASS
EXP08 ahbn_only_traces: PASS
EXP09: rows=20 strategies={'ahbn': 5, 'cluster': 5, 'dcsoc': 5, 'gossip': 5}
EXP09 row_count: PASS
EXP09 all_comparators: PASS
EXP09 one_seed: PASS
EXP09 unique_identities: PASS
EXP09 finite_metrics: PASS
EXP09 delivery_range: PASS
EXP09 ahbn_only_traces: PASS
EXP08/EXP09 ONE-SEED SMOKE GATE: PASS

```

Corrected one-seed Exp08/Exp09 smoke gate: PASS

## Stage 6 — Final pre-run comparator/configuration freeze gate

### Command

```sh
/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python scripts/validate_stage4_prerun_comparators.py
```

### Complete terminal output

```text
EXP08: strategies=['gossip', 'cluster', 'dcsoc', 'ahbn']; dcsoc={'eps': 2.0, 'min_samples': 3, 'fanout': 3, 'inter_fanout': 1}; ahbn.max_fanout=4
EXP09: strategies=['gossip', 'cluster', 'dcsoc', 'ahbn']; dcsoc={'eps': 2.0, 'min_samples': 3, 'fanout': 3, 'inter_fanout': 1}; ahbn.max_fanout=4
EXP10: strategies=['gossip', 'cluster', 'dcsoc', 'ahbn']; dcsoc={'eps': 2.0, 'min_samples': 3, 'fanout': 3, 'inter_fanout': 1}; ahbn.max_fanout=4
EXP11: strategies=['gossip', 'cluster', 'dcsoc', 'ahbn']; dcsoc={'eps': 2.0, 'min_samples': 3, 'fanout': 3, 'inter_fanout': 1}; ahbn.max_fanout=4
EXP12: strategies=['gossip', 'cluster', 'dcsoc', 'ahbn']; dcsoc={'eps': 2.0, 'min_samples': 3, 'fanout': 3, 'inter_fanout': 1}; ahbn.max_fanout=4
STAGE 4 PRE-RUN COMPARATOR VALIDATION: PASS
No simulations were run and no result files were written.

```

Comparator/configuration freeze gate: PASS

## Stage 7 — Official Exp08 rerun

Effective configuration: `configs/exp08_ch_bottleneck.yaml`; 4 strategies × 4 overload factors × 20 seeds = 320 expected rows; seeds 42–61.

### Command

```sh
/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python run_batch.py --config configs/exp08_ch_bottleneck.yaml
```

### Complete terminal output

```text
Saved outputs/csv/exp08_results_20260821_185247.csv
Saved outputs/csv/exp08_adaptive_trace_20260821_185247.csv

```

Official Exp08 execution: PASS

## Stage 8 — Official Exp09 rerun

Effective configuration: `configs/exp09_dense_topology.yaml`; 4 strategies × 5 edge probabilities × 30 seeds = 600 expected rows; seeds 42–71.

### Command

```sh
/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python run_batch.py --config configs/exp09_dense_topology.yaml
```

### Complete terminal output

```text
Saved outputs/csv/exp09_results_20260821_185307.csv
Saved outputs/csv/exp09_adaptive_trace_20260821_185307.csv

```

Official Exp09 execution: PASS

## Stage 9 — Official-output validation

### Command

```sh
head -n 2 outputs/csv/exp08_results_20260821_185247.csv && head -n 2 outputs/csv/exp08_adaptive_trace_20260821_185247.csv && head -n 2 outputs/csv/exp09_results_20260821_185307.csv && head -n 2 outputs/csv/exp09_adaptive_trace_20260821_185307.csv && shasum -a 256 outputs/csv/exp08_results_20260821_185247.csv outputs/csv/exp08_adaptive_trace_20260821_185247.csv outputs/csv/exp09_results_20260821_185307.csv outputs/csv/exp09_adaptive_trace_20260821_185307.csv
```

### Complete terminal output

```text
experiment,strategy,seed,num_nodes,topology_type,topology_param,fanout,num_clusters,ch_overload_factor,delivery_ratio,propagation_delay,duplicates,total_forwards
exp08,gossip,42,100,ba,3,,4,1.0,1.0,3.238671232616496,483,582
experiment,strategy,seed,scenario_tag,time,node_id,message_id,event_type,duplicate_obs,latency_obs,utilization_obs,churn_obs,d_hat,l_hat,u_hat,c_hat,score,weight,mode,fanout,mode_switched,fanout_changed,duplicate_ratio_raw,resource_class,capacity_score,processing_delay,received_new,received_duplicate,forwarded
exp08,ahbn,42,ch_overload_factor=1.0,0.0,0,m1,new_receive,0.0,0.0,0.0,,0.0,0.0,0.0,0.0,0.0,0.5,gossip,3,False,False,0.0,medium,1.0,0.0,1,0,0
experiment,strategy,seed,num_nodes,topology_type,topology_param,fanout,num_clusters,ch_overload_factor,delivery_ratio,propagation_delay,duplicates,total_forwards
exp09,gossip,42,100,er,0.04,4.0,4,,1.0,5.509704822701993,342,440
experiment,strategy,seed,scenario_tag,time,node_id,message_id,event_type,duplicate_obs,latency_obs,utilization_obs,churn_obs,d_hat,l_hat,u_hat,c_hat,score,weight,mode,fanout,mode_switched,fanout_changed,duplicate_ratio_raw,resource_class,capacity_score,processing_delay,received_new,received_duplicate,forwarded
exp09,ahbn,42,edge_prob=0.04,0.0,0,m1,new_receive,0.0,0.0,0.0,,0.0,0.0,0.0,0.0,0.0,0.5,gossip,3,False,False,0.0,medium,1.0,0.0,1,0,0
31042392a1441cee503da63beb283a5aeaf4acee231b507ae3ba23a097573135  outputs/csv/exp08_results_20260821_185247.csv
2e7ab084cf1abe8bcdde28dfe4806940055146eb54011709535a4e972ccb3362  outputs/csv/exp08_adaptive_trace_20260821_185247.csv
625a6cae3b13634108ea05eba363d295417e4edd1c09c6d9e9d463aeb723e6e1  outputs/csv/exp09_results_20260821_185307.csv
ddbd039d986ff9270d6a412eb31382971052afdba04d7f734a5c0b99e45950b3  outputs/csv/exp09_adaptive_trace_20260821_185307.csv

```

Schema/hash inspection: PASS

## Approved Exp09 metadata correction

The first Exp09 rerun (`exp09_results_20260821_185307.csv`) is preserved but invalidated for final use because its `fanout` metadata incorrectly reported the global legacy value. Exp08 remains eligible for validation.

### Command

```sh
git diff -- run_batch.py scripts/validate_exp08_exp09_smoke.py && /Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python -m py_compile run_batch.py scripts/validate_exp08_exp09_smoke.py && /Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python scripts/validate_exp08_exp09_dissemination.py && /Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python scripts/validate_stage4_exp07_execution.py && /Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python scripts/validate_exp08_exp09_smoke.py && /Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python scripts/validate_stage4_prerun_comparators.py
```

### Complete terminal output

```text
diff --git a/v0.6/run_batch.py b/v0.6/run_batch.py
index 79ef362..2b90353 100644
--- a/v0.6/run_batch.py
+++ b/v0.6/run_batch.py
@@ -89,6 +89,7 @@ def run_single(
     nodes = build_nodes_from_graph(graph)
 
     experiment_name = cfg.get("experiment", "")
+    fulfill_all_obligations = experiment_name in {"exp08", "exp09"}
     if experiment_name == "exp12":
         assign_mixed_resources(nodes, cfg, seed=seed, scenario_name=resource_scenario)
 
@@ -96,7 +97,13 @@ def run_single(
     controller = None
 
     if strategy_name == "gossip":
-        strategy = GossipStrategy(fanout=fanout if fanout is not None else 3)
+        strategy = GossipStrategy(
+            fanout=(
+                None
+                if fulfill_all_obligations
+                else (fanout if fanout is not None else 3)
+            )
+        )
 
     elif strategy_name == "cluster":
         cluster_manager = assign_static_clusters(
@@ -170,6 +177,7 @@ def run_single(
                     1,
                 )
             ),
+            fulfill_all_structural_children=fulfill_all_obligations,
         )
 
     else:
@@ -451,7 +459,7 @@ def exp09(cfg: dict) -> tuple[list[ResultRow], list]:
                         num_nodes=num_nodes,
                         topology_type="er",
                         topology_param=edge_prob,
-                        fanout=fanout if strategy_name != "cluster" else None,
+                        fanout=None,
                         num_clusters=num_clusters,
                         ch_overload_factor=None,
                         delivery_ratio=summary["delivery_ratio"],
gossip_all_active_physical_neighbors: PASS
gossip_default_cap_preserved: PASS
dcsoc_all_active_structural_children: PASS
dcsoc_default_cap_preserved: PASS
EXP08/EXP09 DISSEMINATION VALIDATION: PASS
========================================================================
STAGE 4 — EXP07 EXECUTION VALIDATION
========================================================================
Gossip fixed-fanout sweep: [2, 3, 4, 5, 6]
Gossip scheduled runs    : 100
AHBN scheduled runs      : 20
AHBN receives sweep value: NO
AHBN min_fanout          : 2
AHBN max_fanout          : 4
AHBN default_fanout      : 3
AHBN result fanout       : None
Expected total runs      : 120
========================================================================
EXP07 EXECUTION VALIDATION: PASS
========================================================================
EXP08: rows=16 strategies={'ahbn': 4, 'cluster': 4, 'dcsoc': 4, 'gossip': 4}
EXP08 row_count: PASS
EXP08 all_comparators: PASS
EXP08 one_seed: PASS
EXP08 unique_identities: PASS
EXP08 finite_metrics: PASS
EXP08 delivery_range: PASS
EXP08 fanout_metadata_unset: PASS
EXP08 ahbn_only_traces: PASS
EXP09: rows=20 strategies={'ahbn': 5, 'cluster': 5, 'dcsoc': 5, 'gossip': 5}
EXP09 row_count: PASS
EXP09 all_comparators: PASS
EXP09 one_seed: PASS
EXP09 unique_identities: PASS
EXP09 finite_metrics: PASS
EXP09 delivery_range: PASS
EXP09 fanout_metadata_unset: PASS
EXP09 ahbn_only_traces: PASS
EXP08/EXP09 ONE-SEED SMOKE GATE: PASS
EXP08: strategies=['gossip', 'cluster', 'dcsoc', 'ahbn']; dcsoc={'eps': 2.0, 'min_samples': 3, 'fanout': 3, 'inter_fanout': 1}; ahbn.max_fanout=4
EXP09: strategies=['gossip', 'cluster', 'dcsoc', 'ahbn']; dcsoc={'eps': 2.0, 'min_samples': 3, 'fanout': 3, 'inter_fanout': 1}; ahbn.max_fanout=4
EXP10: strategies=['gossip', 'cluster', 'dcsoc', 'ahbn']; dcsoc={'eps': 2.0, 'min_samples': 3, 'fanout': 3, 'inter_fanout': 1}; ahbn.max_fanout=4
EXP11: strategies=['gossip', 'cluster', 'dcsoc', 'ahbn']; dcsoc={'eps': 2.0, 'min_samples': 3, 'fanout': 3, 'inter_fanout': 1}; ahbn.max_fanout=4
EXP12: strategies=['gossip', 'cluster', 'dcsoc', 'ahbn']; dcsoc={'eps': 2.0, 'min_samples': 3, 'fanout': 3, 'inter_fanout': 1}; ahbn.max_fanout=4
STAGE 4 PRE-RUN COMPARATOR VALIDATION: PASS
No simulations were run and no result files were written.

```

Post-correction pre-run gates: PASS

## Stage 10 — Approved Exp09-only replacement rerun

### Command

```sh
/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python run_batch.py --config configs/exp09_dense_topology.yaml
```

### Complete terminal output

```text
Saved outputs/csv/exp09_results_20260821_185523.csv
Saved outputs/csv/exp09_adaptive_trace_20260821_185524.csv

```

Official Exp09 replacement execution: PASS

## Stage 11 — Final official-output validation

### Command

```sh
/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python -m py_compile scripts/validate_exp08_exp09_official.py && /Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python scripts/validate_exp08_exp09_official.py --exp08-results outputs/csv/exp08_results_20260821_185247.csv --exp08-trace outputs/csv/exp08_adaptive_trace_20260821_185247.csv --exp09-results outputs/csv/exp09_results_20260821_185523.csv --exp09-trace outputs/csv/exp09_adaptive_trace_20260821_185524.csv && shasum -a 256 outputs/csv/exp08_results_20260821_185247.csv outputs/csv/exp08_adaptive_trace_20260821_185247.csv outputs/csv/exp09_results_20260821_185523.csv outputs/csv/exp09_adaptive_trace_20260821_185524.csv
```

### Complete terminal output

```text
EXP08 RESULTS: rows=320 cells=16
EXP08 STRATEGY_COUNTS: {'ahbn': 80, 'cluster': 80, 'dcsoc': 80, 'gossip': 80}
EXP08 CONDITION_COUNTS: {'1.0': 80, '1.5': 80, '2.0': 80, '3.0': 80}
EXP08 results experiment: PASS
EXP08 results row_count: PASS
EXP08 results strategies: PASS
EXP08 results conditions: PASS
EXP08 results unique_identities: PASS
EXP08 results complete_cells: PASS
EXP08 results seed_coverage: PASS
EXP08 results finite_metrics: PASS
EXP08 results delivery_range: PASS
EXP08 results nonnegative_counts: PASS
EXP08 results fanout_metadata_unset: PASS
EXP08 TRACE: rows=19985
EXP08 trace nonempty: PASS
EXP08 trace experiment: PASS
EXP08 trace ahbn_only: PASS
EXP08 trace scenarios: PASS
EXP08 trace seeds: PASS
EXP08 trace all_run_cells: PASS
EXP08 trace finite_controller_fields: PASS
EXP08 trace adaptive_fanout_bounds: PASS
EXP09 RESULTS: rows=600 cells=20
EXP09 STRATEGY_COUNTS: {'ahbn': 150, 'cluster': 150, 'dcsoc': 150, 'gossip': 150}
EXP09 CONDITION_COUNTS: {'0.04': 120, '0.06': 120, '0.08': 120, '0.1': 120, '0.12': 120}
EXP09 results experiment: PASS
EXP09 results row_count: PASS
EXP09 results strategies: PASS
EXP09 results conditions: PASS
EXP09 results unique_identities: PASS
EXP09 results complete_cells: PASS
EXP09 results seed_coverage: PASS
EXP09 results finite_metrics: PASS
EXP09 results delivery_range: PASS
EXP09 results nonnegative_counts: PASS
EXP09 results fanout_metadata_unset: PASS
EXP09 TRACE: rows=41367
EXP09 trace nonempty: PASS
EXP09 trace experiment: PASS
EXP09 trace ahbn_only: PASS
EXP09 trace scenarios: PASS
EXP09 trace seeds: PASS
EXP09 trace all_run_cells: PASS
EXP09 trace finite_controller_fields: PASS
EXP09 trace adaptive_fanout_bounds: PASS
OFFICIAL EXP08/EXP09 DATA-INTEGRITY GATE: PASS
31042392a1441cee503da63beb283a5aeaf4acee231b507ae3ba23a097573135  outputs/csv/exp08_results_20260821_185247.csv
2e7ab084cf1abe8bcdde28dfe4806940055146eb54011709535a4e972ccb3362  outputs/csv/exp08_adaptive_trace_20260821_185247.csv
9885a26154e27509ab940b22bfeb13cffc087149a2e1295750793f9690d078c6  outputs/csv/exp09_results_20260821_185523.csv
ddbd039d986ff9270d6a412eb31382971052afdba04d7f734a5c0b99e45950b3  outputs/csv/exp09_adaptive_trace_20260821_185524.csv

```

Official data-integrity gate: PASS

## Stage 12 — Aggregation/plot entry-point inspection

### Command

```sh
ls -1 scripts | rg 'aggregate|plot' && sed -n '1,120p' scripts/aggregate_exp08_e7.py && sed -n '1,120p' scripts/plot_exp08_e8.py && sed -n '1,330p' scripts/plot_results.py
```

### Complete terminal output

```text
aggregate_exp08_e7.py
plot_exp07_publication.py
plot_exp07_side_by_side.py
plot_exp08_e8.py
plot_results.py
#!/usr/bin/env python3
"""Validate and aggregate the final Stage 4 Exp08 rerun for E5."""
import argparse, hashlib, math
from datetime import datetime
from pathlib import Path
import pandas as pd
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[1]
FINAL=(ROOT/'outputs/csv/exp08_results_20260821_164541.csv').resolve()
METRICS=['delivery_ratio','propagation_delay','duplicates','total_forwards']
NAMES={'gossip':'Gossip','cluster':'Structured','dcsoc':'DC-SoC','ahbn':'AHBN'}
SEEDS=set(range(42,62)); OVERLOADS={1.0,1.5,2.0,3.0}
def req(x,m):
    if not x: raise ValueError(m)
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,default=FINAL)
    ap.add_argument('--timestamp',default=datetime.now().strftime('%Y%m%d_%H%M%S')); a=ap.parse_args()
    source=a.input.resolve(); req(source==FINAL,f'wrong input; required {FINAL}, got {source}')
    req(source.is_file(),f'missing input: {source}'); before=digest(source); df=pd.read_csv(source)
    print('E5 Exp08 final aggregation'); print(f'Input: {source}'); print(f'Input SHA-256: {before}')
    req(len(df)==320,f'expected 320 rows, found {len(df)}'); req(set(df.strategy)==set(NAMES),'comparator mismatch')
    req(set(df.seed)==SEEDS,'seeds are not exactly 42..61')
    df.ch_overload_factor=pd.to_numeric(df.ch_overload_factor,errors='coerce')
    req(df.ch_overload_factor.notna().all() and set(df.ch_overload_factor)==OVERLOADS,'overload mismatch/malformed')
    keys=['strategy','ch_overload_factor','seed']; req(not df.duplicated(keys).any(),'duplicate run identities')
    expected={(s,o,z) for s in NAMES for o in OVERLOADS for z in SEEDS}
    req(set(map(tuple,df[keys].itertuples(index=False,name=None)))==expected,'incomplete/extra run grid')
    for m in METRICS:
        req(m in df,f'missing {m}'); df[m]=pd.to_numeric(df[m],errors='coerce')
        req(df[m].notna().all() and df[m].map(math.isfinite).all(),f'invalid {m}')
    counts=df.groupby(['strategy','ch_overload_factor']).size(); req(len(counts)==16 and (counts==20).all(),'bad cells')
    rows=[]
    for (s,o),g in df.groupby(['strategy','ch_overload_factor'],sort=True):
        r={'comparator':NAMES[s],'strategy':s,'overload_factor':o,'n':len(g),'df':len(g)-1}
        for m in METRICS:
            mean=g[m].mean(); sd=g[m].std(ddof=1); se=sd/math.sqrt(len(g)); ci=t.ppf(.975,len(g)-1)*se
            r.update({f'{m}_mean':mean,f'{m}_sd':sd,f'{m}_se':se,f'{m}_ci95':ci,
                      f'{m}_ci95_low':mean-ci,f'{m}_ci95_high':mean+ci})
        rows.append(r)
    out=pd.DataFrame(rows); req(len(out)==16 and (out.n==20).all(),'aggregate validation failed')
    dest=ROOT/'outputs/csv'/f'exp08_final_summary_{a.timestamp}.csv'; out.to_csv(dest,index=False)
    req(before==digest(source),'raw input changed during aggregation')
    print('Raw rows: 320\nComparators: 4\nOverload factors: 4\nConditions: 16\nRuns per condition: 20')
    print('Seeds: 42..61 per condition\nDuplicate identities: 0\nInvalid required metrics: 0')
    print('95% CI: two-sided Student t; df=19'); print(f'Saved: {dest}'); print('E5 RESULT: PASS')
if __name__=='__main__': main()
#!/usr/bin/env python3
"""Generate four final Exp08 figures from the E5 summary only."""
import argparse, math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
COMPS=['Gossip','Structured','DC-SoC','AHBN']; LEVELS=[1.0,1.5,2.0,3.0]
METRICS={'delivery_ratio':'Delivery ratio','propagation_delay':'Propagation delay (s)',
         'duplicates':'Duplicates','total_forwards':'Total forwards'}
STYLES=[('o','-'),('s','--'),('^','-.'),('D',':')]
def req(x,m):
    if not x: raise ValueError(m)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--summary',type=Path,required=True); ap.add_argument('--timestamp',required=True); a=ap.parse_args()
    source=a.summary.resolve(); req(source.parent==(ROOT/'outputs/csv').resolve() and source.name.startswith('exp08_final_summary_'),'not a final summary')
    df=pd.read_csv(source); required={'comparator','overload_factor','n'}
    for m in METRICS: required|={f'{m}_mean',f'{m}_ci95'}
    req(not(required-set(df)),f'missing columns {sorted(required-set(df))}')
    req(len(df)==16 and set(df.comparator)==set(COMPS) and set(df.overload_factor)==set(LEVELS),'invalid grid')
    req((df.n==20).all() and not df.duplicated(['comparator','overload_factor']).any(),'invalid cells')
    outputs=[]; colors=plt.rcParams['axes.prop_cycle'].by_key()['color'][:4]
    for metric,ylabel in METRICS.items():
        fig,ax=plt.subplots(figsize=(7.2,4.8),constrained_layout=True)
        for comp,color,(marker,line) in zip(COMPS,colors,STYLES):
            r=df[df.comparator==comp].sort_values('overload_factor'); req(len(r)==4,f'bad {comp} row count')
            mean=pd.to_numeric(r[f'{metric}_mean'],errors='coerce'); ci=pd.to_numeric(r[f'{metric}_ci95'],errors='coerce')
            req(mean.notna().all() and ci.notna().all() and mean.map(math.isfinite).all() and ci.map(math.isfinite).all() and (ci>=0).all(),f'invalid {metric}')
            ax.errorbar(r.overload_factor,mean,yerr=ci,label=comp,color=color,marker=marker,linestyle=line,linewidth=1.7,markersize=5.5,capsize=3,elinewidth=1.1)
        ax.set_xlabel('CH overload factor'); ax.set_ylabel(ylabel); ax.set_xticks(LEVELS); ax.grid(True,linestyle=':',linewidth=.7,alpha=.65); ax.legend(frameon=False,ncols=2)
        dest=ROOT/'outputs/figures'/f'exp08_final_{metric}_{a.timestamp}.png'; dest.parent.mkdir(parents=True,exist_ok=True); fig.savefig(dest,dpi=300,bbox_inches='tight'); plt.close(fig)
        req(dest.is_file() and dest.stat().st_size>0,f'missing {dest}'); outputs.append(dest)
    print('E6 Exp08 final plotting'); print(f'Summary input only: {source}'); print('Validation: 16 conditions; n=20; 4 comparators x 4 overloads'); print('Error bars: mean +/- Student-t 95% CI')
    for p in outputs: print(f'Saved: {p}')
    print('E6 RESULT: PASS')
if __name__=='__main__': main()
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import t

from ahbn.utils import ensure_dir, extract_timestamp_from_filename


# -----------------------------
# Helpers
# -----------------------------
def get_timestamp(csv_path: str) -> str:
    ts = extract_timestamp_from_filename(csv_path)
    if ts is None:
        raise ValueError(f"Cannot extract timestamp from {csv_path}")
    return ts


def apply_offset(series: pd.Series, offset: float) -> pd.Series:
    return series.astype(float) + offset


def get_plot_output_path(experiment: str, timestamp: str) -> str:
    return f"outputs/plots/{experiment}_combined_{timestamp}.png"


def get_exp07_3panel_output_path(timestamp: str) -> str:
    return f"outputs/plots/exp07_3panel_{timestamp}.png"


def get_adaptive_plot_output_path(experiment: str, timestamp: str) -> str:
    return f"outputs/plots/{experiment}_adaptive_{timestamp}.png"


def make_time_bins(df: pd.DataFrame, bin_width: float = 0.25) -> pd.DataFrame:
    out = df.copy()
    out["time_bin"] = (out["time"] / bin_width).round().astype(float) * bin_width
    return out


def mode_fraction_by_bin(df: pd.DataFrame) -> pd.DataFrame:
    mode_counts = (
        df.groupby(["time_bin", "mode"])["node_id"]
        .nunique()
        .unstack(fill_value=0)
        .reset_index()
    )

    if "gossip" not in mode_counts.columns:
        mode_counts["gossip"] = 0
    if "cluster" not in mode_counts.columns:
        mode_counts["cluster"] = 0

    total = mode_counts["gossip"] + mode_counts["cluster"]
    total = total.replace(0, 1)

    mode_counts["gossip_frac"] = mode_counts["gossip"] / total
    mode_counts["cluster_frac"] = mode_counts["cluster"] / total
    return mode_counts.sort_values("time_bin")


# -----------------------------
# Exp07
# -----------------------------
def plot_exp07(df: pd.DataFrame, ts: str, use_offset: bool) -> None:
    ensure_dir("outputs/plots")

    required_cols = {
        "experiment",
        "strategy",
        "fanout",
        "delivery_ratio",
        "propagation_delay",
        "duplicates",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    exp_values = set(df["experiment"].dropna().unique())
    if exp_values != {"exp07"} and "exp07" not in exp_values:
        print(
            f"Warning: CSV contains experiment values {sorted(exp_values)}. "
            f"This plotting script is intended for exp07."
        )

    df_compare = df[df["strategy"].isin(["gossip", "ahbn"])].copy()

    gossip = df_compare[df_compare["strategy"] == "gossip"].copy()
    ahbn = df_compare[df_compare["strategy"] == "ahbn"].copy()
    if gossip.empty or ahbn.empty:
        raise ValueError("Exp07 requires both Gossip and AHBN result rows.")
    if ahbn["fanout"].notna().any():
        raise ValueError("Exp07 AHBN result fanout must be blank (adaptive reference).")

    metrics = [
        ("delivery_ratio", "Delivery Ratio"),
        ("propagation_delay", "Propagation Delay"),
        ("duplicates", "Duplicates"),
    ]
    x_ticks = sorted(gossip["fanout"].dropna().unique())
    if x_ticks != [2, 3, 4, 5, 6]:
        raise ValueError(f"Expected Gossip fanouts [2, 3, 4, 5, 6], got {x_ticks}")

    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (metric, label) in zip(axes3, metrics):
        grouped = gossip.groupby("fanout")[metric].agg(["count", "mean", "std"]).reset_index()
        critical = grouped["count"].map(lambda n: t.ppf(0.975, n - 1))
        half_width = critical * grouped["std"] / grouped["count"].pow(0.5)
        ahbn_n = ahbn[metric].count()
        ahbn_mean = ahbn[metric].mean()
        ahbn_half_width = t.ppf(0.975, ahbn_n - 1) * ahbn[metric].std() / ahbn_n**0.5

        ax.errorbar(
            grouped["fanout"], grouped["mean"], yerr=half_width,
            marker="o", capsize=4, linewidth=1.8, label="Gossip (fixed fanout)",
        )
        ax.axhspan(
            ahbn_mean - ahbn_half_width,
            ahbn_mean + ahbn_half_width,
            color="tab:orange", alpha=0.16,
        )
        ax.axhline(
            ahbn_mean, color="tab:orange", linestyle="--", linewidth=1.8,
            label="AHBN (adaptive)",
        )
        ax.set_title(f"{label} vs Gossip Fanout")
        ax.set_xlabel("Fixed Gossip Fanout")
        ax.set_ylabel(label)
        ax.set_xticks(x_ticks)
        ax.grid(True, linestyle=":")
        ax.legend()

    fig3.tight_layout()
    out_3panel = get_exp07_3panel_output_path(ts)
    fig3.savefig(out_3panel, dpi=150, bbox_inches="tight")
    plt.close(fig3)

    print(f"Saved {out_3panel}")


# -----------------------------
# Exp08
# -----------------------------
def plot_exp08(df: pd.DataFrame, ts: str, use_offset: bool) -> None:
    ensure_dir("outputs/plots")

    df = df[df["strategy"].isin(["cluster", "ahbn"])].copy()

    grouped = (
        df.groupby(["strategy", "ch_overload_factor"])
        .agg(
            delay_mean=("propagation_delay", "mean"),
            delivery_mean=("delivery_ratio", "mean"),
        )
        .reset_index()
    )

    strategies = grouped["strategy"].unique()
    offsets = {"cluster": -0.03, "ahbn": 0.03} if use_offset else {s: 0.0 for s in strategies}

    x_ticks = sorted(df["ch_overload_factor"].dropna().unique())

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    for s in strategies:
        part = grouped[grouped["strategy"] == s].sort_values("ch_overload_factor")
        x = apply_offset(part["ch_overload_factor"], offsets[s])
        axes[0].plot(x, part["delay_mean"], marker="o", label=s)

    axes[0].set_xlabel("CH Overload Factor")
    axes[0].set_ylabel("Delay")
    axes[0].set_title("Delay vs CH Overload")
    axes[0].set_xticks(x_ticks)
    axes[0].legend()
    axes[0].grid(True, linestyle=":")

    for s in strategies:
        part = grouped[grouped["strategy"] == s].sort_values("ch_overload_factor")
        x = apply_offset(part["ch_overload_factor"], offsets[s])
        axes[1].plot(x, part["delivery_mean"], marker="o", label=s)

    axes[1].set_xlabel("CH Overload Factor")
    axes[1].set_ylabel("Delivery Ratio")
    axes[1].set_title("Delivery vs CH Overload")
    axes[1].set_xticks(x_ticks)
    axes[1].legend()
    axes[1].grid(True, linestyle=":")

    plt.tight_layout()
    out = get_plot_output_path("exp08", ts)
    plt.savefig(out, bbox_inches="tight")
    plt.close()

    print(f"Saved {out}")


# -----------------------------
# Exp09
# -----------------------------
def detect_xlabel(df: pd.DataFrame) -> str:
    topo = df["topology_type"].iloc[0]
    if topo == "er":
        return "ER Edge Probability"
    elif topo == "ba":
        return "BA Attachment Parameter (m)"
    return "Topology Parameter"


def plot_exp09(df: pd.DataFrame, ts: str, use_offset: bool) -> None:
    ensure_dir("outputs/plots")

    xlabel = detect_xlabel(df)

    grouped = (
        df.groupby(["strategy", "topology_param"])
        .agg(
            delay_mean=("propagation_delay", "mean"),
            dup_mean=("duplicates", "mean"),
        )
        .reset_index()
    )

    strategies = grouped["strategy"].unique()

    if use_offset:
        offsets = {"gossip": -0.002, "cluster": 0.0, "ahbn": 0.002}
    else:
        offsets = {s: 0.0 for s in strategies}

    x_ticks = sorted(df["topology_param"].dropna().unique())

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))

    for s in strategies:
        part = grouped[grouped["strategy"] == s].sort_values("topology_param")
        x = apply_offset(part["topology_param"], offsets.get(s, 0.0))
        axes[0].plot(x, part["dup_mean"], marker="o", label=s)

    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Duplicates")
    axes[0].set_title("Duplicates vs Topology")
    axes[0].set_xticks(x_ticks)
    axes[0].legend()
    axes[0].grid(True, linestyle=":")

    for s in strategies:
        part = grouped[grouped["strategy"] == s].sort_values("topology_param")
        x = apply_offset(part["topology_param"], offsets.get(s, 0.0))
        axes[1].plot(x, part["delay_mean"], marker="o", label=s)

    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("Delay")
    axes[1].set_title("Delay vs Topology")
    axes[1].set_xticks(x_ticks)
    axes[1].legend()
    axes[1].grid(True, linestyle=":")

    plt.tight_layout()
    out = get_plot_output_path("exp09", ts)
    plt.savefig(out, bbox_inches="tight")
    plt.close()

    print(f"Saved {out}")


# -----------------------------
# Exp10
# -----------------------------
def plot_exp10(df: pd.DataFrame, ts: str, use_offset: bool) -> None:
    ensure_dir("outputs/plots")

    grouped = (
        df.groupby(["strategy", "failure_mode"])
        .agg(
            delay_mean=("propagation_delay", "mean"),
            delivery_mean=("delivery_ratio", "mean"),
            dup_mean=("duplicates", "mean"),
            recovery_mean=("recovery_time", "mean"),
        )
        .reset_index()
    )

    strategies = grouped["strategy"].unique()
    failure_modes = list(df["failure_mode"].dropna().unique())

    if use_offset:
        offsets = {"gossip": -0.06, "cluster": 0.0, "ahbn": 0.06}
    else:
        offsets = {s: 0.0 for s in strategies}

    x_pos = list(range(len(failure_modes)))
    x_map = {mode: idx for idx, mode in enumerate(failure_modes)}

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))

    for s in strategies:
        part = grouped[grouped["strategy"] == s].copy()
        part["x"] = part["failure_mode"].map(x_map).astype(float) + offsets.get(s, 0.0)
        part = part.sort_values("x")
        axes[0, 0].plot(part["x"], part["delay_mean"], marker="o", label=s)
    axes[0, 0].set_title("Delay vs Failure Mode")
    axes[0, 0].set_ylabel("Propagation Delay")
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(failure_modes)
    axes[0, 0].grid(True, linestyle=":")
    axes[0, 0].legend()

    for s in strategies:
        part = grouped[grouped["strategy"] == s].copy()
        part["x"] = part["failure_mode"].map(x_map).astype(float) + offsets.get(s, 0.0)
        part = part.sort_values("x")
        axes[0, 1].plot(part["x"], part["delivery_mean"], marker="o", label=s)
    axes[0, 1].set_title("Delivery Ratio vs Failure Mode")
    axes[0, 1].set_ylabel("Delivery Ratio")
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(failure_modes)
    axes[0, 1].grid(True, linestyle=":")
    axes[0, 1].legend()

    for s in strategies:

```

Aggregation/plot inspection: PASS

## Stage 13 — Aggregation and plotting

### Command

```sh
MPLCONFIGDIR=/private/tmp/ahbn-stage4-rerun2-mpl /Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python scripts/aggregate_plot_exp08_exp09_rerun2.py --exp08 outputs/csv/exp08_results_20260821_185247.csv --exp09 outputs/csv/exp09_results_20260821_185523.csv --timestamp 20260821_185800
```

### Complete terminal output

```text
Matplotlib is building the font cache; this may take a moment.
EXP08 input: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/outputs/csv/exp08_results_20260821_185247.csv
EXP08 input SHA-256: 31042392a1441cee503da63beb283a5aeaf4acee231b507ae3ba23a097573135
EXP08 summary: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/outputs/csv/exp08_rerun2_summary_20260821_185800.csv
EXP09 input: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/outputs/csv/exp09_results_20260821_185523.csv
EXP09 input SHA-256: 9885a26154e27509ab940b22bfeb13cffc087149a2e1295750793f9690d078c6
EXP09 summary: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/outputs/csv/exp09_rerun2_summary_20260821_185800.csv
EXP08 plot: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/outputs/figures/exp08_rerun2_delivery_ratio_20260821_185800.png
EXP08 plot: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/outputs/figures/exp08_rerun2_propagation_delay_20260821_185800.png
EXP08 plot: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/outputs/figures/exp08_rerun2_duplicates_20260821_185800.png
EXP08 plot: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/outputs/figures/exp08_rerun2_total_forwards_20260821_185800.png
EXP09 plot: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/outputs/figures/exp09_rerun2_delivery_ratio_20260821_185800.png
EXP09 plot: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/outputs/figures/exp09_rerun2_propagation_delay_20260821_185800.png
EXP09 plot: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/outputs/figures/exp09_rerun2_duplicates_20260821_185800.png
EXP09 plot: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/outputs/figures/exp09_rerun2_total_forwards_20260821_185800.png
EXP08/EXP09 RERUN-2 AGGREGATION AND PLOTTING: PASS

```

Aggregation/plot gate: PASS

## Stage 14 — Final hashes, working tree, and sign-off

### Command

```sh
shasum -a 256 outputs/csv/exp08_results_20260821_185247.csv outputs/csv/exp08_adaptive_trace_20260821_185247.csv outputs/csv/exp09_results_20260821_185523.csv outputs/csv/exp09_adaptive_trace_20260821_185524.csv outputs/csv/exp08_rerun2_summary_20260821_185800.csv outputs/csv/exp09_rerun2_summary_20260821_185800.csv outputs/figures/exp08_rerun2_*_20260821_185800.png outputs/figures/exp09_rerun2_*_20260821_185800.png && git diff --check && git status --short
```

### Complete terminal output

```text
31042392a1441cee503da63beb283a5aeaf4acee231b507ae3ba23a097573135  outputs/csv/exp08_results_20260821_185247.csv
2e7ab084cf1abe8bcdde28dfe4806940055146eb54011709535a4e972ccb3362  outputs/csv/exp08_adaptive_trace_20260821_185247.csv
9885a26154e27509ab940b22bfeb13cffc087149a2e1295750793f9690d078c6  outputs/csv/exp09_results_20260821_185523.csv
ddbd039d986ff9270d6a412eb31382971052afdba04d7f734a5c0b99e45950b3  outputs/csv/exp09_adaptive_trace_20260821_185524.csv
550a9e7112a89b0e438b0dc66baebc64da00da47e3dc20b0f923f7849f7055c0  outputs/csv/exp08_rerun2_summary_20260821_185800.csv
78847d05f32970b0f254ed3c513190c2d512352bf78aa9b849522504515ce076  outputs/csv/exp09_rerun2_summary_20260821_185800.csv
388152fde1b8e883ce5a26cbeccda83ea020cecdae68aa1a250ffd47b8c14535  outputs/figures/exp08_rerun2_delivery_ratio_20260821_185800.png
cbcb3a77fadedc4215356a3e5d8af6ee4fa512f1be3cc47979ce68b1844efdc9  outputs/figures/exp08_rerun2_duplicates_20260821_185800.png
6f0b8eb65c4d9d54913cf0ab3e8e427b28f9e4bd686d519ab711e1fe0fc201b5  outputs/figures/exp08_rerun2_propagation_delay_20260821_185800.png
87c131310cdb97d7b35110377d42c6f74404da42098a7ba78b857fb57aaf54e6  outputs/figures/exp08_rerun2_total_forwards_20260821_185800.png
8f9d27723febd63c3243a3cc00900baf850dfea67973f3c09b79894a48dc747f  outputs/figures/exp09_rerun2_delivery_ratio_20260821_185800.png
d94f421b45547a82c0be11ee3ca2848a4545b1d2f5cdf3ce4b3baa9129541635  outputs/figures/exp09_rerun2_duplicates_20260821_185800.png
3985916de3ab8cec2945ea8e5b17f893309639f51f84433d25a4d66fe671b2ac  outputs/figures/exp09_rerun2_propagation_delay_20260821_185800.png
e1748f3eb28903da422912d462eb88e1a7952b95a0f9293b4e3d41be5979b2e7  outputs/figures/exp09_rerun2_total_forwards_20260821_185800.png
 M ahbn/strategies/dcsoc.py
 M ahbn/strategies/gossip.py
 M run_batch.py
 M run_one.py
 M scripts/validate_stage4_exp07_execution.py
?? docs/stage4_exp08_rerun2.md
?? scripts/aggregate_plot_exp08_exp09_rerun2.py
?? scripts/validate_exp08_exp09_dissemination.py
?? scripts/validate_exp08_exp09_official.py
?? scripts/validate_exp08_exp09_smoke.py

```

Final provenance/working-tree gate: PASS

### Files changed and why

- `ahbn/strategies/gossip.py`: explicit uncapped eligible-active-neighbor mode.
- `ahbn/strategies/dcsoc.py`: explicit fulfillment of all active structural children.
- `run_batch.py`, `run_one.py`: enable those modes only for Exp08/Exp09; Exp09 fanout metadata is unset.
- `scripts/validate_stage4_exp07_execution.py`: approved standalone import bootstrap only.
- `scripts/validate_exp08_exp09_dissemination.py`, `scripts/validate_exp08_exp09_smoke.py`, `scripts/validate_exp08_exp09_official.py`: narrow behavioral, smoke, and official-data gates.
- `scripts/aggregate_plot_exp08_exp09_rerun2.py`: validated rerun-2 aggregation and plots.
- `docs/stage4_exp08_rerun2.md`: complete execution record.

No configuration files were changed. Exp07 and Exp10–Exp12 retain capped defaults; Structured, AHBN, topology, clustering/lifecycle, simulator accounting, seeds, scenarios, and metrics were not modified.

### Official validated artifacts

- Exp08 results: `outputs/csv/exp08_results_20260821_185247.csv` (320 rows; 16 conditions; 20 seeds/cell).
- Exp08 AHBN trace: `outputs/csv/exp08_adaptive_trace_20260821_185247.csv` (19,985 AHBN-only rows).
- Exp09 results: `outputs/csv/exp09_results_20260821_185523.csv` (600 rows; 20 conditions; 30 seeds/cell).
- Exp09 AHBN trace: `outputs/csv/exp09_adaptive_trace_20260821_185524.csv` (41,367 AHBN-only rows).
- Summaries: `outputs/csv/exp08_rerun2_summary_20260821_185800.csv`, `outputs/csv/exp09_rerun2_summary_20260821_185800.csv`.
- Plots: eight `outputs/figures/exp08_rerun2_*_20260821_185800.png` / `exp09_rerun2_*_20260821_185800.png` files listed and hashed above.

Preserved invalidated artifact: `outputs/csv/exp09_results_20260821_185307.csv` and its trace remain untouched but are excluded from validation, aggregation, plotting, and sign-off because the result metadata incorrectly reported legacy fanout.

### Final scientific and data-integrity sign-off

Date/time: 2026-08-21 19:03:54 +08

PASS — All required pre-run, Exp07 regression, one-seed smoke, official matrix, identity, coverage, finite-metric, comparator-isolation, metadata, and AHBN-only adaptive-trace gates passed. Only validated Exp08/Exp09 files were aggregated and plotted. No partial validation is claimed.
