#!/usr/bin/env python3
"""Read-only E0 integrity inspection for the frozen v0.61 Exp08 workflow."""
from __future__ import annotations
import inspect
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ahbn.config import load_yaml_config
from ahbn.control import AHBNController
from ahbn.strategies.ahbn import AHBNStrategy
from ahbn.strategies.cluster import ClusterStrategy
from ahbn.strategies.dcsoc import DCSOCStrategy
from ahbn.strategies.gossip import GossipStrategy
from run_batch import build_ahbn_params, run_single
CONFIG = ROOT / "configs/exp08_ch_bottleneck.yaml"
EXPECTED_STRATEGIES = ["gossip", "cluster", "dcsoc", "ahbn"]
EXPECTED_FACTORS = [1.0, 1.5, 2.0, 3.0]
EXPECTED_SEEDS = list(range(42, 62))

def main() -> int:
    cfg = load_yaml_config(CONFIG)
    checks = {
        "v0.61 local paths": ROOT.name == "v0.61" and CONFIG.parent.parent == ROOT,
        "four comparators": cfg["strategies"] == EXPECTED_STRATEGIES,
        "overload factors": [float(x) for x in cfg["ch_overload_factor"]] == EXPECTED_FACTORS,
        "seeds 42-61": [cfg["seed"] + i for i in range(cfg["runs_per_setting"])] == EXPECTED_SEEDS,
        "no Exp07 fanout sweep": "fanouts" not in cfg and "fanout" not in cfg,
        "DC-SoC config has no caps": set(cfg.get("dcsoc", {})) == {"eps", "min_samples"},
        "AHBN bounds": (cfg["ahbn"]["min_fanout"], cfg["ahbn"]["max_fanout"]) == (2, 4),
    }
    summaries = {}
    for strategy in EXPECTED_STRATEGIES:
        summaries[strategy] = run_single(
            cfg=cfg, strategy_name=strategy, seed=42, topology_type=cfg["topology_type"],
            num_nodes=cfg["num_nodes"], use_topology_cache=cfg["use_topology_cache"],
            base_delay=cfg["base_delay"], jitter=cfg["jitter"],
            message_source=cfg["message_source"], num_clusters=cfg["num_clusters"],
            ch_overload_factor=1.0, ba_m=cfg["ba_m"],
            enable_adaptive_trace=(strategy == "ahbn"), scenario_tag="e0-non-authoritative")
    eligible = summaries["dcsoc"]["dcsoc_eligible_overload_nodes"]
    target = summaries["dcsoc"]["dcsoc_overload_target_id"]
    checks.update({
        "DC-SoC source Master": summaries["dcsoc"]["effective_source_id"] == summaries["dcsoc"]["dcsoc_master_id"],
        "DC-SoC eligible target": bool(eligible) and target in eligible,
        "DC-SoC role Master/Core": summaries["dcsoc"]["dcsoc_selected_overload_role"] in {"Master", "Core"},
        "DC-SoC obligations uncapped": summaries["dcsoc"]["max_structural_obligations"] > 3,
        "Gossip normal unbounded": GossipStrategy().fanout is None,
        "Structured uncapped": ClusterStrategy().fanout is None,
        "DC-SoC no fanout API": "fanout" not in inspect.signature(DCSOCStrategy).parameters,
    })
    params = build_ahbn_params(cfg)
    checks["canonical AHBN"] = isinstance(AHBNController(params), AHBNController) and isinstance(AHBNStrategy(), AHBNStrategy) and (params.min_fanout, params.max_fanout) == (2, 4)
    print("E0 — v0.61 Exp08 integrity inspection")
    for name, passed in checks.items(): print(f"{'PASS' if passed else 'FAIL'}  {name}")
    overall = all(checks.values())
    print(f"E0 RESULT: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1
if __name__ == "__main__": raise SystemExit(main())
