"""One-seed, all-condition smoke gate for Exp08 and Exp09."""

from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ahbn.config import load_yaml_config
from ahbn.strategies.dcsoc import DCSOCStrategy
from run_batch import exp08, exp09


EXPECTED_STRATEGIES = {"gossip", "cluster", "dcsoc", "ahbn"}
METRICS = ("delivery_ratio", "propagation_delay", "duplicates", "total_forwards")


def audit_exp08_dcsoc() -> tuple[list, list[dict]]:
    """Observe effective production target selection without changing it."""
    observations: list[dict] = []
    original = DCSOCStrategy.select_targets

    def observed_select_targets(self, node, message, simulator):
        targets = original(self, node, message, simulator)
        if simulator.experiment_name == "exp08" and simulator.strategy_name == "dcsoc":
            active_children = {
                child_id
                for child_id in getattr(node, "dcsoc_children", [])
                if child_id in simulator.nodes and simulator.nodes[child_id].is_active
            }
            observations.append({
                "overload": simulator.ch_overload_factor,
                "node": node.node_id,
                "role": getattr(node, "dcsoc_role", "leaf"),
                "mode": type(self).__name__,
                "fulfill_all": self.fulfill_all_structural_children,
                "active_children": active_children,
                "targets": set(targets),
            })
        return targets

    DCSOCStrategy.select_targets = observed_select_targets
    try:
        cfg = load_yaml_config("configs/exp08_ch_bottleneck.yaml")
        cfg["runs_per_setting"] = 1
        rows, _ = exp08(cfg)
    finally:
        DCSOCStrategy.select_targets = original
    return [row for row in rows if row.strategy == "dcsoc"], observations


def validate(name: str, config_path: str, runner, expected_conditions: int) -> bool:
    cfg = load_yaml_config(config_path)
    cfg["runs_per_setting"] = 1
    rows, traces = runner(cfg)
    identities = [
        (row.strategy, row.seed, row.topology_param, row.ch_overload_factor)
        for row in rows
    ]
    strategies = Counter(row.strategy for row in rows)
    finite = all(
        math.isfinite(float(getattr(row, metric)))
        for row in rows
        for metric in METRICS
    )
    checks = {
        "row_count": len(rows) == expected_conditions,
        "all_comparators": set(strategies) == EXPECTED_STRATEGIES,
        "one_seed": {row.seed for row in rows} == {cfg["seed"]},
        "unique_identities": len(identities) == len(set(identities)),
        "finite_metrics": finite,
        "delivery_range": all(0.0 <= row.delivery_ratio <= 1.0 for row in rows),
        "fanout_metadata_unset": all(row.fanout is None for row in rows),
        "ahbn_only_traces": bool(traces)
        and {row.strategy for row in traces} == {"ahbn"},
    }
    print(f"{name}: rows={len(rows)} strategies={dict(sorted(strategies.items()))}")
    for check, passed in checks.items():
        print(f"{name} {check}: {'PASS' if passed else 'FAIL'}")
    return all(checks.values())


def main() -> int:
    exp08_pass = validate(
        "EXP08", "configs/exp08_ch_bottleneck.yaml", exp08, 16
    )
    exp09_pass = validate(
        "EXP09", "configs/exp09_dense_topology.yaml", exp09, 20
    )
    dcsoc_rows, observations = audit_exp08_dcsoc()
    structural = [item for item in observations if item["active_children"]]
    old_signature = all(
        row.delivery_ratio == 0.04
        and row.duplicates == 3
        and row.total_forwards == 6
        for row in dcsoc_rows
    )
    dcsoc_checks = {
        "effective_strategy_mode": bool(observations)
        and {item["mode"] for item in observations} == {"DCSOCStrategy"},
        "fulfill_all_enabled": bool(observations)
        and all(item["fulfill_all"] for item in observations),
        "reachable_active_structural_edges_fulfilled": bool(structural)
        and all(item["active_children"] <= item["targets"] for item in structural),
        "delivery_not_capped_at_0.040": bool(dcsoc_rows)
        and all(row.delivery_ratio > 0.04 for row in dcsoc_rows),
        "old_0.040_3_6_signature_absent": not old_signature,
    }
    print("EXP08 DC-SoC effective smoke rows:")
    for row in dcsoc_rows:
        print(
            f"  overload={row.ch_overload_factor:g} "
            f"delivery={row.delivery_ratio:.6f} delay={row.propagation_delay:.6f} "
            f"duplicates={row.duplicates} forwards={row.total_forwards}"
        )
    print(
        "EXP08 DC-SoC structural audit: "
        f"observed_calls={len(observations)} "
        f"reached_cores_with_active_children={len(structural)} "
        f"active_edges={sum(len(item['active_children']) for item in structural)}"
    )
    for check, passed_check in dcsoc_checks.items():
        print(f"EXP08 dcsoc_{check}: {'PASS' if passed_check else 'FAIL'}")
    passed = exp08_pass and exp09_pass and all(dcsoc_checks.values())
    print(f"EXP08/EXP09 ONE-SEED SMOKE GATE: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
