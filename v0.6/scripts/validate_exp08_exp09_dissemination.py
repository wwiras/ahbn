"""Validate the narrowly scoped Exp08/Exp09 DC-SoC directionality correction."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ahbn.node import Node
from ahbn.simulator import Simulator
from ahbn.strategies.dcsoc import DCSOCStrategy


class ObservingSimulator(Simulator):
    def __init__(self, *args, **kwargs):
        self.transmissions: list[tuple[int, int]] = []
        super().__init__(*args, **kwargs)

    def send_message(self, src_id, dst_id, message, now):
        self.transmissions.append((src_id, dst_id))
        return super().send_message(src_id, dst_id, message, now)


def node(node_id: int, role: str, parent=None, children=()) -> Node:
    result = Node(node_id=node_id)
    result.is_active = True
    result.cluster_id = 0
    result.dcsoc_role = role
    result.dcsoc_parent = parent
    result.dcsoc_children = list(children)
    return result


def main() -> int:
    nodes = {
        0: node(0, "leaf", parent=4),
        1: node(1, "leaf", parent=4),
        2: node(2, "leaf", parent=4),
        3: node(3, "leaf", parent=5),
        4: node(4, "core", children=(0, 1, 2, 5)),
        5: node(5, "core", parent=4, children=(3, 6)),
        6: node(6, "leaf", parent=5),
    }
    permitted = {(0, 4), (4, 1), (4, 2), (4, 5), (5, 3), (5, 6)}
    simulator = ObservingSimulator(
        nodes=nodes,
        strategy=DCSOCStrategy(fanout=2, fulfill_all_structural_children=True),
        seed=42, base_delay=1.0, jitter=0.0, cluster_manager=object(),
        experiment_name="exp08_directionality", strategy_name="dcsoc",
    )
    simulator.inject_message(0, "m1")
    simulator.run()

    observed = simulator.transmissions
    counts = Counter(observed)
    record = simulator.metrics.messages["m1"]
    checks = {
        "source_leaf_uplinks_exactly_once": counts[(0, 4)] == 1,
        "non_source_leaves_do_not_echo": not [e for e in observed if e[0] in {1, 2, 3, 6}],
        "source_return_absent": (4, 0) not in counts,
        "all_required_edges_observed": set(observed) >= permitted,
        "no_unpermitted_edges": set(observed) <= permitted,
        "no_self_targets": all(src != dst for src, dst in observed),
        "no_duplicate_targets": all(count == 1 for count in counts.values()),
        "no_artificial_fanout_cap": all(counts[e] == 1 for e in {(4, 1), (4, 2), (4, 5)}),
        "full_delivery": record.delivery_ratio(len(nodes)) == 1.0,
        "one_edge_per_non_source_node": record.total_forwards == len(nodes) - 1,
        "zero_duplicates": record.duplicate_count == 0,
        "old_echo_signature_absent": not (
            record.total_forwards == 2 * (len(nodes) - 1)
            and record.duplicate_count == len(nodes) - 1
        ),
    }
    print(f"permitted_edges={len(permitted)} {sorted(permitted)}")
    print(f"observed_edges={len(observed)} {observed}")
    print(f"delivery={record.delivery_ratio(len(nodes)):.3f} forwards={record.total_forwards} duplicates={record.duplicate_count}")
    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    passed = all(checks.values())
    print(f"DC-SOC DIRECTIONAL EDGE-EXCLUSIVITY VALIDATION: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
