from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from ahbn.control import AHBNController, AHBNParams, NodeControlState
from ahbn.strategies.ahbn import AHBNStrategy


EXACT_CASES = (
    (-0.30, 2),
    (-0.25, 2),
    (0.00, 3),
    (0.25, 4),
    (0.50, 4),
    (0.90, 5),
    (1.00, 5),
    (1.50, 6),
    (1.60, 6),
)

ADJACENT_CASES = (
    (-0.250001, 2),
    (-0.250000, 2),
    (-0.249999, 3),
    (0.249999, 3),
    (0.250000, 4),
    (0.250001, 4),
    (0.899999, 4),
    (0.900000, 5),
    (0.900001, 5),
    (1.499999, 5),
    (1.500000, 6),
    (1.500001, 6),
)


def _state_with_score(score: float) -> NodeControlState:
    """Represent the requested score with valid normalized observations."""
    if score < 0.0:
        return NodeControlState(d_hat=-score)
    return NodeControlState(l_hat=min(score, 1.0), u_hat=max(score - 1.0, 0.0))


def _requested_fanout(score: float) -> int:
    state = _state_with_score(score)
    AHBNController(AHBNParams()).decide_mode_and_fanout(state)
    assert state.score == score
    return state.fanout


@pytest.mark.parametrize(("score", "expected"), EXACT_CASES)
def test_exact_s5_boundary_vector(score: float, expected: int) -> None:
    assert _requested_fanout(score) == expected


@pytest.mark.parametrize(("score", "expected"), ADJACENT_CASES)
def test_adjacent_s5_boundary_vector(score: float, expected: int) -> None:
    assert _requested_fanout(score) == expected


def test_default_and_all_supported_requested_gears() -> None:
    assert NodeControlState().fanout == 3
    assert {_requested_fanout(score) for score, _ in EXACT_CASES} == {2, 3, 4, 5, 6}


@pytest.mark.parametrize("requested", (2, 3, 4, 5, 6))
def test_effective_fanout_preserves_controller_selection(requested: int) -> None:
    strategy = AHBNStrategy()
    node = SimpleNamespace(control=SimpleNamespace(fanout=requested))
    assert strategy._get_effective_fanout(node) == requested


def test_gossip_realization_is_limited_by_eligible_neighbors() -> None:
    strategy = AHBNStrategy()
    node = SimpleNamespace(
        node_id=0,
        neighbors=[1, 2, 3],
        control=SimpleNamespace(fanout=6, mode="gossip"),
    )
    simulator = SimpleNamespace(
        nodes={peer_id: SimpleNamespace(is_active=True) for peer_id in node.neighbors},
        rng=random.Random(42),
    )
    targets = strategy.select_targets(node, SimpleNamespace(), simulator)
    assert len(targets) == len(node.neighbors)
    assert set(targets).issubset(node.neighbors)


def _structured_head_targets(score: float, member_count: int) -> list[int]:
    fanout = _requested_fanout(score)
    members = list(range(1, member_count + 1))
    gateway = member_count + 1
    node = SimpleNamespace(
        node_id=0,
        cluster_id=10,
        is_cluster_head=True,
        gateway_neighbors=[gateway],
        control=SimpleNamespace(fanout=fanout, mode="cluster"),
    )
    simulator = SimpleNamespace(
        nodes={
            target_id: SimpleNamespace(is_active=True)
            for target_id in members + [gateway]
        },
        cluster_manager=SimpleNamespace(
            get_cluster_members=lambda cluster_id, exclude: members,
        ),
    )
    return AHBNStrategy().select_targets(node, SimpleNamespace(), simulator)


@pytest.mark.parametrize(("score", "expected"), ((0.90, 5), (1.50, 6)))
def test_structured_head_preserves_controller_fanout(score: float, expected: int) -> None:
    targets = _structured_head_targets(score, member_count=6)
    assert len(targets) == expected


def test_structured_head_realization_is_limited_by_eligible_targets() -> None:
    targets = _structured_head_targets(1.50, member_count=2)
    assert len(targets) == 3
