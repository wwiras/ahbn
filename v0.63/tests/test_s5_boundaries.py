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


@pytest.mark.parametrize(("requested", "effective"), ((6, 4), (3, 3)))
def test_effective_fanout_uses_production_clamp(requested: int, effective: int) -> None:
    strategy = AHBNStrategy()
    node = SimpleNamespace(control=SimpleNamespace(fanout=requested))
    assert strategy._get_effective_fanout(node) == effective


@pytest.mark.parametrize(("requested", "realized"), ((6, 4), (3, 3)))
def test_gossip_realization_obeys_effective_fanout(requested: int, realized: int) -> None:
    strategy = AHBNStrategy()
    node = SimpleNamespace(
        node_id=0,
        neighbors=[1, 2, 3, 4, 5, 6],
        control=SimpleNamespace(fanout=requested, mode="gossip"),
    )
    simulator = SimpleNamespace(
        nodes={peer_id: SimpleNamespace(is_active=True) for peer_id in node.neighbors},
        rng=random.Random(42),
    )
    targets = strategy.select_targets(node, SimpleNamespace(), simulator)
    assert len(targets) == realized
    assert set(targets).issubset(node.neighbors)
