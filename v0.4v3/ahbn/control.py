from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class NodeControlState:
    d_hat: float = 0.0
    u_hat: float = 0.0
    l_hat: float = 0.0
    rho_hat: float = 0.0

    mode: str = "gossip"
    weight: float = 1.0
    fanout: int = 3


@dataclass
class AHBNParams:
    ewma_alpha: float = 0.3

    d0: float = 0.2
    u0: float = 5.0
    l0: float = 2.0
    rho0: float = 0.1

    a_dup: float = -2.0
    a_load: float = -1.5
    a_lat: float = 1.5
    a_churn: float = 1.0

    min_fanout: int = 1
    max_fanout: int = 6
    mode_threshold: float = 0.5


class AHBNController:
    def __init__(self, params: AHBNParams) -> None:
        self.params = params

    def ewma(self, old: float, new: float) -> float:
        a = self.params.ewma_alpha
        return a * new + (1.0 - a) * old

    def sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def update_metrics(
        self,
        state: NodeControlState,
        duplicate_ratio: float,
        load_proxy: float,
        latency_proxy: float,
        churn_proxy: float = 0.0,
    ) -> None:
        state.d_hat = self.ewma(state.d_hat, duplicate_ratio)
        state.u_hat = self.ewma(state.u_hat, load_proxy)
        state.l_hat = self.ewma(state.l_hat, latency_proxy)
        state.rho_hat = self.ewma(state.rho_hat, churn_proxy)

    def compute_weight(self, state: NodeControlState) -> float:
        p = self.params
        x = (
            p.a_churn * (state.rho_hat - p.rho0)
            + p.a_lat * (state.l_hat - p.l0)
            + p.a_dup * (state.d_hat - p.d0)
            + p.a_load * (state.u_hat - p.u0)
        )
        return self.sigmoid(x)

    def decide_mode_and_fanout(self, state: NodeControlState) -> None:
        p = self.params
        state.weight = self.compute_weight(state)

        if state.weight >= p.mode_threshold:
            state.mode = "gossip"
        else:
            state.mode = "cluster"

        raw = round(
            p.max_fanout
            - 2.0 * state.d_hat
            - 0.5 * state.u_hat
            + 0.8 * state.l_hat
        )
        state.fanout = max(p.min_fanout, min(p.max_fanout, raw))