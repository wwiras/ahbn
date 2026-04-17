from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class NodeControlState:
    # Existing EWMA metrics
    d_hat: float = 0.0    # duplicate EWMA
    u_hat: float = 0.0    # load EWMA proxy
    l_hat: float = 0.0    # latency EWMA
    rho_hat: float = 0.0  # churn EWMA (future extension)

    # New Exp09-compatible EWMA metrics
    deg_hat: float = 0.0  # degree EWMA
    ov_hat: float = 0.0   # neighbor-overlap EWMA
    r_hat: float = 0.0    # redundancy EWMA

    # Control outputs
    mode: str = "gossip"  # "gossip" or "cluster"
    weight: float = 1.0   # gossip preference in [0,1]
    fanout: int = 3
    tau: float = 1.0      # forwarding gate threshold in [0,1]


@dataclass
class AHBNParams:
    # EWMA smoothing
    ewma_alpha: float = 0.3

    # Reference points
    d0: float = 0.2
    u0: float = 5.0
    l0: float = 2.0
    rho0: float = 0.1

    # New topology-aware reference points
    deg0: float = 8.0
    ov0: float = 0.25
    r0: float = 0.35

    # Weight equation coefficients
    # Positive => more gossip preference
    # Negative => less gossip preference
    a_dup: float = -2.0
    a_load: float = -1.5
    a_lat: float = 1.5
    a_churn: float = 1.0

    # New Exp09 coefficients
    a_deg: float = -0.4
    a_ov: float = -1.2
    a_red: float = -1.8

    # Redundancy composition
    b_degree: float = 0.25
    b_overlap: float = 0.75

    # Fanout bounds
    min_fanout: int = 1
    max_fanout: int = 6
    mode_threshold: float = 0.5

    # Fanout adaptation coefficients
    fanout_dup_penalty: float = 2.0
    fanout_load_penalty: float = 0.5
    fanout_lat_reward: float = 0.8
    fanout_red_penalty: float = 1.5

    # Forwarding gate parameters
    tau_max: float = 0.90
    tau_min: float = 0.25
    tau_dup_penalty: float = 1.0
    tau_red_penalty: float = 1.5

    # Keep AHBN hybrid, not pure gossip / pure cluster
    min_weight: float = 0.20
    max_weight: float = 0.80


class AHBNController:
    def __init__(self, params: AHBNParams) -> None:
        self.params = params

        # Expose these for simulator convenience if needed
        self.degree_ref = params.deg0
        self.b_degree = params.b_degree
        self.b_overlap = params.b_overlap

    def ewma(self, old: float, new: float) -> float:
        a = self.params.ewma_alpha
        return a * new + (1.0 - a) * old

    def sigmoid(self, x: float) -> float:
        # numerically stable enough for our range
        return 1.0 / (1.0 + math.exp(-x))

    def clamp(self, x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def update_metrics(
        self,
        state: NodeControlState,
        duplicate_ratio: float,
        load_proxy: float,
        latency_proxy: float,
        churn_proxy: float = 0.0,
        degree_proxy: float = 0.0,
        overlap_proxy: float = 0.0,
        redundancy_proxy: float = 0.0,
    ) -> None:
        """
        Backward-compatible metric update.

        Exp07 / Exp08:
            can continue calling with only duplicate_ratio, load_proxy,
            latency_proxy, churn_proxy.

        Exp09:
            can additionally provide degree_proxy, overlap_proxy,
            redundancy_proxy.
        """
        state.d_hat = self.ewma(state.d_hat, duplicate_ratio)
        state.u_hat = self.ewma(state.u_hat, load_proxy)
        state.l_hat = self.ewma(state.l_hat, latency_proxy)
        state.rho_hat = self.ewma(state.rho_hat, churn_proxy)

        state.deg_hat = self.ewma(state.deg_hat, degree_proxy)
        state.ov_hat = self.ewma(state.ov_hat, overlap_proxy)

        # If caller did not explicitly provide redundancy_proxy,
        # derive it from degree + overlap to preserve compatibility.
        if redundancy_proxy <= 0.0:
            redundancy_proxy = (
                self.params.b_overlap * overlap_proxy
                + self.params.b_degree * (degree_proxy / max(1.0, self.params.deg0))
            )

        state.r_hat = self.ewma(state.r_hat, redundancy_proxy)

    def compute_weight(self, state: NodeControlState) -> float:
        """
        Compute gossip preference weight in [0,1].

        High weight  -> more gossip-like behavior
        Low weight   -> more cluster-like behavior
        """
        p = self.params
        x = (
            p.a_churn * (state.rho_hat - p.rho0)
            + p.a_lat * (state.l_hat - p.l0)
            + p.a_dup * (state.d_hat - p.d0)
            + p.a_load * (state.u_hat - p.u0)
            + p.a_deg * (state.deg_hat - p.deg0)
            + p.a_ov * (state.ov_hat - p.ov0)
            + p.a_red * (state.r_hat - p.r0)
        )
        w = self.sigmoid(x)
        return self.clamp(w, p.min_weight, p.max_weight)

    def compute_tau(self, state: NodeControlState) -> float:
        """
        Forwarding gate threshold.
        Lower tau => fewer rebroadcasts.
        """
        p = self.params
        tau = p.tau_max * math.exp(
            -p.tau_dup_penalty * max(0.0, state.d_hat)
            -p.tau_red_penalty * max(0.0, state.r_hat)
        )
        return self.clamp(tau, p.tau_min, p.tau_max)

    def decide_mode_and_fanout(self, state: NodeControlState) -> None:
        p = self.params

        state.weight = self.compute_weight(state)

        if state.weight >= p.mode_threshold:
            state.mode = "gossip"
        else:
            state.mode = "cluster"

        raw = round(
            p.max_fanout
            - p.fanout_dup_penalty * state.d_hat
            - p.fanout_load_penalty * state.u_hat
            + p.fanout_lat_reward * state.l_hat
            - p.fanout_red_penalty * max(0.0, state.r_hat - p.r0)
        )
        state.fanout = max(p.min_fanout, min(p.max_fanout, raw))
        state.tau = self.compute_tau(state)
        
    def snapshot_state(self, state: NodeControlState) -> dict:
        return {
            "mode": state.mode,
            "weight": state.weight,
            "fanout": state.fanout,
            "tau": state.tau,
            "d_hat": state.d_hat,
            "u_hat": state.u_hat,
            "l_hat": state.l_hat,
            "rho_hat": state.rho_hat,
            "deg_hat": state.deg_hat,
            "ov_hat": state.ov_hat,
            "r_hat": state.r_hat,
        }