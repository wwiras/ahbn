from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional
import re

import pandas as pd


# ============================================================
# Final experiment result row
# ============================================================

@dataclass
class ResultRow:
    experiment: str
    strategy: str
    seed: int
    num_nodes: int
    topology_type: str
    topology_param: float | int | None
    fanout: int | None
    num_clusters: int | None
    ch_overload_factor: float | None

    delivery_ratio: float
    propagation_delay: float | None
    duplicates: int
    total_forwards: int


@dataclass
class Exp08ExecutionEvidence:
    """Observational run metadata; never consumed by dissemination logic."""

    experiment: str
    strategy: str
    seed: int
    overload_factor: float
    topology_type: str
    topology_param: float | int | None
    num_nodes: int
    topology_seed: int
    topology_identity: str
    configured_message_source: int
    effective_message_source: int
    dcsoc_master: int | None
    dcsoc_eligible_overload_nodes: str
    dcsoc_selected_overload_node: int | None
    dcsoc_selected_overload_role: str | None
    max_structural_obligations: int | None


# ============================================================
# Canonical AHBN adaptive trace
# ============================================================

@dataclass
class AdaptiveTraceRow:
    """
    One genuine AHBN controller update.

    Each row represents:

        raw normalized observations
            ↓
        EWMA-smoothed observations
            ↓
        controller score
            ↓
        sigmoid weight
            ↓
        mode + fanout

    Therefore one row = one AHBN decision update,
    not merely one simulator/network event.
    """

    # --------------------------------------------------------
    # Experiment context
    # --------------------------------------------------------
    experiment: str
    strategy: str
    seed: int
    scenario_tag: str

    # --------------------------------------------------------
    # Event context
    # --------------------------------------------------------
    time: float
    node_id: int
    message_id: Optional[str]
    event_type: str

    # --------------------------------------------------------
    # Raw normalized observations [0, 1]
    # --------------------------------------------------------
    # duplicate_obs: float
    # latency_obs: float
    # utilization_obs: float
    # churn_obs: float
    
    duplicate_obs: Optional[float]
    latency_obs: Optional[float]
    utilization_obs: Optional[float]
    churn_obs: Optional[float]

    # --------------------------------------------------------
    # EWMA-smoothed observations [0, 1]
    # --------------------------------------------------------
    d_hat: float
    l_hat: float
    u_hat: float
    c_hat: float

    # --------------------------------------------------------
    # Controller computation
    # --------------------------------------------------------
    score: float
    weight: float

    # --------------------------------------------------------
    # Controller decision
    # --------------------------------------------------------
    mode: str
    fanout: int

    # --------------------------------------------------------
    # Adaptation-event indicators
    # --------------------------------------------------------
    mode_switched: bool
    fanout_changed: bool

    # --------------------------------------------------------
    # Supporting diagnostics
    # --------------------------------------------------------
    duplicate_ratio_raw: float
    resource_class: str
    capacity_score: float
    processing_delay: float

    # --------------------------------------------------------
    # Cumulative node counters
    # --------------------------------------------------------
    received_new: int
    received_duplicate: int
    forwarded: int


# ============================================================
# File-system helpers
# ============================================================

def ensure_dir(
    path: str | Path,
) -> None:
    Path(path).mkdir(
        parents=True,
        exist_ok=True,
    )


def current_timestamp() -> str:
    return datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


# ============================================================
# Result CSV
# ============================================================

def save_results_csv(
    rows: Iterable[ResultRow],
    output_path: str | Path,
    add_timestamp: bool = True,
) -> str:

    df = pd.DataFrame(
        [asdict(row) for row in rows]
    )

    output_path = Path(output_path)

    ensure_dir(
        output_path.parent
    )

    if add_timestamp:
        ts = current_timestamp()

        output_path = output_path.with_name(
            f"{output_path.stem}_{ts}"
            f"{output_path.suffix}"
        )

    df.to_csv(
        output_path,
        index=False,
    )

    return str(output_path)


# ============================================================
# AHBN adaptive trace CSV
# ============================================================

def save_adaptive_trace_csv(
    rows: Iterable[AdaptiveTraceRow],
    output_path: str | Path,
    add_timestamp: bool = True,
) -> str:

    rows = list(rows)

    if not rows:
        raise ValueError(
            "No adaptive trace rows to save."
        )

    df = pd.DataFrame(
        [asdict(row) for row in rows]
    )

    output_path = Path(output_path)

    ensure_dir(
        output_path.parent
    )

    if add_timestamp:
        ts = current_timestamp()

        output_path = output_path.with_name(
            f"{output_path.stem}_{ts}"
            f"{output_path.suffix}"
        )

    df.to_csv(
        output_path,
        index=False,
    )

    return str(output_path)


def save_exp08_evidence_csv(
    rows: Iterable[Exp08ExecutionEvidence],
    output_path: str | Path,
    add_timestamp: bool = True,
) -> str:
    df = pd.DataFrame([asdict(row) for row in rows])
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    if add_timestamp:
        ts = current_timestamp()
        output_path = output_path.with_name(f"{output_path.stem}_{ts}{output_path.suffix}")
    df.to_csv(output_path, index=False)
    return str(output_path)


# ============================================================
# Filename helpers
# ============================================================

def extract_timestamp_from_filename(
    path: str | Path,
) -> str | None:

    name = Path(path).name

    match = re.search(
        r"(\d{8}_\d{6})",
        name,
    )

    return (
        match.group(1)
        if match
        else None
    )
