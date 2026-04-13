from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from datetime import datetime
import re

import pandas as pd


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


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def current_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_results_csv(
    rows: Iterable[ResultRow],
    output_path: str | Path,
    add_timestamp: bool = True,
) -> str:
    df = pd.DataFrame([asdict(r) for r in rows])

    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    if add_timestamp:
        ts = current_timestamp()
        output_path = output_path.with_name(f"{output_path.stem}_{ts}{output_path.suffix}")

    df.to_csv(output_path, index=False)
    return str(output_path)


def extract_timestamp_from_filename(path: str | Path) -> str | None:
    """
    Example:
        outputs/csv/exp07_results_20260402_093015.csv
    returns:
        20260402_093015
    """
    name = Path(path).name
    m = re.search(r"(\d{8}_\d{6})", name)
    return m.group(1) if m else None