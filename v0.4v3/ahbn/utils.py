from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


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
    propagation_delay: float
    duplicates: int
    total_forwards: int


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def extract_timestamp_from_filename(path: str | Path) -> str | None:
    """
    Returns YYYYMMDD_HHMMSS if present in filename stem, else None.
    """
    p = Path(path)
    parts = p.stem.split("_")
    if len(parts) >= 2:
        maybe_date = parts[-2]
        maybe_time = parts[-1]
        if (
            len(maybe_date) == 8
            and len(maybe_time) == 6
            and maybe_date.isdigit()
            and maybe_time.isdigit()
        ):
            return f"{maybe_date}_{maybe_time}"
    return None


def add_timestamp_to_path(output_path: str | Path) -> Path:
    """
    Converts:
      outputs/csv/exp07_results.csv
    into:
      outputs/csv/exp07_results_YYYYMMDD_HHMMSS.csv

    If a timestamp already exists in the filename, it is preserved.
    """
    output_path = Path(output_path)

    existing = extract_timestamp_from_filename(output_path)
    if existing is not None:
        return output_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_path.with_name(f"{output_path.stem}_{timestamp}{output_path.suffix}")


def save_results_csv(rows: Iterable[ResultRow], output_path: str | Path) -> Path:
    rows = list(rows)
    if not rows:
        raise ValueError("No rows to save.")

    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    final_path = add_timestamp_to_path(output_path)

    fieldnames = list(asdict(rows[0]).keys())

    with final_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    return final_path