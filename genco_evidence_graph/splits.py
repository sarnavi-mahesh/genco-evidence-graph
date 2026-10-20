"""Authoritative scenario-split loading and membership checks."""

from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq


def load_split_ids(path: str | Path) -> tuple[str, ...]:
    """Load a published graphkit PyTorch split as normalized scenario IDs."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required to read authoritative graphkit split artifacts",
        ) from error
    split_path = Path(path)
    if not split_path.is_file():
        raise FileNotFoundError(f"Split artifact does not exist: {split_path}")
    values = torch.load(split_path, map_location="cpu", weights_only=True)
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"Split artifact must contain a list or tuple: {split_path}")
    normalized = tuple(str(int(value)) for value in values)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"Split artifact contains duplicate scenario IDs: {split_path}")
    return normalized


def verify_split_membership(
    actual_ids: Iterable[str],
    split_path: str | Path,
    label: str,
) -> tuple[str, ...]:
    """Require exact scenario membership against a published split artifact."""
    expected = load_split_ids(split_path)
    actual = tuple(str(value) for value in actual_ids)
    if len(actual) != len(set(actual)):
        raise ValueError(f"{label} contains duplicate scenario IDs")
    missing = sorted(set(expected) - set(actual), key=int)
    extra = sorted(set(actual) - set(expected), key=int)
    if missing or extra or len(actual) != len(expected):
        raise ValueError(
            f"{label} disagrees with authoritative split {split_path}: "
            f"expected={len(expected)}, actual={len(actual)}, "
            f"missing={missing[:10]}, extra={extra[:10]}",
        )
    return expected


def parquet_scenario_ids(
    path: str | Path,
    column: str = "scenario",
) -> tuple[str, ...]:
    """Read unique scenario IDs from a parquet artifact."""
    parquet_path = Path(path)
    values: set[str] = set()
    for batch in pq.ParquetFile(parquet_path).iter_batches(
        columns=[column],
        batch_size=100_000,
    ):
        values.update(str(value.as_py()) for value in batch.column(0))
    return tuple(sorted(values, key=int))
