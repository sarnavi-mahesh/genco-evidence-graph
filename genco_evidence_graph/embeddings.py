"""Final embedding pooling and deterministic reference retrieval."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def load_pooled_keyed_embeddings(
    path: str | Path,
    expected_split: str,
) -> tuple[list[str], np.ndarray]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Embedding artifact does not exist: {path}")
    parquet = pq.ParquetFile(path)
    required = {
        "scenario_id",
        "entity_type",
        "entity_id",
        "correction_step",
        "embedding_stage",
        "embedding",
        "dataset_split",
    }
    if not required.issubset(parquet.schema_arrow.names):
        raise ValueError(
            f"{path} must contain keyed embedding columns: "
            f"{', '.join(sorted(required))}",
        )

    sums: dict[str, dict[str, np.ndarray]] = {"bus": {}, "generator": {}}
    counts: dict[str, dict[str, int]] = {"bus": {}, "generator": {}}
    completed_groups: set[tuple[str, str]] = set()
    current_group: tuple[str, str] | None = None
    current_entity_ids: set[int] = set()
    embedding_dimension: int | None = None

    def complete_group() -> None:
        nonlocal current_group, current_entity_ids
        if current_group is None:
            return
        if current_entity_ids != set(range(len(current_entity_ids))):
            raise ValueError(
                f"{path} has incomplete or non-contiguous entity IDs for "
                f"{current_group[0]} scenario {current_group[1]}",
            )
        completed_groups.add(current_group)

    for batch in parquet.iter_batches(
        columns=[
            "scenario_id",
            "entity_type",
            "entity_id",
            "correction_step",
            "embedding_stage",
            "embedding",
            "dataset_split",
        ],
        batch_size=50_000,
    ):
        scenario_ids = np.asarray(batch.column(0))
        entity_types = np.asarray(batch.column(1))
        entity_ids = np.asarray(batch.column(2))
        correction_steps = batch.column(3)
        embedding_stages = np.asarray(batch.column(4))
        embeddings = np.asarray(batch.column(5).to_pylist(), dtype=float)
        dataset_splits = np.asarray(batch.column(6))
        if embeddings.ndim != 2 or embeddings.shape[1] == 0:
            raise ValueError(f"{path} contains invalid embedding vectors")
        if embedding_dimension is None:
            embedding_dimension = embeddings.shape[1]
        elif embeddings.shape[1] != embedding_dimension:
            raise ValueError(f"{path} contains mixed embedding dimensions")
        if not np.isfinite(embeddings).all():
            raise ValueError(f"{path} contains non-finite embeddings")
        if set(dataset_splits) != {expected_split}:
            raise ValueError(
                f"{path} must contain only dataset_split={expected_split!r}",
            )
        if set(embedding_stages) != {"final"}:
            raise ValueError(f"{path} must contain only embedding_stage='final'")
        if correction_steps.null_count != batch.num_rows:
            raise ValueError(
                f"{path} final embeddings must have null correction_step",
            )
        invalid_types = set(entity_types) - {"bus", "generator"}
        if invalid_types:
            raise ValueError(f"{path} contains unsupported entity types: {invalid_types}")

        for scenario_id, entity_type, entity_id in zip(
            scenario_ids,
            entity_types,
            entity_ids,
        ):
            group = (str(entity_type), str(scenario_id))
            if group != current_group:
                complete_group()
                if group in completed_groups:
                    raise ValueError(
                        f"{path} contains non-contiguous duplicate entity group {group}",
                    )
                current_group = group
                current_entity_ids = set()
            numeric_entity_id = int(entity_id)
            if numeric_entity_id in current_entity_ids:
                raise ValueError(
                    f"{path} contains duplicate entity key "
                    f"({scenario_id}, {entity_type}, {numeric_entity_id})",
                )
            current_entity_ids.add(numeric_entity_id)

        for entity_type in ("bus", "generator"):
            mask = entity_types == entity_type
            for scenario_id in np.unique(scenario_ids[mask]):
                key = str(scenario_id)
                values = embeddings[mask & (scenario_ids == scenario_id)]
                if key in sums[entity_type]:
                    sums[entity_type][key] += values.sum(axis=0)
                    counts[entity_type][key] += len(values)
                else:
                    sums[entity_type][key] = values.sum(axis=0)
                    counts[entity_type][key] = len(values)
    complete_group()

    scenario_ids = sorted(set(sums["bus"]) | set(sums["generator"]))
    if set(sums["bus"]) != set(sums["generator"]):
        raise ValueError(f"Keyed embedding artifact has unaligned entity scenario IDs: {path}")
    pooled = np.stack(
        [
            np.concatenate(
                [
                    sums["bus"][scenario_id] / counts["bus"][scenario_id],
                    sums["generator"][scenario_id] / counts["generator"][scenario_id],
                ],
            )
            for scenario_id in scenario_ids
        ],
    )
    return scenario_ids, pooled

def _mean_entities(values: np.ndarray, label: str) -> np.ndarray:
    if values.ndim != 3:
        raise ValueError(f"{label} must have shape [scenario, entity, dimension]")
    if values.shape[1] == 0:
        raise ValueError(f"{label} cannot contain zero entities")
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains non-finite values")
    return values.mean(axis=1)


def pool_scenario_embeddings(
    bus_embeddings: np.ndarray,
    gen_embeddings: np.ndarray | None = None,
) -> np.ndarray:
    """Pool entity embeddings using mean/concatenate pooling strategy."""
    bus_mean = _mean_entities(np.asarray(bus_embeddings), "bus_embeddings")
    if gen_embeddings is None:
        return bus_mean
    gen_mean = _mean_entities(np.asarray(gen_embeddings), "gen_embeddings")
    if gen_mean.shape[0] != bus_mean.shape[0]:
        raise ValueError("bus and generator scenario counts do not match")
    return np.concatenate([bus_mean, gen_mean], axis=1)


@dataclass(frozen=True)
class ReferenceMatch:
    scenario_id: str
    distance: float
    rank: int


class ReferenceIndex:
    """A small deterministic Euclidean reference index for GENCO use."""

    def __init__(self, scenario_ids: list[str], embeddings: np.ndarray) -> None:
        matrix = np.asarray(embeddings, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != len(scenario_ids):
            raise ValueError("scenario_ids and reference embeddings are not aligned")
        if not len(scenario_ids):
            raise ValueError("reference index cannot be empty")
        if not np.isfinite(matrix).all():
            raise ValueError("reference embeddings contain non-finite values")
        if len(set(scenario_ids)) != len(scenario_ids):
            raise ValueError("reference scenario IDs must be unique")
        self._scenario_ids = tuple(scenario_ids)
        self._embeddings = matrix

    def query(
        self,
        embedding: np.ndarray,
        top_k: int = 5,
        exclude_scenario_id: str | None = None,
    ) -> list[ReferenceMatch]:
        vector = np.asarray(embedding, dtype=float)
        if vector.ndim != 1 or vector.shape[0] != self._embeddings.shape[1]:
            raise ValueError("query embedding dimension does not match the index")
        if not np.isfinite(vector).all():
            raise ValueError("query embedding contains non-finite values")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        distances = np.linalg.norm(self._embeddings - vector, axis=1)
        candidates = [
            (distance, scenario_id)
            for distance, scenario_id in zip(distances, self._scenario_ids)
            if scenario_id != exclude_scenario_id
        ]
        candidates.sort(key=lambda item: (item[0], item[1]))
        return [
            ReferenceMatch(scenario_id=scenario_id, distance=float(distance), rank=rank)
            for rank, (distance, scenario_id) in enumerate(candidates[:top_k], start=1)
        ]
