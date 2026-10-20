"""Final embedding pooling and deterministic reference retrieval."""

from dataclasses import dataclass
import numpy as np


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
