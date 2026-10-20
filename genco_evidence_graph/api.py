"""Typed Python APIs equivalent to CLI operations."""

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

from rdflib import Graph

from .cli import _build_evidence
from .queries import (
    NAMED_QUERIES,
    QUERY_HEADERS,
    high_residual_and_distance,
)
from .validation import validate_evidence_graph


@dataclass(frozen=True)
class EvidenceBuildSpec:
    dataset_id: str
    checkpoint: Path
    prediction_id: str
    prediction_parquet: Path
    generator_prediction_parquet: Path
    target_keyed_embeddings: Path
    reference_keyed_embeddings: Path
    reference_prediction_parquet: Path
    reference_generator_prediction_parquet: Path
    reference_run_manifest: Path
    reference_split_artifact: Path
    reference_config_artifact: Path
    training_run_metadata: Path
    original_upstream_training_metadata_root: Path
    model_repository_path: Path
    raw_dataset_path: Path
    split_artifact: Path
    normalizer_artifact: Path
    config_artifact: Path
    run_manifest: Path
    reproduction_run_manifest: Path
    baseline_tolerance_config: Path
    output: Path
    expected_dataset_revision: str | None = None
    expected_graphkit_commit: str | None = None
    expected_mlflow_run_id: str | None = None
    top_k: int = 5


@dataclass(frozen=True)
class QueryResult:
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


def build_evidence(spec: EvidenceBuildSpec) -> Path:
    """Build and serialize one evidence graph using the CLI implementation."""
    _build_evidence(Namespace(**spec.__dict__))
    return spec.output


def query_evidence(
    graph: Graph | str | Path,
    *,
    named: str | None = None,
    sparql: str | None = None,
    residual_threshold: float | None = None,
    distance_threshold: float | None = None,
) -> QueryResult:
    """Run a named or custom SPARQL query and return stable string rows."""
    if (named is None) == (sparql is None):
        raise ValueError("Provide exactly one of named or sparql")
    if isinstance(graph, Graph):
        parsed = graph
    else:
        parsed = Graph()
        parsed.parse(graph, format="turtle")
    if named == "high-residual-distance":
        if residual_threshold is None or distance_threshold is None:
            raise ValueError(
                "high-residual-distance requires residual and distance thresholds",
            )
        query = high_residual_and_distance(
            residual_threshold,
            distance_threshold,
        )
    elif named is not None:
        try:
            query = NAMED_QUERIES[named]
        except KeyError as error:
            raise ValueError(f"Unknown named query: {named}") from error
    else:
        query = sparql
    results = parsed.query(query)
    headers = (
        QUERY_HEADERS[named]
        if named is not None
        else tuple(str(variable) for variable in results.vars)
    )
    rows = tuple(
        tuple("" if value is None else str(value) for value in row)
        for row in results
    )
    return QueryResult(headers=headers, rows=rows)


def validate_evidence(
    graph: Graph | str | Path,
    shapes_path: str | Path,
) -> None:
    """Validate an in-memory or serialized evidence graph with SHACL."""
    if isinstance(graph, Graph):
        parsed = graph
    else:
        parsed = Graph()
        parsed.parse(graph, format="turtle")
    validate_evidence_graph(parsed, shapes_path)
