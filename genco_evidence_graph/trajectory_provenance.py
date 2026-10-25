"""Verified RDF provenance for cohort latent trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd
from rdflib import Graph
from rdflib.namespace import PROV, RDF

from .provenance import sha256_file
from .rdf import (
    GENCO,
    add_evidence_literal,
    evidence_resource,
    serialize_graph,
)


@dataclass(frozen=True)
class TrajectoryProvenanceSpec:
    trajectory: Path
    capture_manifest: Path
    baseline_run_manifest: Path
    instrumented_run_manifest: Path
    split_artifact: Path
    tolerance_config: Path
    output: Path


@dataclass(frozen=True)
class TrajectoryProvenanceResult:
    output: Path
    triple_count: int
    scenario_count: int
    entity_count: int
    row_count: int


def _load_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _verify_record(
    manifest: dict[str, Any],
    key: str,
    path: Path,
) -> dict[str, Any]:
    record = manifest.get(key)
    if not isinstance(record, dict):
        raise ValueError(
            f"Trajectory capture manifest lacks {key}"
        )
    if Path(record.get("path", "")).resolve() != path.resolve():
        raise ValueError(
            f"Trajectory capture manifest path disagrees for {key}"
        )
    if record.get("sha256") != sha256_file(path):
        raise ValueError(
            f"Trajectory capture manifest hash disagrees for {key}"
        )
    return record


def build_trajectory_provenance_graph(
    spec: TrajectoryProvenanceSpec,
) -> tuple[Graph, dict[str, Any]]:
    """Verify trajectory artifacts and build their provenance graph."""
    manifest = _load_object(
        spec.capture_manifest,
        "Trajectory capture manifest",
    )
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_type")
        != "genco_cohort_latent_trajectory"
    ):
        raise ValueError(
            "Unsupported trajectory capture manifest"
        )
    trajectory_record = _verify_record(
        manifest,
        "trajectory",
        spec.trajectory,
    )
    _verify_record(
        manifest,
        "baseline_run_manifest",
        spec.baseline_run_manifest,
    )
    instrumented_record = _verify_record(
        manifest,
        "instrumented_run_manifest",
        spec.instrumented_run_manifest,
    )
    _verify_record(
        manifest,
        "split_artifact",
        spec.split_artifact,
    )
    tolerance_record = _verify_record(
        manifest,
        "tolerance_policy",
        spec.tolerance_config,
    )
    equivalence = manifest.get("equivalence")
    if not isinstance(equivalence, dict) or (
        equivalence.get("passed") is not True
        or equivalence.get(
            "instrumented_final_snapshot_exact"
        )
        is not True
    ):
        raise ValueError(
            "Trajectory capture equivalence is not verified"
        )

    table = pd.read_parquet(
        spec.trajectory,
        columns=[
            "scenario_id",
            "entity_type",
            "entity_id",
            "correction_step",
            "embedding_stage",
            "embedding",
        ],
    )
    entity_count = table[
        ["scenario_id", "entity_type", "entity_id"]
    ].drop_duplicates().shape[0]
    observed = {
        "row_count": len(table),
        "scenario_count": table[
            "scenario_id"
        ].nunique(),
        "entity_count": entity_count,
        "correction_step_count": table[
            "correction_step"
        ].nunique(),
        "embedding_dimension": len(
            table.iloc[0]["embedding"]
        ),
        "embedding_stage": (
            table["embedding_stage"].unique().tolist()
        ),
    }
    for key in (
        "row_count",
        "scenario_count",
        "entity_count",
        "correction_step_count",
        "embedding_dimension",
    ):
        if trajectory_record.get(key) != observed[key]:
            raise ValueError(
                f"Trajectory manifest {key} disagrees"
            )
    if observed["embedding_stage"] != [
        trajectory_record.get("embedding_stage")
    ]:
        raise ValueError(
            "Trajectory manifest embedding stage disagrees"
        )

    graph = Graph()
    graph.bind("genco", GENCO)
    graph.bind("prov", PROV)
    run_id = instrumented_record["run_id"]
    activity = evidence_resource(
        "trajectory-capture",
        run_id,
    )
    graph.add((activity, RDF.type, PROV.Activity))
    graph.add(
        (activity, RDF.type, GENCO.TrajectoryCaptureRun)
    )

    artifacts = (
        (
            spec.trajectory,
            GENCO.LatentTrajectoryArtifact,
            True,
        ),
        (
            spec.capture_manifest,
            GENCO.TrajectoryCaptureManifest,
            True,
        ),
        (
            spec.baseline_run_manifest,
            GENCO.BaselineRunManifest,
            False,
        ),
        (
            spec.instrumented_run_manifest,
            GENCO.InstrumentedRunManifest,
            False,
        ),
        (
            spec.split_artifact,
            GENCO.CohortInferenceSplit,
            False,
        ),
        (
            spec.tolerance_config,
            GENCO.EquivalenceTolerancePolicy,
            False,
        ),
    )
    for path, artifact_class, generated in artifacts:
        resource = evidence_resource(
            "artifact",
            sha256_file(path),
        )
        graph.add((resource, RDF.type, PROV.Entity))
        graph.add((resource, RDF.type, artifact_class))
        add_evidence_literal(
            graph,
            resource,
            GENCO.artifactPath,
            str(path),
        )
        add_evidence_literal(
            graph,
            resource,
            GENCO.sha256,
            sha256_file(path),
        )
        if generated:
            graph.add(
                (resource, PROV.wasGeneratedBy, activity)
            )
        else:
            graph.add((activity, PROV.used, resource))

    add_evidence_literal(
        graph,
        activity,
        GENCO.scenarioCount,
        trajectory_record["scenario_count"],
    )
    add_evidence_literal(
        graph,
        activity,
        GENCO.entityCount,
        trajectory_record["entity_count"],
    )
    add_evidence_literal(
        graph,
        activity,
        GENCO.trajectoryRowCount,
        trajectory_record["row_count"],
    )
    add_evidence_literal(
        graph,
        activity,
        GENCO.correctionStepCount,
        trajectory_record["correction_step_count"],
    )
    add_evidence_literal(
        graph,
        activity,
        GENCO.embeddingDimension,
        trajectory_record["embedding_dimension"],
    )
    add_evidence_literal(
        graph,
        activity,
        GENCO.embeddingStage,
        trajectory_record["embedding_stage"],
    )
    add_evidence_literal(
        graph,
        activity,
        GENCO.tolerancePolicyId,
        tolerance_record["policy_id"],
    )
    add_evidence_literal(
        graph,
        activity,
        GENCO.comparisonStatus,
        "passed",
    )
    add_evidence_literal(
        graph,
        activity,
        GENCO.maximumEmbeddingDifference,
        equivalence[
            "final_embedding_maximum_absolute_difference"
        ],
    )
    add_evidence_literal(
        graph,
        activity,
        GENCO.finalSnapshotExact,
        True,
    )
    return graph, observed


def write_trajectory_provenance(
    spec: TrajectoryProvenanceSpec,
) -> TrajectoryProvenanceResult:
    """Verify and serialize trajectory provenance."""
    graph, observed = build_trajectory_provenance_graph(
        spec
    )
    serialize_graph(graph, spec.output)
    return TrajectoryProvenanceResult(
        output=spec.output,
        triple_count=len(graph),
        scenario_count=observed["scenario_count"],
        entity_count=observed["entity_count"],
        row_count=observed["row_count"],
    )
