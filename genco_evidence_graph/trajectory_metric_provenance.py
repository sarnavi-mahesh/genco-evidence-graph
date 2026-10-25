"""Verified RDF provenance for derived latent trajectory metrics."""

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
from .trajectory_metrics import METRIC_POLICY_ID


@dataclass(frozen=True)
class TrajectoryMetricProvenanceSpec:
    metric_manifest: Path
    trajectory: Path
    trajectory_manifest: Path
    cohort: Path
    cohort_manifest: Path
    entity_metrics: Path
    step_metrics: Path
    scenario_summary: Path
    group_comparison: Path
    diagnostics: Path
    output: Path


@dataclass(frozen=True)
class TrajectoryMetricProvenanceResult:
    output: Path
    triple_count: int
    entity_count: int
    transition_count: int
    scenario_count: int


def _load_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _verify_record(
    value: dict[str, Any],
    key: str,
    path: Path,
    label: str,
) -> dict[str, Any]:
    record = value.get(key)
    if not isinstance(record, dict):
        raise ValueError(f"{label} lacks {key}")
    if Path(record.get("path", "")).resolve() != path.resolve():
        raise ValueError(f"{label} {key} path disagrees")
    if record.get("sha256") != sha256_file(path):
        raise ValueError(f"{label} {key} hash disagrees")
    return record


def build_trajectory_metric_provenance_graph(
    spec: TrajectoryMetricProvenanceSpec,
) -> tuple[Graph, dict[str, int]]:
    """Verify a trajectory metric bundle and build its PROV-O graph."""
    manifest = _load_object(
        spec.metric_manifest,
        "Trajectory metric manifest",
    )
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_type")
        != "latent_trajectory_metric_bundle"
        or manifest.get("metric_policy_id")
        != METRIC_POLICY_ID
    ):
        raise ValueError(
            "Unsupported trajectory metric manifest"
        )
    inputs = (
        ("trajectory", spec.trajectory),
        ("trajectory_manifest", spec.trajectory_manifest),
        ("cohort", spec.cohort),
        ("cohort_manifest", spec.cohort_manifest),
    )
    for key, path in inputs:
        _verify_record(manifest, key, path, "Metric manifest")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError(
            "Trajectory metric manifest lacks outputs"
        )
    output_specs = (
        ("entity_metrics", spec.entity_metrics),
        ("step_metrics", spec.step_metrics),
        ("scenario_summary", spec.scenario_summary),
        ("group_comparison", spec.group_comparison),
        ("diagnostics", spec.diagnostics),
    )
    records = {
        key: _verify_record(
            outputs,
            key,
            path,
            "Metric manifest outputs",
        )
        for key, path in output_specs
    }
    observed = {
        "entity_count": len(
            pd.read_parquet(spec.entity_metrics)
        ),
        "transition_count": len(
            pd.read_parquet(spec.step_metrics)
        ),
        "scenario_count": pd.read_parquet(
            spec.scenario_summary,
            columns=["scenario_id"],
        )["scenario_id"].nunique(),
    }
    if records["entity_metrics"].get(
        "row_count"
    ) != observed["entity_count"]:
        raise ValueError(
            "Entity metric row count disagrees"
        )
    if records["step_metrics"].get(
        "row_count"
    ) != observed["transition_count"]:
        raise ValueError(
            "Step metric row count disagrees"
        )

    graph = Graph()
    graph.bind("genco", GENCO)
    graph.bind("prov", PROV)
    activity = evidence_resource(
        "trajectory-metric-computation",
        sha256_file(spec.metric_manifest),
    )
    graph.add((activity, RDF.type, PROV.Activity))
    graph.add(
        (
            activity,
            RDF.type,
            GENCO.TrajectoryMetricComputation,
        )
    )
    input_classes = (
        (spec.trajectory, GENCO.LatentTrajectoryArtifact),
        (
            spec.trajectory_manifest,
            GENCO.TrajectoryCaptureManifest,
        ),
        (spec.cohort, GENCO.ContrastingScenarioCohort),
        (spec.cohort_manifest, GENCO.CohortManifest),
    )
    output_classes = (
        (
            spec.entity_metrics,
            GENCO.EntityTrajectoryMetricTable,
        ),
        (
            spec.step_metrics,
            GENCO.StepTrajectoryMetricTable,
        ),
        (
            spec.scenario_summary,
            GENCO.ScenarioTrajectorySummary,
        ),
        (
            spec.group_comparison,
            GENCO.CohortGroupTrajectoryComparison,
        ),
        (
            spec.diagnostics,
            GENCO.TrajectoryMetricDiagnostics,
        ),
        (
            spec.metric_manifest,
            GENCO.TrajectoryMetricManifest,
        ),
    )
    for path, artifact_class in (
        *input_classes,
        *output_classes,
    ):
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
        if (path, artifact_class) in input_classes:
            graph.add((activity, PROV.used, resource))
        else:
            graph.add(
                (resource, PROV.wasGeneratedBy, activity)
            )
    policy = manifest["policy"]
    add_evidence_literal(
        graph,
        activity,
        GENCO.metricPolicyId,
        METRIC_POLICY_ID,
    )
    add_evidence_literal(
        graph,
        activity,
        GENCO.entityCount,
        observed["entity_count"],
    )
    add_evidence_literal(
        graph,
        activity,
        GENCO.stepTransitionCount,
        observed["transition_count"],
    )
    add_evidence_literal(
        graph,
        activity,
        GENCO.scenarioCount,
        observed["scenario_count"],
    )
    add_evidence_literal(
        graph,
        activity,
        GENCO.distanceMetric,
        policy["distance_metric"],
    )
    add_evidence_literal(
        graph,
        activity,
        GENCO.localDirectionMetric,
        policy["local_directional_consistency"],
    )
    add_evidence_literal(
        graph,
        activity,
        GENCO.globalDirectionMetric,
        policy["global_directional_alignment"],
    )
    return graph, observed


def write_trajectory_metric_provenance(
    spec: TrajectoryMetricProvenanceSpec,
) -> TrajectoryMetricProvenanceResult:
    """Verify and serialize trajectory metric provenance."""
    graph, observed = (
        build_trajectory_metric_provenance_graph(spec)
    )
    serialize_graph(graph, spec.output)
    return TrajectoryMetricProvenanceResult(
        output=spec.output,
        triple_count=len(graph),
        entity_count=observed["entity_count"],
        transition_count=observed["transition_count"],
        scenario_count=observed["scenario_count"],
    )
