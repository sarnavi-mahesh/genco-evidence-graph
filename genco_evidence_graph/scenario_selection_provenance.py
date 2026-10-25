"""Verified RDF provenance for frozen contrasting scenario cohorts."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
import re
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


_GROUPS = (
    "familiar_good",
    "familiar_bad",
    "unfamiliar_good",
    "unfamiliar_bad",
)

_SCENARIO_FIELDS = (
    "scenario_id",
    "cohort_group",
    "selection_reason",
    "selection_rank",
    "nearest_reference_scenario_id",
    "nearest_reference_distance",
    "mean_absolute_pbe",
    "max_absolute_pbe",
    "max_pbe_bus",
    "distance_percentile",
    "residual_percentile",
    "max_pbe_percentile",
    "corner_score",
    "boundary_score",
)

_TEXT_FIELDS = {
    "scenario_id",
    "cohort_group",
    "selection_reason",
    "nearest_reference_scenario_id",
}

_INTEGER_FIELDS = {
    "selection_rank",
    "max_pbe_bus",
}


@dataclass(frozen=True)
class CohortProvenanceSpec:
    scores: Path
    score_manifest: Path
    summary: Path
    cohort: Path
    cohort_manifest: Path
    diagnostics: Path
    training_reference_ids: Path
    output: Path

    def __post_init__(self) -> None:
        paths = (
            self.scores,
            self.score_manifest,
            self.summary,
            self.cohort,
            self.cohort_manifest,
            self.diagnostics,
            self.training_reference_ids,
            self.output,
        )
        if len(set(paths)) != len(paths):
            raise ValueError(
                "Cohort provenance inputs and output must all differ"
            )


@dataclass(frozen=True)
class VerifiedCohortProvenance:
    scoring_policy_id: str
    selection_policy_id: str
    checkpoint_sha256: str
    dataset_revision: str
    graphkit_commit: str
    target_run_id: str
    reference_run_id: str
    selected_scenarios: tuple[dict[str, Any], ...]
    group_candidate_counts: dict[str, int]


@dataclass(frozen=True)
class CohortProvenanceResult:
    output: Path
    triple_count: int
    selected_scenario_count: int
    selected_reference_count: int


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _require_value(
    value: dict[str, Any],
    key: str,
    expected: Any,
    label: str,
) -> None:
    actual = value.get(key)
    if actual != expected:
        raise ValueError(
            f"{label} has invalid {key}: expected={expected!r}, "
            f"actual={actual!r}"
        )


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[0-9a-f]{64}", value) is None
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_revision(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[0-9a-f]{40}", value) is None
    ):
        raise ValueError(f"{label} must be a lowercase 40-character revision")
    return value


def _require_finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _verify_artifact_record(
    value: dict[str, Any],
    key: str,
    expected_path: Path,
    label: str,
) -> dict[str, str]:
    record = value.get(key)
    if not isinstance(record, dict):
        raise ValueError(f"{label} lacks an object record for {key}")
    if set(record) != {"path", "sha256"}:
        raise ValueError(
            f"{label} {key} record must contain only path and sha256"
        )
    recorded_path = record.get("path")
    recorded_hash = record.get("sha256")
    if not isinstance(recorded_path, str) or not recorded_path:
        raise ValueError(f"{label} {key} path must be non-empty")
    _require_sha256(recorded_hash, f"{label} {key} sha256")
    if Path(recorded_path).resolve() != expected_path.resolve():
        raise ValueError(
            f"{label} {key} path disagrees with the explicit input"
        )
    actual_hash = sha256_file(expected_path)
    if recorded_hash != actual_hash:
        raise ValueError(
            f"{label} {key} hash disagrees with {expected_path}"
        )
    return {
        "path": recorded_path,
        "sha256": actual_hash,
    }


def _normalize_scenario_record(
    value: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    if set(value) != set(_SCENARIO_FIELDS):
        missing = sorted(set(_SCENARIO_FIELDS) - set(value))
        extra = sorted(set(value) - set(_SCENARIO_FIELDS))
        raise ValueError(
            f"{label} has invalid fields: missing={missing}, extra={extra}"
        )

    normalized: dict[str, Any] = {}
    for field in _SCENARIO_FIELDS:
        raw = value[field]
        if field in _TEXT_FIELDS:
            if not isinstance(raw, (str, int)) or str(raw) == "":
                raise ValueError(f"{label} {field} must be non-empty")
            normalized[field] = str(raw)
        elif field in _INTEGER_FIELDS:
            if isinstance(raw, bool):
                raise ValueError(f"{label} {field} must be an integer")
            integer = int(raw)
            if float(raw) != integer:
                raise ValueError(f"{label} {field} must be an integer")
            normalized[field] = integer
        else:
            number = float(raw)
            if not math.isfinite(number):
                raise ValueError(f"{label} {field} must be finite")
            normalized[field] = number
    return normalized


def _cohort_records(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.is_file():
        raise FileNotFoundError(f"Cohort artifact does not exist: {path}")
    table = pd.read_parquet(path)
    missing = sorted(set(_SCENARIO_FIELDS) - set(table.columns))
    if missing:
        raise ValueError(f"Cohort artifact lacks columns: {missing}")
    records = table[list(_SCENARIO_FIELDS)].to_dict(orient="records")
    return tuple(
        _normalize_scenario_record(record, f"Cohort row {index}")
        for index, record in enumerate(records)
    )


def _reference_ids(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Training reference ID artifact does not exist: {path}"
        )
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["scenario_id"]:
            raise ValueError(
                "Training reference ID artifact must have only "
                "a scenario_id column"
            )
        rows = list(reader)
    values = tuple(str(row["scenario_id"]) for row in rows)
    if any(not value or not value.isdigit() for value in values):
        raise ValueError(
            "Training reference IDs must be non-negative integer strings"
        )
    if len(values) != len(set(values)):
        raise ValueError("Training reference IDs must be deduplicated")
    if values != tuple(sorted(values, key=int)):
        raise ValueError("Training reference IDs must be numerically sorted")
    return values


def verify_cohort_provenance_inputs(
    spec: CohortProvenanceSpec,
) -> VerifiedCohortProvenance:
    """Verify the complete frozen cohort chain before RDF construction."""
    score_manifest = _load_json_object(
        spec.score_manifest,
        "Score manifest",
    )
    summary = _load_json_object(spec.summary, "Distribution summary")
    diagnostics = _load_json_object(
        spec.diagnostics,
        "Selection diagnostics",
    )
    cohort_manifest = _load_json_object(
        spec.cohort_manifest,
        "Cohort manifest",
    )

    _require_value(score_manifest, "schema_version", 1, "Score manifest")
    _require_value(
        score_manifest,
        "artifact_type",
        "scenario_scoring_table",
        "Score manifest",
    )
    _require_value(summary, "schema_version", 1, "Distribution summary")
    _require_value(
        summary,
        "artifact_type",
        "scenario_score_distribution_summary",
        "Distribution summary",
    )
    _require_value(diagnostics, "schema_version", 1, "Selection diagnostics")
    _require_value(
        diagnostics,
        "artifact_type",
        "contrasting_scenario_selection_diagnostics",
        "Selection diagnostics",
    )
    _require_value(diagnostics, "status", "selected", "Selection diagnostics")
    _require_value(cohort_manifest, "schema_version", 1, "Cohort manifest")
    _require_value(
        cohort_manifest,
        "artifact_type",
        "frozen_contrasting_scenario_cohort",
        "Cohort manifest",
    )

    _verify_artifact_record(
        score_manifest,
        "output",
        spec.scores,
        "Score manifest",
    )
    _verify_artifact_record(
        summary,
        "score_table",
        spec.scores,
        "Distribution summary",
    )
    _verify_artifact_record(
        summary,
        "score_manifest",
        spec.score_manifest,
        "Distribution summary",
    )
    _verify_artifact_record(
        diagnostics,
        "score_table",
        spec.scores,
        "Selection diagnostics",
    )
    _verify_artifact_record(
        diagnostics,
        "score_manifest",
        spec.score_manifest,
        "Selection diagnostics",
    )
    _verify_artifact_record(
        diagnostics,
        "distribution_summary",
        spec.summary,
        "Selection diagnostics",
    )
    for key, path in (
        ("score_table", spec.scores),
        ("score_manifest", spec.score_manifest),
        ("distribution_summary", spec.summary),
        ("diagnostics", spec.diagnostics),
        ("cohort_artifact", spec.cohort),
        ("training_reference_ids", spec.training_reference_ids),
    ):
        _verify_artifact_record(
            cohort_manifest,
            key,
            path,
            "Cohort manifest",
        )

    scoring_policy_id = score_manifest.get("scoring_policy_id")
    selection_policy_id = cohort_manifest.get("selection_policy_id")
    if not isinstance(scoring_policy_id, str) or not scoring_policy_id:
        raise ValueError("Score manifest lacks a scoring policy ID")
    if not isinstance(selection_policy_id, str) or not selection_policy_id:
        raise ValueError("Cohort manifest lacks a selection policy ID")
    for value, label in (
        (summary, "Distribution summary"),
        (cohort_manifest, "Cohort manifest"),
    ):
        _require_value(
            value,
            "scoring_policy_id",
            scoring_policy_id,
            label,
        )
    _require_value(
        diagnostics,
        "selection_policy_id",
        selection_policy_id,
        "Selection diagnostics",
    )

    inputs = score_manifest.get("inputs")
    if not isinstance(inputs, dict) or len(inputs) != 7:
        raise ValueError(
            "Score manifest inputs must contain seven artifact records"
        )
    for name, record in inputs.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Score manifest input names must be non-empty")
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError(
                f"Score manifest input {name!r} must contain path and sha256"
            )
        if not isinstance(record["path"], str) or not record["path"]:
            raise ValueError(
                f"Score manifest input {name!r} path must be non-empty"
            )
        _require_sha256(
            record["sha256"],
            f"Score manifest input {name!r} sha256",
        )

    required_scoring_text = (
        "distance_metric",
        "pooling_method",
        "residual_metric",
        "numeric_dtype",
    )
    for key in required_scoring_text:
        if (
            not isinstance(score_manifest.get(key), str)
            or not score_manifest[key]
        ):
            raise ValueError(f"Score manifest has invalid {key}")
    for key in (
        "target_scenario_count",
        "reference_scenario_count",
        "entity_embedding_dimension",
        "pooled_embedding_dimension",
    ):
        value = score_manifest.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Score manifest has invalid {key}")

    for key in (
        "scenario_count",
        "assigned_scenario_count",
        "excluded_scenario_count",
    ):
        value = summary.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Distribution summary has invalid {key}")
    if summary["scenario_count"] <= 0:
        raise ValueError(
            "Distribution summary scenario_count must be positive"
        )
    if (
        summary["assigned_scenario_count"]
        + summary["excluded_scenario_count"]
        != summary["scenario_count"]
    ):
        raise ValueError(
            "Distribution summary assigned and excluded counts "
            "must equal scenario_count"
        )
    if (
        not isinstance(summary.get("quantile_method"), str)
        or not summary["quantile_method"]
    ):
        raise ValueError("Distribution summary has invalid quantile_method")
    _require_finite_number(
        summary.get("spearman_distance_vs_mean_absolute_pbe"),
        "Distribution summary Spearman correlation",
    )
    percentile_policy = summary.get("percentile_policy")
    if percentile_policy != {
        "rank_method": "average",
        "low_max": 0.25,
        "high_min": 0.75,
    }:
        raise ValueError(
            "Distribution summary percentile_policy disagrees "
            "with the selection policy"
        )

    threshold_policy = cohort_manifest.get("threshold_policy")
    required_thresholds = (
        "low_percentile",
        "high_percentile",
        "low_distance_max",
        "high_distance_min",
        "low_residual_max",
        "high_residual_min",
    )
    if (
        not isinstance(threshold_policy, dict)
        or set(threshold_policy) != set(required_thresholds)
    ):
        raise ValueError("Cohort manifest has invalid threshold_policy")
    for key in required_thresholds:
        _require_finite_number(
            threshold_policy[key],
            f"Cohort manifest threshold_policy {key}",
        )
    if (
        float(threshold_policy["low_percentile"]) != 0.25
        or float(threshold_policy["high_percentile"]) != 0.75
    ):
        raise ValueError(
            "Cohort manifest percentile thresholds disagree "
            "with the selection policy"
        )

    selection_policy = cohort_manifest.get("selection_policy")
    expected_policy_values = {
        "per_group": 5,
        "corner_representatives": 3,
        "inner_boundary_representatives": 1,
        "physical_diversity_representatives": 1,
        "corner_metric": (
            "euclidean_percentile_corner_distance"
        ),
        "boundary_metric": (
            "euclidean_percentile_inner_boundary_distance"
        ),
        "physical_diversity_priority": [
            "novel_max_pbe_bus",
            "novel_nearest_reference",
            "max_pbe_percentile_separation",
            "corner_score",
            "numeric_scenario_id",
        ],
    }
    if selection_policy != expected_policy_values:
        raise ValueError(
            "Cohort manifest selection_policy disagrees "
            "with the declared policy"
        )

    raw_manifest_records = cohort_manifest.get("selected_scenarios")
    if not isinstance(raw_manifest_records, list):
        raise ValueError("Cohort manifest selected_scenarios must be a list")
    manifest_records = tuple(
        _normalize_scenario_record(
            record,
            f"Cohort manifest scenario {index}",
        )
        for index, record in enumerate(raw_manifest_records)
        if isinstance(record, dict)
    )
    if len(manifest_records) != len(raw_manifest_records):
        raise ValueError(
            "Every cohort manifest selected_scenarios entry must be an object"
        )

    cohort_records = _cohort_records(spec.cohort)
    if manifest_records != cohort_records:
        raise ValueError(
            "Cohort manifest scenario records disagree with the cohort artifact"
        )

    selected_ids = [record["scenario_id"] for record in cohort_records]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Cohort contains duplicate selected scenario IDs")
    if len(cohort_records) != 20:
        raise ValueError("Cohort must contain exactly 20 selected scenarios")

    for group in _GROUPS:
        records = [
            record
            for record in cohort_records
            if record["cohort_group"] == group
        ]
        if len(records) != 5:
            raise ValueError(f"Cohort group {group!r} must contain 5 scenarios")
        if sorted(record["selection_rank"] for record in records) != [
            1,
            2,
            3,
            4,
            5,
        ]:
            raise ValueError(
                f"Cohort group {group!r} must have selection ranks 1 through 5"
            )
        reasons = {
            reason: sum(
                record["selection_reason"] == reason
                for record in records
            )
            for reason in ("corner", "inner_boundary", "physical_diversity")
        }
        if reasons != {
            "corner": 3,
            "inner_boundary": 1,
            "physical_diversity": 1,
        }:
            raise ValueError(
                f"Cohort group {group!r} has invalid selection composition"
            )

    reference_ids = _reference_ids(spec.training_reference_ids)
    expected_reference_ids = tuple(
        sorted(
            {
                record["nearest_reference_scenario_id"]
                for record in cohort_records
            },
            key=int,
        )
    )
    if reference_ids != expected_reference_ids:
        raise ValueError(
            "Training reference IDs disagree with the selected scenarios"
        )

    _require_value(
        cohort_manifest,
        "selected_test_scenario_count",
        len(cohort_records),
        "Cohort manifest",
    )
    _require_value(
        cohort_manifest,
        "selected_training_reference_count",
        len(reference_ids),
        "Cohort manifest",
    )
    _require_value(
        diagnostics,
        "selected_scenario_count",
        len(cohort_records),
        "Selection diagnostics",
    )
    _require_value(
        diagnostics,
        "selected_training_reference_count",
        len(reference_ids),
        "Selection diagnostics",
    )

    group_counts = cohort_manifest.get("group_candidate_counts")
    if (
        not isinstance(group_counts, dict)
        or set(group_counts) != set(_GROUPS)
        or any(
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 5
            for count in group_counts.values()
        )
    ):
        raise ValueError("Cohort manifest has invalid group candidate counts")
    _require_value(
        summary,
        "group_candidate_counts",
        group_counts,
        "Distribution summary",
    )
    _require_value(
        diagnostics,
        "group_candidate_counts",
        group_counts,
        "Selection diagnostics",
    )

    identity = score_manifest.get("verified_identity")
    if not isinstance(identity, dict):
        raise ValueError("Score manifest lacks verified_identity")
    required_identity = (
        "checkpoint_sha256",
        "dataset_revision",
        "graphkit_commit",
        "target_run_id",
        "reference_run_id",
    )
    if any(
        not isinstance(identity.get(key), str) or not identity[key]
        for key in required_identity
    ):
        raise ValueError("Score manifest has incomplete verified_identity")
    _require_sha256(
        identity["checkpoint_sha256"],
        "Score manifest checkpoint_sha256",
    )
    _require_revision(
        identity["dataset_revision"],
        "Score manifest dataset_revision",
    )
    _require_revision(
        identity["graphkit_commit"],
        "Score manifest graphkit_commit",
    )

    return VerifiedCohortProvenance(
        scoring_policy_id=scoring_policy_id,
        selection_policy_id=selection_policy_id,
        checkpoint_sha256=identity["checkpoint_sha256"],
        dataset_revision=identity["dataset_revision"],
        graphkit_commit=identity["graphkit_commit"],
        target_run_id=identity["target_run_id"],
        reference_run_id=identity["reference_run_id"],
        selected_scenarios=cohort_records,
        group_candidate_counts={
            key: int(group_counts[key])
            for key in _GROUPS
        },
    )


def _artifact(
    graph: Graph,
    path: Path,
    artifact_class,
    *,
    generated_by=None,
):
    resource = evidence_resource("artifact", str(path))
    graph.add((resource, RDF.type, PROV.Entity))
    graph.add((resource, RDF.type, GENCO.Artifact))
    graph.add((resource, RDF.type, artifact_class))
    add_evidence_literal(graph, resource, GENCO.artifactPath, str(path))
    add_evidence_literal(graph, resource, GENCO.sha256, sha256_file(path))
    if generated_by is not None:
        graph.add((resource, PROV.wasGeneratedBy, generated_by))
    return resource


def build_cohort_provenance_graph(
    spec: CohortProvenanceSpec,
) -> tuple[Graph, VerifiedCohortProvenance]:
    """Build cohort provenance using the shared GENCO RDF conventions."""
    verified = verify_cohort_provenance_inputs(spec)
    score_manifest_value = _load_json_object(
        spec.score_manifest,
        "Score manifest",
    )
    cohort_manifest_value = _load_json_object(
        spec.cohort_manifest,
        "Cohort manifest",
    )

    graph = Graph()
    graph.bind("genco", GENCO)
    graph.bind("prov", PROV)

    scoring = evidence_resource(
        "scenario-scoring-run",
        sha256_file(spec.score_manifest),
    )
    summarization = evidence_resource(
        "score-summarization-run",
        sha256_file(spec.summary),
    )
    selection = evidence_resource(
        "cohort-selection-run",
        sha256_file(spec.cohort_manifest),
    )
    for activity, activity_class in (
        (scoring, GENCO.ScenarioScoringRun),
        (summarization, GENCO.ScoreSummarizationRun),
        (selection, GENCO.CohortSelectionRun),
    ):
        graph.add((activity, RDF.type, PROV.Activity))
        graph.add((activity, RDF.type, activity_class))

    score_table = _artifact(
        graph,
        spec.scores,
        GENCO.ScenarioScoreTable,
        generated_by=scoring,
    )
    score_manifest = _artifact(
        graph,
        spec.score_manifest,
        GENCO.ScenarioScoreManifest,
        generated_by=scoring,
    )
    summary = _artifact(
        graph,
        spec.summary,
        GENCO.ScoreDistributionSummary,
        generated_by=summarization,
    )
    cohort = _artifact(
        graph,
        spec.cohort,
        GENCO.ContrastingScenarioCohort,
        generated_by=selection,
    )
    cohort_manifest = _artifact(
        graph,
        spec.cohort_manifest,
        GENCO.CohortManifest,
        generated_by=selection,
    )
    diagnostics = _artifact(
        graph,
        spec.diagnostics,
        GENCO.CohortSelectionDiagnostics,
        generated_by=selection,
    )
    reference_ids = _artifact(
        graph,
        spec.training_reference_ids,
        GENCO.TrainingReferenceIdSet,
        generated_by=selection,
    )

    for used in (score_table, score_manifest):
        graph.add((summarization, PROV.used, used))
    for used in (score_table, score_manifest, summary):
        graph.add((selection, PROV.used, used))

    for record in score_manifest_value["inputs"].values():
        source_path = str(record["path"])
        source = evidence_resource("artifact", source_path)
        graph.add((source, RDF.type, PROV.Entity))
        graph.add((source, RDF.type, GENCO.Artifact))
        graph.add((source, RDF.type, GENCO.ScenarioScoringInputArtifact))
        add_evidence_literal(graph, source, GENCO.artifactPath, source_path)
        add_evidence_literal(graph, source, GENCO.sha256, record["sha256"])
        graph.add((scoring, PROV.used, source))

    add_evidence_literal(
        graph,
        scoring,
        GENCO.scoringPolicyId,
        verified.scoring_policy_id,
    )
    scoring_predicates = {
        "distance_metric": GENCO.distanceMetric,
        "pooling_method": GENCO.poolingMethod,
        "residual_metric": GENCO.residualMetric,
        "numeric_dtype": GENCO.numericDtype,
        "target_scenario_count": GENCO.targetScenarioCount,
        "reference_scenario_count": GENCO.referenceScenarioCount,
        "entity_embedding_dimension": GENCO.entityEmbeddingDimension,
        "pooled_embedding_dimension": GENCO.pooledEmbeddingDimension,
    }
    for key, predicate in scoring_predicates.items():
        add_evidence_literal(
            graph,
            scoring,
            predicate,
            score_manifest_value[key],
        )
    add_evidence_literal(
        graph,
        scoring,
        GENCO.checkpointHash,
        verified.checkpoint_sha256,
    )
    add_evidence_literal(
        graph,
        scoring,
        GENCO.datasetRevision,
        verified.dataset_revision,
    )
    add_evidence_literal(
        graph,
        scoring,
        GENCO.graphkitCommit,
        verified.graphkit_commit,
    )
    add_evidence_literal(
        graph,
        scoring,
        GENCO.targetRunId,
        verified.target_run_id,
    )
    add_evidence_literal(
        graph,
        scoring,
        GENCO.referenceRunId,
        verified.reference_run_id,
    )
    add_evidence_literal(
        graph,
        summarization,
        GENCO.scoringPolicyId,
        verified.scoring_policy_id,
    )
    summary_value = _load_json_object(
        spec.summary,
        "Distribution summary",
    )
    summary_predicates = {
        "scenario_count": GENCO.scenarioCount,
        "assigned_scenario_count": GENCO.assignedScenarioCount,
        "excluded_scenario_count": GENCO.excludedScenarioCount,
        "quantile_method": GENCO.quantileMethod,
        "spearman_distance_vs_mean_absolute_pbe": (
            GENCO.spearmanDistanceVsMeanAbsolutePbe
        ),
    }
    for key, predicate in summary_predicates.items():
        add_evidence_literal(
            graph,
            summarization,
            predicate,
            summary_value[key],
        )
    add_evidence_literal(
        graph,
        summarization,
        GENCO.percentileRankMethod,
        summary_value["percentile_policy"]["rank_method"],
    )
    add_evidence_literal(
        graph,
        selection,
        GENCO.selectionPolicyId,
        verified.selection_policy_id,
    )
    selection_policy = cohort_manifest_value["selection_policy"]
    selection_predicates = {
        "per_group": GENCO.representativesPerGroup,
        "corner_representatives": GENCO.cornerRepresentativeCount,
        "inner_boundary_representatives": (
            GENCO.innerBoundaryRepresentativeCount
        ),
        "physical_diversity_representatives": (
            GENCO.physicalDiversityRepresentativeCount
        ),
        "corner_metric": GENCO.cornerMetric,
        "boundary_metric": GENCO.boundaryMetric,
    }
    for key, predicate in selection_predicates.items():
        add_evidence_literal(
            graph,
            selection,
            predicate,
            selection_policy[key],
        )
    for priority in selection_policy["physical_diversity_priority"]:
        add_evidence_literal(
            graph,
            selection,
            GENCO.physicalDiversityPriority,
            priority,
        )

    threshold_policy = cohort_manifest_value["threshold_policy"]
    for key, predicate in (
        ("low_percentile", GENCO.lowPercentileThreshold),
        ("high_percentile", GENCO.highPercentileThreshold),
        ("low_distance_max", GENCO.lowDistanceMaximum),
        ("high_distance_min", GENCO.highDistanceMinimum),
        ("low_residual_max", GENCO.lowResidualMaximum),
        ("high_residual_min", GENCO.highResidualMinimum),
    ):
        add_evidence_literal(graph, selection, predicate, threshold_policy[key])

    group_count_predicates = {
        "familiar_good": GENCO.familiarGoodCandidateCount,
        "familiar_bad": GENCO.familiarBadCandidateCount,
        "unfamiliar_good": GENCO.unfamiliarGoodCandidateCount,
        "unfamiliar_bad": GENCO.unfamiliarBadCandidateCount,
    }
    for group, count in verified.group_candidate_counts.items():
        add_evidence_literal(
            graph,
            selection,
            group_count_predicates[group],
            count,
        )

    cohort_manifest_hash = sha256_file(spec.cohort_manifest)
    for record in verified.selected_scenarios:
        scenario_id = record["scenario_id"]
        reference_id = record["nearest_reference_scenario_id"]
        scenario = evidence_resource("scenario", scenario_id)
        reference = evidence_resource("scenario", reference_id)
        membership = evidence_resource(
            "cohort-membership",
            f"{cohort_manifest_hash}:{scenario_id}",
        )

        graph.add((scenario, RDF.type, PROV.Entity))
        graph.add((scenario, RDF.type, GENCO.Scenario))
        graph.add((scenario, RDF.type, GENCO.SelectedTestScenario))
        graph.add((reference, RDF.type, PROV.Entity))
        graph.add((reference, RDF.type, GENCO.Scenario))
        graph.add((reference, RDF.type, GENCO.TrainingReferenceScenario))
        graph.add((membership, RDF.type, PROV.Entity))
        graph.add((membership, RDF.type, GENCO.CohortMembership))
        graph.add((membership, PROV.wasGeneratedBy, selection))
        graph.add((cohort, PROV.hadMember, scenario))
        graph.add((cohort, GENCO.hasCohortMembership, membership))
        graph.add((membership, GENCO.selectedScenario, scenario))
        graph.add((membership, GENCO.nearestTrainingReference, reference))

        add_evidence_literal(graph, scenario, GENCO.scenarioId, scenario_id)
        add_evidence_literal(graph, scenario, GENCO.datasetSplit, "test")
        add_evidence_literal(graph, reference, GENCO.scenarioId, reference_id)
        add_evidence_literal(graph, reference, GENCO.datasetSplit, "train")

        predicates = {
            "cohort_group": GENCO.cohortGroup,
            "selection_reason": GENCO.selectionReason,
            "selection_rank": GENCO.selectionRank,
            "nearest_reference_distance": GENCO.nearestReferenceDistance,
            "mean_absolute_pbe": GENCO.meanAbsolutePbe,
            "max_absolute_pbe": GENCO.maxAbsolutePbe,
            "max_pbe_bus": GENCO.maxPbeBus,
            "distance_percentile": GENCO.distancePercentile,
            "residual_percentile": GENCO.residualPercentile,
            "max_pbe_percentile": GENCO.maxPbePercentile,
            "corner_score": GENCO.cornerScore,
            "boundary_score": GENCO.boundaryScore,
        }
        for key, predicate in predicates.items():
            add_evidence_literal(graph, membership, predicate, record[key])

    for artifact in (
        cohort_manifest,
        diagnostics,
        reference_ids,
    ):
        graph.add((cohort, GENCO.hasProvenanceArtifact, artifact))

    return graph, verified


def write_cohort_provenance(
    spec: CohortProvenanceSpec,
) -> CohortProvenanceResult:
    """Verify, build, and serialize one cohort provenance graph."""
    graph, verified = build_cohort_provenance_graph(spec)
    serialize_graph(graph, spec.output)
    references = {
        record["nearest_reference_scenario_id"]
        for record in verified.selected_scenarios
    }
    return CohortProvenanceResult(
        output=spec.output,
        triple_count=len(graph),
        selected_scenario_count=len(verified.selected_scenarios),
        selected_reference_count=len(references),
    )
