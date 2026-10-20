"""Typed contracts used at the evidence-graph boundaries."""

from dataclasses import dataclass, field
import math
import re
from typing import Any

from .attestation import VerifiedTrainingAttestation


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _validate_finite(value: float | None, name: str) -> None:
    if value is not None and not math.isfinite(value):
        raise ValueError(f"{name} must be finite when provided")


def _require_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class DatasetProvenance:
    dataset_id: str
    grid_name: str
    task: str
    corpus_scenario_count: int | None = None
    dataset_split: str | None = None
    split_scenario_count: int | None = None
    perturbations: tuple[str, ...] = ()
    config_path: str | None = None
    config_hash: str | None = None
    source_artifacts: tuple[str, ...] = ()
    artifact_records: tuple[dict[str, str], ...] = ()
    dataset_revision: str | None = None
    software_version: str | None = None
    generation_method: str | None = None
    generation_seed: int | None = None
    generation_seed_status: str | None = None
    provenance_limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.dataset_id, "dataset_id")
        _require_text(self.grid_name, "grid_name")
        _require_text(self.task, "task")
        if self.corpus_scenario_count is not None and self.corpus_scenario_count < 0:
            raise ValueError("corpus_scenario_count cannot be negative")
        if self.split_scenario_count is not None and self.split_scenario_count < 0:
            raise ValueError("split_scenario_count cannot be negative")
        if self.generation_seed is not None and self.generation_seed < 0:
            raise ValueError("generation_seed cannot be negative")


@dataclass(frozen=True)
class TrainingEvidenceAssociation:
    """Non-causal evidence associated with a checkpoint publication."""

    association_status: str
    evidence_basis: tuple[str, ...]
    provenance_limitations: tuple[str, ...]
    production_identity_verified: bool = False
    configuration_compatible: bool = True

    def __post_init__(self) -> None:
        _require_text(self.association_status, "association_status")
        if not self.evidence_basis:
            raise ValueError("evidence_basis must not be empty")
        if not self.provenance_limitations:
            raise ValueError("provenance_limitations must not be empty")
        for value in self.evidence_basis:
            _require_text(value, "evidence_basis")
        for value in self.provenance_limitations:
            _require_text(value, "provenance_limitations")
        if self.production_identity_verified:
            raise ValueError(
                "associated training evidence cannot claim verified production identity",
            )
        if not self.configuration_compatible:
            raise ValueError(
                "associated training evidence must be configuration compatible",
            )


@dataclass(frozen=True)
class ModelProvenance:
    training_run_id: str
    checkpoint_id: str
    dataset_id: str
    model_variant: str | None = None
    hidden_dim: int | None = None
    correction_steps: int | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    checkpoint_hash: str | None = None
    normalizer_hash: str | None = None
    config_hash: str | None = None
    graphkit_commit: str | None = None
    mlflow_run_id: str | None = None
    training_dataset_id: str | None = None
    training_split_scenario_count: int | None = None
    training_split_hash: str | None = None
    training_start_time_ms: int | None = None
    training_end_time_ms: int | None = None
    training_seed: int | None = None
    training_epochs: int | None = None
    training_batch_size: int | None = None
    training_learning_rate: float | None = None
    training_optimizer: str | None = None
    training_selected_metrics: dict[str, float] = field(default_factory=dict)
    training_software_revision_status: str | None = None
    checkpoint_binding_status: str | None = None
    training_params_hash: str | None = None
    training_metrics_hash: str | None = None
    model_repository_revision: str | None = None
    checkpoint_repository_path: str | None = None
    training_evidence_association: TrainingEvidenceAssociation | None = None
    authoritative_training_attestation: VerifiedTrainingAttestation | None = None

    def __post_init__(self) -> None:
        _require_text(self.training_run_id, "training_run_id")
        _require_text(self.checkpoint_id, "checkpoint_id")
        _require_text(self.dataset_id, "dataset_id")
        if self.hidden_dim is not None and self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.correction_steps is not None and self.correction_steps < 0:
            raise ValueError("correction_steps cannot be negative")
        if (
            self.training_split_scenario_count is not None
            and self.training_split_scenario_count <= 0
        ):
            raise ValueError("training_split_scenario_count must be positive")
        if self.training_epochs is not None and self.training_epochs <= 0:
            raise ValueError("training_epochs must be positive")
        if self.training_batch_size is not None and self.training_batch_size <= 0:
            raise ValueError("training_batch_size must be positive")
        _validate_finite(
            self.training_learning_rate,
            "training_learning_rate",
        )
        if self.training_learning_rate is not None and self.training_learning_rate <= 0:
            raise ValueError("training_learning_rate must be positive")
        if self.training_optimizer is not None:
            _require_text(self.training_optimizer, "training_optimizer")
        if self.training_software_revision_status is not None:
            _require_text(
                self.training_software_revision_status,
                "training_software_revision_status",
            )
        if self.checkpoint_binding_status is not None:
            _require_text(
                self.checkpoint_binding_status,
                "checkpoint_binding_status",
            )
        if self.authoritative_training_attestation is not None:
            if not isinstance(
                self.authoritative_training_attestation,
                VerifiedTrainingAttestation,
            ):
                raise ValueError(
                    "authoritative training attestation must be cryptographically "
                    "verified",
                )
            if self.training_evidence_association is not None:
                raise ValueError(
                    "authoritative training attestation and unverified association "
                    "are mutually exclusive",
                )
            attestation = self.authoritative_training_attestation
            if self.checkpoint_hash != attestation.checkpoint_sha256:
                raise ValueError(
                    "authoritative training attestation checkpoint hash disagrees",
                )
            if self.checkpoint_id != attestation.checkpoint_id:
                raise ValueError(
                    "authoritative training attestation checkpoint ID disagrees",
                )
            if self.training_run_id != attestation.training_run_id:
                raise ValueError(
                    "authoritative training attestation run ID disagrees",
                )
        for name, value in self.metrics.items():
            _validate_finite(value, f"metrics[{name!r}]")
        for name, value in self.training_selected_metrics.items():
            _validate_finite(value, f"training_selected_metrics[{name!r}]")


@dataclass(frozen=True)
class EmbeddingBatch:
    scenario_ids: tuple[str, ...]
    bus_embeddings: Any
    gen_embeddings: Any | None = None
    correction_step: int = 0

    def __post_init__(self) -> None:
        _require_text(self.scenario_ids[0], "scenario_ids[0]") if self.scenario_ids else None
        if self.correction_step < 0:
            raise ValueError("correction_step cannot be negative")
        if getattr(self.bus_embeddings, "ndim", None) != 3:
            raise ValueError("bus_embeddings must have shape [scenario, entity, dimension]")
        if self.bus_embeddings.shape[0] != len(self.scenario_ids):
            raise ValueError("scenario_ids and bus_embeddings are not aligned")
        if self.gen_embeddings is not None:
            if getattr(self.gen_embeddings, "ndim", None) != 3:
                raise ValueError(
                    "gen_embeddings must have shape [scenario, entity, dimension]",
                )
            if self.gen_embeddings.shape[0] != len(self.scenario_ids):
                raise ValueError("scenario_ids and gen_embeddings are not aligned")


@dataclass(frozen=True)
class MetricSemantics:
    metric_name: str
    unit: str
    treatment: str
    aggregation: str
    population: str
    comparison_tolerance: str = "not applicable"

    def __post_init__(self) -> None:
        for name in ("metric_name", "unit", "treatment", "aggregation", "population"):
            _require_text(getattr(self, name), name)


@dataclass(frozen=True)
class PhysicalEvidence:
    mean_residual: float | None = None
    max_residual: float | None = None
    active_residual: float | None = None
    reactive_residual: float | None = None
    violation_count: int | None = None
    metric_semantics: tuple[MetricSemantics, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "mean_residual",
            "max_residual",
            "active_residual",
            "reactive_residual",
        ):
            _validate_finite(getattr(self, name), name)
        if self.violation_count is not None and self.violation_count < 0:
            raise ValueError("violation_count cannot be negative")
        names = [semantics.metric_name for semantics in self.metric_semantics]
        if len(names) != len(set(names)):
            raise ValueError("metric_semantics metric names must be unique")


@dataclass(frozen=True)
class PredictionEvidence:
    prediction_id: str
    provenance_summary: dict[str, Any]
    reference_cases: tuple[dict[str, Any], ...]
    reference_distance: float | None
    reference_distance_percentile: float | None
    physical_evidence: PhysicalEvidence
    verification: dict[str, Any] | None = None
    inference_run_id: str | None = None
    inference_start_time_ms: int | None = None
    inference_end_time_ms: int | None = None
    inference_graphkit_commit: str | None = None
    reference_extraction_run_id: str | None = None
    reference_extraction_start_time_ms: int | None = None
    reference_extraction_end_time_ms: int | None = None
    reference_extraction_graphkit_commit: str | None = None
    reproduction_run_id: str | None = None
    reproduction_start_time_ms: int | None = None
    reproduction_end_time_ms: int | None = None
    reproduction_graphkit_commit: str | None = None
    reproduction_environment_hash: str | None = None
    reproduction_runtime: dict[str, str] = field(default_factory=dict)
    reproduction_comparison_status: str | None = None
    reproduction_tolerance_policy_id: str | None = None
    reproduction_tolerance_policy_hash: str | None = None
    reproduction_maximum_voltage_difference: float | None = None
    reproduction_maximum_power_difference: float | None = None

    def __post_init__(self) -> None:
        _require_text(self.prediction_id, "prediction_id")
        if self.inference_run_id is not None:
            _require_text(self.inference_run_id, "inference_run_id")
        if self.reproduction_run_id is not None:
            _require_text(self.reproduction_run_id, "reproduction_run_id")
        if self.reproduction_comparison_status is not None:
            _require_text(
                self.reproduction_comparison_status,
                "reproduction_comparison_status",
            )
            if self.reproduction_comparison_status != "passed":
                raise ValueError(
                    "reproduction_comparison_status must be 'passed' when provided",
                )
        if self.reproduction_tolerance_policy_id is not None:
            _require_text(
                self.reproduction_tolerance_policy_id,
                "reproduction_tolerance_policy_id",
            )
        if self.reproduction_tolerance_policy_hash is not None:
            _require_sha256(
                self.reproduction_tolerance_policy_hash,
                "reproduction_tolerance_policy_hash",
            )
        _validate_finite(self.reference_distance, "reference_distance")
        _validate_finite(
            self.reference_distance_percentile,
            "reference_distance_percentile",
        )
        _validate_finite(
            self.reproduction_maximum_voltage_difference,
            "reproduction_maximum_voltage_difference",
        )
        _validate_finite(
            self.reproduction_maximum_power_difference,
            "reproduction_maximum_power_difference",
        )
