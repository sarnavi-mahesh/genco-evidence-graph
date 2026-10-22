"""Authoritative local metadata adapters for dataset, config, and MLflow provenance."""

from ast import literal_eval
from dataclasses import dataclass
from pathlib import Path
import numpy as np

import yaml

from .provenance import sha256_path
from .training_metadata import (
    TRAINING_METADATA_NORMALIZATION as _TRAINING_METADATA_NORMALIZATION,
    extract_huggingface_revision as _extract_huggingface_revision,
    verify_normalized_upstream_metadata as _verify_normalized_upstream_metadata,
)

TRAINING_METADATA_NORMALIZATION = _TRAINING_METADATA_NORMALIZATION
extract_huggingface_revision = _extract_huggingface_revision
verify_normalized_upstream_metadata = _verify_normalized_upstream_metadata

def _read(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Required provenance metadata does not exist: {path}")
    return path.read_text().strip()


def _parse_mlflow_value(path: Path):
    value = _read(path)
    try:
        return literal_eval(value)
    except (SyntaxError, ValueError):
        return value


@dataclass(frozen=True)
class ConfigMetadata:
    seed: int
    grid_name: str
    task: str
    corpus_scenario_count: int
    hidden_dimension: int
    model_variant: str


@dataclass(frozen=True)
class MLflowRunMetadata:
    run_id: str
    run_name: str
    status: int
    start_time_ms: int
    end_time_ms: int
    graphkit_commit: str
    seed: int
    get_embeddings: bool
    grid_name: str
    task: str
    corpus_scenario_count: int
    hidden_dimension: int


@dataclass(frozen=True)
class TrainingRunMetadata:
    run_id: str
    status: int
    start_time_ms: int
    end_time_ms: int
    seed: int
    grid_name: str
    task: str
    corpus_scenario_count: int
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str
    selected_metrics: dict[str, float]
    software_revision_status: str
    checkpoint_binding_status: str
    parameters_hash: str
    metrics_hash: str


def load_config_metadata(path: str | Path) -> ConfigMetadata:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration does not exist: {config_path}")
    value = yaml.safe_load(config_path.read_text())
    try:
        networks = value["data"]["networks"]
        scenarios = value["data"]["scenarios"]
        return ConfigMetadata(
            seed=int(value["seed"]),
            grid_name=str(networks[0]),
            task=str(value["task"]["task_name"]),
            corpus_scenario_count=int(scenarios[0]),
            hidden_dimension=int(value["model"]["hidden_size"]),
            model_variant=str(value["model"]["type"]),
        )
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError(f"Configuration metadata is incomplete: {config_path}") from error


def load_mlflow_run_metadata(path: str | Path) -> MLflowRunMetadata:
    run_path = Path(path)
    meta_path = run_path / "meta.yaml"
    if not meta_path.is_file():
        raise FileNotFoundError(f"MLflow run metadata does not exist: {meta_path}")
    meta = yaml.safe_load(meta_path.read_text())
    params = run_path / "params" / "args"
    tags = run_path / "tags"
    networks = _parse_mlflow_value(params / "data" / "networks")
    scenarios = _parse_mlflow_value(params / "data" / "scenarios")
    try:
        return MLflowRunMetadata(
            run_id=str(meta["run_id"]),
            run_name=str(meta["run_name"]),
            status=int(meta["status"]),
            start_time_ms=int(meta["start_time"]),
            end_time_ms=int(meta["end_time"]),
            graphkit_commit=_read(tags / "mlflow.source.git.commit"),
            seed=int(_parse_mlflow_value(params / "seed")),
            get_embeddings=bool(_parse_mlflow_value(params / "get_embeddings")),
            grid_name=str(networks[0]),
            task=str(_parse_mlflow_value(params / "task" / "task_name")),
            corpus_scenario_count=int(scenarios[0]),
            hidden_dimension=int(_parse_mlflow_value(params / "model" / "hidden_size")),
        )
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError(f"MLflow run metadata is incomplete: {run_path}") from error



def _load_final_mlflow_metric(path: Path) -> float:
    if not path.is_file():
        raise FileNotFoundError(f"MLflow metric does not exist: {path}")
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"MLflow metric is empty: {path}")
    parts = lines[-1].split()
    if len(parts) < 2:
        raise ValueError(f"Invalid MLflow metric record: {lines[-1]!r}")
    try:
        value = float(parts[1])
    except ValueError as error:
        raise ValueError(
            f"Invalid MLflow metric value: {lines[-1]!r}",
        ) from error
    if not np.isfinite(value):
        raise ValueError(f"MLflow metric is non-finite: {path}")
    return value


def load_training_run_metadata(path: str | Path) -> TrainingRunMetadata:
    """Load authoritative checkpoint-training metadata from an MLflow export."""
    run_path = Path(path)
    meta = yaml.safe_load(_read(run_path / "meta.yaml"))
    params = run_path / "params/args"
    networks = _parse_mlflow_value(params / "data/networks")
    scenarios = _parse_mlflow_value(params / "data/scenarios")
    try:
        metadata = TrainingRunMetadata(
            run_id=str(meta["run_id"]),
            status=int(meta["status"]),
            start_time_ms=int(meta["start_time"]),
            end_time_ms=int(meta["end_time"]),
            seed=int(_parse_mlflow_value(params / "seed")),
            grid_name=str(networks[0]),
            task=str(_parse_mlflow_value(params / "task/task_name")),
            corpus_scenario_count=int(scenarios[0]),
            epochs=int(_parse_mlflow_value(params / "training/epochs")),
            batch_size=int(_parse_mlflow_value(params / "training/batch_size")),
            learning_rate=float(
                _parse_mlflow_value(params / "optimizer/learning_rate"),
            ),
            optimizer=str(
                _parse_mlflow_value(params / "optimizer/type"),
            ),
            selected_metrics={
                "training_loss": _load_final_mlflow_metric(
                    run_path / "metrics/Training Loss",
                ),
                "validation_loss": _load_final_mlflow_metric(
                    run_path / "metrics/Validation loss",
                ),
                "validation_bus_mse_loss": _load_final_mlflow_metric(
                    run_path / "metrics/Validation Masked bus MSE loss",
                ),
                "validation_generator_mse_loss": _load_final_mlflow_metric(
                    run_path / "metrics/Validation Masked generator MSE loss",
                ),
                "validation_physics_loss": _load_final_mlflow_metric(
                    run_path
                    / "metrics/Validation Layered Weighted Physics Loss",
                ),
                "validation_qg_violation_penalty_loss": (
                    _load_final_mlflow_metric(
                        run_path
                        / "metrics/Validation Qg Violation Penalty loss",
                    )
                ),
            },
            software_revision_status=(
                "unavailable upstream: MLflow source_version is empty and "
                "mlflow.source.git.commit was not published"
            ),
            checkpoint_binding_status=(
                "associated training evidence is co-published and "
                "configuration-compatible but production identity is "
                "unverified because the run does not record a checkpoint hash"
            ),
            parameters_hash=sha256_path(params),
            metrics_hash=sha256_path(run_path / "metrics"),
        )
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError(
            f"Training MLflow metadata is incomplete: {run_path}",
        ) from error
    if metadata.status != 3:
        raise ValueError(f"Training run {metadata.run_id} did not complete")
    return metadata




def verify_run_matches_config(
    run: MLflowRunMetadata,
    config: ConfigMetadata,
    *,
    require_embeddings: bool = True,
) -> None:
    if run.status != 3:
        raise ValueError(f"MLflow run {run.run_id} did not finish successfully")
    if require_embeddings and not run.get_embeddings:
        raise ValueError(f"MLflow run {run.run_id} did not enable embedding export")
    comparisons = {
        "seed": (run.seed, config.seed),
        "grid_name": (run.grid_name, config.grid_name),
        "task": (run.task, config.task),
        "corpus_scenario_count": (
            run.corpus_scenario_count,
            config.corpus_scenario_count,
        ),
        "hidden_dimension": (run.hidden_dimension, config.hidden_dimension),
    }
    mismatches = {
        name: values
        for name, values in comparisons.items()
        if values[0] != values[1]
    }
    if mismatches:
        raise ValueError(
            f"MLflow run {run.run_id} disagrees with configuration: {mismatches}",
        )
