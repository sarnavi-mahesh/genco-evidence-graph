"""Authoritative local metadata adapters for dataset, config, and MLflow provenance."""

from ast import literal_eval
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import numpy as np

import yaml

from .provenance import sha256_path

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


def _huggingface_etag(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(
        f"blob {len(data)}\0".encode() + data,
    ).hexdigest()


TRAINING_METADATA_NORMALIZATION = {
    "policy_id": "genco-training-metadata-normalization-v1",
    "transforms": {
        "meta.yaml": "portable-artifact-uri-and-upstream-user",
        "params/args/data/split_from_existing_files": "repository-relative-split-path",
        "tags/mlflow.source.name": "portable-source-name",
        "tags/mlflow.user": "upstream-user",
    },
}


def _normalized_training_metadata_bytes(
    declared_path: str,
    original_bytes: bytes,
) -> bytes:
    if declared_path == "meta.yaml":
        value = yaml.safe_load(original_bytes)
        if not isinstance(value, dict):
            raise ValueError(
                "Original upstream training metadata meta.yaml is not a mapping",
            )
        value["artifact_uri"] = "artifacts"
        value["user_id"] = "upstream"
        return yaml.safe_dump(value, sort_keys=False).encode()
    replacements = {
        "params/args/data/split_from_existing_files": b"genco-opfdata/splits",
        "tags/mlflow.source.name": b"gridfm_graphkit",
        "tags/mlflow.user": b"upstream",
    }
    return replacements.get(declared_path, original_bytes)


def verify_normalized_upstream_metadata(
    path: str | Path,
    original_repository_root: str | Path,
    expected_repository_revision: str,
) -> None:
    """Verify normalized training metadata against its pinned upstream bytes."""
    run_path = Path(path)
    original_root = Path(original_repository_root)
    manifest_path = run_path / "upstream_source_manifest.json"
    manifest = json.loads(_read(manifest_path))

    if manifest.get("format_version") != 2:
        raise ValueError("Unsupported training metadata source manifest format")
    if manifest.get("source_repository") != "gridfm/genco-opfdata":
        raise ValueError("Unexpected training metadata source repository")
    if manifest.get("source_revision") != expected_repository_revision:
        raise ValueError(
            "Training metadata source revision disagrees with model repository",
        )
    if manifest.get("source_run_id") != run_path.name:
        raise ValueError("Training metadata source run ID disagrees with run path")
    if manifest.get("normalization") != TRAINING_METADATA_NORMALIZATION:
        raise ValueError(
            "Training metadata normalization policy is not the approved "
            f"{TRAINING_METADATA_NORMALIZATION['policy_id']!r}",
        )

    try:
        mlflow_index = run_path.parts.index("mlflow")
    except ValueError as error:
        raise ValueError(
            f"Training metadata path is not under an MLflow tree: {run_path}",
        ) from error

    run_relative = Path(*run_path.parts[mlflow_index:])
    original_run_path = original_root / run_relative
    cache_run_path = (
        original_root
        / ".cache/huggingface/download"
        / run_relative
    )

    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("Training metadata source manifest has no files")

    seen_paths: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Invalid training metadata source manifest record")
        try:
            declared_path = str(record["path"])
            upstream_sha256 = str(record["upstream_sha256"])
            normalized_sha256 = str(record["normalized_sha256"])
        except KeyError as error:
            raise ValueError(
                "Incomplete training metadata source manifest record",
            ) from error

        relative_path = Path(declared_path)
        if (
            relative_path.is_absolute()
            or declared_path in {"", "."}
            or ".." in relative_path.parts
        ):
            raise ValueError(
                f"Unsafe training metadata manifest path: {declared_path}",
            )
        if declared_path in seen_paths:
            raise ValueError(
                f"Duplicate training metadata manifest path: {declared_path}",
            )
        seen_paths.add(declared_path)

        original = original_run_path / relative_path
        normalized = run_path / relative_path
        metadata = cache_run_path / f"{declared_path}.metadata"

        if not original.is_file():
            raise FileNotFoundError(
                "Original upstream training metadata does not exist: "
                f"{original}",
            )
        if not normalized.is_file():
            raise FileNotFoundError(
                f"Normalized training metadata does not exist: {normalized}",
            )

        metadata_lines = _read(metadata).splitlines()
        if len(metadata_lines) < 2:
            raise ValueError(
                f"Incomplete Hugging Face cache metadata: {metadata}",
            )
        if metadata_lines[0] != expected_repository_revision:
            raise ValueError(
                f"Hugging Face revision disagrees for: {original}",
            )

        etag = metadata_lines[1]
        if not re.fullmatch(r"[0-9a-f]{40}", etag):
            raise ValueError(
                f"Unsupported Hugging Face cache ETag for: {original}",
            )
        if _huggingface_etag(original) != etag:
            raise ValueError(
                "Original upstream training metadata disagrees with cache ETag: "
                f"{original}",
            )
        if sha256_path(original) != upstream_sha256:
            raise ValueError(
                f"Original upstream training metadata hash disagrees: {original}",
            )
        expected_bytes = _normalized_training_metadata_bytes(
            declared_path,
            original.read_bytes(),
        )
        expected_hash = hashlib.sha256(expected_bytes).hexdigest()
        if normalized_sha256 != expected_hash:
            raise ValueError(
                f"Declared normalized training metadata hash disagrees with "
                f"deterministic transform: {normalized}",
            )
        if normalized.read_bytes() != expected_bytes:
            raise ValueError(
                f"Normalized training metadata disagrees with deterministic "
                f"transform: {normalized}",
            )

    normalized_inventory = {
        item.relative_to(run_path).as_posix()
        for item in run_path.rglob("*")
        if item.is_file() and item != manifest_path
    }
    if seen_paths != normalized_inventory:
        raise ValueError(
            "Training metadata manifest does not match the exact normalized "
            "file inventory: "
            f"undeclared_normalized={sorted(normalized_inventory - seen_paths)}, "
            f"missing_normalized={sorted(seen_paths - normalized_inventory)}",
        )

    transformed_paths = {
        path
        for path in seen_paths
        if _normalized_training_metadata_bytes(
            path,
            (original_run_path / path).read_bytes(),
        )
        != (original_run_path / path).read_bytes()
    }
    if transformed_paths != set(TRAINING_METADATA_NORMALIZATION["transforms"]):
        raise ValueError(
            "Training metadata transformed path set disagrees with the approved "
            "normalization policy",
        )


def extract_huggingface_revision(raw_dataset_path: str | Path) -> str:
    tree_dir = Path(raw_dataset_path) / ".cache" / "huggingface" / "trees"
    revisions = sorted(
        path.stem
        for path in tree_dir.glob("*.json")
        if re.fullmatch(r"[0-9a-f]{40}", path.stem)
    )
    if len(revisions) != 1:
        raise ValueError(
            f"Expected exactly one Hugging Face revision under {tree_dir}, "
            f"found {revisions}",
        )
    return revisions[0]


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
