"""Authenticated, deterministic preparation of upstream MLflow metadata."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from types import MappingProxyType
from typing import Mapping

import yaml


TRAINING_METADATA_MANIFEST_FORMAT_VERSION = 2
TRAINING_METADATA_SOURCE_REPOSITORY = "gridfm/genco-opfdata"
TRAINING_METADATA_NORMALIZATION: Mapping[str, object] = MappingProxyType(
    {
        "policy_id": "genco-training-metadata-normalization-v1",
        "transforms": MappingProxyType(
            {
                "meta.yaml": "portable-artifact-uri-and-upstream-user",
                "params/args/data/split_from_existing_files": (
                    "repository-relative-split-path"
                ),
                "tags/mlflow.source.name": "portable-source-name",
                "tags/mlflow.user": "upstream-user",
            },
        ),
    },
)

# Pinned inventory copied from the approved release manifest. Do not derive this
# set from the data being verified.
EXPECTED_TRAINING_METADATA_PATHS = frozenset(
    {
        "artifacts/stats/case14_ieee_scenario_splits.json",
        "artifacts/stats/normalization_stats.txt",
        "artifacts/test/case14_ieee_metrics.csv",
        "artifacts/test/case14_ieee_RMSE.csv",
        "artifacts/test_plots/case14_ieee/case14_ieee_correlation_PQ.png",
        "artifacts/test_plots/case14_ieee/case14_ieee_correlation_PV.png",
        "artifacts/test_plots/case14_ieee/case14_ieee_correlation_REF.png",
        "artifacts/test_plots/case14_ieee/case14_ieee_objective.png",
        "artifacts/test_plots/case14_ieee/max_residual_P.png",
        "artifacts/test_plots/case14_ieee/max_residual_Q.png",
        "artifacts/test_plots/case14_ieee/mean_residual_P.png",
        "artifacts/test_plots/case14_ieee/mean_residual_Q.png",
        "meta.yaml",
        "metrics/Learning Rate",
        "metrics/Training Loss",
        "metrics/Validation Layered Weighted Physics Loss",
        "metrics/Validation Masked bus MSE loss",
        "metrics/Validation Masked generator MSE loss",
        "metrics/Validation Qg Violation Penalty loss",
        "metrics/Validation layer_0_residual",
        "metrics/Validation layer_0_weight",
        "metrics/Validation layer_10_residual",
        "metrics/Validation layer_10_weight",
        "metrics/Validation layer_11_residual",
        "metrics/Validation layer_11_weight",
        "metrics/Validation layer_1_residual",
        "metrics/Validation layer_1_weight",
        "metrics/Validation layer_2_residual",
        "metrics/Validation layer_2_weight",
        "metrics/Validation layer_3_residual",
        "metrics/Validation layer_3_weight",
        "metrics/Validation layer_4_residual",
        "metrics/Validation layer_4_weight",
        "metrics/Validation layer_5_residual",
        "metrics/Validation layer_5_weight",
        "metrics/Validation layer_6_residual",
        "metrics/Validation layer_6_weight",
        "metrics/Validation layer_7_residual",
        "metrics/Validation layer_7_weight",
        "metrics/Validation layer_8_residual",
        "metrics/Validation layer_8_weight",
        "metrics/Validation layer_9_residual",
        "metrics/Validation layer_9_weight",
        "metrics/Validation loss",
        "metrics/epoch",
        "params/args/callbacks/optuna_early_prune",
        "params/args/callbacks/patience",
        "params/args/callbacks/tol",
        "params/args/data/baseMVA",
        "params/args/data/mask_value",
        "params/args/data/networks",
        "params/args/data/normalization",
        "params/args/data/scenarios",
        "params/args/data/split_from_existing_files",
        "params/args/data/test_ratio",
        "params/args/data/val_ratio",
        "params/args/data/workers",
        "params/args/optimizer/learning_rate",
        "params/args/optimizer/optimizer_params/betas",
        "params/args/optimizer/scheduler_params/factor",
        "params/args/optimizer/scheduler_params/mode",
        "params/args/optimizer/scheduler_params/patience",
        "params/args/optimizer/scheduler_type",
        "params/args/optimizer/type",
        "params/args/seed",
        "params/args/task/task_name",
        "params/args/training/accelerator",
        "params/args/training/batch_size",
        "params/args/training/devices",
        "params/args/training/epochs",
        "params/args/training/loss_args",
        "params/args/training/loss_weights",
        "params/args/training/losses",
        "params/args/training/strategy",
        "params/args/verbose",
        "tags/mlflow.runName",
        "tags/mlflow.source.name",
        "tags/mlflow.source.type",
        "tags/mlflow.user",
    },
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_GIT_BLOB_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_MANIFEST_FIELDS = frozenset(
    {
        "format_version",
        "source_repository",
        "source_revision",
        "source_run_id",
        "normalization",
        "files",
    },
)
_FILE_FIELDS = frozenset({"path", "upstream_sha256", "normalized_sha256"})


@dataclass(frozen=True)
class TrainingMetadataFileRecord:
    path: str
    upstream_sha256: str
    normalized_sha256: str

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) for value in (self.path, self.upstream_sha256, self.normalized_sha256)):
            raise TypeError("Training metadata file record fields must be strings")


@dataclass(frozen=True)
class TrainingMetadataNormalization:
    policy_id: str
    transforms: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.policy_id, str) or not isinstance(self.transforms, tuple):
            raise TypeError("Training metadata normalization must use immutable values")
        if any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not all(isinstance(part, str) for part in item)
            for item in self.transforms
        ):
            raise TypeError("Training metadata transforms must be immutable pairs")


@dataclass(frozen=True)
class TrainingMetadataManifest:
    format_version: int
    source_repository: str
    source_revision: str
    source_run_id: str
    normalization: TrainingMetadataNormalization
    files: tuple[TrainingMetadataFileRecord, ...]

    def __post_init__(self) -> None:
        if isinstance(self.format_version, bool) or not isinstance(self.format_version, int):
            raise TypeError("Training metadata format version must be an integer")
        if not all(
            isinstance(value, str)
            for value in (self.source_repository, self.source_revision, self.source_run_id)
        ):
            raise TypeError("Training metadata identity fields must be strings")
        if not isinstance(self.files, tuple) or any(
            not isinstance(record, TrainingMetadataFileRecord) for record in self.files
        ):
            raise TypeError("Training metadata files must be an immutable record tuple")
        if not isinstance(self.normalization, TrainingMetadataNormalization):
            raise TypeError("Training metadata normalization has an invalid model type")


def _normalization_as_dict() -> dict[str, object]:
    return {
        "policy_id": TRAINING_METADATA_NORMALIZATION["policy_id"],
        "transforms": dict(TRAINING_METADATA_NORMALIZATION["transforms"]),
    }


def training_metadata_manifest_to_dict_unchecked(
    manifest: TrainingMetadataManifest,
) -> dict[str, object]:
    """Convert a model to plain containers; validation is performed by callers."""
    if not isinstance(manifest, TrainingMetadataManifest):
        raise ValueError("Expected a TrainingMetadataManifest")
    normalization = manifest.normalization
    files = manifest.files
    if not isinstance(normalization, TrainingMetadataNormalization):
        raise ValueError("Invalid training metadata normalization model")
    if not isinstance(files, tuple) or not isinstance(normalization.transforms, tuple):
        raise ValueError("Training metadata model must contain immutable tuples")
    return {
        "format_version": manifest.format_version,
        "source_repository": manifest.source_repository,
        "source_revision": manifest.source_revision,
        "source_run_id": manifest.source_run_id,
        "normalization": {
            "policy_id": normalization.policy_id,
            "transforms": dict(normalization.transforms),
        },
        "files": [
            {
                "path": record.path,
                "upstream_sha256": record.upstream_sha256,
                "normalized_sha256": record.normalized_sha256,
            }
            for record in sorted(files, key=lambda item: item.path)
            if isinstance(record, TrainingMetadataFileRecord)
        ],
    }


def _expect_exact_keys(value: object, expected: frozenset[str], label: str) -> dict:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object")
    keys = set(value)
    if keys != expected:
        missing = sorted(expected - keys)
        unknown = sorted(keys - expected)
        raise ValueError(f"Invalid {label} fields: missing={missing}, unknown={unknown}")
    return value


def _validate_manifest_path(path: object) -> str:
    if not isinstance(path, str) or not path:
        raise ValueError("Training metadata path must be a non-empty string")
    if "\\" in path or "\x00" in path:
        raise ValueError(f"Unsafe training metadata manifest path: {path!r}")
    relative = PurePosixPath(path)
    if (
        relative.is_absolute()
        or path in {"", "."}
        or any(part in {"", ".", ".."} for part in path.split("/"))
        or relative.as_posix() != path
    ):
        raise ValueError(f"Unsafe training metadata manifest path: {path!r}")
    return path


def _validate_relative_run_path(path: str | Path) -> Path:
    value = str(path)
    if "\\" in value or "\x00" in value:
        raise ValueError("Training metadata run path must use safe POSIX separators")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or not value
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or relative.as_posix() != value
    ):
        raise ValueError(f"Unsafe training metadata run path: {value}")
    return Path(*value.split("/"))


def _reject_symlinked_relative_path(root: Path, relative: Path) -> None:
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Training metadata input contains a symlink: {current}")


def training_metadata_manifest_from_dict(value: object) -> TrainingMetadataManifest:
    """Strictly parse and validate a manifest represented by plain JSON values."""
    top = _expect_exact_keys(value, _MANIFEST_FIELDS, "training metadata manifest")
    version = top["format_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("Training metadata format_version must be an integer")
    if version != TRAINING_METADATA_MANIFEST_FORMAT_VERSION:
        raise ValueError(f"Unsupported training metadata manifest format: {version}")
    if top["source_repository"] != TRAINING_METADATA_SOURCE_REPOSITORY:
        raise ValueError("Unexpected training metadata source repository")
    revision = top["source_revision"]
    if not isinstance(revision, str) or not _REVISION_RE.fullmatch(revision):
        raise ValueError("Invalid training metadata source revision")
    run_id = top["source_run_id"]
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("Training metadata source run ID must be a non-empty string")

    norm = _expect_exact_keys(
        top["normalization"],
        frozenset({"policy_id", "transforms"}),
        "training metadata normalization",
    )
    if norm != _normalization_as_dict():
        raise ValueError("Training metadata normalization policy is not approved")
    transforms_value = norm["transforms"]
    if not isinstance(transforms_value, dict):
        raise ValueError("Training metadata normalization transforms must be an object")
    transforms = tuple(sorted(transforms_value.items()))
    normalization = TrainingMetadataNormalization(norm["policy_id"], transforms)

    records_value = top["files"]
    if not isinstance(records_value, list) or not records_value:
        raise ValueError("Training metadata manifest files must be a non-empty array")
    records: list[TrainingMetadataFileRecord] = []
    seen: set[str] = set()
    for item in records_value:
        record = _expect_exact_keys(item, _FILE_FIELDS, "training metadata file record")
        path = _validate_manifest_path(record["path"])
        if path in seen:
            raise ValueError(f"Duplicate training metadata manifest path: {path}")
        seen.add(path)
        upstream_hash = record["upstream_sha256"]
        normalized_hash = record["normalized_sha256"]
        if not isinstance(upstream_hash, str) or not _SHA256_RE.fullmatch(upstream_hash):
            raise ValueError(f"Invalid upstream SHA-256 for training metadata path: {path}")
        if not isinstance(normalized_hash, str) or not _SHA256_RE.fullmatch(normalized_hash):
            raise ValueError(f"Invalid normalized SHA-256 for training metadata path: {path}")
        records.append(TrainingMetadataFileRecord(path, upstream_hash, normalized_hash))
    paths = [record.path for record in records]
    if paths != sorted(paths):
        raise ValueError("Training metadata file records are not in canonical path order")
    if set(paths) != EXPECTED_TRAINING_METADATA_PATHS:
        missing = sorted(EXPECTED_TRAINING_METADATA_PATHS - set(paths))
        unexpected = sorted(set(paths) - EXPECTED_TRAINING_METADATA_PATHS)
        raise ValueError(
            "Training metadata manifest does not match the approved inventory: "
            f"missing={missing}, unexpected={unexpected}",
        )
    return TrainingMetadataManifest(
        version,
        TRAINING_METADATA_SOURCE_REPOSITORY,
        revision,
        run_id,
        normalization,
        tuple(records),
    )


def training_metadata_manifest_to_dict(
    manifest: TrainingMetadataManifest,
) -> dict[str, object]:
    """Return a validated, deterministic plain-dictionary representation."""
    value = training_metadata_manifest_to_dict_unchecked(manifest)
    validated = training_metadata_manifest_from_dict(value)
    return training_metadata_manifest_to_dict_unchecked(validated)


def _manifest_json_bytes(manifest: TrainingMetadataManifest) -> bytes:
    return (
        json.dumps(training_metadata_manifest_to_dict(manifest), indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def load_training_metadata_manifest(path: str | Path) -> TrainingMetadataManifest:
    """Load strict JSON, rejecting duplicate object keys and malformed records."""
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON object key in manifest: {key}")
            result[key] = item
        return result

    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"Invalid training metadata manifest JSON: {path}") from error
    return training_metadata_manifest_from_dict(value)


def write_training_metadata_manifest(
    manifest: TrainingMetadataManifest,
    path: str | Path,
) -> None:
    Path(path).write_bytes(_manifest_json_bytes(manifest))


def normalized_training_metadata_bytes(
    declared_path: str,
    original_bytes: bytes,
) -> bytes:
    """Apply only the code-approved path-specific normalization transforms."""
    if declared_path == "meta.yaml":
        try:
            value = yaml.safe_load(original_bytes)
        except (yaml.YAMLError, UnicodeDecodeError) as error:
            raise ValueError("Original upstream training metadata meta.yaml is invalid YAML") from error
        if not isinstance(value, dict):
            raise ValueError("Original upstream training metadata meta.yaml is not a mapping")
        value["artifact_uri"] = "artifacts"
        value["user_id"] = "upstream"
        return yaml.safe_dump(value, sort_keys=False).encode("utf-8")
    replacements = {
        "params/args/data/split_from_existing_files": b"genco-opfdata/splits",
        "tags/mlflow.source.name": b"gridfm_graphkit",
        "tags/mlflow.user": b"upstream",
    }
    return replacements.get(declared_path, original_bytes)


def _git_blob_etag(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def extract_huggingface_revision(raw_dataset_path: str | Path) -> str:
    tree_dir = Path(raw_dataset_path) / ".cache" / "huggingface" / "trees"
    revisions = sorted(
        path.stem
        for path in tree_dir.glob("*.json")
        if _REVISION_RE.fullmatch(path.stem)
    )
    if len(revisions) != 1:
        raise ValueError(
            f"Expected exactly one Hugging Face revision under {tree_dir}, found {revisions}",
        )
    return revisions[0]


def _regular_file_inventory(root: Path) -> set[str]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"Training metadata run is not a regular directory: {root}")
    paths: set[str] = set()
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in list(directories):
            item = current_path / name
            if item.is_symlink() or not item.is_dir():
                raise ValueError(f"Training metadata contains a symlink or non-directory: {item}")
        for name in filenames:
            item = current_path / name
            if item.is_symlink() or not item.is_file():
                raise ValueError(f"Training metadata contains a symlink or non-regular file: {item}")
            paths.add(item.relative_to(root).as_posix())
    return paths


def _check_inventory(paths: set[str], label: str) -> None:
    if paths != EXPECTED_TRAINING_METADATA_PATHS:
        raise ValueError(
            f"{label} does not match the approved training metadata inventory: "
            f"missing={sorted(EXPECTED_TRAINING_METADATA_PATHS - paths)}, "
            f"unexpected={sorted(paths - EXPECTED_TRAINING_METADATA_PATHS)}",
        )


def _cache_record(
    original: Path,
    metadata_path: Path,
    expected_revision: str,
) -> bytes:
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise FileNotFoundError(f"Hugging Face cache metadata does not exist: {metadata_path}")
    try:
        lines = metadata_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"Invalid Hugging Face cache metadata encoding: {metadata_path}") from error
    if len(lines) < 2:
        raise ValueError(f"Incomplete Hugging Face cache metadata: {metadata_path}")
    if lines[0] != expected_revision:
        raise ValueError(f"Hugging Face revision disagrees for: {original}")
    etag = lines[1]
    is_git_blob_sha1 = _GIT_BLOB_SHA1_RE.fullmatch(etag) is not None
    is_sha256 = _SHA256_RE.fullmatch(etag) is not None
    if not (is_git_blob_sha1 or is_sha256):
        raise ValueError(f"Unsupported Hugging Face cache ETag for: {original}")

    data = original.read_bytes()
    actual_etag = (
        _git_blob_etag(data)
        if is_git_blob_sha1
        else hashlib.sha256(data).hexdigest()
    )
    if actual_etag != etag:
        raise ValueError(
            f"Original upstream training metadata disagrees with cache ETag: {original}"
        )
    return data


def verify_normalized_upstream_metadata(
    path: str | Path,
    original_repository_root: str | Path,
    expected_repository_revision: str,
    *,
    run_relative_path: str | Path | None = None,
) -> None:
    """Authenticate and verify the normalized run against pinned upstream bytes."""
    requested_run_path = Path(path)
    if requested_run_path.is_symlink():
        raise ValueError(f"Normalized training metadata run is a symlink: {requested_run_path}")
    run_path = requested_run_path.resolve()
    original_root = Path(original_repository_root).resolve()
    if run_relative_path is None:
        try:
            mlflow_index = run_path.parts.index("mlflow")
        except ValueError as error:
            raise ValueError(f"Training metadata path is not under an MLflow tree: {run_path}") from error
        run_relative = Path(*run_path.parts[mlflow_index:])
    else:
        run_relative = _validate_relative_run_path(run_relative_path)
    manifest_path = run_path / "upstream_source_manifest.json"
    manifest = load_training_metadata_manifest(manifest_path)
    if manifest.source_revision != expected_repository_revision:
        raise ValueError("Training metadata source revision disagrees with model repository")
    if manifest.source_run_id != run_relative.name:
        raise ValueError("Training metadata source run ID disagrees with run path")
    if not _REVISION_RE.fullmatch(expected_repository_revision):
        raise ValueError("Invalid expected Hugging Face repository revision")

    original_run_path = original_root / run_relative
    cache_run_path = original_root / ".cache/huggingface/download" / run_relative
    _reject_symlinked_relative_path(original_root, run_relative)
    _check_inventory(_regular_file_inventory(original_run_path), "Original upstream training metadata")
    normalized_inventory = _regular_file_inventory(run_path) - {"upstream_source_manifest.json"}
    _check_inventory(normalized_inventory, "Normalized training metadata")

    seen_transformed: set[str] = set()
    for record in manifest.files:
        original = original_run_path / record.path
        normalized = run_path / record.path
        metadata_path = cache_run_path / f"{record.path}.metadata"
        _reject_symlinked_relative_path(
            original_root,
            Path(".cache/huggingface/download")
            / run_relative
            / f"{record.path}.metadata",
        )
        original_bytes = _cache_record(original, metadata_path, expected_repository_revision)
        if hashlib.sha256(original_bytes).hexdigest() != record.upstream_sha256:
            raise ValueError(f"Original upstream training metadata hash disagrees: {original}")
        expected_bytes = normalized_training_metadata_bytes(record.path, original_bytes)
        if expected_bytes != original_bytes:
            seen_transformed.add(record.path)
        if hashlib.sha256(expected_bytes).hexdigest() != record.normalized_sha256:
            raise ValueError(
                "Declared normalized training metadata hash disagrees with deterministic "
                f"transform: {normalized}",
            )
        if normalized.read_bytes() != expected_bytes:
            raise ValueError(
                "Normalized training metadata disagrees with deterministic "
                f"transform: {normalized}",
            )
    approved_transforms = set(TRAINING_METADATA_NORMALIZATION["transforms"])
    if seen_transformed != approved_transforms:
        raise ValueError(
            "Training metadata transformed path set disagrees with the approved "
            f"normalization policy: {sorted(seen_transformed)}",
        )


def prepare_normalized_training_metadata(
    *,
    original_repository_root: str | Path,
    model_repository_path: str | Path,
    run_relative_path: str | Path,
    output_run_path: str | Path,
) -> Path:
    """Generate and verify normalized training metadata atomically."""
    original_root = Path(original_repository_root).resolve()
    model_root = Path(model_repository_path).resolve()
    run_relative = _validate_relative_run_path(run_relative_path)
    if run_relative.name in {"", ".", ".."}:
        raise ValueError("Training metadata run path must identify a run ID")

    raw_output = Path(output_run_path)
    if raw_output.is_symlink():
        raise ValueError(f"Output training metadata path is a symlink: {raw_output}")
    output = raw_output.resolve()
    _reject_symlinked_relative_path(original_root, run_relative)
    original_run = original_root / run_relative
    if original_root == output or original_root in output.parents or output in original_root.parents:
        raise ValueError("Original repository root and output path must not overlap")
    if original_run == output or original_run in output.parents or output in original_run.parents:
        raise ValueError("Original training run and output path must not overlap")
    if output.name != run_relative.name:
        raise ValueError("Output training metadata directory name disagrees with run ID")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output training metadata directory is not empty: {output}")
    if run_relative.name != original_run.name:
        raise ValueError("Training metadata run ID disagrees with run-relative path")

    revision = extract_huggingface_revision(model_root)
    if not original_run.is_dir():
        raise FileNotFoundError(f"Original upstream training run does not exist: {original_run}")
    original_paths = _regular_file_inventory(original_run)
    _check_inventory(original_paths, "Original upstream training metadata")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{run_relative.name}.tmp-", dir=output.parent),
    )
    try:
        records: list[TrainingMetadataFileRecord] = []
        transformed_paths: set[str] = set()
        for declared_path in sorted(EXPECTED_TRAINING_METADATA_PATHS):
            original = original_run / declared_path
            cache_record_path = (
                original_root
                / ".cache/huggingface/download"
                / run_relative
                / f"{declared_path}.metadata"
            )
            _reject_symlinked_relative_path(
                original_root,
                Path(".cache/huggingface/download")
                / run_relative
                / f"{declared_path}.metadata",
            )
            original_bytes = _cache_record(original, cache_record_path, revision)
            normalized_bytes = normalized_training_metadata_bytes(declared_path, original_bytes)
            if normalized_bytes != original_bytes:
                transformed_paths.add(declared_path)
            destination = temporary / declared_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(normalized_bytes)
            records.append(
                TrainingMetadataFileRecord(
                    declared_path,
                    hashlib.sha256(original_bytes).hexdigest(),
                    hashlib.sha256(normalized_bytes).hexdigest(),
                ),
            )
        approved_transforms = set(TRAINING_METADATA_NORMALIZATION["transforms"])
        if transformed_paths != approved_transforms:
            raise ValueError(
                "Training metadata transformed path set disagrees with the approved "
                f"normalization policy: {sorted(transformed_paths)}",
            )
        manifest = TrainingMetadataManifest(
            TRAINING_METADATA_MANIFEST_FORMAT_VERSION,
            TRAINING_METADATA_SOURCE_REPOSITORY,
            revision,
            run_relative.name,
            TrainingMetadataNormalization(
                str(TRAINING_METADATA_NORMALIZATION["policy_id"]),
                tuple(sorted(dict(TRAINING_METADATA_NORMALIZATION["transforms"]).items())),
            ),
            tuple(records),
        )
        manifest_path = temporary / "upstream_source_manifest.json"
        write_training_metadata_manifest(manifest, manifest_path)
        verify_normalized_upstream_metadata(
            temporary,
            original_root,
            revision,
            run_relative_path=run_relative,
        )
        if output.exists():
            output.rmdir()
        os.replace(temporary, output)
        return output / "upstream_source_manifest.json"
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)










