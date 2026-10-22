"""Reproducible graphkit baseline inference orchestration."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from .adapters import (
    extract_huggingface_revision,
    load_config_metadata,
    load_mlflow_run_metadata,
    verify_run_matches_config,
)
from .provenance import sha256_file, sha256_path
from .splits import parquet_scenario_ids, verify_split_membership


@dataclass(frozen=True)
class BaselineInferenceSpec:
    graphkit_python: Path
    graphkit_workdir: Path
    config: Path
    checkpoint: Path
    normalizer: Path
    data_path: Path
    split_artifact: Path
    output_dir: Path
    run_name: str
    graphkit_requirements_lock: Path
    graphkit_source_record: Path
    experiment_name: str = "genco-evidence-baseline"
    num_workers: int = 0
    reuse_verified_run: bool = False
    dataset_split: str = "test"
    export_embeddings: bool = True


@dataclass(frozen=True)
class BaselineInferenceResult:
    output_dir: Path
    manifest_path: Path
    mlflow_run_dir: Path
    run_id: str


@dataclass(frozen=True)
class VerifiedRunManifest:
    manifest_path: Path
    mlflow_run_dir: Path
    run_id: str
    graphkit_commit: str
    dataset_revision: str
    embedding_dimension: int
    runtime: dict[str, object]


@dataclass(frozen=True)
class VerifiedReproductionManifest:
    manifest_path: Path
    mlflow_run_dir: Path
    run_id: str
    graphkit_commit: str
    dataset_revision: str
    start_time_ms: int
    end_time_ms: int
    runtime_environment_sha256: str
    runtime: dict[str, str]


NATIVE_EMBEDDING_FILENAMES = (
    "case14_ieee_bus_embeddings.parquet",
    "case14_ieee_gen_embeddings.parquet",
)


def prepare_reference_prediction_splits(
    *,
    train_split: str | Path,
    validation_split: str | Path,
    test_split: str | Path,
    output_dir: str | Path,
) -> Path:
    """Create a graphkit view that predicts the authoritative train split."""
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Reference split view must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    mappings = {
        Path(test_split): output / "train.pt",
        Path(validation_split): output / "val.pt",
        Path(train_split): output / "test.pt",
    }
    for source, target in mappings.items():
        shutil.copy2(source, target)
        if sha256_file(source) != sha256_file(target):
            raise ValueError(f"Copied split hash disagrees: {source} -> {target}")
    return output


def verify_manifest_artifacts(
    manifest_path: str | Path,
    artifact_paths: tuple[str | Path, ...],
) -> None:
    """Verify selected files against a run manifest before comparison."""
    path = Path(manifest_path).resolve()
    value = json.loads(path.read_text())
    root = path.parent
    recorded = {
        (root / item["path"]).resolve(): item["sha256"]
        for item in value.get("artifacts", ())
    }
    for artifact_path in artifact_paths:
        artifact = Path(artifact_path).resolve()
        if artifact not in recorded:
            raise ValueError(f"Artifact is absent from run manifest: {artifact}")
        if sha256_file(artifact) != recorded[artifact]:
            raise ValueError(f"Artifact hash disagrees with run manifest: {artifact}")


@dataclass(frozen=True)
class BaselineComparisonResult:
    absolute_tolerance: float
    active_residual_actual: float
    active_residual_expected: float
    reactive_residual_actual: float
    reactive_residual_expected: float


@dataclass(frozen=True)
class KeyedBaselineComparisonResult:
    bus_row_count: int
    generator_row_count: int
    maximum_absolute_differences: dict[str, float]
    tolerance_policy_id: str
    tolerance_policy_sha256: str


APPROVED_TOLERANCE_POLICY = {
    "format_version": 1,
    "policy_id": "genco-case14-baseline-equivalence-v1",
    "absolute_tolerances": {
        "bus": {
            "Vm_pred": 2e-6,
            "Va_pred": 2e-6,
            "Pg_pred": 2.5e-3,
            "Qg_pred": 2.5e-3,
            "active res. (MW)": 2.5e-3,
            "reactive res. (MVar)": 2.5e-3,
            "PBE": 2.5e-3,
        },
        "generator": {"p_mw_pred": 2.5e-3},
    },
    "rationale": (
        "Bounds define accepted keyed row-level numerical variation between "
        "manifest-bound embedding-enabled and uninstrumented inference runs. "
        "Observed maxima and comparison status are recomputed during evidence "
        "construction and are not asserted in this configuration."
    ),
}
APPROVED_TOLERANCE_POLICY_SHA256 = hashlib.sha256(
    json.dumps(
        APPROVED_TOLERANCE_POLICY,
        sort_keys=True,
        separators=(",", ":"),
    ).encode(),
).hexdigest()


def _compare_keyed_table(
    actual_path: str | Path,
    expected_path: str | Path,
    keys: tuple[str, ...],
    tolerances: dict[str, float],
) -> tuple[int, dict[str, float]]:
    actual = pd.read_parquet(actual_path).sort_values(list(keys)).reset_index(drop=True)
    expected = (
        pd.read_parquet(expected_path).sort_values(list(keys)).reset_index(drop=True)
    )
    if actual.duplicated(list(keys)).any() or expected.duplicated(list(keys)).any():
        raise ValueError(f"Baseline comparison keys are not unique: {keys}")
    if not actual[list(keys)].equals(expected[list(keys)]):
        raise ValueError(f"Baseline row keys disagree for {actual_path}")
    if set(actual.columns) != set(expected.columns):
        raise ValueError(f"Baseline schemas disagree for {actual_path}")
    missing_fields = sorted(set(tolerances) - set(actual.columns))
    if missing_fields:
        raise ValueError(
            f"Baseline table is missing required comparison fields: {missing_fields}",
        )
    differences: dict[str, float] = {}
    for column in actual.columns:
        if column in keys:
            continue
        actual_series = actual[column]
        expected_series = expected[column]
        is_numeric = (
            pd.api.types.is_numeric_dtype(actual_series)
            or pd.api.types.is_numeric_dtype(expected_series)
            or column in tolerances
        )
        if is_numeric:
            actual_numeric = pd.to_numeric(actual_series, errors="coerce")
            expected_numeric = pd.to_numeric(expected_series, errors="coerce")
            for label, table, values in (
                ("actual", actual, actual_numeric),
                ("expected", expected, expected_numeric),
            ):
                invalid = ~np.isfinite(values.to_numpy(dtype=float))
                if invalid.any():
                    affected_keys = (
                        table.loc[invalid, list(keys)]
                        .head(10)
                        .to_dict(orient="records")
                    )
                    raise ValueError(
                        f"Non-finite values in {label} baseline field "
                        f"{column!r} at keys {affected_keys}",
                    )
            actual_values = actual_numeric.to_numpy(dtype=float)
            expected_values = expected_numeric.to_numpy(dtype=float)
        else:
            actual_values = actual_series.to_numpy()
            expected_values = expected_series.to_numpy()
        if column in tolerances:
            difference = float(np.max(np.abs(actual_values - expected_values)))
            differences[column] = difference
            if difference > tolerances[column]:
                raise ValueError(
                    f"Baseline field {column!r} exceeded absolute tolerance "
                    f"{tolerances[column]}: {difference}",
                )
        elif is_numeric:
            if not np.array_equal(actual_values, expected_values):
                raise ValueError(f"Invariant baseline field {column!r} changed")
        elif not actual_series.equals(expected_series):
            raise ValueError(f"Invariant baseline field {column!r} changed")
    return len(actual), differences



def _verify_comparison_runtime(
    manifest_path: str | Path,
    manifest: dict[str, object],
    label: str,
) -> dict[str, object]:
    runtime = manifest.get("runtime")
    required = {
        "python_version",
        "platform",
        "graphkit",
        "torch",
        "torchvision",
        "torchaudio",
        "torch_scatter",
        "python_executable",
        "environment_path",
        "environment_sha256",
        "requirements_lock_path",
        "requirements_lock_sha256",
        "source_record_path",
        "source_record_sha256",
        "source_repository",
        "source_commit",
    }
    if not isinstance(runtime, dict):
        raise ValueError(f"{label} run manifest lacks runtime provenance")
    missing = sorted(required - runtime.keys())
    if missing:
        raise ValueError(
            f"{label} run manifest runtime lacks required fields: {missing}",
        )

    root = Path(manifest_path).resolve().parent
    runtime_artifacts = (
        root / str(runtime["environment_path"]),
        root / str(runtime["requirements_lock_path"]),
        root / str(runtime["source_record_path"]),
    )
    verify_manifest_artifacts(manifest_path, runtime_artifacts)

    hash_fields = (
        ("environment_path", "environment_sha256"),
        ("requirements_lock_path", "requirements_lock_sha256"),
        ("source_record_path", "source_record_sha256"),
    )
    for path_field, hash_field in hash_fields:
        artifact = root / str(runtime[path_field])
        actual_hash = sha256_file(artifact)
        if actual_hash != runtime[hash_field]:
            raise ValueError(
                f"{label} runtime {hash_field} disagrees with {artifact}",
            )

    source_repository = str(runtime["source_repository"])
    if not source_repository.startswith("https://github.com/"):
        raise ValueError(f"{label} runtime source repository is not public HTTPS")
    if runtime["source_commit"] != manifest.get("graphkit_commit"):
        raise ValueError(
            f"{label} runtime source commit disagrees with graphkit commit",
        )
    return runtime


def _load_approved_tolerance_policy(
    tolerance_config: str | Path | None,
) -> dict[str, object]:
    if tolerance_config is None:
        return APPROVED_TOLERANCE_POLICY
    supplied = yaml.safe_load(Path(tolerance_config).read_text())
    if supplied != APPROVED_TOLERANCE_POLICY:
        raise ValueError(
            "Baseline tolerance policy is not the approved immutable policy "
            f"{APPROVED_TOLERANCE_POLICY['policy_id']!r} "
            f"({APPROVED_TOLERANCE_POLICY_SHA256})",
        )
    return supplied


def _verify_comparison_compatibility(
    actual_manifest_path: str | Path,
    actual_manifest: dict[str, object],
    expected_manifest_path: str | Path,
    expected_manifest: dict[str, object],
) -> None:
    actual_runtime = _verify_comparison_runtime(
        actual_manifest_path,
        actual_manifest,
        "Actual",
    )
    expected_runtime = _verify_comparison_runtime(
        expected_manifest_path,
        expected_manifest,
        "Expected",
    )

    for label, manifest in (
        ("Actual", actual_manifest),
        ("Expected", expected_manifest),
    ):
        if manifest.get("format_version") != 1 or manifest.get("run_status") != 3:
            raise ValueError(f"{label} run manifest is not a completed format-1 run")
    for label, manifest_path, manifest in (
        ("Actual", actual_manifest_path, actual_manifest),
        ("Expected", expected_manifest_path, expected_manifest),
    ):
        effective_config = Path(manifest_path).resolve().parent / "effective_config.yaml"
        verify_manifest_artifacts(manifest_path, (effective_config,))
        if sha256_file(effective_config) != manifest.get("effective_config_sha256"):
            raise ValueError(
                f"{label} effective configuration hash disagrees with run manifest",
            )

    if actual_manifest.get("export_embeddings") is not False:
        raise ValueError(
            "Actual run must be the uninstrumented reproduction "
            "with export_embeddings=false",
        )
    if actual_manifest.get("embedding_dimension") is not None:
        raise ValueError(
            "Actual uninstrumented reproduction must not declare "
            "an embedding dimension",
        )
    if expected_manifest.get("export_embeddings") is not True:
        raise ValueError(
            "Expected run must be the embedding-enabled target "
            "with export_embeddings=true",
        )
    expected_dimension = expected_manifest.get("embedding_dimension")
    if (
        not isinstance(expected_dimension, int)
        or isinstance(expected_dimension, bool)
        or expected_dimension <= 0
    ):
        raise ValueError(
            "Expected embedding-enabled target lacks a valid embedding dimension",
        )

    manifest_identity_fields = (
        "checkpoint_sha256",
        "normalizer_sha256",
        "config_sha256",
        "effective_config_sha256",
        "split_sha256",
        "raw_dataset_sha256",
        "dataset_revision",
        "graphkit_commit",
        "dataset_split",
        "grid_name",
        "task",
        "seed",
        "corpus_scenario_count",
    )
    for field in manifest_identity_fields:
        if field not in actual_manifest:
            raise ValueError(f"Actual run manifest lacks required identity {field!r}")
        if field not in expected_manifest:
            raise ValueError(f"Expected run manifest lacks required identity {field!r}")
        if actual_manifest[field] != expected_manifest[field]:
            raise ValueError(
                f"Baseline run identity {field!r} disagrees: "
                f"actual={actual_manifest[field]!r}, "
                f"expected={expected_manifest[field]!r}",
            )

    runtime_identity_fields = (
        "python_version",
        "platform",
        "graphkit",
        "torch",
        "torchvision",
        "torchaudio",
        "torch_scatter",
        "python_executable",
        "environment_sha256",
        "requirements_lock_sha256",
        "source_record_sha256",
        "source_repository",
        "source_commit",
    )
    for field in runtime_identity_fields:
        if actual_runtime[field] != expected_runtime[field]:
            raise ValueError(
                f"Baseline runtime identity {field!r} disagrees: "
                f"actual={actual_runtime[field]!r}, "
                f"expected={expected_runtime[field]!r}",
            )


def compare_keyed_baseline(
    *,
    prediction_path: str | Path,
    generator_prediction_path: str | Path,
    expected_prediction_path: str | Path,
    expected_generator_prediction_path: str | Path,
    actual_run_manifest: str | Path,
    expected_run_manifest: str | Path,
    tolerance_config: str | Path | None = None,
) -> KeyedBaselineComparisonResult:
    """Compare two manifest-bound keyed prediction runs."""
    actual_manifest = json.loads(Path(actual_run_manifest).read_text())
    expected_manifest = json.loads(Path(expected_run_manifest).read_text())
    _verify_comparison_compatibility(
        actual_run_manifest,
        actual_manifest,
        expected_run_manifest,
        expected_manifest,
    )
    verify_manifest_artifacts(
        actual_run_manifest,
        (
            prediction_path,
            generator_prediction_path,
        ),
    )
    verify_manifest_artifacts(
        expected_run_manifest,
        (
            expected_prediction_path,
            expected_generator_prediction_path,
        ),
    )
    policy = _load_approved_tolerance_policy(tolerance_config)
    tolerances = policy["absolute_tolerances"]
    bus_count, bus_differences = _compare_keyed_table(
        prediction_path,
        expected_prediction_path,
        ("scenario", "bus"),
        {name: float(value) for name, value in tolerances["bus"].items()},
    )
    generator_count, generator_differences = _compare_keyed_table(
        generator_prediction_path,
        expected_generator_prediction_path,
        ("scenario", "idx"),
        {name: float(value) for name, value in tolerances["generator"].items()},
    )
    return KeyedBaselineComparisonResult(
        bus_row_count=bus_count,
        generator_row_count=generator_count,
        maximum_absolute_differences={
            **{f"bus.{key}": value for key, value in bus_differences.items()},
            **{
                f"generator.{key}": value
                for key, value in generator_differences.items()
            },
        },
        tolerance_policy_id=str(APPROVED_TOLERANCE_POLICY["policy_id"]),
        tolerance_policy_sha256=APPROVED_TOLERANCE_POLICY_SHA256,
    )


def compare_baseline_residuals(
    prediction_path: str | Path,
    expected_metrics_path: str | Path,
    absolute_tolerance: float = 1e-6,
) -> BaselineComparisonResult:
    """Compare coherent predictions with published aggregate residual metrics."""
    if absolute_tolerance < 0:
        raise ValueError("Baseline comparison tolerance cannot be negative")
    predictions = pd.read_parquet(
        prediction_path,
        columns=["active res. (MW)", "reactive res. (MVar)"],
    )
    expected_table = pd.read_csv(expected_metrics_path)
    expected = dict(zip(expected_table["Metric"], expected_table["Value"], strict=True))
    required = {
        "Avg. active res. (MW)",
        "Avg. reactive res. (MVar)",
    }
    missing = required - expected.keys()
    if missing:
        raise ValueError(f"Published baseline metrics are missing: {sorted(missing)}")
    actual_active = float(predictions["active res. (MW)"].abs().mean())
    actual_reactive = float(predictions["reactive res. (MVar)"].abs().mean())
    expected_active = float(expected["Avg. active res. (MW)"])
    expected_reactive = float(expected["Avg. reactive res. (MVar)"])
    differences = {
        "active residual": abs(actual_active - expected_active),
        "reactive residual": abs(actual_reactive - expected_reactive),
    }
    failed = {
        name: difference
        for name, difference in differences.items()
        if difference > absolute_tolerance
    }
    if failed:
        raise ValueError(
            "Baseline residual comparison exceeded absolute tolerance "
            f"{absolute_tolerance}: {failed}",
        )
    return BaselineComparisonResult(
        absolute_tolerance=absolute_tolerance,
        active_residual_actual=actual_active,
        active_residual_expected=expected_active,
        reactive_residual_actual=actual_reactive,
        reactive_residual_expected=expected_reactive,
    )


def keyed_embedding_dimension(path: str | Path) -> int:
    """Return one positive embedding dimension shared by every keyed row."""
    parquet = pq.ParquetFile(path)
    if "embedding" not in parquet.schema_arrow.names:
        raise ValueError(f"Keyed embedding artifact lacks embedding values: {path}")

    dimension: int | None = None
    row_count = 0
    for batch in parquet.iter_batches(
        columns=["embedding"],
        batch_size=50_000,
    ):
        values = batch.column(0)
        if values.null_count:
            raise ValueError(f"Keyed embedding artifact contains null values: {path}")
        if not (
            pa.types.is_list(values.type)
            or pa.types.is_large_list(values.type)
        ):
            raise ValueError(
                f"Keyed embedding column must contain lists: {path}",
            )

        lengths = np.diff(
            values.offsets.to_numpy(zero_copy_only=False),
        )
        unique_lengths = set(int(length) for length in lengths)
        if len(unique_lengths) != 1:
            raise ValueError(
                f"Keyed embedding dimensions are inconsistent: {path}",
            )
        batch_dimension = unique_lengths.pop()
        if batch_dimension <= 0:
            raise ValueError(
                f"Keyed embedding dimension must be positive: {path}",
            )
        if dimension is None:
            dimension = batch_dimension
        elif dimension != batch_dimension:
            raise ValueError(
                f"Keyed embedding dimensions are inconsistent: {path}",
            )
        row_count += batch.num_rows

    if row_count == 0 or dimension is None:
        raise ValueError(f"Keyed embedding artifact is empty: {path}")
    return dimension


def verify_embedding_entity_coverage(
    *,
    embedding_path: str | Path,
    prediction_path: str | Path,
    generator_prediction_path: str | Path,
    label: str,
) -> None:
    embedding = pd.read_parquet(
        embedding_path,
        columns=["scenario_id", "entity_type", "entity_id"],
    )
    bus = pd.read_parquet(
        prediction_path,
        columns=["scenario", "bus"],
    )
    generator = pd.read_parquet(
        generator_prediction_path,
        columns=["scenario", "idx"],
    )

    if embedding.duplicated(
        ["scenario_id", "entity_type", "entity_id"],
    ).any():
        raise ValueError(f"{label} contains duplicate embedding entity keys")
    if bus.duplicated(["scenario", "bus"]).any():
        raise ValueError(f"{label} contains duplicate bus prediction keys")
    if generator.duplicated(["scenario", "idx"]).any():
        raise ValueError(f"{label} contains duplicate generator prediction keys")

    embedding_keys = {
        (str(scenario), str(entity_type), int(entity_id))
        for scenario, entity_type, entity_id in embedding.itertuples(
            index=False,
            name=None,
        )
    }
    expected_keys = {
        (str(scenario), "bus", int(entity_id))
        for scenario, entity_id in bus.itertuples(index=False, name=None)
    }
    expected_keys.update(
        {
            (str(scenario), "generator", int(entity_id))
            for scenario, entity_id in generator.itertuples(
                index=False,
                name=None,
            )
        },
    )

    missing = sorted(expected_keys - embedding_keys)
    extra = sorted(embedding_keys - expected_keys)
    if missing or extra:
        raise ValueError(
            f"{label} entity coverage disagrees with predictions: "
            f"missing={missing[:10]}, extra={extra[:10]}",
        )


def canonicalize_graphkit_embeddings(
    output_dir: str | Path,
    dataset_split: str,
) -> Path:
    """Normalize native graphkit bus/gen embedding tables into the keyed schema."""
    root = Path(output_dir)
    target = root / "case14_ieee_keyed_embeddings.parquet"
    temporary = target.with_suffix(".parquet.tmp")
    schema = pa.schema(
        [
            ("scenario_id", pa.int64()),
            ("entity_type", pa.string()),
            ("entity_id", pa.int64()),
            ("correction_step", pa.int8()),
            ("embedding_stage", pa.string()),
            ("embedding", pa.list_(pa.float32())),
            ("dataset_split", pa.string()),
        ],
    )
    writer = pq.ParquetWriter(temporary, schema, compression="zstd")
    try:
        for entity_type, filename, id_column in (
            ("bus", "case14_ieee_bus_embeddings.parquet", "bus"),
            ("generator", "case14_ieee_gen_embeddings.parquet", "idx"),
        ):
            source = root / filename
            if not source.is_file():
                raise FileNotFoundError(f"Graphkit embedding output is missing: {source}")
            for batch in pq.ParquetFile(source).iter_batches(batch_size=50_000):
                names = batch.schema.names
                embedding_names = sorted(
                    name for name in names if name.startswith("emb_")
                )
                if not embedding_names:
                    raise ValueError(f"No embedding columns found in {source}")
                values = np.column_stack(
                    [
                        batch.column(names.index(name)).to_numpy(
                            zero_copy_only=False,
                        )
                        for name in embedding_names
                    ],
                ).astype(np.float32, copy=False)
                count = batch.num_rows
                writer.write_table(
                    pa.Table.from_arrays(
                        [
                            batch.column(names.index("scenario")).cast(pa.int64()),
                            pa.array([entity_type] * count),
                            batch.column(names.index(id_column)).cast(pa.int64()),
                            pa.nulls(count, type=pa.int8()),
                            pa.array(["final"] * count),
                            pa.ListArray.from_arrays(
                                pa.array(
                                    np.arange(
                                        0,
                                        (count + 1) * values.shape[1],
                                        values.shape[1],
                                        dtype=np.int64,
                                    ),
                                ),
                                pa.array(values.reshape(-1), type=pa.float32()),
                            ),
                            pa.array([dataset_split] * count),
                        ],
                        schema=schema,
                    ),
                )
    finally:
        writer.close()
    temporary.replace(target)
    return target


def _compact_native_embedding_artifacts(output_dir: str | Path) -> None:
    root = Path(output_dir)
    for filename in NATIVE_EMBEDDING_FILENAMES:
        native = root / filename
        if not native.is_file():
            raise FileNotFoundError(
                f"Native embedding artifact cannot be compacted because it is "
                f"missing: {native}",
            )
        native.unlink()


def _find_mlflow_run(mlflow_root: Path, run_name: str) -> Path:
    matches = []
    for meta_path in mlflow_root.glob("*/*/meta.yaml"):
        try:
            summary = yaml.safe_load(meta_path.read_text())
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(summary, dict) or summary.get("run_name") != run_name:
            continue
        if summary.get("status") != 3:
            continue
        load_mlflow_run_metadata(meta_path.parent)
        matches.append(meta_path.parent)
    if len(matches) != 1:
        raise ValueError(
            f"Expected one generated MLflow run named {run_name!r}, found {matches}",
        )
    return matches[0]


def _normalize_copied_mlflow_run(run_path: Path, split_directory: str) -> None:
    meta_path = run_path / "meta.yaml"
    metadata = yaml.safe_load(meta_path.read_text())
    metadata["artifact_uri"] = "artifacts"
    meta_path.write_text(yaml.safe_dump(metadata, sort_keys=False))
    (run_path / "params/args/data/split_from_existing_files").write_text(
        split_directory,
    )
    source_name = run_path / "tags/mlflow.source.name"
    if source_name.is_file():
        source_name.write_text("gridfm_graphkit.__main__")


def _load_graphkit_source_record(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        if "=" not in line:
            raise ValueError(f"Invalid graphkit source record line: {line!r}")
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    if set(values) != {"repository", "commit"}:
        raise ValueError(
            "Graphkit source record must contain exactly repository and commit",
        )
    if not values["repository"].startswith("https://github.com/"):
        raise ValueError("Graphkit source repository must be a public GitHub HTTPS URL")
    commit = values["commit"]
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("Graphkit source commit must be a lowercase 40-character SHA")
    return values


def _graphkit_python_identity(
    graphkit_python: str | Path,
    graphkit_workdir: str | Path,
) -> tuple[Path, str]:
    """Identify a selected venv executable without following its final symlink."""
    selected = Path(os.path.abspath(graphkit_python))
    workdir = Path(graphkit_workdir).resolve()
    if not selected.is_file():
        raise FileNotFoundError(
            f"Graphkit Python executable does not exist: {selected}",
        )
    try:
        relative = selected.relative_to(workdir)
    except ValueError as error:
        raise ValueError(
            "Graphkit Python executable must be inside the graphkit checkout",
        ) from error
    return selected, relative.as_posix()


def _capture_runtime_environment(
    graphkit_python: Path,
    graphkit_workdir: Path,
    requirements_lock: Path,
    source_record: Path,
    output_path: Path,
) -> dict[str, object]:
    graphkit_python, executable = _graphkit_python_identity(
        graphkit_python,
        graphkit_workdir,
    )
    graphkit_workdir = graphkit_workdir.resolve()
    requirements_lock = requirements_lock.resolve()
    source_record = source_record.resolve()
    if not requirements_lock.is_file():
        raise FileNotFoundError(
            f"Graphkit requirements lock does not exist: {requirements_lock}",
        )
    if not source_record.is_file():
        raise FileNotFoundError(
            f"Graphkit source record does not exist: {source_record}",
        )

    active_freeze = subprocess.run(
        [
            str(graphkit_python),
            "-m",
            "pip",
            "freeze",
            "--all",
            "--exclude-editable",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if active_freeze != requirements_lock.read_text():
        raise ValueError(
            "Active graphkit environment does not match the supplied "
            "portable requirements lock",
        )
    if any(
        marker in active_freeze
        for marker in ("git+ssh", "file://", "-e ", "sarnavi-mahesh")
    ):
        raise ValueError("Graphkit requirements lock contains a non-portable source")

    declared_source = _load_graphkit_source_record(source_record)
    actual_repository = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        check=True,
        capture_output=True,
        text=True,
        cwd=graphkit_workdir,
    ).stdout.strip()
    actual_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=graphkit_workdir,
    ).stdout.strip()
    if actual_repository != declared_source["repository"]:
        raise ValueError(
            "Graphkit checkout origin disagrees with the source record: "
            f"{actual_repository!r} != {declared_source['repository']!r}",
        )
    if actual_commit != declared_source["commit"]:
        raise ValueError(
            "Graphkit checkout commit disagrees with the source record: "
            f"{actual_commit!r} != {declared_source['commit']!r}",
        )

    output_path.write_text(active_freeze)
    copied_lock = output_path.parent / "graphkit-requirements.lock"
    copied_source = output_path.parent / "graphkit-source.txt"
    shutil.copy2(requirements_lock, copied_lock)
    shutil.copy2(source_record, copied_source)

    script = """
import importlib.metadata
import json
import platform

packages = {
    name: importlib.metadata.version(distribution)
    for name, distribution in {
        "graphkit": "gridfm-graphkit",
        "torch": "torch",
        "torchvision": "torchvision",
        "torchaudio": "torchaudio",
        "torch_scatter": "torch-scatter",
    }.items()
}
print(json.dumps({
    "python_version": platform.python_version(),
    "platform": platform.platform(),
    **packages,
}, sort_keys=True))
"""
    runtime = json.loads(
        subprocess.run(
            [str(graphkit_python), "-c", script],
            check=True,
            capture_output=True,
            text=True,
        ).stdout,
    )
    return {
        **runtime,
        "python_executable": executable,
        "environment_path": output_path.name,
        "environment_sha256": sha256_file(output_path),
        "requirements_lock_path": copied_lock.name,
        "requirements_lock_sha256": sha256_file(copied_lock),
        "source_record_path": copied_source.name,
        "source_record_sha256": sha256_file(copied_source),
        "source_repository": declared_source["repository"],
        "source_commit": declared_source["commit"],
    }



def _verify_selected_runtime(
    *,
    graphkit_python: Path,
    graphkit_workdir: Path,
    requirements_lock: Path,
    source_record: Path,
    recorded_runtime: dict[str, object],
) -> None:
    selected_graphkit_python = graphkit_python
    graphkit_workdir = graphkit_workdir.resolve()
    requirements_lock = requirements_lock.resolve()
    source_record = source_record.resolve()

    if sha256_file(requirements_lock) != recorded_runtime["requirements_lock_sha256"]:
        raise ValueError(
            "Selected graphkit requirements lock disagrees with run manifest",
        )
    if sha256_file(source_record) != recorded_runtime["source_record_sha256"]:
        raise ValueError(
            "Selected graphkit source record disagrees with run manifest",
        )

    graphkit_python, executable = _graphkit_python_identity(
        selected_graphkit_python,
        graphkit_workdir,
    )

    active_freeze = subprocess.run(
        [
            str(graphkit_python),
            "-m",
            "pip",
            "freeze",
            "--all",
            "--exclude-editable",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if active_freeze != requirements_lock.read_text():
        raise ValueError(
            "Selected graphkit executable does not match the requirements lock",
        )

    declared_source = _load_graphkit_source_record(source_record)
    if declared_source["repository"] != recorded_runtime["source_repository"]:
        raise ValueError(
            "Selected graphkit source repository disagrees with run manifest",
        )
    if declared_source["commit"] != recorded_runtime["source_commit"]:
        raise ValueError(
            "Selected graphkit source commit disagrees with run manifest",
        )

    actual_repository = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        check=True,
        capture_output=True,
        text=True,
        cwd=graphkit_workdir,
    ).stdout.strip()
    actual_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=graphkit_workdir,
    ).stdout.strip()
    if actual_repository != declared_source["repository"]:
        raise ValueError(
            "Selected graphkit checkout origin disagrees with run manifest",
        )
    if actual_commit != declared_source["commit"]:
        raise ValueError(
            "Selected graphkit checkout commit disagrees with run manifest",
        )

    if executable != recorded_runtime["python_executable"]:
        raise ValueError(
            "Selected graphkit executable path disagrees with run manifest",
        )

    script = """
import importlib.metadata
import json
import platform

print(json.dumps({
    "python_version": platform.python_version(),
    "platform": platform.platform(),
    "graphkit": importlib.metadata.version("gridfm-graphkit"),
    "torch": importlib.metadata.version("torch"),
    "torchvision": importlib.metadata.version("torchvision"),
    "torchaudio": importlib.metadata.version("torchaudio"),
    "torch_scatter": importlib.metadata.version("torch-scatter"),
}, sort_keys=True))
"""
    active_runtime = json.loads(
        subprocess.run(
            [str(graphkit_python), "-c", script],
            check=True,
            capture_output=True,
            text=True,
        ).stdout,
    )
    for field, actual in active_runtime.items():
        if actual != recorded_runtime[field]:
            raise ValueError(
                f"Selected graphkit runtime {field!r} disagrees with "
                f"run manifest: {actual!r} != {recorded_runtime[field]!r}",
            )


def run_baseline_inference(spec: BaselineInferenceSpec) -> BaselineInferenceResult:
    """Run graphkit prediction and emit a hash-bound provenance manifest."""
    output_dir = spec.output_dir.resolve()
    graphkit_python = spec.graphkit_python.absolute()
    config_path = spec.config.resolve()
    checkpoint_path = spec.checkpoint.resolve()
    normalizer_path = spec.normalizer.resolve()
    data_path = spec.data_path.resolve()
    split_path = spec.split_artifact.resolve()
    manifest_path = output_dir / "run_manifest.json"
    if spec.reuse_verified_run:
        if not spec.export_embeddings:
            raise ValueError(
                "Verified-run reuse is supported only for manifested embedding runs",
            )
        verified = verify_run_manifest(
            manifest_path,
            prediction_path=output_dir / "case14_ieee_predictions.parquet",
            generator_prediction_path=(
                output_dir / "case14_ieee_gen_predictions.parquet"
            ),
            target_embedding_path=(
                output_dir / "case14_ieee_keyed_embeddings.parquet"
            ),
            checkpoint_path=checkpoint_path,
            normalizer_path=normalizer_path,
            config_path=config_path,
            split_path=split_path,
            raw_dataset_path=data_path / "case14_ieee/raw",
        )
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("run_name") != spec.run_name:
            raise ValueError("Reused run name disagrees with existing manifest")
        if manifest.get("dataset_split") != spec.dataset_split:
            raise ValueError("Reused dataset split disagrees with existing manifest")
        _verify_selected_runtime(
            graphkit_python=graphkit_python,
            graphkit_workdir=spec.graphkit_workdir,
            requirements_lock=spec.graphkit_requirements_lock,
            source_record=spec.graphkit_source_record,
            recorded_runtime=verified.runtime,
        )
        return BaselineInferenceResult(
            output_dir=output_dir,
            manifest_path=manifest_path,
            mlflow_run_dir=verified.mlflow_run_dir,
            run_id=verified.run_id,
        )
    if (
        output_dir.exists()
        and any(output_dir.iterdir())
        and not spec.reuse_verified_run
    ):
        raise ValueError(f"Baseline output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    graphkit_workdir = spec.graphkit_workdir.resolve()
    mlflow_source_root = graphkit_workdir / "mlruns"
    effective_config_path = output_dir / "effective_config.yaml"
    effective_config = yaml.safe_load(config_path.read_text())
    effective_config["data"]["split_from_existing_files"] = str(split_path.parent)
    effective_config_path.write_text(
        yaml.safe_dump(effective_config, sort_keys=False),
    )
    command = [
        str(graphkit_python),
        "-m",
        "gridfm_graphkit",
        "predict",
        "--config",
        str(effective_config_path),
        "--config_conversion",
        "auto_inline",
        "--model_path",
        str(checkpoint_path),
        "--normalizer_stats",
        str(normalizer_path),
        "--data_path",
        str(data_path),
        "--output_path",
        str(output_dir),
        "--num_workers",
        str(spec.num_workers),
        "--log_dir",
        "mlruns",
        "--exp_name",
        spec.experiment_name,
        "--run_name",
        spec.run_name,
    ]
    if spec.export_embeddings:
        command.append("--get_embeddings")
    environment = os.environ.copy()
    environment["MLFLOW_ALLOW_FILE_STORE"] = "true"
    subprocess.run(
        command,
        check=True,
        env=environment,
        cwd=graphkit_workdir,
    )

    portable_config = yaml.safe_load(config_path.read_text())
    portable_config["data"]["split_from_existing_files"] = (
        spec.split_artifact.parent.as_posix()
    )
    effective_config_path.write_text(
        yaml.safe_dump(portable_config, sort_keys=False),
    )
    verify_split_membership(
        parquet_scenario_ids(output_dir / "case14_ieee_predictions.parquet"),
        split_path,
        "Bus predictions",
    )
    verify_split_membership(
        parquet_scenario_ids(output_dir / "case14_ieee_gen_predictions.parquet"),
        split_path,
        "Generator predictions",
    )
    keyed_path = None
    embedding_dimension = None
    if spec.export_embeddings:
        keyed_path = canonicalize_graphkit_embeddings(
            output_dir,
            spec.dataset_split,
        )
        verify_split_membership(
            parquet_scenario_ids(keyed_path, "scenario_id"),
            split_path,
            "Target embeddings",
        )
        verify_embedding_entity_coverage(
            embedding_path=keyed_path,
            prediction_path=output_dir / "case14_ieee_predictions.parquet",
            generator_prediction_path=(
                output_dir / "case14_ieee_gen_predictions.parquet"
            ),
            label="Generated embeddings",
        )
        embedding_dimension = keyed_embedding_dimension(keyed_path)
    source_run_path = _find_mlflow_run(mlflow_source_root, spec.run_name)
    run_path = output_dir / "mlflow_run"
    if not run_path.exists():
        shutil.copytree(source_run_path, run_path)
    _normalize_copied_mlflow_run(
        run_path,
        spec.split_artifact.parent.as_posix(),
    )
    config = load_config_metadata(effective_config_path)
    run = load_mlflow_run_metadata(run_path)
    verify_run_matches_config(
        run,
        config,
        require_embeddings=spec.export_embeddings,
    )
    raw_dataset_path = data_path / config.grid_name / "raw"
    runtime_environment_path = output_dir / "runtime_environment.txt"
    runtime = _capture_runtime_environment(
        graphkit_python,
        graphkit_workdir,
        spec.graphkit_requirements_lock,
        spec.graphkit_source_record,
        runtime_environment_path,
    )
    if spec.export_embeddings:
        _compact_native_embedding_artifacts(output_dir)
    artifact_paths = [
        output_dir / "case14_ieee_predictions.parquet",
        output_dir / "case14_ieee_gen_predictions.parquet",
        effective_config_path,
        runtime_environment_path,
        output_dir / "graphkit-requirements.lock",
        output_dir / "graphkit-source.txt",
    ]
    if spec.export_embeddings:
        artifact_paths.append(keyed_path)
    artifact_paths = sorted(artifact_paths)
    manifest = {
        "format_version": 1,
        "run_id": run.run_id,
        "run_name": run.run_name,
        "run_status": run.status,
        "start_time_ms": run.start_time_ms,
        "end_time_ms": run.end_time_ms,
        "graphkit_commit": run.graphkit_commit,
        "seed": run.seed,
        "grid_name": run.grid_name,
        "task": run.task,
        "dataset_split": spec.dataset_split,
        "export_embeddings": spec.export_embeddings,
        "native_embeddings_compacted": spec.export_embeddings,
        "embedding_dimension": embedding_dimension,
        "corpus_scenario_count": run.corpus_scenario_count,
        "dataset_revision": extract_huggingface_revision(raw_dataset_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "normalizer_sha256": sha256_file(normalizer_path),
        "config_sha256": sha256_file(config_path),
        "effective_config_sha256": sha256_file(effective_config_path),
        "split_sha256": sha256_file(split_path),
        "raw_dataset_sha256": sha256_path(raw_dataset_path),
        "mlflow_run_path": run_path.relative_to(output_dir).as_posix(),
        "mlflow_run_sha256": sha256_path(run_path),
        "runtime": runtime,
        "artifacts": [
            {
                "path": path.relative_to(output_dir).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in artifact_paths
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return BaselineInferenceResult(
        output_dir=output_dir,
        manifest_path=manifest_path,
        mlflow_run_dir=run_path,
        run_id=run.run_id,
    )


def verify_run_manifest(
    manifest_path: str | Path,
    *,
    prediction_path: str | Path,
    generator_prediction_path: str | Path,
    target_embedding_path: str | Path,
    checkpoint_path: str | Path,
    normalizer_path: str | Path,
    config_path: str | Path,
    split_path: str | Path,
    raw_dataset_path: str | Path,
) -> VerifiedRunManifest:
    """Verify that target evidence artifacts belong to one manifested run."""
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Run manifest does not exist: {path}")
    value = json.loads(path.read_text())
    if value.get("format_version") != 1:
        raise ValueError(f"Unsupported run manifest format: {path}")
    runtime = _verify_comparison_runtime(path, value, "Target")
    root = path.parent.resolve()
    recorded_artifacts = {
        (root / record["path"]).resolve(): record["sha256"]
        for record in value.get("artifacts", ())
    }
    compacted = value.get("native_embeddings_compacted")
    if compacted not in (None, False, True):
        raise ValueError("Run manifest has invalid native embedding compaction state")
    if compacted is True:
        for filename in NATIVE_EMBEDDING_FILENAMES:
            native = (root / filename).resolve()
            if native in recorded_artifacts or native.is_file():
                raise ValueError(
                    f"Compacted run retains native embedding artifact: {native}",
                )
    for artifact, expected_hash in recorded_artifacts.items():
        if not artifact.is_file():
            raise FileNotFoundError(f"Manifest artifact does not exist: {artifact}")
        if sha256_file(artifact) != expected_hash:
            raise ValueError(f"Manifest artifact hash disagrees: {artifact}")

    effective_config_path = (root / "effective_config.yaml").resolve()
    effective_config_hash = recorded_artifacts.get(effective_config_path)
    if effective_config_hash is None:
        raise ValueError(
            "Effective configuration is absent from run manifest artifacts",
        )
    if value.get("effective_config_sha256") != effective_config_hash:
        raise ValueError(
            "Effective configuration hash disagrees with run manifest",
        )

    for artifact in (
        prediction_path,
        generator_prediction_path,
        target_embedding_path,
    ):
        resolved = Path(artifact).resolve()
        expected_hash = recorded_artifacts.get(resolved)
        if expected_hash is None:
            raise ValueError(
                f"Target evidence artifact is not recorded by run manifest: {artifact}",
            )
    source_hashes = {
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "normalizer_sha256": sha256_file(normalizer_path),
        "config_sha256": sha256_file(config_path),
        "split_sha256": sha256_file(split_path),
        "raw_dataset_sha256": sha256_path(raw_dataset_path),
    }
    for name, actual_hash in source_hashes.items():
        if value.get(name) != actual_hash:
            raise ValueError(f"{name} disagrees with run manifest {path}")
    mlflow_run_dir = (root / value["mlflow_run_path"]).resolve()
    run = load_mlflow_run_metadata(mlflow_run_dir)
    if run.status != 3 or not run.get_embeddings:
        raise ValueError(f"Target run {run.run_id} must be completed with embeddings")
    if run.run_id != value.get("run_id"):
        raise ValueError("MLflow run ID disagrees with run manifest")
    if run.graphkit_commit != value.get("graphkit_commit"):
        raise ValueError("Graphkit commit disagrees with run manifest")
    if sha256_path(mlflow_run_dir) != value.get("mlflow_run_sha256"):
        raise ValueError("MLflow metadata hash disagrees with run manifest")
    effective_config = load_config_metadata(effective_config_path)
    verify_run_matches_config(run, effective_config)
    allowed_files = set(recorded_artifacts)
    allowed_files.add(path.resolve())
    allowed_files.update(
        item.resolve()
        for item in mlflow_run_dir.rglob("*")
        if item.is_file()
    )
    allowed_files.update(
        Path(item).resolve()
        for item in (
            checkpoint_path,
            normalizer_path,
            config_path,
            split_path,
        )
    )
    raw_root = Path(raw_dataset_path).resolve()
    allowed_files.update(
        item.resolve()
        for item in raw_root.rglob("*")
        if item.is_file()
    )
    actual_files = {
        item.resolve()
        for item in root.rglob("*")
        if item.is_file()
    }
    unknown_files = sorted(actual_files - allowed_files)
    if unknown_files:
        raise ValueError(f"Run directory contains unmanifested files: {unknown_files}")
    recorded_dimension = value.get("embedding_dimension")
    if (
        not isinstance(recorded_dimension, int)
        or isinstance(recorded_dimension, bool)
        or recorded_dimension <= 0
    ):
        raise ValueError("Run manifest lacks a valid embedding dimension")
    actual_dimension = keyed_embedding_dimension(target_embedding_path)
    if actual_dimension != recorded_dimension:
        raise ValueError(
            "Embedding dimension disagrees with run manifest: "
            f"{actual_dimension} != {recorded_dimension}",
        )
    verify_embedding_entity_coverage(
        embedding_path=target_embedding_path,
        prediction_path=prediction_path,
        generator_prediction_path=generator_prediction_path,
        label="Target embeddings",
    )
    verify_split_membership(
        parquet_scenario_ids(target_embedding_path, "scenario_id"),
        split_path,
        "Target embeddings",
    )
    return VerifiedRunManifest(
        manifest_path=path,
        mlflow_run_dir=mlflow_run_dir,
        run_id=run.run_id,
        graphkit_commit=run.graphkit_commit,
        dataset_revision=str(value["dataset_revision"]),
        embedding_dimension=recorded_dimension,
        runtime=runtime,
    )


def verify_reproduction_manifest(
    manifest_path: str | Path,
    *,
    prediction_path: str | Path,
    generator_prediction_path: str | Path,
    checkpoint_path: str | Path,
    normalizer_path: str | Path,
    config_path: str | Path,
    split_path: str | Path,
    raw_dataset_path: str | Path,
) -> VerifiedReproductionManifest:
    """Verify an uninstrumented clean-runtime reproduction and its environment."""
    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Reproduction manifest does not exist: {path}")
    value = json.loads(path.read_text())
    if (
        value.get("format_version") != 1
        or value.get("dataset_split") != "test"
        or value.get("export_embeddings") is not False
        or value.get("run_status") != 3
    ):
        raise ValueError(f"Unsupported reproduction manifest: {path}")
    runtime = value.get("runtime")
    required_runtime = {
        "python_version",
        "platform",
        "torch",
        "torchvision",
        "torchaudio",
        "torch_scatter",
        "environment_path",
        "environment_sha256",
    }
    if not isinstance(runtime, dict) or not required_runtime <= runtime.keys():
        raise ValueError("Reproduction manifest lacks complete runtime provenance")
    environment_path = (path.parent / runtime["environment_path"]).resolve()
    if not environment_path.is_file():
        raise FileNotFoundError(
            f"Runtime environment artifact does not exist: {environment_path}",
        )
    if sha256_file(environment_path) != runtime["environment_sha256"]:
        raise ValueError("Runtime environment hash disagrees with reproduction manifest")
    verify_manifest_artifacts(
        path,
        (
            prediction_path,
            generator_prediction_path,
            environment_path,
        ),
    )
    source_hashes = {
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "normalizer_sha256": sha256_file(normalizer_path),
        "config_sha256": sha256_file(config_path),
        "split_sha256": sha256_file(split_path),
        "raw_dataset_sha256": sha256_path(raw_dataset_path),
    }
    for name, actual_hash in source_hashes.items():
        if value.get(name) != actual_hash:
            raise ValueError(f"{name} disagrees with reproduction manifest {path}")
    run_path = (path.parent / value["mlflow_run_path"]).resolve()
    run = load_mlflow_run_metadata(run_path)
    if run.status != 3 or run.get_embeddings:
        raise ValueError(
            f"Reproduction run {run.run_id} must be completed without embeddings",
        )
    if run.run_id != value.get("run_id"):
        raise ValueError("MLflow run ID disagrees with reproduction manifest")
    if run.graphkit_commit != value.get("graphkit_commit"):
        raise ValueError("Graphkit commit disagrees with reproduction manifest")
    if sha256_path(run_path) != value.get("mlflow_run_sha256"):
        raise ValueError("MLflow metadata hash disagrees with reproduction manifest")
    verify_split_membership(
        parquet_scenario_ids(prediction_path),
        split_path,
        "Reproduction bus predictions",
    )
    verify_split_membership(
        parquet_scenario_ids(generator_prediction_path),
        split_path,
        "Reproduction generator predictions",
    )
    return VerifiedReproductionManifest(
        manifest_path=path,
        mlflow_run_dir=run_path,
        run_id=run.run_id,
        graphkit_commit=run.graphkit_commit,
        dataset_revision=str(value["dataset_revision"]),
        start_time_ms=run.start_time_ms,
        end_time_ms=run.end_time_ms,
        runtime_environment_sha256=str(runtime["environment_sha256"]),
        runtime={key: str(runtime[key]) for key in required_runtime},
    )
