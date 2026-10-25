"""Verified preparation and capture inputs for cohort trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from .provenance import sha256_file
from .splits import load_split_ids


@dataclass(frozen=True)
class CohortInferenceSplitSpec:
    train_split: Path
    validation_split: Path
    test_split: Path
    cohort: Path
    cohort_manifest: Path
    output_dir: Path

    def __post_init__(self) -> None:
        inputs = (
            self.train_split,
            self.validation_split,
            self.test_split,
            self.cohort,
            self.cohort_manifest,
        )
        if len(set(inputs)) != len(inputs):
            raise ValueError(
                "Cohort inference split inputs must all differ"
            )
        if self.output_dir in inputs:
            raise ValueError(
                "Cohort inference split output must differ from its inputs"
            )


@dataclass(frozen=True)
class CohortInferenceSplitResult:
    output_dir: Path
    train_split: Path
    validation_split: Path
    test_split: Path
    manifest: Path
    selected_scenario_count: int


@dataclass(frozen=True)
class TrajectoryCaptureSpec:
    baseline_dir: Path
    instrumented_dir: Path
    split_artifact: Path
    tolerance_config: Path
    output: Path
    manifest_output: Path


@dataclass(frozen=True)
class TrajectoryCaptureResult:
    trajectory: Path
    manifest: Path
    row_count: int
    scenario_count: int
    correction_step_count: int
    embedding_dimension: int


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


def _verify_manifest_artifact(
    manifest: dict[str, Any],
    key: str,
    expected_path: Path,
) -> None:
    record = manifest.get(key)
    if (
        not isinstance(record, dict)
        or set(record) != {"path", "sha256"}
    ):
        raise ValueError(
            f"Cohort manifest has an invalid {key} record"
        )
    recorded_path = record.get("path")
    if (
        not isinstance(recorded_path, str)
        or Path(recorded_path).resolve()
        != expected_path.resolve()
    ):
        raise ValueError(
            f"Cohort manifest {key} path disagrees "
            "with the explicit input"
        )
    if record.get("sha256") != sha256_file(expected_path):
        raise ValueError(
            f"Cohort manifest {key} hash disagrees "
            f"with {expected_path}"
        )


def _selected_cohort_ids(
    cohort_path: Path,
    manifest: dict[str, Any],
) -> tuple[str, ...]:
    if not cohort_path.is_file():
        raise FileNotFoundError(
            f"Cohort artifact does not exist: {cohort_path}"
        )
    table = pd.read_parquet(
        cohort_path,
        columns=["scenario_id"],
    )
    values = pd.to_numeric(
        table["scenario_id"],
        errors="raise",
    )
    if values.isna().any() or not (
        values == values.astype("int64")
    ).all():
        raise ValueError(
            "Cohort scenario IDs must be finite integers"
        )
    selected = tuple(
        str(value)
        for value in values.astype("int64")
    )
    if len(selected) != 20:
        raise ValueError(
            "Cohort inference requires exactly 20 scenarios"
        )
    if len(selected) != len(set(selected)):
        raise ValueError(
            "Cohort inference scenarios must be unique"
        )

    records = manifest.get("selected_scenarios")
    if not isinstance(records, list):
        raise ValueError(
            "Cohort manifest selected_scenarios must be a list"
        )
    manifest_ids = tuple(
        str(record.get("scenario_id"))
        for record in records
        if isinstance(record, dict)
    )
    if len(manifest_ids) != len(records):
        raise ValueError(
            "Cohort manifest selected_scenarios entries "
            "must be objects"
        )
    if selected != manifest_ids:
        raise ValueError(
            "Cohort scenario order disagrees with the cohort manifest"
        )
    if manifest.get("selected_test_scenario_count") != len(
        selected
    ):
        raise ValueError(
            "Cohort manifest selected scenario count disagrees"
        )
    return selected


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    temporary.replace(path)


def prepare_cohort_inference_splits(
    spec: CohortInferenceSplitSpec,
) -> CohortInferenceSplitResult:
    """Create a Graphkit split view whose test set is the frozen cohort."""
    output_dir = spec.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(
            "Cohort inference split directory must be empty: "
            f"{output_dir}"
        )

    train_ids = load_split_ids(spec.train_split)
    validation_ids = load_split_ids(spec.validation_split)
    test_ids = load_split_ids(spec.test_split)
    split_sets = {
        "train": set(train_ids),
        "validation": set(validation_ids),
        "test": set(test_ids),
    }
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        overlap = split_sets[left] & split_sets[right]
        if overlap:
            raise ValueError(
                f"Authoritative {left} and {right} splits overlap "
                f"for {len(overlap)} scenarios"
            )

    manifest = _load_json_object(
        spec.cohort_manifest,
        "Cohort manifest",
    )
    if manifest.get("schema_version") != 1:
        raise ValueError(
            "Cohort manifest has an unsupported schema version"
        )
    if manifest.get("artifact_type") != (
        "frozen_contrasting_scenario_cohort"
    ):
        raise ValueError(
            "Cohort manifest has an unexpected artifact type"
        )
    _verify_manifest_artifact(
        manifest,
        "cohort_artifact",
        spec.cohort,
    )
    selected_ids = _selected_cohort_ids(
        spec.cohort,
        manifest,
    )

    missing = sorted(
        set(selected_ids) - split_sets["test"],
        key=int,
    )
    if missing:
        raise ValueError(
            "Frozen cohort contains scenarios outside the "
            f"authoritative test split: {missing}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_train = output_dir / "train.pt"
    output_validation = output_dir / "val.pt"
    output_test = output_dir / "test.pt"
    output_manifest = (
        output_dir / "cohort_split_manifest.json"
    )

    shutil.copy2(spec.train_split, output_train)
    shutil.copy2(spec.validation_split, output_validation)
    temporary_test = output_test.with_suffix(".pt.tmp")
    torch.save(
        [
            torch.tensor(
                int(scenario_id),
                dtype=torch.int64,
            )
            for scenario_id in selected_ids
        ],
        temporary_test,
    )
    temporary_test.replace(output_test)

    if sha256_file(output_train) != sha256_file(
        spec.train_split
    ):
        raise ValueError(
            "Copied training split hash disagrees"
        )
    if sha256_file(output_validation) != sha256_file(
        spec.validation_split
    ):
        raise ValueError(
            "Copied validation split hash disagrees"
        )
    if load_split_ids(output_test) != selected_ids:
        raise ValueError(
            "Generated cohort test split has incorrect membership "
            "or ordering"
        )

    split_manifest = {
        "schema_version": 1,
        "artifact_type": "cohort_inference_split_view",
        "cohort_artifact": {
            "path": str(spec.cohort),
            "sha256": sha256_file(spec.cohort),
        },
        "cohort_manifest": {
            "path": str(spec.cohort_manifest),
            "sha256": sha256_file(
                spec.cohort_manifest
            ),
        },
        "authoritative_splits": {
            "train": {
                "path": str(spec.train_split),
                "sha256": sha256_file(
                    spec.train_split
                ),
                "scenario_count": len(train_ids),
            },
            "validation": {
                "path": str(
                    spec.validation_split
                ),
                "sha256": sha256_file(
                    spec.validation_split
                ),
                "scenario_count": len(
                    validation_ids
                ),
            },
            "test": {
                "path": str(spec.test_split),
                "sha256": sha256_file(
                    spec.test_split
                ),
                "scenario_count": len(test_ids),
            },
        },
        "selected_scenario_count": len(
            selected_ids
        ),
        "selected_scenario_ids": list(
            selected_ids
        ),
        "outputs": {
            "train": {
                "path": str(output_train),
                "sha256": sha256_file(
                    output_train
                ),
            },
            "validation": {
                "path": str(
                    output_validation
                ),
                "sha256": sha256_file(
                    output_validation
                ),
            },
            "test": {
                "path": str(output_test),
                "sha256": sha256_file(
                    output_test
                ),
            },
        },
    }
    _write_json(
        output_manifest,
        split_manifest,
    )

    return CohortInferenceSplitResult(
        output_dir=output_dir,
        train_split=output_train,
        validation_split=output_validation,
        test_split=output_test,
        manifest=output_manifest,
        selected_scenario_count=len(
            selected_ids
        ),
    )


def _artifact_record(
    manifest: dict[str, Any],
    root: Path,
    name: str,
) -> Path:
    matches = [
        record
        for record in manifest.get("artifacts", ())
        if record.get("path") == name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Run manifest must record exactly one {name} artifact"
        )
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(
            f"Run artifact does not exist: {path}"
        )
    if matches[0].get("sha256") != sha256_file(path):
        raise ValueError(
            f"Run manifest hash disagrees for {path}"
        )
    return path


def _trajectory_frame(
    path: Path,
    *,
    entity_type: str,
    entity_column: str,
) -> pd.DataFrame:
    table = pd.read_parquet(path)
    embedding_columns = sorted(
        column
        for column in table.columns
        if column.startswith("emb_")
    )
    if embedding_columns != [
        f"emb_{index:03d}"
        for index in range(len(embedding_columns))
    ]:
        raise ValueError(
            f"Trajectory embedding columns are not contiguous: {path}"
        )
    required = {
        "scenario",
        entity_column,
        "correction_step",
        "embedding_stage",
        *embedding_columns,
    }
    if not embedding_columns or not required.issubset(table.columns):
        raise ValueError(
            f"Trajectory table has an invalid schema: {path}"
        )
    embeddings = table[embedding_columns].to_numpy(
        dtype=np.float32,
        copy=True,
    )
    if not np.isfinite(embeddings).all():
        raise ValueError(
            f"Trajectory table contains non-finite embeddings: {path}"
        )
    result = pd.DataFrame(
        {
            "scenario_id": table["scenario"].astype(
                "int64"
            ),
            "entity_type": entity_type,
            "entity_id": table[entity_column].astype(
                "int64"
            ),
            "correction_step": table[
                "correction_step"
            ].astype("int16"),
            "embedding_stage": table[
                "embedding_stage"
            ].astype(str),
            "embedding": list(embeddings),
        }
    )
    return result


def _compare_prediction_tables(
    baseline: Path,
    instrumented: Path,
    *,
    keys: list[str],
    tolerances: dict[str, float],
) -> dict[str, float]:
    left = pd.read_parquet(baseline).sort_values(
        keys
    ).reset_index(drop=True)
    right = pd.read_parquet(instrumented).sort_values(
        keys
    ).reset_index(drop=True)
    if not left[keys].equals(right[keys]):
        raise ValueError(
            f"Prediction row keys disagree: {baseline.name}"
        )
    maxima: dict[str, float] = {}
    for column, tolerance in tolerances.items():
        if column not in left or column not in right:
            raise ValueError(
                f"Tolerance column is missing: {column}"
            )
        maximum = float(
            np.max(
                np.abs(
                    left[column].to_numpy(dtype=float)
                    - right[column].to_numpy(dtype=float)
                )
            )
        )
        maxima[column] = maximum
        if maximum > float(tolerance):
            raise ValueError(
                f"{column} difference {maximum} exceeds "
                f"tolerance {tolerance}"
            )
    return maxima


def finalize_trajectory_capture(
    spec: TrajectoryCaptureSpec,
) -> TrajectoryCaptureResult:
    """Canonicalize and verify a cohort trajectory capture run."""
    baseline_dir = spec.baseline_dir.resolve()
    instrumented_dir = spec.instrumented_dir.resolve()
    output = spec.output.resolve()
    manifest_output = spec.manifest_output.resolve()
    if output == manifest_output:
        raise ValueError(
            "Trajectory artifact and manifest outputs must differ"
        )
    for path in (output, manifest_output):
        if path.exists():
            raise ValueError(
                f"Trajectory capture output already exists: {path}"
            )

    baseline_manifest_path = baseline_dir / "run_manifest.json"
    instrumented_manifest_path = (
        instrumented_dir / "run_manifest.json"
    )
    baseline_manifest = _load_json_object(
        baseline_manifest_path,
        "Baseline run manifest",
    )
    instrumented_manifest = _load_json_object(
        instrumented_manifest_path,
        "Instrumented run manifest",
    )
    if instrumented_manifest.get("plugin_modules") != [
        "genco_graphkit_capture.plugin"
    ]:
        raise ValueError(
            "Instrumented run does not record the trajectory plugin"
        )
    identity_fields = (
        "checkpoint_sha256",
        "normalizer_sha256",
        "split_sha256",
        "raw_dataset_sha256",
        "graphkit_commit",
        "seed",
        "grid_name",
        "task",
        "dataset_split",
    )
    for field in identity_fields:
        if baseline_manifest.get(field) != (
            instrumented_manifest.get(field)
        ):
            raise ValueError(
                f"Run identity field disagrees: {field}"
            )
    if instrumented_manifest.get(
        "embedding_dimension"
    ) != 384:
        raise ValueError(
            "Trajectory capture requires 384-dimensional embeddings"
        )

    tolerance_policy = yaml.safe_load(
        spec.tolerance_config.read_text()
    )
    tolerances = tolerance_policy[
        "absolute_tolerances"
    ]
    bus_maxima = _compare_prediction_tables(
        _artifact_record(
            baseline_manifest,
            baseline_dir,
            "case14_ieee_predictions.parquet",
        ),
        _artifact_record(
            instrumented_manifest,
            instrumented_dir,
            "case14_ieee_predictions.parquet",
        ),
        keys=["scenario", "bus"],
        tolerances=tolerances["bus"],
    )
    generator_maxima = _compare_prediction_tables(
        _artifact_record(
            baseline_manifest,
            baseline_dir,
            "case14_ieee_gen_predictions.parquet",
        ),
        _artifact_record(
            instrumented_manifest,
            instrumented_dir,
            "case14_ieee_gen_predictions.parquet",
        ),
        keys=["scenario", "idx", "bus"],
        tolerances=tolerances["generator"],
    )

    bus = _trajectory_frame(
        _artifact_record(
            instrumented_manifest,
            instrumented_dir,
            "case14_ieee_bus_trajectory_predictions.parquet",
        ),
        entity_type="bus",
        entity_column="bus",
    )
    generator = _trajectory_frame(
        _artifact_record(
            instrumented_manifest,
            instrumented_dir,
            "case14_ieee_gen_trajectory_predictions.parquet",
        ),
        entity_type="generator",
        entity_column="idx",
    )
    trajectory = pd.concat(
        [bus, generator],
        ignore_index=True,
    ).sort_values(
        [
            "scenario_id",
            "entity_type",
            "entity_id",
            "correction_step",
        ],
        kind="stable",
    ).reset_index(drop=True)
    keys = [
        "scenario_id",
        "entity_type",
        "entity_id",
        "correction_step",
    ]
    if trajectory.duplicated(keys).any():
        raise ValueError(
            "Trajectory contains duplicate entity-step rows"
        )
    if set(trajectory["embedding_stage"]) != {
        "prediction_head_input"
    }:
        raise ValueError(
            "Trajectory has an unexpected embedding stage"
        )

    split_ids = load_split_ids(spec.split_artifact)
    if {
        str(value)
        for value in trajectory["scenario_id"]
    } != set(split_ids):
        raise ValueError(
            "Trajectory scenarios disagree with the cohort split"
        )
    step_counts = trajectory.groupby(
        [
            "scenario_id",
            "entity_type",
            "entity_id",
        ]
    )["correction_step"].agg(list)
    expected_steps = list(range(1, 13))
    if not all(
        steps == expected_steps
        for steps in step_counts
    ):
        raise ValueError(
            "Every trajectory entity must contain correction steps 1-12"
        )

    instrumented_embeddings = pd.read_parquet(
        _artifact_record(
            instrumented_manifest,
            instrumented_dir,
            "case14_ieee_keyed_embeddings.parquet",
        )
    )
    final_rows = trajectory[
        trajectory["correction_step"] == 12
    ].sort_values(keys[:3]).reset_index(drop=True)
    expected_final = instrumented_embeddings.sort_values(
        keys[:3]
    ).reset_index(drop=True)
    if not final_rows[keys[:3]].equals(
        expected_final[keys[:3]]
    ):
        raise ValueError(
            "Final trajectory keys disagree with final embeddings"
        )
    final_values = np.stack(final_rows["embedding"])
    expected_values = np.stack(
        expected_final["embedding"]
    )
    if not np.array_equal(final_values, expected_values):
        raise ValueError(
            "Final trajectory snapshots disagree with "
            "instrumented final embeddings"
        )

    baseline_embeddings = pd.read_parquet(
        _artifact_record(
            baseline_manifest,
            baseline_dir,
            "case14_ieee_keyed_embeddings.parquet",
        )
    ).sort_values(keys[:3]).reset_index(drop=True)
    if not baseline_embeddings[keys[:3]].equals(
        expected_final[keys[:3]]
    ):
        raise ValueError(
            "Baseline and instrumented final embedding keys disagree"
        )
    embedding_maximum = float(
        np.max(
            np.abs(
                np.stack(baseline_embeddings["embedding"])
                - expected_values
            )
        )
    )
    embedding_tolerance = float(
        tolerances["final_embedding"]
    )
    if embedding_maximum > embedding_tolerance:
        raise ValueError(
            "Final embedding difference "
            f"{embedding_maximum} exceeds tolerance "
            f"{embedding_tolerance}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    trajectory.to_parquet(output, index=False)
    capture_manifest = {
        "schema_version": 1,
        "artifact_type": "genco_cohort_latent_trajectory",
        "trajectory": {
            "path": str(spec.output),
            "sha256": sha256_file(output),
            "row_count": len(trajectory),
            "scenario_count": len(split_ids),
            "entity_count": len(step_counts),
            "correction_step_count": 12,
            "embedding_dimension": 384,
            "embedding_stage": "prediction_head_input",
        },
        "baseline_run_manifest": {
            "path": str(
                spec.baseline_dir / "run_manifest.json"
            ),
            "sha256": sha256_file(
                baseline_manifest_path
            ),
            "run_id": baseline_manifest["run_id"],
        },
        "instrumented_run_manifest": {
            "path": str(
                spec.instrumented_dir / "run_manifest.json"
            ),
            "sha256": sha256_file(
                instrumented_manifest_path
            ),
            "run_id": instrumented_manifest["run_id"],
        },
        "split_artifact": {
            "path": str(spec.split_artifact),
            "sha256": sha256_file(
                spec.split_artifact
            ),
        },
        "tolerance_policy": {
            "path": str(spec.tolerance_config),
            "sha256": sha256_file(
                spec.tolerance_config
            ),
            "policy_id": tolerance_policy["policy_id"],
        },
        "equivalence": {
            "passed": True,
            "bus_maximum_absolute_differences": bus_maxima,
            "generator_maximum_absolute_differences": (
                generator_maxima
            ),
            "final_embedding_maximum_absolute_difference": (
                embedding_maximum
            ),
            "final_embedding_tolerance": embedding_tolerance,
            "instrumented_final_snapshot_exact": True,
        },
    }
    _write_json(manifest_output, capture_manifest)
    return TrajectoryCaptureResult(
        trajectory=output,
        manifest=manifest_output,
        row_count=len(trajectory),
        scenario_count=len(split_ids),
        correction_step_count=12,
        embedding_dimension=384,
    )
