"""Verified inputs for scenario scoring and cohort selection."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from .baseline import verify_manifest_artifacts
from .embeddings import load_pooled_keyed_embeddings
from .provenance import sha256_file
from .splits import (
    load_split_ids,
    verify_split_membership,
)

_COMPATIBILITY_FIELDS = (
    "checkpoint_sha256",
    "config_sha256",
    "corpus_scenario_count",
    "dataset_revision",
    "embedding_dimension",
    "graphkit_commit",
    "grid_name",
    "normalizer_sha256",
    "raw_dataset_sha256",
    "seed",
    "task",
)

_RUNTIME_COMPATIBILITY_FIELDS = (
    "environment_sha256",
    "graphkit",
    "platform",
    "python_version",
    "requirements_lock_sha256",
    "source_commit",
    "source_record_sha256",
    "source_repository",
    "torch",
    "torch_scatter",
    "torchaudio",
    "torchvision",
)

_RUNTIME_PATH_FIELDS = (
    "environment_path",
    "requirements_lock_path",
    "source_record_path",
)

ProgressCallback = Callable[[int, int], None]

SCORING_POLICY_ID = "genco-scenario-scoring-v1"
POOLING_METHOD = "mean_bus_concatenate_mean_generator"
DISTANCE_METRIC = "euclidean"
RESIDUAL_METRIC = "mean_absolute_bus_pbe"

LOW_PERCENTILE = 0.25
HIGH_PERCENTILE = 0.75
PERCENTILE_RANK_METHOD = "average"
QUANTILE_METHOD = "linear"

QUANTILES = (
    0.0,
    0.01,
    0.05,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
    1.0,
)

GROUP_CORNERS = {
    "familiar_good": (0.0, 0.0),
    "familiar_bad": (0.0, 1.0),
    "unfamiliar_good": (1.0, 0.0),
    "unfamiliar_bad": (1.0, 1.0),
}

SELECTION_POLICY_ID = (
    "genco-contrasting-cohort-v2"
)

CORNER_REPRESENTATIVE_COUNT = 3
BOUNDARY_REPRESENTATIVE_COUNT = 1
DIVERSITY_REPRESENTATIVE_COUNT = 1

REPRESENTATIVES_PER_GROUP = (
    CORNER_REPRESENTATIVE_COUNT
    + BOUNDARY_REPRESENTATIVE_COUNT
    + DIVERSITY_REPRESENTATIVE_COUNT
)

GROUP_BOUNDARIES = {
    "familiar_good": (
        LOW_PERCENTILE,
        LOW_PERCENTILE,
    ),
    "familiar_bad": (
        LOW_PERCENTILE,
        HIGH_PERCENTILE,
    ),
    "unfamiliar_good": (
        HIGH_PERCENTILE,
        LOW_PERCENTILE,
    ),
    "unfamiliar_bad": (
        HIGH_PERCENTILE,
        HIGH_PERCENTILE,
    ),
}

@dataclass(frozen=True)
class ScenarioScoringInputSpec:
    target_keyed_embeddings: Path
    reference_keyed_embeddings: Path
    prediction_parquet: Path
    test_split: Path
    train_split: Path
    target_run_manifest: Path
    reference_run_manifest: Path


@dataclass(frozen=True)
class VerifiedScenarioScoringInputs:
    target_run_id: str
    reference_run_id: str
    checkpoint_sha256: str
    dataset_revision: str
    graphkit_commit: str
    embedding_dimension: int
    target_scenario_count: int
    reference_scenario_count: int

@dataclass(frozen=True)
class ScenarioScoringSpec(ScenarioScoringInputSpec):
    output: Path
    manifest_output: Path
    device: str = "auto"
    query_block_size: int = 256
    reference_block_size: int = 8192

    def __post_init__(self) -> None:
        if self.query_block_size <= 0:
            raise ValueError(
                "query_block_size must be positive"
            )
        if self.reference_block_size <= 0:
            raise ValueError(
                "reference_block_size must be positive"
            )
        if self.output == self.manifest_output:
            raise ValueError(
                "Score table and manifest paths must differ"
            )


@dataclass(frozen=True)
class ScenarioScoringResult:
    output: Path
    manifest: Path
    scenario_count: int
    reference_count: int
    device: str
    timings_seconds: dict[str, float]

@dataclass(frozen=True)
class ScoreSummarySpec:
    scores: Path
    score_manifest: Path
    output: Path

    def __post_init__(self) -> None:
        if self.output in {
            self.scores,
            self.score_manifest,
        }:
            raise ValueError(
                "Distribution summary output must differ "
                "from its inputs"
            )


@dataclass(frozen=True)
class ScoreSummaryResult:
    output: Path
    scenario_count: int
    thresholds: dict[str, float]
    group_candidate_counts: dict[str, int]
    excluded_scenario_count: int
    spearman: float


@dataclass(frozen=True)
class ContrastingScenarioSelectionSpec:
    scores: Path
    score_manifest: Path
    summary: Path
    output: Path
    manifest_output: Path
    diagnostics_output: Path
    training_reference_ids_output: Path

    def __post_init__(self) -> None:
        paths = (
            self.scores,
            self.score_manifest,
            self.summary,
            self.output,
            self.manifest_output,
            self.diagnostics_output,
            self.training_reference_ids_output,
        )
        if len(set(paths)) != len(paths):
            raise ValueError(
                "Selection input and output paths must "
                "all differ"
            )


@dataclass(frozen=True)
class ContrastingScenarioSelectionResult:
    output: Path
    manifest: Path
    diagnostics: Path
    training_reference_ids: Path
    selected_scenario_count: int
    selected_reference_count: int


def _load_run_manifest(
    path: Path,
    label: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"{label} run manifest does not exist: {path}"
        )

    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{label} run manifest is not valid JSON: {path}"
        ) from error

    if not isinstance(value, dict):
        raise ValueError(
            f"{label} run manifest must contain a JSON object"
        )

    required = {
        "artifacts",
        "checkpoint_sha256",
        "config_sha256",
        "corpus_scenario_count",
        "dataset_revision",
        "dataset_split",
        "embedding_dimension",
        "export_embeddings",
        "format_version",
        "graphkit_commit",
        "grid_name",
        "normalizer_sha256",
        "raw_dataset_sha256",
        "run_id",
        "run_status",
        "runtime",
        "seed",
        "split_sha256",
        "task",
    }
    missing = sorted(required - value.keys())
    if missing:
        raise ValueError(
            f"{label} run manifest lacks required fields: {missing}"
        )

    if value["format_version"] != 1:
        raise ValueError(
            f"{label} run manifest has unsupported format_version: "
            f"{value['format_version']!r}"
        )

    if value["run_status"] != 3:
        raise ValueError(
            f"{label} run manifest does not describe a completed run"
        )

    if value["export_embeddings"] is not True:
        raise ValueError(
            f"{label} run manifest does not declare embedding export"
        )

    embedding_dimension = value["embedding_dimension"]
    if (
        not isinstance(embedding_dimension, int)
        or isinstance(embedding_dimension, bool)
        or embedding_dimension <= 0
    ):
        raise ValueError(
            f"{label} run manifest has invalid embedding_dimension"
        )

    artifacts = value["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError(
            f"{label} run manifest has no artifact inventory"
        )

    observed_paths: set[str] = set()

    for index, record in enumerate(artifacts):
        if not isinstance(record, dict):
            raise ValueError(
                f"{label} artifact record {index} must be an object"
            )

        if set(record) != {"path", "sha256"}:
            raise ValueError(
                f"{label} artifact record {index} must contain "
                "exactly path and sha256"
            )

        relative = record["path"]
        digest = record["sha256"]

        if not isinstance(relative, str) or not relative:
            raise ValueError(
                f"{label} artifact record {index} has invalid path"
            )

        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"{label} artifact record {index} has unsafe path"
            )

        if relative in observed_paths:
            raise ValueError(
                f"{label} run manifest contains duplicate artifact "
                f"path: {relative}"
            )
        observed_paths.add(relative)

        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in digest
            )
        ):
            raise ValueError(
                f"{label} artifact record {index} has invalid SHA-256"
            )

    runtime = value["runtime"]
    if not isinstance(runtime, dict):
        raise ValueError(
            f"{label} run manifest lacks runtime provenance"
        )

    required_runtime = (
        set(_RUNTIME_COMPATIBILITY_FIELDS)
        | set(_RUNTIME_PATH_FIELDS)
    )
    missing_runtime = sorted(
        required_runtime - runtime.keys()
    )
    if missing_runtime:
        raise ValueError(
            f"{label} runtime provenance lacks fields: "
            f"{missing_runtime}"
        )

    if runtime["source_commit"] != value["graphkit_commit"]:
        raise ValueError(
            f"{label} runtime source commit disagrees with "
            "graphkit_commit"
        )

    return value


def _verify_manifest_role(
    manifest: dict[str, Any],
    *,
    expected_split: str,
    label: str,
) -> None:
    if manifest["dataset_split"] != expected_split:
        raise ValueError(
            f"{label} run manifest must describe "
            f"dataset_split={expected_split!r}"
        )

def _verify_runtime_artifacts(
    manifest_path: Path,
    manifest: dict[str, Any],
) -> None:
    root = manifest_path.resolve().parent
    runtime = manifest["runtime"]

    paths = tuple(
        root / str(runtime[field])
        for field in _RUNTIME_PATH_FIELDS
    )
    verify_manifest_artifacts(
        manifest_path,
        paths,
    )


def _verify_compatible_runs(
    target: dict[str, Any],
    reference: dict[str, Any],
) -> None:
    disagreements = {
        field: (target[field], reference[field])
        for field in _COMPATIBILITY_FIELDS
        if target[field] != reference[field]
    }
    if disagreements:
        raise ValueError(
            "Target and reference run manifests are incompatible: "
            f"{disagreements}"
        )

    target_runtime = target["runtime"]
    reference_runtime = reference["runtime"]

    runtime_disagreements = {
        field: (
            target_runtime[field],
            reference_runtime[field],
        )
        for field in _RUNTIME_COMPATIBILITY_FIELDS
        if target_runtime[field] != reference_runtime[field]
    }
    if runtime_disagreements:
        raise ValueError(
            "Target and reference runtime provenance is "
            f"incompatible: {runtime_disagreements}"
        )


def verify_scenario_scoring_inputs(
    spec: ScenarioScoringInputSpec,
) -> VerifiedScenarioScoringInputs:
    target_manifest = _load_run_manifest(
        spec.target_run_manifest,
        "Target",
    )
    reference_manifest = _load_run_manifest(
        spec.reference_run_manifest,
        "Reference",
    )

    _verify_manifest_role(
        target_manifest,
        expected_split="test",
        label="Target",
    )
    _verify_manifest_role(
        reference_manifest,
        expected_split="train",
        label="Reference",
    )
    _verify_compatible_runs(
        target_manifest,
        reference_manifest,
    )

    verify_manifest_artifacts(
        spec.target_run_manifest,
        (
            spec.target_keyed_embeddings,
            spec.prediction_parquet,
        ),
    )
    verify_manifest_artifacts(
        spec.reference_run_manifest,
        (spec.reference_keyed_embeddings,),
    )

    _verify_runtime_artifacts(
        spec.target_run_manifest,
        target_manifest,
    )
    _verify_runtime_artifacts(
        spec.reference_run_manifest,
        reference_manifest,
    )

    target_split_hash = sha256_file(spec.test_split)
    if target_split_hash != target_manifest["split_sha256"]:
        raise ValueError(
            "Test split hash disagrees with the target run manifest"
        )

    reference_split_hash = sha256_file(spec.train_split)
    if reference_split_hash != reference_manifest["split_sha256"]:
        raise ValueError(
            "Train split hash disagrees with the reference run manifest"
        )

    test_ids = load_split_ids(spec.test_split)
    train_ids = load_split_ids(spec.train_split)

    overlap = set(test_ids) & set(train_ids)
    if overlap:
        examples = sorted(overlap, key=int)[:10]
        raise ValueError(
            "Train and test splits overlap: "
            f"count={len(overlap)}, examples={examples}"
        )

    return VerifiedScenarioScoringInputs(
        target_run_id=str(target_manifest["run_id"]),
        reference_run_id=str(reference_manifest["run_id"]),
        checkpoint_sha256=str(
            target_manifest["checkpoint_sha256"]
        ),
        dataset_revision=str(
            target_manifest["dataset_revision"]
        ),
        graphkit_commit=str(
            target_manifest["graphkit_commit"]
        ),
        embedding_dimension=int(
            target_manifest["embedding_dimension"]
        ),
        target_scenario_count=len(test_ids),
        reference_scenario_count=len(train_ids),
    )
def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            separators=(",", ": "),
        )
        + "\n"
    ).encode("utf-8")


def _write_json(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )
    try:
        temporary.write_bytes(
            _json_bytes(value)
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()

def _write_text(
    path: Path,
    value: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )
    try:
        temporary.write_text(value)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_json_object(
    path: Path,
    label: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"{label} does not exist: {path}"
        )

    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{label} is not valid JSON: {path}"
        ) from error

    if not isinstance(value, dict):
        raise ValueError(
            f"{label} must contain a JSON object"
        )

    return value

def _write_parquet(
    path: Path,
    table: pd.DataFrame,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )
    try:
        table.to_parquet(
            temporary,
            index=False,
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    try:
        selected = torch.device(requested)
    except (RuntimeError, TypeError) as error:
        raise ValueError(
            f"Unsupported computation device: {requested!r}"
        ) from error

    if selected.type not in {"cpu", "mps", "cuda"}:
        raise ValueError(
            f"Unsupported computation device: {requested!r}"
        )

    if (
        selected.type == "mps"
        and not torch.backends.mps.is_available()
    ):
        raise ValueError(
            "MPS was requested but is unavailable"
        )

    if (
        selected.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise ValueError(
            "CUDA was requested but is unavailable"
        )

    return selected

def _exact_nearest_references(
    queries: np.ndarray,
    references: np.ndarray,
    reference_ids: list[str],
    *,
    device: torch.device,
    query_block_size: int,
    reference_block_size: int,
    progress: ProgressCallback | None = None,
) -> tuple[np.ndarray, list[str]]:
    if query_block_size <= 0:
        raise ValueError(
            "query_block_size must be positive"
        )
    if reference_block_size <= 0:
        raise ValueError(
            "reference_block_size must be positive"
        )

    query_matrix = np.asarray(
        queries,
        dtype=np.float32,
    )
    reference_matrix = np.asarray(
        references,
        dtype=np.float32,
    )

    if query_matrix.ndim != 2:
        raise ValueError(
            "Query embeddings must be two-dimensional"
        )
    if reference_matrix.ndim != 2:
        raise ValueError(
            "Reference embeddings must be two-dimensional"
        )
    if query_matrix.shape[0] == 0:
        raise ValueError(
            "Query embeddings cannot be empty"
        )
    if reference_matrix.shape[0] == 0:
        raise ValueError(
            "Reference embeddings cannot be empty"
        )
    if query_matrix.shape[1] == 0:
        raise ValueError(
            "Embedding dimension must be positive"
        )
    if (
        query_matrix.shape[1]
        != reference_matrix.shape[1]
    ):
        raise ValueError(
            "Target and reference embedding dimensions disagree"
        )
    if reference_matrix.shape[0] != len(reference_ids):
        raise ValueError(
            "Reference IDs and embedding rows are not aligned"
        )
    if len(reference_ids) != len(set(reference_ids)):
        raise ValueError(
            "Reference scenario IDs must be unique"
        )
    if not np.isfinite(query_matrix).all():
        raise ValueError(
            "Target embeddings contain non-finite values"
        )
    if not np.isfinite(reference_matrix).all():
        raise ValueError(
            "Reference embeddings contain non-finite values"
        )

    normalized_reference_ids = [
        str(scenario_id)
        for scenario_id in reference_ids
    ]

    query_tensor = torch.from_numpy(
        query_matrix,
    ).to(device)
    query_norms = (
        query_tensor * query_tensor
    ).sum(dim=1)

    best_squared = np.full(
        query_matrix.shape[0],
        np.inf,
        dtype=np.float32,
    )
    best_indices = np.full(
        query_matrix.shape[0],
        -1,
        dtype=np.int64,
    )

    with torch.inference_mode():
        for reference_start in range(
            0,
            len(reference_matrix),
            reference_block_size,
        ):
            reference_end = min(
                reference_start + reference_block_size,
                len(reference_matrix),
            )

            reference_block = torch.from_numpy(
                reference_matrix[
                    reference_start:reference_end
                ],
            ).to(device)
            reference_norms = (
                reference_block * reference_block
            ).sum(dim=1)

            for query_start in range(
                0,
                len(query_matrix),
                query_block_size,
            ):
                query_end = min(
                    query_start + query_block_size,
                    len(query_matrix),
                )

                squared = (
                    query_norms[
                        query_start:query_end,
                        None,
                    ]
                    + reference_norms[None, :]
                    - 2.0
                    * query_tensor[
                        query_start:query_end
                    ]
                    @ reference_block.T
                )
                squared.clamp_(min=0.0)

                values, indices = torch.min(
                    squared,
                    dim=1,
                )

                candidate_values = (
                    values.detach()
                    .cpu()
                    .numpy()
                    .astype(
                        np.float32,
                        copy=False,
                    )
                )
                candidate_indices = (
                    indices.detach()
                    .cpu()
                    .numpy()
                    .astype(
                        np.int64,
                        copy=False,
                    )
                    + reference_start
                )

                for offset, candidate_index in enumerate(
                    candidate_indices
                ):
                    query_index = query_start + offset
                    candidate_value = candidate_values[offset]
                    current_value = best_squared[query_index]
                    current_index = best_indices[query_index]

                    better_distance = (
                        candidate_value < current_value
                    )
                    equal_distance = (
                        candidate_value == current_value
                    )
                    better_tie = (
                        equal_distance
                        and (
                            current_index < 0
                            or normalized_reference_ids[
                                candidate_index
                            ]
                            < normalized_reference_ids[
                                current_index
                            ]
                        )
                    )

                    if better_distance or better_tie:
                        best_squared[
                            query_index
                        ] = candidate_value
                        best_indices[
                            query_index
                        ] = candidate_index

            del reference_block
            del reference_norms

            if progress is not None:
                progress(
                    reference_end,
                    len(reference_matrix),
                )

    if np.any(best_indices < 0):
        raise RuntimeError(
            "At least one target has no reference match"
        )

    distances = np.sqrt(
        best_squared.astype(np.float64)
    )
    nearest_ids = [
        normalized_reference_ids[index]
        for index in best_indices
    ]

    return distances, nearest_ids


def _load_residuals(
    prediction_path: Path,
    expected_ids: set[str],
) -> pd.DataFrame:
    if not prediction_path.is_file():
        raise FileNotFoundError(
            "Prediction artifact does not exist: "
            f"{prediction_path}"
        )
    if not expected_ids:
        raise ValueError(
            "Expected scenario IDs cannot be empty"
        )

    table = pd.read_parquet(
        prediction_path,
        columns=["scenario", "bus", "PBE"],
    )

    if table.empty:
        raise ValueError(
            "Prediction artifact is empty"
        )

    scenario_values = pd.to_numeric(
        table["scenario"],
        errors="raise",
    )
    if not np.isfinite(
        scenario_values.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Prediction artifact contains non-finite "
            "scenario IDs"
        )
    if not np.equal(
        scenario_values,
        np.floor(scenario_values),
    ).all():
        raise ValueError(
            "Prediction artifact contains non-integer "
            "scenario IDs"
        )

    bus_values = pd.to_numeric(
        table["bus"],
        errors="raise",
    )
    if not np.isfinite(
        bus_values.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Prediction artifact contains non-finite bus IDs"
        )
    if not np.equal(
        bus_values,
        np.floor(bus_values),
    ).all():
        raise ValueError(
            "Prediction artifact contains non-integer bus IDs"
        )

    table["scenario_id"] = (
        scenario_values.astype("int64").astype(str)
    )
    table["bus"] = bus_values.astype("int64")

    if (table["bus"] < 0).any():
        raise ValueError(
            "Prediction artifact contains negative bus IDs"
        )

    pbe_values = pd.to_numeric(
        table["PBE"],
        errors="raise",
    )
    if not np.isfinite(
        pbe_values.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Prediction artifact contains non-finite PBE values"
        )

    table["absolute_pbe"] = pbe_values.abs()

    if table.duplicated(
        ["scenario_id", "bus"]
    ).any():
        raise ValueError(
            "Prediction artifact contains duplicate "
            "scenario/bus keys"
        )

    normalized_expected_ids = {
        str(int(value))
        for value in expected_ids
    }
    observed_ids = set(table["scenario_id"])

    missing = sorted(
        normalized_expected_ids - observed_ids,
        key=int,
    )
    extra = sorted(
        observed_ids - normalized_expected_ids,
        key=int,
    )

    if missing or extra:
        raise ValueError(
            "Prediction scenarios disagree with the test split: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )

    observed_buses = set(
        int(value)
        for value in table["bus"].unique()
    )
    expected_buses = set(
        range(len(observed_buses))
    )

    if observed_buses != expected_buses:
        raise ValueError(
            "Prediction artifact has non-contiguous bus IDs: "
            f"{sorted(observed_buses)}"
        )

    bus_counts = table.groupby(
        "scenario_id",
        sort=False,
    )["bus"].nunique()

    incomplete = bus_counts[
        bus_counts != len(expected_buses)
    ]
    if not incomplete.empty:
        examples = sorted(
            incomplete.index.astype(str),
            key=int,
        )[:10]
        raise ValueError(
            "Prediction artifact has incomplete bus coverage "
            f"for scenarios: {examples}"
        )

    summaries = (
        table.groupby(
            "scenario_id",
            sort=False,
        )["absolute_pbe"]
        .agg(
            mean_absolute_pbe="mean",
            max_absolute_pbe="max",
        )
        .reset_index()
    )

    maximum_rows = (
        table.sort_values(
            [
                "scenario_id",
                "absolute_pbe",
                "bus",
            ],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .drop_duplicates(
            "scenario_id",
            keep="first",
        )
        [["scenario_id", "bus"]]
        .rename(
            columns={"bus": "max_pbe_bus"}
        )
    )

    result = summaries.merge(
        maximum_rows,
        on="scenario_id",
        validate="one_to_one",
    )

    result["_scenario_order"] = pd.to_numeric(
        result["scenario_id"],
        errors="raise",
    ).astype("int64")
    result = (
        result.sort_values(
            "_scenario_order",
            kind="mergesort",
        )
        .drop(columns="_scenario_order")
        .reset_index(drop=True)
    )

    if len(result) != len(normalized_expected_ids):
        raise ValueError(
            "Residual aggregation did not produce one row "
            "per test scenario"
        )

    numeric_columns = [
        "mean_absolute_pbe",
        "max_absolute_pbe",
        "max_pbe_bus",
    ]
    if not np.isfinite(
        result[numeric_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Residual aggregation produced non-finite values"
        )

    return result

def score_scenarios(
    spec: ScenarioScoringSpec,
    *,
    progress: ProgressCallback | None = None,
) -> ScenarioScoringResult:
    total_start = perf_counter()
    timings: dict[str, float] = {}

    phase_start = perf_counter()
    verified = verify_scenario_scoring_inputs(spec)
    timings["input_verification"] = (
        perf_counter() - phase_start
    )

    device = _select_device(spec.device)

    phase_start = perf_counter()
    target_ids, target_matrix = (
        load_pooled_keyed_embeddings(
            spec.target_keyed_embeddings,
            expected_split="test",
        )
    )
    verify_split_membership(
        target_ids,
        spec.test_split,
        "Target embeddings",
    )
    timings["target_embedding_load"] = (
        perf_counter() - phase_start
    )

    phase_start = perf_counter()
    reference_ids, reference_matrix = (
        load_pooled_keyed_embeddings(
            spec.reference_keyed_embeddings,
            expected_split="train",
        )
    )
    verify_split_membership(
        reference_ids,
        spec.train_split,
        "Reference embeddings",
    )
    timings["reference_embedding_load"] = (
        perf_counter() - phase_start
    )

    if len(target_ids) != verified.target_scenario_count:
        raise ValueError(
            "Target embedding count disagrees with the "
            "verified test split"
        )
    if (
        len(reference_ids)
        != verified.reference_scenario_count
    ):
        raise ValueError(
            "Reference embedding count disagrees with the "
            "verified train split"
        )

    if target_matrix.ndim != 2:
        raise ValueError(
            "Target pooled embeddings must be two-dimensional"
        )
    if reference_matrix.ndim != 2:
        raise ValueError(
            "Reference pooled embeddings must be "
            "two-dimensional"
        )
    if (
        target_matrix.shape[1]
        != reference_matrix.shape[1]
    ):
        raise ValueError(
            "Target and reference pooled dimensions disagree"
        )

    expected_pooled_dimension = (
        verified.embedding_dimension * 2
    )
    if (
        target_matrix.shape[1]
        != expected_pooled_dimension
    ):
        raise ValueError(
            "Pooled embedding dimension disagrees with "
            "the run manifests: "
            f"expected={expected_pooled_dimension}, "
            f"actual={target_matrix.shape[1]}"
        )

    phase_start = perf_counter()
    distances, nearest_ids = (
        _exact_nearest_references(
            target_matrix,
            reference_matrix,
            reference_ids,
            device=device,
            query_block_size=spec.query_block_size,
            reference_block_size=(
                spec.reference_block_size
            ),
            progress=progress,
        )
    )
    timings["nearest_reference_search"] = (
        perf_counter() - phase_start
    )

    phase_start = perf_counter()
    residuals = _load_residuals(
        spec.prediction_parquet,
        set(target_ids),
    )
    timings["residual_aggregation"] = (
        perf_counter() - phase_start
    )

    scores = pd.DataFrame(
        {
            "scenario_id": target_ids,
            "nearest_reference_scenario_id": nearest_ids,
            "nearest_reference_distance": distances,
        }
    ).merge(
        residuals,
        on="scenario_id",
        validate="one_to_one",
    )

    scores["distance_metric"] = DISTANCE_METRIC
    scores["pooling_method"] = POOLING_METHOD
    scores["residual_metric"] = RESIDUAL_METRIC

    scores["_scenario_order"] = pd.to_numeric(
        scores["scenario_id"],
        errors="raise",
    ).astype("int64")
    scores = (
        scores.sort_values(
            "_scenario_order",
            kind="mergesort",
        )
        .drop(columns="_scenario_order")
        .reset_index(drop=True)
    )

    expected_columns = [
        "scenario_id",
        "nearest_reference_scenario_id",
        "nearest_reference_distance",
        "mean_absolute_pbe",
        "max_absolute_pbe",
        "max_pbe_bus",
        "distance_metric",
        "pooling_method",
        "residual_metric",
    ]
    scores = scores[expected_columns]

    if len(scores) != verified.target_scenario_count:
        raise ValueError(
            "Scenario scoring did not produce one row "
            "per verified test scenario"
        )
    if scores["scenario_id"].duplicated().any():
        raise ValueError(
            "Scenario scoring produced duplicate scenario IDs"
        )

    numeric_columns = [
        "nearest_reference_distance",
        "mean_absolute_pbe",
        "max_absolute_pbe",
        "max_pbe_bus",
    ]
    if not np.isfinite(
        scores[numeric_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Scenario scores contain non-finite values"
        )
    if (
        scores["nearest_reference_distance"] < 0
    ).any():
        raise ValueError(
            "Scenario scores contain negative distances"
        )

    phase_start = perf_counter()
    _write_parquet(
        spec.output,
        scores,
    )
    timings["score_table_write"] = (
        perf_counter() - phase_start
    )

    timings["total_before_manifest"] = (
        perf_counter() - total_start
    )

    manifest = {
        "schema_version": 1,
        "artifact_type": "scenario_scoring_table",
        "scoring_policy_id": SCORING_POLICY_ID,
        "row_count": len(scores),
        "target_split": "test",
        "reference_split": "train",
        "target_scenario_count": len(target_ids),
        "reference_scenario_count": len(reference_ids),
        "entity_embedding_dimension": int(
            verified.embedding_dimension
        ),
        "pooled_embedding_dimension": int(
            target_matrix.shape[1]
        ),
        "distance_metric": DISTANCE_METRIC,
        "pooling_method": POOLING_METHOD,
        "residual_metric": RESIDUAL_METRIC,
        "numeric_dtype": "float32-neighbor-search",
        "device": str(device),
        "query_block_size": spec.query_block_size,
        "reference_block_size": (
            spec.reference_block_size
        ),
        "verified_identity": {
            "target_run_id": verified.target_run_id,
            "reference_run_id": (
                verified.reference_run_id
            ),
            "checkpoint_sha256": (
                verified.checkpoint_sha256
            ),
            "dataset_revision": (
                verified.dataset_revision
            ),
            "graphkit_commit": (
                verified.graphkit_commit
            ),
        },
        "inputs": {
            "target_keyed_embeddings": {
                "path": str(
                    spec.target_keyed_embeddings
                ),
                "sha256": sha256_file(
                    spec.target_keyed_embeddings
                ),
            },
            "reference_keyed_embeddings": {
                "path": str(
                    spec.reference_keyed_embeddings
                ),
                "sha256": sha256_file(
                    spec.reference_keyed_embeddings
                ),
            },
            "prediction_parquet": {
                "path": str(
                    spec.prediction_parquet
                ),
                "sha256": sha256_file(
                    spec.prediction_parquet
                ),
            },
            "test_split": {
                "path": str(spec.test_split),
                "sha256": sha256_file(
                    spec.test_split
                ),
            },
            "train_split": {
                "path": str(spec.train_split),
                "sha256": sha256_file(
                    spec.train_split
                ),
            },
            "target_run_manifest": {
                "path": str(
                    spec.target_run_manifest
                ),
                "sha256": sha256_file(
                    spec.target_run_manifest
                ),
            },
            "reference_run_manifest": {
                "path": str(
                    spec.reference_run_manifest
                ),
                "sha256": sha256_file(
                    spec.reference_run_manifest
                ),
            },
        },
        "timings_seconds": {
            name: round(value, 6)
            for name, value in timings.items()
        },
        "output": {
            "path": str(spec.output),
            "sha256": sha256_file(spec.output),
        },
    }

    phase_start = perf_counter()
    _write_json(
        spec.manifest_output,
        manifest,
    )
    timings["manifest_write"] = (
        perf_counter() - phase_start
    )
    timings["total"] = (
        perf_counter() - total_start
    )

    return ScenarioScoringResult(
        output=spec.output,
        manifest=spec.manifest_output,
        scenario_count=len(scores),
        reference_count=len(reference_ids),
        device=str(device),
        timings_seconds={
            name: round(value, 6)
            for name, value in timings.items()
        },
    )

def _load_verified_score_table(
    scores_path: Path,
    manifest_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not scores_path.is_file():
        raise FileNotFoundError(
            f"Scenario score table does not exist: "
            f"{scores_path}"
        )

    manifest = _load_json_object(
        manifest_path,
        "Scenario scoring manifest",
    )

    required_manifest_fields = {
        "artifact_type",
        "scoring_policy_id",
        "row_count",
        "target_scenario_count",
        "reference_scenario_count",
        "target_split",
        "reference_split",
        "distance_metric",
        "pooling_method",
        "residual_metric",
        "output",
    }
    missing_manifest_fields = sorted(
        required_manifest_fields - manifest.keys()
    )
    if missing_manifest_fields:
        raise ValueError(
            "Scenario scoring manifest lacks fields: "
            f"{missing_manifest_fields}"
        )

    if (
        manifest["artifact_type"]
        != "scenario_scoring_table"
    ):
        raise ValueError(
            "Scenario scoring manifest has the wrong "
            "artifact type"
        )

    if (
        manifest["scoring_policy_id"]
        != SCORING_POLICY_ID
    ):
        raise ValueError(
            "Scenario scoring manifest has an unsupported "
            "scoring policy"
        )

    if manifest["target_split"] != "test":
        raise ValueError(
            "Scenario scoring manifest must use the test "
            "target split"
        )

    if manifest["reference_split"] != "train":
        raise ValueError(
            "Scenario scoring manifest must use the train "
            "reference split"
        )

    expected_manifest_values = {
        "distance_metric": DISTANCE_METRIC,
        "pooling_method": POOLING_METHOD,
        "residual_metric": RESIDUAL_METRIC,
    }
    for field, expected in expected_manifest_values.items():
        if manifest[field] != expected:
            raise ValueError(
                "Scenario scoring manifest has unexpected "
                f"{field}: {manifest[field]!r}"
            )

    output_record = manifest["output"]
    if not isinstance(output_record, dict):
        raise ValueError(
            "Scenario scoring manifest output must be "
            "an object"
        )

    recorded_hash = output_record.get("sha256")
    if (
        not isinstance(recorded_hash, str)
        or len(recorded_hash) != 64
    ):
        raise ValueError(
            "Scenario scoring manifest has an invalid "
            "output SHA-256"
        )

    actual_hash = sha256_file(scores_path)
    if actual_hash != recorded_hash:
        raise ValueError(
            "Scenario score table hash disagrees with "
            "its scoring manifest"
        )

    scores = pd.read_parquet(scores_path)

    required_columns = {
        "scenario_id",
        "nearest_reference_scenario_id",
        "nearest_reference_distance",
        "mean_absolute_pbe",
        "max_absolute_pbe",
        "max_pbe_bus",
        "distance_metric",
        "pooling_method",
        "residual_metric",
    }
    missing_columns = sorted(
        required_columns - set(scores.columns)
    )
    if missing_columns:
        raise ValueError(
            "Scenario score table lacks columns: "
            f"{missing_columns}"
        )

    if scores.empty:
        raise ValueError(
            "Scenario score table is empty"
        )

    if len(scores) != manifest["row_count"]:
        raise ValueError(
            "Scenario score-table row count disagrees "
            "with its manifest"
        )

    if (
        len(scores)
        != manifest["target_scenario_count"]
    ):
        raise ValueError(
            "Scenario score-table target count disagrees "
            "with its manifest"
        )

    scores = scores.copy()
    scores["scenario_id"] = (
        scores["scenario_id"].astype(str)
    )
    scores["nearest_reference_scenario_id"] = (
        scores[
            "nearest_reference_scenario_id"
        ].astype(str)
    )

    if scores["scenario_id"].duplicated().any():
        raise ValueError(
            "Scenario score table contains duplicate "
            "scenario IDs"
        )

    scenario_numbers = pd.to_numeric(
        scores["scenario_id"],
        errors="raise",
    )
    reference_numbers = pd.to_numeric(
        scores["nearest_reference_scenario_id"],
        errors="raise",
    )

    for label, values in (
        ("scenario IDs", scenario_numbers),
        ("nearest-reference IDs", reference_numbers),
    ):
        array = values.to_numpy(dtype=float)
        if not np.isfinite(array).all():
            raise ValueError(
                f"Scenario score table contains "
                f"non-finite {label}"
            )
        if not np.equal(
            array,
            np.floor(array),
        ).all():
            raise ValueError(
                f"Scenario score table contains "
                f"non-integer {label}"
            )

    numeric_columns = [
        "nearest_reference_distance",
        "mean_absolute_pbe",
        "max_absolute_pbe",
        "max_pbe_bus",
    ]
    numeric_values = scores[
        numeric_columns
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "Scenario score table contains non-finite "
            "numeric values"
        )

    if (
        scores["nearest_reference_distance"] < 0
    ).any():
        raise ValueError(
            "Scenario score table contains negative "
            "reference distances"
        )

    if (scores["mean_absolute_pbe"] < 0).any():
        raise ValueError(
            "Scenario score table contains negative "
            "mean absolute PBE"
        )

    if (scores["max_absolute_pbe"] < 0).any():
        raise ValueError(
            "Scenario score table contains negative "
            "maximum absolute PBE"
        )

    if (scores["max_pbe_bus"] < 0).any():
        raise ValueError(
            "Scenario score table contains negative "
            "maximum-PBE bus IDs"
        )

    if (
        scores["mean_absolute_pbe"]
        > scores["max_absolute_pbe"]
    ).any():
        raise ValueError(
            "Mean absolute PBE exceeds maximum absolute "
            "PBE"
        )

    expected_table_values = {
        "distance_metric": DISTANCE_METRIC,
        "pooling_method": POOLING_METHOD,
        "residual_metric": RESIDUAL_METRIC,
    }
    for column, expected in expected_table_values.items():
        observed = set(
            scores[column].dropna().astype(str)
        )
        if observed != {expected}:
            raise ValueError(
                "Scenario score table has unexpected "
                f"{column} values: {sorted(observed)}"
            )

    return scores, manifest

def _load_verified_distribution_summary(
    path: Path,
    *,
    scores_path: Path,
    score_manifest_path: Path,
) -> dict[str, Any]:
    summary = _load_json_object(
        path,
        "Score distribution summary",
    )

    required = {
        "artifact_type",
        "scoring_policy_id",
        "scenario_count",
        "score_table",
        "score_manifest",
        "percentile_policy",
        "quantile_method",
        "selection_thresholds",
        "group_candidate_counts",
        "assigned_scenario_count",
        "excluded_scenario_count",
    }
    missing = sorted(required - summary.keys())
    if missing:
        raise ValueError(
            "Score distribution summary lacks fields: "
            f"{missing}"
        )

    if (
        summary["artifact_type"]
        != "scenario_score_distribution_summary"
    ):
        raise ValueError(
            "Score distribution summary has the wrong "
            "artifact type"
        )

    if (
        summary["scoring_policy_id"]
        != SCORING_POLICY_ID
    ):
        raise ValueError(
            "Score distribution summary uses an "
            "unsupported scoring policy"
        )

    score_record = summary["score_table"]
    manifest_record = summary["score_manifest"]

    if not isinstance(score_record, dict):
        raise ValueError(
            "Summary score-table record must be an object"
        )
    if not isinstance(manifest_record, dict):
        raise ValueError(
            "Summary score-manifest record must be "
            "an object"
        )

    if score_record.get("sha256") != sha256_file(
        scores_path
    ):
        raise ValueError(
            "Score distribution summary does not match "
            "the scenario score table"
        )

    if manifest_record.get("sha256") != sha256_file(
        score_manifest_path
    ):
        raise ValueError(
            "Score distribution summary does not match "
            "the scoring manifest"
        )

    percentile_policy = summary[
        "percentile_policy"
    ]
    if not isinstance(percentile_policy, dict):
        raise ValueError(
            "Summary percentile policy must be an object"
        )

    expected_percentile_policy = {
        "rank_method": PERCENTILE_RANK_METHOD,
        "low_max": LOW_PERCENTILE,
        "high_min": HIGH_PERCENTILE,
    }
    if percentile_policy != expected_percentile_policy:
        raise ValueError(
            "Score distribution summary has an "
            "unsupported percentile policy"
        )

    if summary["quantile_method"] != QUANTILE_METHOD:
        raise ValueError(
            "Score distribution summary has an "
            "unsupported quantile method"
        )

    group_counts = summary[
        "group_candidate_counts"
    ]
    if not isinstance(group_counts, dict):
        raise ValueError(
            "Summary group counts must be an object"
        )

    if set(group_counts) != set(GROUP_CORNERS):
        raise ValueError(
            "Summary group counts do not contain the "
            "declared groups"
        )

    for group_name, count in group_counts.items():
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
        ):
            raise ValueError(
                "Summary has an invalid candidate count "
                f"for {group_name!r}"
            )

    thresholds = summary["selection_thresholds"]
    required_thresholds = {
        "low_distance_max",
        "high_distance_min",
        "low_residual_max",
        "high_residual_min",
    }
    if (
        not isinstance(thresholds, dict)
        or set(thresholds) != required_thresholds
    ):
        raise ValueError(
            "Summary has invalid selection thresholds"
        )

    threshold_values = np.asarray(
        list(thresholds.values()),
        dtype=float,
    )
    if not np.isfinite(threshold_values).all():
        raise ValueError(
            "Summary selection thresholds are "
            "non-finite"
        )

    return summary

def _with_score_percentiles(
    scores: pd.DataFrame,
) -> pd.DataFrame:
    result = scores.copy()

    result["distance_percentile"] = result[
        "nearest_reference_distance"
    ].rank(
        method=PERCENTILE_RANK_METHOD,
        pct=True,
    )
    result["residual_percentile"] = result[
        "mean_absolute_pbe"
    ].rank(
        method=PERCENTILE_RANK_METHOD,
        pct=True,
    )
    result["max_pbe_percentile"] = result[
        "max_absolute_pbe"
    ].rank(
        method=PERCENTILE_RANK_METHOD,
        pct=True,
    )

    percentile_columns = [
        "distance_percentile",
        "residual_percentile",
        "max_pbe_percentile",
    ]
    values = result[
        percentile_columns
    ].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Percentile calculation produced non-finite "
            "values"
        )

    if ((values <= 0.0) | (values > 1.0)).any():
        raise ValueError(
            "Percentile calculation produced values "
            "outside (0, 1]"
        )

    return result

def _assign_group(
    row: pd.Series,
) -> str | None:
    distance = float(
        row["distance_percentile"]
    )
    residual = float(
        row["residual_percentile"]
    )

    if (
        distance <= LOW_PERCENTILE
        and residual <= LOW_PERCENTILE
    ):
        return "familiar_good"

    if (
        distance <= LOW_PERCENTILE
        and residual >= HIGH_PERCENTILE
    ):
        return "familiar_bad"

    if (
        distance >= HIGH_PERCENTILE
        and residual <= LOW_PERCENTILE
    ):
        return "unfamiliar_good"

    if (
        distance >= HIGH_PERCENTILE
        and residual >= HIGH_PERCENTILE
    ):
        return "unfamiliar_bad"

    return None

def _group_candidate_counts(
    scores: pd.DataFrame,
) -> dict[str, int]:
    return {
        group_name: int(
            (
                scores["cohort_group"]
                == group_name
            ).sum()
        )
        for group_name in GROUP_CORNERS
    }

def _distribution(
    values: pd.Series,
) -> dict[str, float]:
    array = values.to_numpy(dtype=float)

    if not np.isfinite(array).all():
        raise ValueError(
            f"{values.name} contains non-finite values"
        )

    return {
        f"q{int(quantile * 100):02d}": float(
            np.quantile(
                array,
                quantile,
                method=QUANTILE_METHOD,
            )
        )
        for quantile in QUANTILES
    }


def summarize_score_distributions(
    spec: ScoreSummarySpec,
) -> ScoreSummaryResult:
    scores, score_manifest = (
        _load_verified_score_table(
            spec.scores,
            spec.score_manifest,
        )
    )

    ranked = _with_score_percentiles(scores)
    ranked["cohort_group"] = ranked.apply(
        _assign_group,
        axis=1,
    )

    group_counts = _group_candidate_counts(
        ranked
    )
    assigned_count = sum(group_counts.values())
    excluded_count = len(ranked) - assigned_count

    distance_rank = ranked[
        "distance_percentile"
    ].to_numpy(dtype=float)
    residual_rank = ranked[
        "residual_percentile"
    ].to_numpy(dtype=float)

    if (
        np.std(distance_rank) == 0.0
        or np.std(residual_rank) == 0.0
    ):
        raise ValueError(
            "Spearman correlation is undefined for "
            "a constant score distribution"
        )

    spearman = float(
        np.corrcoef(
            distance_rank,
            residual_rank,
        )[0, 1]
    )
    if not np.isfinite(spearman):
        raise ValueError(
            "Spearman correlation is non-finite"
        )

    thresholds = _selection_thresholds(scores)

    summary = {
        "schema_version": 1,
        "artifact_type": (
            "scenario_score_distribution_summary"
        ),
        "scoring_policy_id": (
            score_manifest["scoring_policy_id"]
        ),
        "scenario_count": len(scores),
        "score_table": {
            "path": str(spec.scores),
            "sha256": sha256_file(spec.scores),
        },
        "score_manifest": {
            "path": str(spec.score_manifest),
            "sha256": sha256_file(
                spec.score_manifest
            ),
        },
        "percentile_policy": {
            "rank_method": (
                PERCENTILE_RANK_METHOD
            ),
            "low_max": LOW_PERCENTILE,
            "high_min": HIGH_PERCENTILE,
        },
        "quantile_method": QUANTILE_METHOD,
        "distance": _distribution(
            scores[
                "nearest_reference_distance"
            ]
        ),
        "mean_absolute_pbe": _distribution(
            scores["mean_absolute_pbe"]
        ),
        "max_absolute_pbe": _distribution(
            scores["max_absolute_pbe"]
        ),
        "spearman_distance_vs_mean_absolute_pbe": (
            spearman
        ),
        "selection_thresholds": thresholds,
        "group_candidate_counts": group_counts,
        "assigned_scenario_count": assigned_count,
        "excluded_scenario_count": excluded_count,
    }

    _write_json(
        spec.output,
        summary,
    )

    return ScoreSummaryResult(
        output=spec.output,
        scenario_count=len(scores),
        thresholds=thresholds,
        group_candidate_counts=group_counts,
        excluded_scenario_count=excluded_count,
        spearman=spearman,
    )

def _selection_thresholds(
    scores: pd.DataFrame,
) -> dict[str, float]:
    return {
        "low_distance_max": float(
            scores[
                "nearest_reference_distance"
            ].quantile(
                LOW_PERCENTILE,
                interpolation=QUANTILE_METHOD,
            )
        ),
        "high_distance_min": float(
            scores[
                "nearest_reference_distance"
            ].quantile(
                HIGH_PERCENTILE,
                interpolation=QUANTILE_METHOD,
            )
        ),
        "low_residual_max": float(
            scores[
                "mean_absolute_pbe"
            ].quantile(
                LOW_PERCENTILE,
                interpolation=QUANTILE_METHOD,
            )
        ),
        "high_residual_min": float(
            scores[
                "mean_absolute_pbe"
            ].quantile(
                HIGH_PERCENTILE,
                interpolation=QUANTILE_METHOD,
            )
        ),
    }


def _select_group(
    candidates: pd.DataFrame,
    group_name: str,
) -> pd.DataFrame:
    if group_name not in GROUP_CORNERS:
        raise ValueError(
            f"Unknown scenario group: {group_name!r}"
        )

    if len(candidates) < REPRESENTATIVES_PER_GROUP:
        raise ValueError(
            f"Group {group_name!r} contains only "
            f"{len(candidates)} scenarios; "
            f"{REPRESENTATIVES_PER_GROUP} are required"
        )

    corner_distance, corner_residual = (
        GROUP_CORNERS[group_name]
    )
    boundary_distance, boundary_residual = (
        GROUP_BOUNDARIES[group_name]
    )

    ranked = candidates.copy()
    ranked["_scenario_order"] = pd.to_numeric(
        ranked["scenario_id"],
        errors="raise",
    ).astype("int64")

    ranked["corner_score"] = np.hypot(
        ranked["distance_percentile"]
        - corner_distance,
        ranked["residual_percentile"]
        - corner_residual,
    )
    ranked["boundary_score"] = np.hypot(
        ranked["distance_percentile"]
        - boundary_distance,
        ranked["residual_percentile"]
        - boundary_residual,
    )

    corner = (
        ranked.sort_values(
            [
                "corner_score",
                "_scenario_order",
            ],
            ascending=[True, True],
            kind="mergesort",
        )
        .head(CORNER_REPRESENTATIVE_COUNT)
        .copy()
    )
    corner["selection_reason"] = "corner"
    corner["selection_rank"] = np.arange(
        1,
        CORNER_REPRESENTATIVE_COUNT + 1,
    )

    remaining = ranked.drop(index=corner.index)

    boundary = (
        remaining.sort_values(
            [
                "boundary_score",
                "corner_score",
                "_scenario_order",
            ],
            ascending=[True, True, True],
            kind="mergesort",
        )
        .head(BOUNDARY_REPRESENTATIVE_COUNT)
        .copy()
    )
    boundary[
        "selection_reason"
    ] = "inner_boundary"
    boundary["selection_rank"] = (
        CORNER_REPRESENTATIVE_COUNT + 1
    )

    remaining = remaining.drop(
        index=boundary.index
    )

    already_selected = pd.concat(
        [corner, boundary],
        ignore_index=False,
    )

    selected_buses = set(
        already_selected[
            "max_pbe_bus"
        ].astype(int)
    )
    selected_references = set(
        already_selected[
            "nearest_reference_scenario_id"
        ].astype(str)
    )
    selected_max_pbe_percentiles = (
        already_selected[
            "max_pbe_percentile"
        ].to_numpy(dtype=float)
    )

    diversity_candidates = remaining.copy()

    diversity_candidates[
        "_novel_max_pbe_bus"
    ] = ~diversity_candidates[
        "max_pbe_bus"
    ].astype(int).isin(selected_buses)

    diversity_candidates[
        "_novel_reference"
    ] = ~diversity_candidates[
        "nearest_reference_scenario_id"
    ].astype(str).isin(selected_references)

    diversity_candidates[
        "_max_pbe_separation"
    ] = diversity_candidates[
        "max_pbe_percentile"
    ].astype(float).map(
        lambda value: float(
            np.min(
                np.abs(
                    selected_max_pbe_percentiles
                    - value
                )
            )
        )
    )

    diversity = (
        diversity_candidates.sort_values(
            [
                "_novel_max_pbe_bus",
                "_novel_reference",
                "_max_pbe_separation",
                "corner_score",
                "_scenario_order",
            ],
            ascending=[
                False,
                False,
                False,
                True,
                True,
            ],
            kind="mergesort",
        )
        .head(DIVERSITY_REPRESENTATIVE_COUNT)
        .copy()
    )
    diversity[
        "selection_reason"
    ] = "physical_diversity"
    diversity["selection_rank"] = (
        REPRESENTATIVES_PER_GROUP
    )

    selected = pd.concat(
        [corner, boundary, diversity],
        ignore_index=True,
    )

    selected = selected.drop(
        columns=[
            column
            for column in (
                "_scenario_order",
                "_novel_max_pbe_bus",
                "_novel_reference",
                "_max_pbe_separation",
            )
            if column in selected.columns
        ]
    )

    if len(selected) != REPRESENTATIVES_PER_GROUP:
        raise RuntimeError(
            f"Selection for {group_name!r} produced "
            f"{len(selected)} scenarios instead of "
            f"{REPRESENTATIVES_PER_GROUP}"
        )

    if selected[
        "scenario_id"
    ].astype(str).duplicated().any():
        raise RuntimeError(
            f"Selection for {group_name!r} contains "
            "duplicate scenarios"
        )

    expected_reasons = {
        "corner": CORNER_REPRESENTATIVE_COUNT,
        "inner_boundary": (
            BOUNDARY_REPRESENTATIVE_COUNT
        ),
        "physical_diversity": (
            DIVERSITY_REPRESENTATIVE_COUNT
        ),
    }
    observed_reasons = (
        selected["selection_reason"]
        .value_counts()
        .to_dict()
    )
    if observed_reasons != expected_reasons:
        raise RuntimeError(
            f"Selection for {group_name!r} has an "
            f"invalid composition: {observed_reasons}"
        )

    return selected


def select_contrasting_scenarios(
    spec: ContrastingScenarioSelectionSpec,
) -> ContrastingScenarioSelectionResult:
    scores, score_manifest = (
        _load_verified_score_table(
            spec.scores,
            spec.score_manifest,
        )
    )
    summary = _load_verified_distribution_summary(
        spec.summary,
        scores_path=spec.scores,
        score_manifest_path=spec.score_manifest,
    )

    if summary["scenario_count"] != len(scores):
        raise ValueError(
            "Distribution summary scenario count "
            "disagrees with the score table"
        )

    expected_thresholds = _selection_thresholds(
        scores
    )
    recorded_thresholds = summary[
        "selection_thresholds"
    ]

    for name, expected in expected_thresholds.items():
        recorded = float(
            recorded_thresholds[name]
        )
        if recorded != expected:
            raise ValueError(
                "Distribution summary threshold "
                f"{name!r} disagrees with the score table"
            )

    ranked = _with_score_percentiles(scores)
    ranked["cohort_group"] = ranked.apply(
        _assign_group,
        axis=1,
    )

    group_counts = _group_candidate_counts(
        ranked
    )
    if (
        group_counts
        != summary["group_candidate_counts"]
    ):
        raise ValueError(
            "Distribution summary group counts disagree "
            "with the score table"
        )

    sparse_groups = {
        group_name: count
        for group_name, count in group_counts.items()
        if count < REPRESENTATIVES_PER_GROUP
    }

    diagnostics: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": (
            "contrasting_scenario_selection_diagnostics"
        ),
        "selection_policy_id": (
            SELECTION_POLICY_ID
        ),
        "status": (
            "insufficient_candidates"
            if sparse_groups
            else "ready"
        ),
        "score_table": {
            "path": str(spec.scores),
            "sha256": sha256_file(spec.scores),
        },
        "score_manifest": {
            "path": str(spec.score_manifest),
            "sha256": sha256_file(
                spec.score_manifest
            ),
        },
        "distribution_summary": {
            "path": str(spec.summary),
            "sha256": sha256_file(spec.summary),
        },
        "percentile_policy": {
            "rank_method": (
                PERCENTILE_RANK_METHOD
            ),
            "low_max": LOW_PERCENTILE,
            "high_min": HIGH_PERCENTILE,
        },
        "required_per_group": (
            REPRESENTATIVES_PER_GROUP
        ),
        "required_composition": {
            "corner": (
                CORNER_REPRESENTATIVE_COUNT
            ),
            "inner_boundary": (
                BOUNDARY_REPRESENTATIVE_COUNT
            ),
            "physical_diversity": (
                DIVERSITY_REPRESENTATIVE_COUNT
            ),
        },
        "group_candidate_counts": group_counts,
        "sparse_groups": sparse_groups,
    }

    if sparse_groups:
        _write_json(
            spec.diagnostics_output,
            diagnostics,
        )
        details = ", ".join(
            f"{name}={count}"
            for name, count in sorted(
                sparse_groups.items()
            )
        )
        raise ValueError(
            "Contrasting scenarios cannot be selected "
            "without changing the declared thresholds: "
            f"{details}. Diagnostics: "
            f"{spec.diagnostics_output}"
        )

    selected_groups = [
        _select_group(
            ranked[
                ranked["cohort_group"]
                == group_name
            ].copy(),
            group_name,
        )
        for group_name in GROUP_CORNERS
    ]

    selected = pd.concat(
        selected_groups,
        ignore_index=True,
    )

    group_order = {
        group_name: index
        for index, group_name in enumerate(
            GROUP_CORNERS
        )
    }
    selected["_group_order"] = selected[
        "cohort_group"
    ].map(group_order)
    selected["_scenario_order"] = pd.to_numeric(
        selected["scenario_id"],
        errors="raise",
    ).astype("int64")

    selected = (
        selected.sort_values(
            [
                "_group_order",
                "selection_rank",
                "_scenario_order",
            ],
            kind="mergesort",
        )
        .drop(
            columns=[
                "_group_order",
                "_scenario_order",
            ]
        )
        .reset_index(drop=True)
    )

    expected_total = (
        len(GROUP_CORNERS)
        * REPRESENTATIVES_PER_GROUP
    )
    if len(selected) != expected_total:
        raise RuntimeError(
            f"Selected {len(selected)} scenarios; "
            f"{expected_total} were required"
        )

    if selected[
        "scenario_id"
    ].astype(str).duplicated().any():
        raise RuntimeError(
            "Selected scenarios contain duplicate IDs"
        )

    reference_ids = sorted(
        set(
            selected[
                "nearest_reference_scenario_id"
            ].astype(str)
        ),
        key=int,
    )

    _write_parquet(
        spec.output,
        selected,
    )

    _write_text(
        spec.training_reference_ids_output,
        (
            "scenario_id\n"
            + "\n".join(reference_ids)
            + "\n"
        ),
    )

    diagnostics["status"] = "selected"
    diagnostics[
        "selected_scenario_count"
    ] = len(selected)
    diagnostics[
        "selected_training_reference_count"
    ] = len(reference_ids)
    diagnostics[
        "selection_reason_counts"
    ] = {
        group_name: {
            reason: int(count)
            for reason, count in (
                selected[
                    selected["cohort_group"]
                    == group_name
                ]["selection_reason"]
                .value_counts()
                .sort_index()
                .items()
            )
        }
        for group_name in GROUP_CORNERS
    }

    _write_json(
        spec.diagnostics_output,
        diagnostics,
    )

    manifest = {
        "schema_version": 1,
        "artifact_type": (
            "frozen_contrasting_scenario_cohort"
        ),
        "selection_policy_id": (
            SELECTION_POLICY_ID
        ),
        "scoring_policy_id": (
            score_manifest["scoring_policy_id"]
        ),
        "score_table": {
            "path": str(spec.scores),
            "sha256": sha256_file(spec.scores),
        },
        "score_manifest": {
            "path": str(spec.score_manifest),
            "sha256": sha256_file(
                spec.score_manifest
            ),
        },
        "distribution_summary": {
            "path": str(spec.summary),
            "sha256": sha256_file(spec.summary),
        },
        "diagnostics": {
            "path": str(spec.diagnostics_output),
            "sha256": sha256_file(
                spec.diagnostics_output
            ),
        },
        "threshold_policy": {
            "low_percentile": LOW_PERCENTILE,
            "high_percentile": HIGH_PERCENTILE,
            **expected_thresholds,
        },
        "selection_policy": {
            "per_group": (
                REPRESENTATIVES_PER_GROUP
            ),
            "corner_representatives": (
                CORNER_REPRESENTATIVE_COUNT
            ),
            "inner_boundary_representatives": (
                BOUNDARY_REPRESENTATIVE_COUNT
            ),
            "physical_diversity_representatives": (
                DIVERSITY_REPRESENTATIVE_COUNT
            ),
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
        },
        "group_candidate_counts": group_counts,
        "selected_test_scenario_count": len(
            selected
        ),
        "selected_training_reference_count": len(
            reference_ids
        ),
        "cohort_artifact": {
            "path": str(spec.output),
            "sha256": sha256_file(spec.output),
        },
        "training_reference_ids": {
            "path": str(
                spec.training_reference_ids_output
            ),
            "sha256": sha256_file(
                spec.training_reference_ids_output
            ),
        },
        "selected_scenarios": [
            {
                "scenario_id": str(
                    row.scenario_id
                ),
                "cohort_group": str(
                    row.cohort_group
                ),
                "selection_reason": str(
                    row.selection_reason
                ),
                "selection_rank": int(
                    row.selection_rank
                ),
                "nearest_reference_scenario_id": str(
                    row.nearest_reference_scenario_id
                ),
                "nearest_reference_distance": float(
                    row.nearest_reference_distance
                ),
                "mean_absolute_pbe": float(
                    row.mean_absolute_pbe
                ),
                "max_absolute_pbe": float(
                    row.max_absolute_pbe
                ),
                "max_pbe_bus": int(
                    row.max_pbe_bus
                ),
                "distance_percentile": float(
                    row.distance_percentile
                ),
                "residual_percentile": float(
                    row.residual_percentile
                ),
                "max_pbe_percentile": float(
                    row.max_pbe_percentile
                ),
                "corner_score": float(
                    row.corner_score
                ),
                "boundary_score": float(
                    row.boundary_score
                ),
            }
            for row in selected.itertuples(
                index=False
            )
        ],
    }

    _write_json(
        spec.manifest_output,
        manifest,
    )

    return ContrastingScenarioSelectionResult(
        output=spec.output,
        manifest=spec.manifest_output,
        diagnostics=spec.diagnostics_output,
        training_reference_ids=(
            spec.training_reference_ids_output
        ),
        selected_scenario_count=len(selected),
        selected_reference_count=len(
            reference_ids
        ),
    )