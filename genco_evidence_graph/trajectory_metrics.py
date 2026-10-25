"""Derived, queryable metrics for verified GENCO latent trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .provenance import sha256_file


METRIC_POLICY_ID = "genco-latent-trajectory-metrics-v1"
QUANTILE_METHOD = "linear"
QUANTILES = (
    0.0,
    0.01,
    0.05,
    0.1,
    0.25,
    0.5,
    0.75,
    0.9,
    0.95,
    0.99,
    1.0,
)
ZERO_NORM_EPSILON = 1e-12
ENTITY_METRICS = (
    "path_length",
    "net_displacement",
    "path_efficiency",
    "mean_step_movement",
    "max_step_movement",
    "local_directional_consistency",
    "global_directional_alignment",
)


@dataclass(frozen=True)
class TrajectoryMetricSpec:
    trajectory: Path
    trajectory_manifest: Path
    cohort: Path
    cohort_manifest: Path
    entity_metrics_output: Path
    step_metrics_output: Path
    scenario_summary_output: Path
    group_comparison_output: Path
    diagnostics_output: Path
    manifest_output: Path

    def __post_init__(self) -> None:
        paths = (
            self.trajectory,
            self.trajectory_manifest,
            self.cohort,
            self.cohort_manifest,
            self.entity_metrics_output,
            self.step_metrics_output,
            self.scenario_summary_output,
            self.group_comparison_output,
            self.diagnostics_output,
            self.manifest_output,
        )
        if len(set(paths)) != len(paths):
            raise ValueError(
                "Trajectory metric inputs and outputs must all differ"
            )


@dataclass(frozen=True)
class TrajectoryMetricResult:
    entity_metrics: Path
    step_metrics: Path
    scenario_summary: Path
    group_comparison: Path
    diagnostics: Path
    manifest: Path
    entity_count: int
    step_transition_count: int
    scenario_count: int


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _verify_record(
    manifest: dict[str, Any],
    key: str,
    path: Path,
    label: str,
) -> dict[str, Any]:
    record = manifest.get(key)
    if not isinstance(record, dict):
        raise ValueError(f"{label} lacks {key}")
    if Path(record.get("path", "")).resolve() != path.resolve():
        raise ValueError(f"{label} {key} path disagrees")
    if record.get("sha256") != sha256_file(path):
        raise ValueError(f"{label} {key} hash disagrees")
    return record


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    )
    temporary.replace(path)


def _write_parquet(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_parquet(temporary, index=False)
    temporary.replace(path)


def _cosine(
    left: np.ndarray,
    right: np.ndarray,
) -> float | None:
    denominator = float(
        np.linalg.norm(left) * np.linalg.norm(right)
    )
    if denominator <= ZERO_NORM_EPSILON:
        return None
    return float(
        np.clip(
            np.dot(left, right) / denominator,
            -1.0,
            1.0,
        )
    )


def _entity_trajectory_metrics(
    group: pd.DataFrame,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ordered = group.sort_values("correction_step")
    steps = ordered["correction_step"].astype(int).tolist()
    if steps != list(range(1, 13)):
        raise ValueError(
            "Every entity trajectory must contain steps 1-12"
        )
    vectors = np.stack(ordered["embedding"]).astype(
        np.float64,
        copy=False,
    )
    if vectors.shape != (12, 384):
        raise ValueError(
            "Every entity trajectory must have shape (12, 384)"
        )
    if not np.isfinite(vectors).all():
        raise ValueError(
            "Trajectory embeddings must be finite"
        )

    deltas = np.diff(vectors, axis=0)
    movements = np.linalg.norm(deltas, axis=1)
    path_length = float(movements.sum())
    displacement = vectors[-1] - vectors[0]
    net_displacement = float(
        np.linalg.norm(displacement)
    )
    path_efficiency = (
        net_displacement / path_length
        if path_length > ZERO_NORM_EPSILON
        else 1.0
    )

    local_cosines = [
        _cosine(deltas[index], deltas[index + 1])
        for index in range(len(deltas) - 1)
    ]
    valid_local = [
        value
        for value in local_cosines
        if value is not None
    ]
    local_consistency = (
        float(np.mean(valid_local))
        if valid_local
        else 1.0
    )
    global_cosines = [
        _cosine(delta, displacement)
        for delta in deltas
    ]
    valid_global = [
        value
        for value in global_cosines
        if value is not None
    ]
    global_alignment = (
        float(np.mean(valid_global))
        if valid_global
        and net_displacement > ZERO_NORM_EPSILON
        else 0.0
    )

    first = ordered.iloc[0]
    entity = {
        "scenario_id": int(first["scenario_id"]),
        "entity_type": str(first["entity_type"]),
        "entity_id": int(first["entity_id"]),
        "path_length": path_length,
        "net_displacement": net_displacement,
        "path_efficiency": float(
            np.clip(path_efficiency, 0.0, 1.0)
        ),
        "mean_step_movement": float(movements.mean()),
        "max_step_movement": float(movements.max()),
        "max_movement_start_step": int(
            np.argmax(movements) + 1
        ),
        "local_directional_consistency": (
            local_consistency
        ),
        "global_directional_alignment": (
            global_alignment
        ),
    }
    transitions = []
    for index, movement in enumerate(movements):
        transitions.append(
            {
                "scenario_id": entity["scenario_id"],
                "entity_type": entity["entity_type"],
                "entity_id": entity["entity_id"],
                "start_step": index + 1,
                "end_step": index + 2,
                "step_movement": float(movement),
                "previous_step_cosine": (
                    np.nan
                    if index == 0
                    else local_cosines[index - 1]
                ),
                "global_displacement_cosine": (
                    global_cosines[index]
                    if global_cosines[index] is not None
                    and net_displacement
                    > ZERO_NORM_EPSILON
                    else np.nan
                ),
            }
        )
    return entity, transitions


def _distribution(values: pd.Series) -> dict[str, float]:
    array = values.to_numpy(dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(
            "Metric distributions require finite values"
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


def _scenario_summaries(
    entities: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario_id, scenario in entities.groupby(
        "scenario_id",
        sort=True,
    ):
        for entity_type in ("bus", "generator", "all"):
            selected = (
                scenario
                if entity_type == "all"
                else scenario[
                    scenario["entity_type"] == entity_type
                ]
            )
            if selected.empty:
                raise ValueError(
                    f"Scenario {scenario_id} has no {entity_type} entities"
                )
            row: dict[str, Any] = {
                "scenario_id": int(scenario_id),
                "cohort_group": selected[
                    "cohort_group"
                ].iloc[0],
                "entity_type": entity_type,
                "entity_count": len(selected),
                "total_path_length": float(
                    selected["path_length"].sum()
                ),
                "total_net_displacement": float(
                    selected["net_displacement"].sum()
                ),
            }
            for metric in ENTITY_METRICS:
                row[f"mean_{metric}"] = float(
                    selected[metric].mean()
                )
                row[f"median_{metric}"] = float(
                    selected[metric].median()
                )
                row[f"max_{metric}"] = float(
                    selected[metric].max()
                )
            localized = selected.sort_values(
                ["path_length", "entity_type", "entity_id"],
                ascending=[False, True, True],
                kind="stable",
            ).iloc[0]
            row["highest_movement_entity_type"] = localized[
                "entity_type"
            ]
            row["highest_movement_entity_id"] = int(
                localized["entity_id"]
            )
            row["highest_movement_path_length"] = float(
                localized["path_length"]
            )
            row["highest_movement_path_share"] = float(
                localized["path_length"]
                / row["total_path_length"]
                if row["total_path_length"]
                > ZERO_NORM_EPSILON
                else 0.0
            )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["scenario_id", "entity_type"],
        kind="stable",
    ).reset_index(drop=True)


def _group_comparisons(
    scenarios: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    summary_metrics = [
        "total_path_length",
        "total_net_displacement",
        *[
            f"mean_{metric}"
            for metric in ENTITY_METRICS
        ],
    ]
    for (
        cohort_group,
        entity_type,
    ), selected in scenarios.groupby(
        ["cohort_group", "entity_type"],
        sort=True,
    ):
        row: dict[str, Any] = {
            "cohort_group": cohort_group,
            "entity_type": entity_type,
            "scenario_count": selected[
                "scenario_id"
            ].nunique(),
        }
        for metric in summary_metrics:
            row[f"mean_{metric}"] = float(
                selected[metric].mean()
            )
            row[f"median_{metric}"] = float(
                selected[metric].median()
            )
            row[f"min_{metric}"] = float(
                selected[metric].min()
            )
            row[f"max_{metric}"] = float(
                selected[metric].max()
            )
            row[f"std_{metric}"] = float(
                selected[metric].std(ddof=0)
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["cohort_group", "entity_type"],
        kind="stable",
    ).reset_index(drop=True)


def derive_trajectory_metrics(
    spec: TrajectoryMetricSpec,
) -> TrajectoryMetricResult:
    """Derive entity, transition, scenario, and group trajectory metrics."""
    output_paths = (
        spec.entity_metrics_output,
        spec.step_metrics_output,
        spec.scenario_summary_output,
        spec.group_comparison_output,
        spec.diagnostics_output,
        spec.manifest_output,
    )
    for path in output_paths:
        if path.exists():
            raise ValueError(
                f"Trajectory metric output already exists: {path}"
            )

    trajectory_manifest = _load_json(
        spec.trajectory_manifest,
        "Trajectory capture manifest",
    )
    if (
        trajectory_manifest.get("schema_version") != 1
        or trajectory_manifest.get("artifact_type")
        != "genco_cohort_latent_trajectory"
    ):
        raise ValueError(
            "Unsupported trajectory capture manifest"
        )
    trajectory_record = _verify_record(
        trajectory_manifest,
        "trajectory",
        spec.trajectory,
        "Trajectory capture manifest",
    )
    if (
        trajectory_record.get("correction_step_count")
        != 12
        or trajectory_record.get("embedding_dimension")
        != 384
        or trajectory_record.get("embedding_stage")
        != "prediction_head_input"
    ):
        raise ValueError(
            "Trajectory capture metadata is incompatible "
            "with the metric policy"
        )
    equivalence = trajectory_manifest.get("equivalence")
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

    cohort_manifest = _load_json(
        spec.cohort_manifest,
        "Cohort manifest",
    )
    if (
        cohort_manifest.get("schema_version") != 1
        or cohort_manifest.get("artifact_type")
        != "frozen_contrasting_scenario_cohort"
    ):
        raise ValueError("Unsupported cohort manifest")
    _verify_record(
        cohort_manifest,
        "cohort_artifact",
        spec.cohort,
        "Cohort manifest",
    )
    cohort = pd.read_parquet(
        spec.cohort,
        columns=[
            "scenario_id",
            "cohort_group",
            "selection_reason",
            "selection_rank",
        ],
    )
    cohort["scenario_id"] = cohort[
        "scenario_id"
    ].astype("int64")
    if len(cohort) != 20 or cohort[
        "scenario_id"
    ].duplicated().any():
        raise ValueError(
            "Trajectory metrics require the frozen 20-scenario cohort"
        )

    trajectory = pd.read_parquet(spec.trajectory)
    required = {
        "scenario_id",
        "entity_type",
        "entity_id",
        "correction_step",
        "embedding_stage",
        "embedding",
    }
    if set(trajectory.columns) != required:
        raise ValueError(
            "Canonical trajectory table has an unexpected schema"
        )
    if set(trajectory["embedding_stage"]) != {
        "prediction_head_input"
    }:
        raise ValueError(
            "Canonical trajectory embedding stage is unsupported"
        )
    trajectory["scenario_id"] = trajectory[
        "scenario_id"
    ].astype("int64")
    if set(trajectory["scenario_id"]) != set(
        cohort["scenario_id"]
    ):
        raise ValueError(
            "Trajectory and cohort scenario membership disagree"
        )
    keys = [
        "scenario_id",
        "entity_type",
        "entity_id",
        "correction_step",
    ]
    if trajectory.duplicated(keys).any():
        raise ValueError(
            "Canonical trajectory contains duplicate keys"
        )

    entity_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    for _, group in trajectory.groupby(
        keys[:3],
        sort=True,
    ):
        entity, transitions = _entity_trajectory_metrics(
            group
        )
        entity_rows.append(entity)
        transition_rows.extend(transitions)
    entities = pd.DataFrame(entity_rows).merge(
        cohort,
        on="scenario_id",
        how="left",
        validate="many_to_one",
    )
    entities["path_length_rank_in_scenario"] = (
        entities.groupby("scenario_id")[
            "path_length"
        ].rank(method="first", ascending=False).astype("int64")
    )
    entities["path_length_rank_in_entity_type"] = (
        entities.groupby(
            ["scenario_id", "entity_type"]
        )["path_length"]
        .rank(method="first", ascending=False)
        .astype("int64")
    )
    scenario_path_totals = entities.groupby(
        "scenario_id"
    )["path_length"].transform("sum")
    type_path_totals = entities.groupby(
        ["scenario_id", "entity_type"]
    )["path_length"].transform("sum")
    entities["path_length_share_in_scenario"] = np.where(
        scenario_path_totals > ZERO_NORM_EPSILON,
        entities["path_length"] / scenario_path_totals,
        0.0,
    )
    entities["path_length_share_in_entity_type"] = np.where(
        type_path_totals > ZERO_NORM_EPSILON,
        entities["path_length"] / type_path_totals,
        0.0,
    )
    entities = entities.sort_values(
        ["scenario_id", "entity_type", "entity_id"],
        kind="stable",
    ).reset_index(drop=True)

    transitions = pd.DataFrame(
        transition_rows
    ).merge(
        cohort[["scenario_id", "cohort_group"]],
        on="scenario_id",
        how="left",
        validate="many_to_one",
    ).sort_values(
        [
            "scenario_id",
            "entity_type",
            "entity_id",
            "start_step",
        ],
        kind="stable",
    ).reset_index(drop=True)
    scenarios = _scenario_summaries(entities)
    groups = _group_comparisons(scenarios)

    distributions: dict[str, Any] = {
        "overall": {
            metric: _distribution(entities[metric])
            for metric in ENTITY_METRICS
        },
        "by_entity_type": {},
        "by_cohort_group": {},
    }
    for entity_type, selected in entities.groupby(
        "entity_type",
        sort=True,
    ):
        distributions["by_entity_type"][
            entity_type
        ] = {
            metric: _distribution(selected[metric])
            for metric in ENTITY_METRICS
        }
    for cohort_group, selected in entities.groupby(
        "cohort_group",
        sort=True,
    ):
        distributions["by_cohort_group"][
            cohort_group
        ] = {
            metric: _distribution(selected[metric])
            for metric in ENTITY_METRICS
        }
    localized = entities.sort_values(
        [
            "path_length",
            "scenario_id",
            "entity_type",
            "entity_id",
        ],
        ascending=[False, True, True, True],
        kind="stable",
    ).head(20)
    diagnostics = {
        "schema_version": 1,
        "artifact_type": (
            "latent_trajectory_metric_diagnostics"
        ),
        "metric_policy_id": METRIC_POLICY_ID,
        "quantile_method": QUANTILE_METHOD,
        "entity_count": len(entities),
        "step_transition_count": len(transitions),
        "scenario_count": cohort["scenario_id"].nunique(),
        "cohort_group_counts": {
            str(key): int(value)
            for key, value in cohort[
                "cohort_group"
            ].value_counts().sort_index().items()
        },
        "distributions": distributions,
        "highest_movement_entities": [
            {
                "scenario_id": str(row.scenario_id),
                "cohort_group": row.cohort_group,
                "entity_type": row.entity_type,
                "entity_id": int(row.entity_id),
                "path_length": float(row.path_length),
                "path_length_share_in_scenario": float(
                    row.path_length_share_in_scenario
                ),
            }
            for row in localized.itertuples(index=False)
        ],
    }

    _write_parquet(spec.entity_metrics_output, entities)
    _write_parquet(spec.step_metrics_output, transitions)
    _write_parquet(spec.scenario_summary_output, scenarios)
    _write_parquet(spec.group_comparison_output, groups)
    _write_json(spec.diagnostics_output, diagnostics)

    outputs = {
        "entity_metrics": {
            "path": str(spec.entity_metrics_output),
            "sha256": sha256_file(
                spec.entity_metrics_output
            ),
            "row_count": len(entities),
        },
        "step_metrics": {
            "path": str(spec.step_metrics_output),
            "sha256": sha256_file(
                spec.step_metrics_output
            ),
            "row_count": len(transitions),
        },
        "scenario_summary": {
            "path": str(spec.scenario_summary_output),
            "sha256": sha256_file(
                spec.scenario_summary_output
            ),
            "row_count": len(scenarios),
        },
        "group_comparison": {
            "path": str(spec.group_comparison_output),
            "sha256": sha256_file(
                spec.group_comparison_output
            ),
            "row_count": len(groups),
        },
        "diagnostics": {
            "path": str(spec.diagnostics_output),
            "sha256": sha256_file(
                spec.diagnostics_output
            ),
        },
    }
    manifest = {
        "schema_version": 1,
        "artifact_type": "latent_trajectory_metric_bundle",
        "metric_policy_id": METRIC_POLICY_ID,
        "trajectory": {
            "path": str(spec.trajectory),
            "sha256": sha256_file(spec.trajectory),
        },
        "trajectory_manifest": {
            "path": str(spec.trajectory_manifest),
            "sha256": sha256_file(
                spec.trajectory_manifest
            ),
        },
        "cohort": {
            "path": str(spec.cohort),
            "sha256": sha256_file(spec.cohort),
        },
        "cohort_manifest": {
            "path": str(spec.cohort_manifest),
            "sha256": sha256_file(
                spec.cohort_manifest
            ),
        },
        "policy": {
            "layer_snapshot_count": 12,
            "transition_count": 11,
            "embedding_dimension": 384,
            "distance_metric": "euclidean_l2",
            "path_length": "sum_l2_consecutive_deltas",
            "net_displacement": "l2_final_minus_initial",
            "path_efficiency": (
                "net_displacement_divided_by_path_length"
            ),
            "local_directional_consistency": (
                "mean_cosine_of_consecutive_deltas"
            ),
            "global_directional_alignment": (
                "mean_cosine_of_each_delta_with_net_displacement"
            ),
            "zero_path_efficiency": 1.0,
            "zero_local_consistency": 1.0,
            "zero_global_alignment": 0.0,
            "zero_norm_epsilon": ZERO_NORM_EPSILON,
            "scenario_aggregation": (
                "separate_bus_generator_and_all_entity_summaries"
            ),
            "group_aggregation": (
                "unweighted_across_five_scenario_summaries"
            ),
        },
        "outputs": outputs,
    }
    _write_json(spec.manifest_output, manifest)
    return TrajectoryMetricResult(
        entity_metrics=spec.entity_metrics_output,
        step_metrics=spec.step_metrics_output,
        scenario_summary=spec.scenario_summary_output,
        group_comparison=spec.group_comparison_output,
        diagnostics=spec.diagnostics_output,
        manifest=spec.manifest_output,
        entity_count=len(entities),
        step_transition_count=len(transitions),
        scenario_count=cohort["scenario_id"].nunique(),
    )
