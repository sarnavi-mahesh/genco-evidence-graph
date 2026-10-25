import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from genco_evidence_graph.cli import build_parser
from genco_evidence_graph.provenance import sha256_file
from genco_evidence_graph.trajectory_metrics import (
    METRIC_POLICY_ID,
    TrajectoryMetricSpec,
    _entity_trajectory_metrics,
    derive_trajectory_metrics,
)
from genco_evidence_graph.trajectory_metric_provenance import (
    TrajectoryMetricProvenanceSpec,
    build_trajectory_metric_provenance_graph,
    write_trajectory_metric_provenance,
)


def _write_fixture(tmp_path: Path) -> TrajectoryMetricSpec:
    trajectory = tmp_path / "trajectory.parquet"
    trajectory_manifest = tmp_path / "trajectory_manifest.json"
    cohort = tmp_path / "cohort.parquet"
    cohort_manifest = tmp_path / "cohort_manifest.json"
    groups = (
        "familiar_good",
        "familiar_bad",
        "unfamiliar_good",
        "unfamiliar_bad",
    )
    cohort_table = pd.DataFrame(
        {
            "scenario_id": np.arange(1, 21),
            "cohort_group": [
                groups[(value - 1) // 5]
                for value in range(1, 21)
            ],
            "selection_reason": ["corner"] * 20,
            "selection_rank": [
                ((value - 1) % 5) + 1
                for value in range(1, 21)
            ],
        }
    )
    cohort_table.to_parquet(cohort, index=False)
    cohort_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": (
                    "frozen_contrasting_scenario_cohort"
                ),
                "cohort_artifact": {
                    "path": str(cohort),
                    "sha256": sha256_file(cohort),
                },
            }
        )
    )

    rows = []
    for scenario_id in range(1, 21):
        for entity_type in ("bus", "generator"):
            for step in range(1, 13):
                embedding = np.zeros(384, dtype=np.float32)
                embedding[0] = float(step - 1)
                rows.append(
                    {
                        "scenario_id": scenario_id,
                        "entity_type": entity_type,
                        "entity_id": 0,
                        "correction_step": step,
                        "embedding_stage": (
                            "prediction_head_input"
                        ),
                        "embedding": embedding,
                    }
                )
    pd.DataFrame(rows).to_parquet(
        trajectory,
        index=False,
    )
    trajectory_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": (
                    "genco_cohort_latent_trajectory"
                ),
                "trajectory": {
                    "path": str(trajectory),
                    "sha256": sha256_file(trajectory),
                    "correction_step_count": 12,
                    "embedding_dimension": 384,
                    "embedding_stage": (
                        "prediction_head_input"
                    ),
                },
                "equivalence": {
                    "passed": True,
                    "instrumented_final_snapshot_exact": True,
                },
            }
        )
    )
    return TrajectoryMetricSpec(
        trajectory=trajectory,
        trajectory_manifest=trajectory_manifest,
        cohort=cohort,
        cohort_manifest=cohort_manifest,
        entity_metrics_output=tmp_path / "entities.parquet",
        step_metrics_output=tmp_path / "steps.parquet",
        scenario_summary_output=tmp_path / "scenarios.parquet",
        group_comparison_output=tmp_path / "groups.parquet",
        diagnostics_output=tmp_path / "diagnostics.json",
        manifest_output=tmp_path / "manifest.json",
    )


def test_derive_straight_trajectory_metrics(
    tmp_path: Path,
) -> None:
    spec = _write_fixture(tmp_path)

    result = derive_trajectory_metrics(spec)

    assert result.entity_count == 40
    assert result.step_transition_count == 440
    assert result.scenario_count == 20
    entities = pd.read_parquet(result.entity_metrics)
    assert np.allclose(entities["path_length"], 11.0)
    assert np.allclose(
        entities["net_displacement"],
        11.0,
    )
    assert np.allclose(entities["path_efficiency"], 1.0)
    assert np.allclose(
        entities["local_directional_consistency"],
        1.0,
    )
    assert np.allclose(
        entities["global_directional_alignment"],
        1.0,
    )
    scenarios = pd.read_parquet(result.scenario_summary)
    groups = pd.read_parquet(result.group_comparison)
    assert len(scenarios) == 60
    assert len(groups) == 12
    manifest = json.loads(result.manifest.read_text())
    assert manifest["metric_policy_id"] == METRIC_POLICY_ID


def test_local_and_global_direction_metrics_are_distinct() -> None:
    vectors = [np.zeros(384, dtype=np.float32)]
    for index in range(11):
        delta = np.zeros(384, dtype=np.float32)
        delta[index % 2] = 1.0
        vectors.append(vectors[-1] + delta)
    table = pd.DataFrame(
        {
            "scenario_id": [1] * 12,
            "entity_type": ["bus"] * 12,
            "entity_id": [0] * 12,
            "correction_step": range(1, 13),
            "embedding": vectors,
        }
    )

    entity, _ = _entity_trajectory_metrics(table)

    assert entity[
        "local_directional_consistency"
    ] == pytest.approx(0.0)
    assert entity[
        "global_directional_alignment"
    ] == pytest.approx(np.sqrt(61.0) / 11.0)


def test_trajectory_metrics_reject_modified_input(
    tmp_path: Path,
) -> None:
    spec = _write_fixture(tmp_path)
    spec.trajectory.write_bytes(b"modified")

    with pytest.raises(
        ValueError,
        match="trajectory hash disagrees",
    ):
        derive_trajectory_metrics(spec)


def test_trajectory_metric_command_is_registered() -> None:
    args = build_parser().parse_args(
        [
            "compute-trajectory-metrics",
            "--trajectory", "trajectory.parquet",
            "--trajectory-manifest", "trajectory.json",
            "--cohort", "cohort.parquet",
            "--cohort-manifest", "cohort.json",
            "--entity-metrics-output", "entities.parquet",
            "--step-metrics-output", "steps.parquet",
            "--scenario-summary-output", "scenarios.parquet",
            "--group-comparison-output", "groups.parquet",
            "--diagnostics-output", "diagnostics.json",
            "--manifest-output", "manifest.json",
        ]
    )

    assert args.handler.__name__ == (
        "_compute_trajectory_metrics"
    )


def test_trajectory_metric_api_is_public() -> None:
    import genco_evidence_graph as package

    assert "compute_trajectory_metrics" in package.__all__
    assert callable(package.compute_trajectory_metrics)


def _provenance_spec(
    metric_spec: TrajectoryMetricSpec,
    output: Path,
) -> TrajectoryMetricProvenanceSpec:
    return TrajectoryMetricProvenanceSpec(
        metric_manifest=metric_spec.manifest_output,
        trajectory=metric_spec.trajectory,
        trajectory_manifest=(
            metric_spec.trajectory_manifest
        ),
        cohort=metric_spec.cohort,
        cohort_manifest=metric_spec.cohort_manifest,
        entity_metrics=metric_spec.entity_metrics_output,
        step_metrics=metric_spec.step_metrics_output,
        scenario_summary=(
            metric_spec.scenario_summary_output
        ),
        group_comparison=(
            metric_spec.group_comparison_output
        ),
        diagnostics=metric_spec.diagnostics_output,
        output=output,
    )


def test_write_trajectory_metric_provenance(
    tmp_path: Path,
) -> None:
    metric_spec = _write_fixture(tmp_path)
    derive_trajectory_metrics(metric_spec)
    spec = _provenance_spec(
        metric_spec,
        tmp_path / "metric_provenance.ttl",
    )

    result = write_trajectory_metric_provenance(spec)

    assert result.triple_count > 0
    assert result.entity_count == 40
    assert result.transition_count == 440
    assert result.scenario_count == 20
    assert result.output.is_file()


def test_metric_provenance_rejects_modified_output(
    tmp_path: Path,
) -> None:
    metric_spec = _write_fixture(tmp_path)
    derive_trajectory_metrics(metric_spec)
    spec = _provenance_spec(
        metric_spec,
        tmp_path / "metric_provenance.ttl",
    )
    spec.entity_metrics.write_bytes(b"modified")

    with pytest.raises(
        ValueError,
        match="entity_metrics hash disagrees",
    ):
        build_trajectory_metric_provenance_graph(spec)


def test_metric_provenance_command_is_registered() -> None:
    args = build_parser().parse_args(
        [
            "build-trajectory-metric-provenance",
            "--metric-manifest", "manifest.json",
            "--trajectory", "trajectory.parquet",
            "--trajectory-manifest", "trajectory.json",
            "--cohort", "cohort.parquet",
            "--cohort-manifest", "cohort.json",
            "--entity-metrics", "entities.parquet",
            "--step-metrics", "steps.parquet",
            "--scenario-summary", "scenarios.parquet",
            "--group-comparison", "groups.parquet",
            "--diagnostics", "diagnostics.json",
            "--output", "metrics.ttl",
        ]
    )

    assert args.handler.__name__ == (
        "_build_trajectory_metric_provenance"
    )


def test_metric_provenance_api_is_public() -> None:
    import genco_evidence_graph as package

    assert (
        "build_trajectory_metric_provenance"
        in package.__all__
    )
    assert callable(
        package.build_trajectory_metric_provenance
    )
