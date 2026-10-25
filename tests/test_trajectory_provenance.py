import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from genco_evidence_graph.cli import build_parser
from genco_evidence_graph.provenance import sha256_file
from genco_evidence_graph.trajectory_provenance import (
    TrajectoryProvenanceSpec,
    build_trajectory_provenance_graph,
    write_trajectory_provenance,
)


def _record(path: Path, **values) -> dict:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        **values,
    }


def _fixture(tmp_path: Path) -> TrajectoryProvenanceSpec:
    trajectory = tmp_path / "trajectory.parquet"
    baseline = tmp_path / "baseline.json"
    instrumented = tmp_path / "instrumented.json"
    split = tmp_path / "test.pt"
    tolerances = tmp_path / "tolerances.yaml"
    capture = tmp_path / "capture.json"
    output = tmp_path / "trajectory.ttl"

    pd.DataFrame(
        {
            "scenario_id": [1],
            "entity_type": ["bus"],
            "entity_id": [0],
            "correction_step": [1],
            "embedding_stage": [
                "prediction_head_input"
            ],
            "embedding": [
                np.array([1.0, 2.0], dtype=np.float32)
            ],
        }
    ).to_parquet(trajectory, index=False)
    baseline.write_text("{}\n")
    instrumented.write_text("{}\n")
    split.write_bytes(b"split")
    tolerances.write_text("policy: test\n")
    capture.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": (
                    "genco_cohort_latent_trajectory"
                ),
                "trajectory": _record(
                    trajectory,
                    row_count=1,
                    scenario_count=1,
                    entity_count=1,
                    correction_step_count=1,
                    embedding_dimension=2,
                    embedding_stage=(
                        "prediction_head_input"
                    ),
                ),
                "baseline_run_manifest": _record(
                    baseline,
                    run_id="baseline",
                ),
                "instrumented_run_manifest": _record(
                    instrumented,
                    run_id="instrumented",
                ),
                "split_artifact": _record(split),
                "tolerance_policy": _record(
                    tolerances,
                    policy_id="test-policy",
                ),
                "equivalence": {
                    "passed": True,
                    "instrumented_final_snapshot_exact": True,
                    "final_embedding_maximum_absolute_difference": 0.0,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return TrajectoryProvenanceSpec(
        trajectory=trajectory,
        capture_manifest=capture,
        baseline_run_manifest=baseline,
        instrumented_run_manifest=instrumented,
        split_artifact=split,
        tolerance_config=tolerances,
        output=output,
    )


def test_write_trajectory_provenance(
    tmp_path: Path,
) -> None:
    spec = _fixture(tmp_path)

    result = write_trajectory_provenance(spec)

    assert result.triple_count > 0
    assert result.row_count == 1
    assert result.scenario_count == 1
    assert result.entity_count == 1
    assert result.output.is_file()


def test_trajectory_provenance_rejects_tampering(
    tmp_path: Path,
) -> None:
    spec = _fixture(tmp_path)
    spec.trajectory.write_bytes(b"modified")

    with pytest.raises(
        ValueError,
        match="hash disagrees",
    ):
        build_trajectory_provenance_graph(spec)


def test_trajectory_provenance_command_is_registered() -> None:
    args = build_parser().parse_args(
        [
            "build-trajectory-provenance",
            "--trajectory", "trajectory.parquet",
            "--capture-manifest", "capture.json",
            "--baseline-run-manifest", "baseline.json",
            "--instrumented-run-manifest", "instrumented.json",
            "--split-artifact", "test.pt",
            "--tolerance-config", "tolerances.yaml",
            "--output", "trajectory.ttl",
        ]
    )

    assert args.handler.__name__ == (
        "_build_trajectory_provenance"
    )


def test_trajectory_provenance_api_is_public() -> None:
    import genco_evidence_graph as package

    assert "build_trajectory_provenance" in package.__all__
    assert callable(package.build_trajectory_provenance)
