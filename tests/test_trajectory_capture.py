import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from genco_evidence_graph.cli import build_parser
from genco_evidence_graph.provenance import sha256_file
from genco_evidence_graph.splits import load_split_ids
from genco_evidence_graph.trajectory_capture import (
    CohortInferenceSplitSpec,
    prepare_cohort_inference_splits,
)


def _write_split(path: Path, values: list[int]) -> None:
    torch.save(
        [
            torch.tensor(value, dtype=torch.int64)
            for value in values
        ],
        path,
    )


def _write_fixture(
    tmp_path: Path,
) -> CohortInferenceSplitSpec:
    train = tmp_path / "source_train.pt"
    validation = tmp_path / "source_val.pt"
    test = tmp_path / "source_test.pt"
    cohort = tmp_path / "cohort.parquet"
    cohort_manifest = (
        tmp_path / "cohort_manifest.json"
    )

    _write_split(train, list(range(1, 31)))
    _write_split(validation, list(range(31, 41)))
    selected = list(range(41, 61))
    _write_split(test, selected + list(range(61, 71)))
    pd.DataFrame(
        {
            "scenario_id": [
                str(value)
                for value in selected
            ],
        }
    ).to_parquet(cohort, index=False)
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
                "selected_test_scenario_count": 20,
                "selected_scenarios": [
                    {"scenario_id": str(value)}
                    for value in selected
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    return CohortInferenceSplitSpec(
        train_split=train,
        validation_split=validation,
        test_split=test,
        cohort=cohort,
        cohort_manifest=cohort_manifest,
        output_dir=tmp_path / "cohort_splits",
    )


def test_prepare_cohort_inference_splits(
    tmp_path: Path,
) -> None:
    spec = _write_fixture(tmp_path)

    result = prepare_cohort_inference_splits(spec)

    assert result.selected_scenario_count == 20
    assert sha256_file(result.train_split) == (
        sha256_file(spec.train_split)
    )
    assert sha256_file(result.validation_split) == (
        sha256_file(spec.validation_split)
    )
    assert load_split_ids(result.test_split) == tuple(
        str(value)
        for value in range(41, 61)
    )

    manifest = json.loads(result.manifest.read_text())
    assert manifest["selected_scenario_ids"] == [
        str(value)
        for value in range(41, 61)
    ]
    assert manifest["outputs"]["test"][
        "sha256"
    ] == sha256_file(result.test_split)


def test_prepare_cohort_inference_rejects_non_test_scenario(
    tmp_path: Path,
) -> None:
    spec = _write_fixture(tmp_path)
    table = pd.read_parquet(spec.cohort)
    table.loc[0, "scenario_id"] = "1"
    table.to_parquet(spec.cohort, index=False)

    manifest = json.loads(
        spec.cohort_manifest.read_text()
    )
    manifest["cohort_artifact"]["sha256"] = (
        sha256_file(spec.cohort)
    )
    manifest["selected_scenarios"][0][
        "scenario_id"
    ] = "1"
    spec.cohort_manifest.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    with pytest.raises(
        ValueError,
        match="outside the authoritative test split",
    ):
        prepare_cohort_inference_splits(spec)


def test_prepare_cohort_inference_rejects_modified_cohort(
    tmp_path: Path,
) -> None:
    spec = _write_fixture(tmp_path)
    spec.cohort.write_bytes(b"modified")

    with pytest.raises(
        ValueError,
        match="cohort_artifact hash disagrees",
    ):
        prepare_cohort_inference_splits(spec)


def test_prepare_cohort_inference_command_is_registered() -> None:
    args = build_parser().parse_args(
        [
            "prepare-cohort-inference-splits",
            "--train-split",
            "train.pt",
            "--validation-split",
            "val.pt",
            "--test-split",
            "test.pt",
            "--cohort",
            "cohort.parquet",
            "--cohort-manifest",
            "cohort_manifest.json",
            "--output-dir",
            "cohort_splits",
        ]
    )

    assert args.command == (
        "prepare-cohort-inference-splits"
    )
    assert args.handler.__name__ == (
        "_prepare_cohort_inference_splits"
    )


def test_prepare_cohort_inference_splits_is_public() -> None:
    import genco_evidence_graph as package

    assert (
        "prepare_cohort_inference_splits"
        in package.__all__
    )
    assert callable(
        package.prepare_cohort_inference_splits
    )


def test_finalize_trajectory_capture_command_is_registered() -> None:
    args = build_parser().parse_args(
        [
            "finalize-trajectory-capture",
            "--baseline-dir",
            "baseline",
            "--instrumented-dir",
            "instrumented",
            "--split-artifact",
            "test.pt",
            "--tolerance-config",
            "tolerances.yaml",
            "--output",
            "trajectories.parquet",
            "--manifest-output",
            "trajectory_manifest.json",
        ]
    )

    assert args.command == (
        "finalize-trajectory-capture"
    )
    assert args.handler.__name__ == (
        "_finalize_trajectory_capture"
    )


def test_finalize_trajectory_capture_is_public() -> None:
    import genco_evidence_graph as package

    assert (
        "finalize_trajectory_capture"
        in package.__all__
    )
    assert callable(
        package.finalize_trajectory_capture
    )
