import json
from pathlib import Path

import pytest
import torch

import numpy as np
import pandas as pd

from genco_evidence_graph.scenario_selection import (
    ScenarioScoringInputSpec,
    ScenarioScoringSpec,
    _exact_nearest_references,
    _load_residuals,
    _select_device,
    score_scenarios,
    verify_scenario_scoring_inputs,
    ScoreSummarySpec,
    summarize_score_distributions,
    ContrastingScenarioSelectionSpec,
    select_contrasting_scenarios,
)
from genco_evidence_graph.provenance import sha256_file
from genco_evidence_graph.cli import build_parser

def _write_run(
    root: Path,
    *,
    dataset_split: str,
    split_path: Path,
    embedding_name: str,
    prediction_name: str,
    checkpoint_sha256: str = "a" * 64,
    embedding_dimension: int = 384,
) -> Path:
    root.mkdir(parents=True)

    embedding = root / embedding_name
    prediction = root / prediction_name
    environment = root / "runtime_environment.txt"
    requirements = root / "graphkit-requirements.lock"
    source = root / "graphkit-source.txt"

    embedding.write_bytes(
        f"{dataset_split}-embeddings".encode()
    )
    prediction.write_bytes(
        f"{dataset_split}-predictions".encode()
    )
    environment.write_text("environment\n")
    requirements.write_text("requirements\n")
    source.write_text("source\n")

    artifacts = [
        {
            "path": embedding.name,
            "sha256": sha256_file(embedding),
        },
        {
            "path": prediction.name,
            "sha256": sha256_file(prediction),
        },
        {
            "path": environment.name,
            "sha256": sha256_file(environment),
        },
        {
            "path": requirements.name,
            "sha256": sha256_file(requirements),
        },
        {
            "path": source.name,
            "sha256": sha256_file(source),
        },
    ]

    manifest = {
        "artifacts": artifacts,
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": "b" * 64,
        "corpus_scenario_count": 8,
        "dataset_revision": "c" * 40,
        "dataset_split": dataset_split,
        "embedding_dimension": embedding_dimension,
        "export_embeddings": True,
        "format_version": 1,
        "graphkit_commit": "d" * 40,
        "grid_name": "case14_ieee",
        "normalizer_sha256": "e" * 64,
        "raw_dataset_sha256": "f" * 64,
        "run_id": f"{dataset_split}-run",
        "run_status": 3,
        "runtime": {
            "environment_path": environment.name,
            "environment_sha256": sha256_file(
                environment
            ),
            "graphkit": "0.9.0.3",
            "platform": "test-platform",
            "python_version": "3.12.10",
            "requirements_lock_path": requirements.name,
            "requirements_lock_sha256": sha256_file(
                requirements
            ),
            "source_commit": "d" * 40,
            "source_record_path": source.name,
            "source_record_sha256": sha256_file(source),
            "source_repository": (
                "https://example.invalid/graphkit.git"
            ),
            "torch": "2.11.0",
            "torch_scatter": "2.1.2",
            "torchaudio": "2.11.0",
            "torchvision": "0.26.0",
        },
        "seed": 42,
        "split_sha256": sha256_file(split_path),
        "task": "OptimalPowerFlow",
    }

    manifest_path = root / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return manifest_path


def _verified_fixture(
    tmp_path: Path,
) -> tuple[ScenarioScoringInputSpec, Path, Path]:
    test_split = tmp_path / "test.pt"
    train_split = tmp_path / "train.pt"

    torch.save([5, 6], test_split)
    torch.save([1, 2, 3, 4], train_split)

    target_root = tmp_path / "target"
    reference_root = tmp_path / "reference"

    target_manifest = _write_run(
        target_root,
        dataset_split="test",
        split_path=test_split,
        embedding_name="test_embeddings.parquet",
        prediction_name="test_predictions.parquet",
    )
    reference_manifest = _write_run(
        reference_root,
        dataset_split="train",
        split_path=train_split,
        embedding_name="train_embeddings.parquet",
        prediction_name="train_predictions.parquet",
    )

    spec = ScenarioScoringInputSpec(
        target_keyed_embeddings=(
            target_root / "test_embeddings.parquet"
        ),
        reference_keyed_embeddings=(
            reference_root / "train_embeddings.parquet"
        ),
        prediction_parquet=(
            target_root / "test_predictions.parquet"
        ),
        test_split=test_split,
        train_split=train_split,
        target_run_manifest=target_manifest,
        reference_run_manifest=reference_manifest,
    )
    return spec, target_manifest, reference_manifest


def _write_keyed_embeddings(
    path: Path,
    *,
    dataset_split: str,
    rows: list[tuple[int, str, int, list[float]]],
) -> None:
    pd.DataFrame(
        [
            {
                "scenario_id": scenario_id,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "correction_step": None,
                "embedding_stage": "final",
                "embedding": embedding,
                "dataset_split": dataset_split,
            }
            for (
                scenario_id,
                entity_type,
                entity_id,
                embedding,
            ) in rows
        ]
    ).to_parquet(
        path,
        index=False,
    )


def _refresh_artifact_hash(
    manifest_path: Path,
    artifact_path: Path,
) -> None:
    value = json.loads(manifest_path.read_text())

    matched = False
    for record in value["artifacts"]:
        if record["path"] == artifact_path.name:
            record["sha256"] = sha256_file(
                artifact_path
            )
            matched = True
            break

    if not matched:
        raise AssertionError(
            f"Artifact is absent from fixture manifest: "
            f"{artifact_path}"
        )

    manifest_path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

def test_scenario_inputs_are_manifest_bound(
    tmp_path: Path,
) -> None:
    spec, _, _ = _verified_fixture(tmp_path)

    result = verify_scenario_scoring_inputs(spec)

    assert result.target_run_id == "test-run"
    assert result.reference_run_id == "train-run"
    assert result.checkpoint_sha256 == "a" * 64
    assert result.dataset_revision == "c" * 40
    assert result.graphkit_commit == "d" * 40
    assert result.embedding_dimension == 384
    assert result.target_scenario_count == 2
    assert result.reference_scenario_count == 4


def test_scenario_inputs_reject_modified_artifact(
    tmp_path: Path,
) -> None:
    spec, _, _ = _verified_fixture(tmp_path)
    spec.prediction_parquet.write_text("modified")

    with pytest.raises(
        ValueError,
        match="Artifact hash disagrees",
    ):
        verify_scenario_scoring_inputs(spec)


def test_scenario_inputs_reject_incompatible_runs(
    tmp_path: Path,
) -> None:
    spec, _, reference_manifest = _verified_fixture(
        tmp_path
    )

    value = json.loads(reference_manifest.read_text())
    value["checkpoint_sha256"] = "9" * 64
    reference_manifest.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    )

    with pytest.raises(
        ValueError,
        match="run manifests are incompatible",
    ):
        verify_scenario_scoring_inputs(spec)


def test_scenario_inputs_reject_wrong_split_hash(
    tmp_path: Path,
) -> None:
    spec, _, _ = _verified_fixture(tmp_path)
    torch.save([5, 6, 7], spec.test_split)

    with pytest.raises(
        ValueError,
        match="Test split hash disagrees",
    ):
        verify_scenario_scoring_inputs(spec)


def test_scenario_inputs_reject_split_overlap(
    tmp_path: Path,
) -> None:
    spec, target_manifest, _ = _verified_fixture(
        tmp_path
    )

    torch.save([4, 5], spec.test_split)

    target_value = json.loads(target_manifest.read_text())
    target_value["split_sha256"] = sha256_file(
        spec.test_split
    )
    target_manifest.write_text(
        json.dumps(
            target_value,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    with pytest.raises(
        ValueError,
        match="Train and test splits overlap",
    ):
        verify_scenario_scoring_inputs(spec)


def test_exact_nearest_references_across_blocks() -> None:
    queries = np.array(
        [
            [0.0, 0.0],
            [9.0, 9.0],
        ],
        dtype=np.float32,
    )
    references = np.array(
        [
            [0.0, 1.0],
            [10.0, 10.0],
            [0.0, -1.0],
        ],
        dtype=np.float32,
    )

    progress: list[tuple[int, int]] = []

    distances, scenario_ids = (
        _exact_nearest_references(
            queries,
            references,
            ["10", "20", "30"],
            device=torch.device("cpu"),
            query_block_size=1,
            reference_block_size=2,
            progress=lambda completed, total: (
                progress.append((completed, total))
            ),
        )
    )

    np.testing.assert_allclose(
        distances,
        [1.0, np.sqrt(2.0)],
        rtol=1e-6,
        atol=1e-6,
    )
    assert scenario_ids == ["10", "20"]
    assert progress == [(2, 3), (3, 3)]


def test_exact_nearest_references_tie_breaks_by_id() -> None:
    queries = np.array(
        [[0.0, 0.0]],
        dtype=np.float32,
    )
    references = np.array(
        [
            [0.0, 1.0],
            [0.0, -1.0],
        ],
        dtype=np.float32,
    )

    distances, scenario_ids = (
        _exact_nearest_references(
            queries,
            references,
            ["30", "10"],
            device=torch.device("cpu"),
            query_block_size=1,
            reference_block_size=1,
        )
    )

    np.testing.assert_allclose(distances, [1.0])
    assert scenario_ids == ["10"]


@pytest.mark.parametrize(
    ("queries", "references", "message"),
    [
        (
            np.empty((0, 2), dtype=np.float32),
            np.ones((1, 2), dtype=np.float32),
            "cannot be empty",
        ),
        (
            np.ones((1, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            "cannot be empty",
        ),
        (
            np.ones((1, 2), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
            "dimensions disagree",
        ),
        (
            np.array(
                [[np.nan, 0.0]],
                dtype=np.float32,
            ),
            np.ones((1, 2), dtype=np.float32),
            "non-finite",
        ),
    ],
)
def test_exact_nearest_references_rejects_invalid_matrices(
    queries: np.ndarray,
    references: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _exact_nearest_references(
            queries,
            references,
            ["1"],
            device=torch.device("cpu"),
            query_block_size=1,
            reference_block_size=1,
        )


def test_select_device_accepts_cpu() -> None:
    assert _select_device("cpu") == torch.device("cpu")


def test_select_device_rejects_unknown_device() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported computation device",
    ):
        _select_device("unknown-device")


def test_load_residuals_aggregates_absolute_pbe(
    tmp_path: Path,
) -> None:
    prediction_path = tmp_path / "predictions.parquet"

    pd.DataFrame(
        {
            "scenario": [2, 1, 2, 1],
            "bus": [1, 1, 0, 0],
            "PBE": [-2.0, -3.0, 2.0, 1.0],
        }
    ).to_parquet(
        prediction_path,
        index=False,
    )

    result = _load_residuals(
        prediction_path,
        {"1", "2"},
    )

    assert result.to_dict(orient="records") == [
        {
            "scenario_id": "1",
            "mean_absolute_pbe": 2.0,
            "max_absolute_pbe": 3.0,
            "max_pbe_bus": 1,
        },
        {
            "scenario_id": "2",
            "mean_absolute_pbe": 2.0,
            "max_absolute_pbe": 2.0,
            "max_pbe_bus": 0,
        },
    ]


def test_load_residuals_rejects_duplicate_bus_key(
    tmp_path: Path,
) -> None:
    prediction_path = tmp_path / "predictions.parquet"

    pd.DataFrame(
        {
            "scenario": [1, 1],
            "bus": [0, 0],
            "PBE": [1.0, 2.0],
        }
    ).to_parquet(
        prediction_path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="duplicate scenario/bus",
    ):
        _load_residuals(
            prediction_path,
            {"1"},
        )


def test_load_residuals_rejects_split_mismatch(
    tmp_path: Path,
) -> None:
    prediction_path = tmp_path / "predictions.parquet"

    pd.DataFrame(
        {
            "scenario": [1, 1, 3, 3],
            "bus": [0, 1, 0, 1],
            "PBE": [1.0, 2.0, 3.0, 4.0],
        }
    ).to_parquet(
        prediction_path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="missing=.*2.*extra=.*3",
    ):
        _load_residuals(
            prediction_path,
            {"1", "2"},
        )


def test_load_residuals_rejects_non_finite_pbe(
    tmp_path: Path,
) -> None:
    prediction_path = tmp_path / "predictions.parquet"

    pd.DataFrame(
        {
            "scenario": [1],
            "bus": [0],
            "PBE": [np.nan],
        }
    ).to_parquet(
        prediction_path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="non-finite PBE",
    ):
        _load_residuals(
            prediction_path,
            {"1"},
        )


def test_load_residuals_rejects_incomplete_bus_coverage(
    tmp_path: Path,
) -> None:
    prediction_path = tmp_path / "predictions.parquet"

    pd.DataFrame(
        {
            "scenario": [1, 1, 2],
            "bus": [0, 1, 0],
            "PBE": [1.0, 2.0, 3.0],
        }
    ).to_parquet(
        prediction_path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="incomplete bus coverage",
    ):
        _load_residuals(
            prediction_path,
            {"1", "2"},
        )


def test_score_scenarios_writes_verified_table_and_manifest(
    tmp_path: Path,
) -> None:
    test_split = tmp_path / "test.pt"
    train_split = tmp_path / "train.pt"

    torch.save([5, 6], test_split)
    torch.save([1, 2, 3], train_split)

    target_root = tmp_path / "target"
    reference_root = tmp_path / "reference"

    target_manifest = _write_run(
        target_root,
        dataset_split="test",
        split_path=test_split,
        embedding_name="test_embeddings.parquet",
        prediction_name="test_predictions.parquet",
        embedding_dimension=1,
    )
    reference_manifest = _write_run(
        reference_root,
        dataset_split="train",
        split_path=train_split,
        embedding_name="train_embeddings.parquet",
        prediction_name="train_predictions.parquet",
        embedding_dimension=1,
    )

    target_embeddings = (
        target_root / "test_embeddings.parquet"
    )
    reference_embeddings = (
        reference_root / "train_embeddings.parquet"
    )
    predictions = (
        target_root / "test_predictions.parquet"
    )

    _write_keyed_embeddings(
        target_embeddings,
        dataset_split="test",
        rows=[
            (5, "bus", 0, [0.0]),
            (6, "bus", 0, [9.0]),
            (5, "generator", 0, [0.0]),
            (6, "generator", 0, [9.0]),
        ],
    )
    _write_keyed_embeddings(
        reference_embeddings,
        dataset_split="train",
        rows=[
            (1, "bus", 0, [0.0]),
            (2, "bus", 0, [10.0]),
            (3, "bus", 0, [0.0]),
            (1, "generator", 0, [1.0]),
            (2, "generator", 0, [10.0]),
            (3, "generator", 0, [-1.0]),
        ],
    )

    pd.DataFrame(
        {
            "scenario": [5, 5, 6, 6],
            "bus": [0, 1, 0, 1],
            "PBE": [1.0, -3.0, 2.0, -2.0],
        }
    ).to_parquet(
        predictions,
        index=False,
    )

    _refresh_artifact_hash(
        target_manifest,
        target_embeddings,
    )
    _refresh_artifact_hash(
        target_manifest,
        predictions,
    )
    _refresh_artifact_hash(
        reference_manifest,
        reference_embeddings,
    )

    output = tmp_path / "scores.parquet"
    manifest_output = tmp_path / "scores_manifest.json"

    progress: list[tuple[int, int]] = []

    result = score_scenarios(
        ScenarioScoringSpec(
            target_keyed_embeddings=target_embeddings,
            reference_keyed_embeddings=(
                reference_embeddings
            ),
            prediction_parquet=predictions,
            test_split=test_split,
            train_split=train_split,
            target_run_manifest=target_manifest,
            reference_run_manifest=reference_manifest,
            output=output,
            manifest_output=manifest_output,
            device="cpu",
            query_block_size=1,
            reference_block_size=2,
        ),
        progress=lambda completed, total: (
            progress.append((completed, total))
        ),
    )

    scores = pd.read_parquet(output)
    manifest = json.loads(
        manifest_output.read_text()
    )

    assert result.scenario_count == 2
    assert result.reference_count == 3
    assert result.device == "cpu"
    assert progress == [(2, 3), (3, 3)]

    assert scores["scenario_id"].tolist() == [
        "5",
        "6",
    ]
    assert scores[
        "nearest_reference_scenario_id"
    ].tolist() == ["1", "2"]

    np.testing.assert_allclose(
        scores[
            "nearest_reference_distance"
        ].to_numpy(),
        [1.0, np.sqrt(2.0)],
        rtol=1e-6,
        atol=1e-6,
    )

    assert scores["mean_absolute_pbe"].tolist() == [
        2.0,
        2.0,
    ]
    assert scores["max_absolute_pbe"].tolist() == [
        3.0,
        2.0,
    ]
    assert scores["max_pbe_bus"].tolist() == [
        1,
        0,
    ]

    assert manifest["scoring_policy_id"] == (
        "genco-scenario-scoring-v1"
    )
    assert manifest["row_count"] == 2
    assert manifest["target_scenario_count"] == 2
    assert manifest["reference_scenario_count"] == 3
    assert manifest[
        "entity_embedding_dimension"
    ] == 1
    assert manifest[
        "pooled_embedding_dimension"
    ] == 2
    assert manifest["device"] == "cpu"
    assert manifest["output"]["sha256"] == (
        sha256_file(output)
    )

def test_score_scenarios_command_is_registered() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "score-scenarios",
            "--target-keyed-embeddings",
            "target.parquet",
            "--reference-keyed-embeddings",
            "reference.parquet",
            "--prediction-parquet",
            "predictions.parquet",
            "--test-split",
            "test.pt",
            "--train-split",
            "train.pt",
            "--target-run-manifest",
            "target_manifest.json",
            "--reference-run-manifest",
            "reference_manifest.json",
            "--output",
            "scores.parquet",
            "--manifest-output",
            "scores_manifest.json",
        ]
    )

    assert args.command == "score-scenarios"
    assert args.handler.__name__ == "_score_scenarios"
    assert args.device == "auto"
    assert args.query_block_size == 256
    assert args.reference_block_size == 8192


def test_generate_scenario_scores_is_public() -> None:
    import genco_evidence_graph as package

    assert "generate_scenario_scores" in package.__all__
    assert callable(package.generate_scenario_scores)

def _write_score_fixture(
    tmp_path: Path,
) -> tuple[Path, Path]:
    rows: list[dict[str, object]] = []
    scenario_id = 1

    for distance_level in range(1, 5):
        for residual_level in range(1, 5):
            rows.append(
                {
                    "scenario_id": str(
                        scenario_id
                    ),
                    "nearest_reference_scenario_id": (
                        str(1000 + scenario_id)
                    ),
                    "nearest_reference_distance": (
                        float(distance_level)
                    ),
                    "mean_absolute_pbe": (
                        float(residual_level)
                    ),
                    "max_absolute_pbe": (
                        float(residual_level + 1)
                    ),
                    "max_pbe_bus": (
                        scenario_id % 14
                    ),
                    "distance_metric": (
                        "euclidean"
                    ),
                    "pooling_method": (
                        "mean_bus_concatenate_"
                        "mean_generator"
                    ),
                    "residual_metric": (
                        "mean_absolute_bus_pbe"
                    ),
                }
            )
            scenario_id += 1

    scores = pd.DataFrame(rows)
    score_path = tmp_path / "scores.parquet"
    manifest_path = (
        tmp_path / "scores_manifest.json"
    )

    scores.to_parquet(
        score_path,
        index=False,
    )

    manifest = {
        "schema_version": 1,
        "artifact_type": (
            "scenario_scoring_table"
        ),
        "scoring_policy_id": (
            "genco-scenario-scoring-v1"
        ),
        "row_count": len(scores),
        "target_split": "test",
        "reference_split": "train",
        "target_scenario_count": len(scores),
        "reference_scenario_count": 100,
        "distance_metric": "euclidean",
        "pooling_method": (
            "mean_bus_concatenate_mean_generator"
        ),
        "residual_metric": (
            "mean_absolute_bus_pbe"
        ),
        "output": {
            "path": str(score_path),
            "sha256": sha256_file(
                score_path
            ),
        },
    }
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    return score_path, manifest_path

def test_summarize_scores_writes_bound_distributions(
    tmp_path: Path,
) -> None:
    score_path, manifest_path = (
        _write_score_fixture(tmp_path)
    )
    output = tmp_path / "summary.json"

    result = summarize_score_distributions(
        ScoreSummarySpec(
            scores=score_path,
            score_manifest=manifest_path,
            output=output,
        )
    )

    summary = json.loads(output.read_text())

    assert result.scenario_count == 16
    assert result.group_candidate_counts == {
        "familiar_good": 1,
        "familiar_bad": 1,
        "unfamiliar_good": 1,
        "unfamiliar_bad": 1,
    }
    assert result.excluded_scenario_count == 12
    assert np.isfinite(result.spearman)

    assert summary["scenario_count"] == 16
    assert summary[
        "group_candidate_counts"
    ] == result.group_candidate_counts
    assert summary[
        "excluded_scenario_count"
    ] == 12
    assert summary["score_table"]["sha256"] == (
        sha256_file(score_path)
    )
    assert summary[
        "score_manifest"
    ]["sha256"] == sha256_file(
        manifest_path
    )


def test_summarize_scores_rejects_modified_table(
    tmp_path: Path,
) -> None:
    score_path, manifest_path = (
        _write_score_fixture(tmp_path)
    )

    scores = pd.read_parquet(score_path)
    scores.loc[
        0,
        "nearest_reference_distance",
    ] = 999.0
    scores.to_parquet(
        score_path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="hash disagrees",
    ):
        summarize_score_distributions(
            ScoreSummarySpec(
                scores=score_path,
                score_manifest=manifest_path,
                output=tmp_path / "summary.json",
            )
        )


def test_summarize_scores_rejects_constant_distribution(
    tmp_path: Path,
) -> None:
    score_path, manifest_path = (
        _write_score_fixture(tmp_path)
    )

    scores = pd.read_parquet(score_path)
    scores["nearest_reference_distance"] = 1.0
    scores.to_parquet(
        score_path,
        index=False,
    )

    manifest = json.loads(
        manifest_path.read_text()
    )
    manifest["output"]["sha256"] = sha256_file(
        score_path
    )
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    with pytest.raises(
        ValueError,
        match="correlation is undefined",
    ):
        summarize_score_distributions(
            ScoreSummarySpec(
                scores=score_path,
                score_manifest=manifest_path,
                output=tmp_path / "summary.json",
            )
        )

def test_summarize_scores_command_is_registered() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "summarize-scores",
            "--scores",
            "scores.parquet",
            "--score-manifest",
            "scores_manifest.json",
            "--output",
            "summary.json",
        ]
    )

    assert args.command == "summarize-scores"
    assert args.handler.__name__ == (
        "_summarize_scores"
    )
    assert args.scores == "scores.parquet"
    assert args.score_manifest == (
        "scores_manifest.json"
    )
    assert args.output == "summary.json"


def test_inspect_score_distributions_is_public() -> None:
    import genco_evidence_graph as package

    assert (
        "inspect_score_distributions"
        in package.__all__
    )
    assert callable(
        package.inspect_score_distributions
    )

def _write_populated_selection_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path]:
    rows: list[dict[str, object]] = []
    scenario_id = 1

    for distance_level in range(1, 13):
        for residual_level in range(1, 13):
            rows.append(
                {
                    "scenario_id": str(
                        scenario_id
                    ),
                    "nearest_reference_scenario_id": (
                        str(1000 + scenario_id)
                    ),
                    "nearest_reference_distance": (
                        float(distance_level)
                    ),
                    "mean_absolute_pbe": (
                        float(residual_level)
                    ),
                    "max_absolute_pbe": (
                        float(residual_level)
                        + (scenario_id % 5) / 10.0
                    ),
                    "max_pbe_bus": (
                        scenario_id % 14
                    ),
                    "distance_metric": "euclidean",
                    "pooling_method": (
                        "mean_bus_concatenate_"
                        "mean_generator"
                    ),
                    "residual_metric": (
                        "mean_absolute_bus_pbe"
                    ),
                }
            )
            scenario_id += 1

    scores = pd.DataFrame(rows)
    score_path = tmp_path / "scores.parquet"
    score_manifest_path = (
        tmp_path / "scores_manifest.json"
    )
    summary_path = tmp_path / "summary.json"

    scores.to_parquet(
        score_path,
        index=False,
    )

    score_manifest = {
        "schema_version": 1,
        "artifact_type": (
            "scenario_scoring_table"
        ),
        "scoring_policy_id": (
            "genco-scenario-scoring-v1"
        ),
        "row_count": len(scores),
        "target_split": "test",
        "reference_split": "train",
        "target_scenario_count": len(scores),
        "reference_scenario_count": 1000,
        "distance_metric": "euclidean",
        "pooling_method": (
            "mean_bus_concatenate_mean_generator"
        ),
        "residual_metric": (
            "mean_absolute_bus_pbe"
        ),
        "output": {
            "path": str(score_path),
            "sha256": sha256_file(
                score_path
            ),
        },
    }
    score_manifest_path.write_text(
        json.dumps(
            score_manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    summarize_score_distributions(
        ScoreSummarySpec(
            scores=score_path,
            score_manifest=score_manifest_path,
            output=summary_path,
        )
    )

    return (
        score_path,
        score_manifest_path,
        summary_path,
    )


def test_select_contrasting_scenarios_freezes_composition(
    tmp_path: Path,
) -> None:
    (
        score_path,
        score_manifest_path,
        summary_path,
    ) = _write_populated_selection_fixture(
        tmp_path
    )

    cohort_path = tmp_path / "cohort.parquet"
    manifest_path = tmp_path / "manifest.json"
    diagnostics_path = (
        tmp_path / "diagnostics.json"
    )
    references_path = (
        tmp_path / "references.csv"
    )

    result = select_contrasting_scenarios(
        ContrastingScenarioSelectionSpec(
            scores=score_path,
            score_manifest=score_manifest_path,
            summary=summary_path,
            output=cohort_path,
            manifest_output=manifest_path,
            diagnostics_output=diagnostics_path,
            training_reference_ids_output=(
                references_path
            ),
        )
    )

    selected = pd.read_parquet(cohort_path)
    manifest = json.loads(
        manifest_path.read_text()
    )
    diagnostics = json.loads(
        diagnostics_path.read_text()
    )

    assert result.selected_scenario_count == 20
    assert selected["scenario_id"].is_unique
    assert diagnostics["status"] == "selected"
    assert manifest["selection_policy_id"] == (
        "genco-contrasting-cohort-v2"
    )

    for group_name, group in selected.groupby(
        "cohort_group"
    ):
        assert group_name in {
            "familiar_good",
            "familiar_bad",
            "unfamiliar_good",
            "unfamiliar_bad",
        }
        assert len(group) == 5
        assert (
            group["selection_reason"]
            .value_counts()
            .to_dict()
        ) == {
            "corner": 3,
            "inner_boundary": 1,
            "physical_diversity": 1,
        }

    assert manifest[
        "cohort_artifact"
    ]["sha256"] == sha256_file(cohort_path)

    assert manifest[
        "training_reference_ids"
    ]["sha256"] == sha256_file(
        references_path
    )


def test_sparse_selection_writes_diagnostics_before_failure(
    tmp_path: Path,
) -> None:
    score_path, manifest_path = (
        _write_score_fixture(tmp_path)
    )
    summary_path = tmp_path / "summary.json"

    summarize_score_distributions(
        ScoreSummarySpec(
            scores=score_path,
            score_manifest=manifest_path,
            output=summary_path,
        )
    )

    diagnostics_path = (
        tmp_path / "diagnostics.json"
    )
    cohort_path = tmp_path / "cohort.parquet"
    selection_manifest = (
        tmp_path / "selection_manifest.json"
    )
    references_path = (
        tmp_path / "references.csv"
    )

    with pytest.raises(
        ValueError,
        match="Diagnostics:",
    ):
        select_contrasting_scenarios(
            ContrastingScenarioSelectionSpec(
                scores=score_path,
                score_manifest=manifest_path,
                summary=summary_path,
                output=cohort_path,
                manifest_output=(
                    selection_manifest
                ),
                diagnostics_output=(
                    diagnostics_path
                ),
                training_reference_ids_output=(
                    references_path
                ),
            )
        )

    diagnostics = json.loads(
        diagnostics_path.read_text()
    )

    assert diagnostics["status"] == (
        "insufficient_candidates"
    )
    assert diagnostics["sparse_groups"]
    assert not cohort_path.exists()
    assert not selection_manifest.exists()
    assert not references_path.exists()


def test_selection_rejects_modified_summary(
    tmp_path: Path,
) -> None:
    (
        score_path,
        score_manifest_path,
        summary_path,
    ) = _write_populated_selection_fixture(
        tmp_path
    )

    summary = json.loads(
        summary_path.read_text()
    )
    summary["group_candidate_counts"][
        "familiar_good"
    ] += 1
    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    with pytest.raises(
        ValueError,
        match="group counts disagree",
    ):
        select_contrasting_scenarios(
            ContrastingScenarioSelectionSpec(
                scores=score_path,
                score_manifest=(
                    score_manifest_path
                ),
                summary=summary_path,
                output=tmp_path / "cohort.parquet",
                manifest_output=(
                    tmp_path / "manifest.json"
                ),
                diagnostics_output=(
                    tmp_path / "diagnostics.json"
                ),
                training_reference_ids_output=(
                    tmp_path / "references.csv"
                ),
            )
        )


def test_select_cohort_command_is_registered() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "select-cohort",
            "--scores",
            "scores.parquet",
            "--score-manifest",
            "scores_manifest.json",
            "--summary",
            "summary.json",
            "--output",
            "cohort.parquet",
            "--manifest-output",
            "cohort_manifest.json",
            "--diagnostics-output",
            "diagnostics.json",
            "--training-reference-ids-output",
            "references.csv",
        ]
    )

    assert args.command == "select-cohort"
    assert args.handler.__name__ == (
        "_select_cohort"
    )
    assert args.scores == "scores.parquet"
    assert args.score_manifest == (
        "scores_manifest.json"
    )
    assert args.summary == "summary.json"
    assert args.output == "cohort.parquet"
    assert args.manifest_output == (
        "cohort_manifest.json"
    )
    assert args.diagnostics_output == (
        "diagnostics.json"
    )
    assert (
        args.training_reference_ids_output
        == "references.csv"
    )


def test_freeze_contrasting_cohort_is_public() -> None:
    import genco_evidence_graph as package

    assert (
        "freeze_contrasting_cohort"
        in package.__all__
    )
    assert callable(
        package.freeze_contrasting_cohort
    )