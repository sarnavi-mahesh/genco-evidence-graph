import os
from pathlib import Path

import pytest

from genco_evidence_graph.cli import main
from genco_evidence_graph.training_metadata import (
    EXPECTED_TRAINING_METADATA_PATHS,
    load_training_metadata_manifest,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_RELATIVE_PATH = Path(
    "mlflow/train/case14_ieee/seed42/"
    "1927fe80558a48539ee37be478706781"
)

EXTRA_UPSTREAM_PATHS = {
    "artifacts/test/case14_ieee_RMSE.csv",
    "artifacts/test_plots/case14_ieee/case14_ieee_correlation_PQ.png",
    "artifacts/test_plots/case14_ieee/case14_ieee_correlation_PV.png",
    "artifacts/test_plots/case14_ieee/case14_ieee_correlation_REF.png",
    "artifacts/test_plots/case14_ieee/case14_ieee_objective.png",
    "artifacts/test_plots/case14_ieee/max_residual_P.png",
    "artifacts/test_plots/case14_ieee/max_residual_Q.png",
    "artifacts/test_plots/case14_ieee/mean_residual_P.png",
    "artifacts/test_plots/case14_ieee/mean_residual_Q.png",
}


def test_case14_build_evidence_from_supplied_79_file_metadata(tmp_path) -> None:
    if os.environ.get("GENCO_RUN_CASE14_INTEGRATION") != "1":
        pytest.skip("Set GENCO_RUN_CASE14_INTEGRATION=1 to run the real-data test")

    normalized_run = (
        REPO_ROOT / "genco-opfdata" / RUN_RELATIVE_PATH
    )
    original_root = REPO_ROOT / "artifacts/original_upstream_training_metadata"
    manifest_path = normalized_run / "upstream_source_manifest.json"

    frozen_graph = (
            REPO_ROOT
            / "artifacts/case14_scenario_215429_evidence.ttl"
    )

    required_paths = [
        frozen_graph,
        manifest_path,
        original_root / ".cache/huggingface",
        REPO_ROOT / "artifacts/case14_seed42_test_embeddings/run_manifest.json",
        REPO_ROOT / "artifacts/case14_seed42_test_no_embeddings/run_manifest.json",
        REPO_ROOT / "artifacts/case14_seed42_train_embeddings/run_manifest.json",
        REPO_ROOT / "data/case14_ieee/raw",
        REPO_ROOT / "genco-opfdata/splits/test.pt",
        REPO_ROOT / "genco-opfdata/splits/train.pt",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        pytest.skip("Missing real-data integration inputs: " + ", ".join(missing))

    manifest = load_training_metadata_manifest(manifest_path)
    manifest_paths = {record.path for record in manifest.files}

    assert len(EXPECTED_TRAINING_METADATA_PATHS) == 79
    assert len(manifest.files) == 79
    assert EXTRA_UPSTREAM_PATHS <= manifest_paths

    output_graph = tmp_path / "case14_scenario_215429_evidence.ttl"

    build_args = [
        "build-evidence",
        "--dataset-id", "opfdata_case14_ieee",
        "--checkpoint", str(REPO_ROOT / "genco-opfdata/case14_ieee/seed42/best_model_state_dict.pt"),
        "--prediction-id", "215429",
        "--prediction-parquet", str(REPO_ROOT / "artifacts/case14_seed42_test_embeddings/case14_ieee_predictions.parquet"),
        "--generator-prediction-parquet", str(REPO_ROOT / "artifacts/case14_seed42_test_embeddings/case14_ieee_gen_predictions.parquet"),
        "--target-keyed-embeddings", str(REPO_ROOT / "artifacts/case14_seed42_test_embeddings/case14_ieee_keyed_embeddings.parquet"),
        "--reference-keyed-embeddings", str(REPO_ROOT / "artifacts/case14_seed42_train_embeddings/case14_ieee_keyed_embeddings.parquet"),
        "--reference-prediction-parquet", str(REPO_ROOT / "artifacts/case14_seed42_train_embeddings/case14_ieee_predictions.parquet"),
        "--reference-generator-prediction-parquet", str(REPO_ROOT / "artifacts/case14_seed42_train_embeddings/case14_ieee_gen_predictions.parquet"),
        "--reference-run-manifest", str(REPO_ROOT / "artifacts/case14_seed42_train_embeddings/run_manifest.json"),
        "--reference-split-artifact", str(REPO_ROOT / "genco-opfdata/splits/train.pt"),
        "--reference-config-artifact", str(REPO_ROOT / "configs/case14_seed42_test_embeddings.yaml"),
        "--training-run-metadata", str(normalized_run),
        "--original-upstream-training-metadata-root", str(original_root),
        "--model-repository-path", str(REPO_ROOT / "genco-opfdata"),
        "--raw-dataset-path", str(REPO_ROOT / "data/case14_ieee/raw"),
        "--split-artifact", str(REPO_ROOT / "genco-opfdata/splits/test.pt"),
        "--normalizer-artifact", str(REPO_ROOT / "genco-opfdata/case14_ieee/seed42/normalizer_stats.pt"),
        "--config-artifact", str(REPO_ROOT / "configs/case14_seed42_test_embeddings.yaml"),
        "--run-manifest", str(REPO_ROOT / "artifacts/case14_seed42_test_embeddings/run_manifest.json"),
        "--reproduction-run-manifest", str(REPO_ROOT / "artifacts/case14_seed42_test_no_embeddings/run_manifest.json"),
        "--baseline-tolerance-config", str(REPO_ROOT / "configs/baseline_equivalence_tolerances.yaml"),
        "--expected-dataset-revision", "78c3af1e915284e8c7c31f84879878bcf06db6dd",
        "--expected-graphkit-commit", "f104900245cba26d1380037eea562cf633a97484",
        "--output", str(output_graph),
    ]

    assert main(build_args) == 0
    assert output_graph.is_file()
    assert output_graph.read_bytes() == frozen_graph.read_bytes()

    assert main([
        "validate",
        "--graph", str(output_graph),
        "--shapes", str(REPO_ROOT / "ontology/shapes.ttl"),
    ]) == 0