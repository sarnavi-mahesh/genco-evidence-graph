import hashlib
import base64
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from rdflib import BNode, Graph, Literal
from rdflib.namespace import PROV, RDF
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from genco_evidence_graph.adapters import (
    TRAINING_METADATA_NORMALIZATION,
    extract_huggingface_revision,
    load_config_metadata,
    load_mlflow_run_metadata,
    load_training_run_metadata,
    verify_normalized_upstream_metadata,
    verify_run_matches_config,
)
from genco_evidence_graph.attestation import (
    PRODUCTION_ATTESTATION_TRUST_POLICY,
    VerifiedTrainingAttestation,
    canonical_attestation_payload,
    verify_production_training_attestation,
    verify_training_attestation,
)
from genco_evidence_graph.api import query_evidence, validate_evidence
from genco_evidence_graph.baseline import (
    APPROVED_TOLERANCE_POLICY,
    APPROVED_TOLERANCE_POLICY_SHA256,
    _compact_native_embedding_artifacts,
    _find_mlflow_run,
    _graphkit_python_identity,
    _verify_selected_runtime,
    canonicalize_graphkit_embeddings,
    compare_baseline_residuals,
    compare_keyed_baseline,
    prepare_reference_prediction_splits,
    verify_embedding_entity_coverage,
    verify_reproduction_manifest,
    verify_run_manifest,
)
from genco_evidence_graph.cli import (
    _comparison_maxima,
    _keyed_embedding_table,
    _load_verified_training_run,
    _physical_evidence,
    build_parser,
    main,
)
from genco_evidence_graph.contracts import MetricSemantics, PhysicalEvidence
from genco_evidence_graph.embeddings import ReferenceIndex, pool_scenario_embeddings
from genco_evidence_graph.provenance import (
    canonical_json_hash,
    sha256_file,
    sha256_path,
    stable_uri,
)
from genco_evidence_graph.queries import (
    ASSOCIATED_TRAINING_EVIDENCE,
    AVAILABLE_RESIDUAL_TYPES,
    AVAILABLE_VIOLATION_TYPES,
    CHECKPOINT,
    EMBEDDING_DISTANCE,
    REFERENCE_CASES,
    REPRODUCTION,
    RESIDUAL,
    VIOLATIONS,
    high_residual_and_distance,
)
from genco_evidence_graph.rdf import GENCO, build_evidence_graph
from genco_evidence_graph.validation import (
    validate_evidence_graph,
    validate_synthetic_evidence_graph,
)
from genco_evidence_graph.splits import load_split_ids, verify_split_membership
from genco_evidence_graph.training_metadata import (
    EXPECTED_TRAINING_METADATA_PATHS,
    TRAINING_METADATA_NORMALIZATION,
    load_training_metadata_manifest,
    prepare_normalized_training_metadata,
    training_metadata_manifest_from_dict,
    training_metadata_manifest_to_dict,
)
from genco_evidence_graph.contracts import (
    DatasetProvenance,
    ModelProvenance,
    PredictionEvidence,
    TrainingEvidenceAssociation,
)


def _unverified_training_association() -> TrainingEvidenceAssociation:
    return TrainingEvidenceAssociation(
        association_status="co-published-compatible-unverified-production",
        evidence_basis=("co-publication at a pinned repository revision",),
        production_identity_verified=False,
        configuration_compatible=True,
        provenance_limitations=(
            "training run has no checkpoint hash; historical commit unavailable",
        ),
    )


def _synthetic_training_attestation(
    checkpoint_hash: str = "a" * 64,
) -> tuple[
    VerifiedTrainingAttestation,
    dict[str, bytes],
]:
    private_key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    issuer = "synthetic-review-issuer"
    document = {
        "format_version": 1,
        "attestation_id": "synthetic-attestation",
        "issuer": issuer,
        "source_uri": "https://example.invalid/synthetic-attestation.json",
        "checkpoint_id": "checkpoint",
        "checkpoint_sha256": checkpoint_hash,
        "training_run_id": "run",
        "content_sha256": "9" * 64,
        "verification_status": "verified",
    }
    signature = base64.b64encode(
        private_key.sign(canonical_attestation_payload(document)),
    ).decode()
    trusted_keys = {
        issuer: private_key.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        ),
    }
    return (
        verify_training_attestation(document, signature, trusted_keys),
        trusted_keys,
    )


def test_pooling_is_deterministic_and_concatenates_entity_means() -> None:
    bus = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    gen = np.array([[[5.0, 6.0]]])
    np.testing.assert_allclose(pool_scenario_embeddings(bus, gen), [[2.0, 3.0, 5.0, 6.0]])


def test_reference_index_has_stable_tie_breaking_and_ranks() -> None:
    index = ReferenceIndex(["b", "a"], np.array([[1.0, 0.0], [-1.0, 0.0]]))
    matches = index.query(np.array([0.0, 0.0]), top_k=2)
    assert [(match.scenario_id, match.rank) for match in matches] == [("a", 1), ("b", 2)]


def test_reference_index_rejects_dimension_mismatch() -> None:
    index = ReferenceIndex(["a"], np.array([[1.0, 2.0]]))
    with pytest.raises(ValueError, match="dimension"):
        index.query(np.array([1.0]))


def test_graphkit_python_identity_preserves_venv_symlink(tmp_path) -> None:
    checkout = tmp_path / "gridfm-graphkit"
    executable = checkout / ".venv/bin/python"
    executable.parent.mkdir(parents=True)
    system_python = tmp_path / "python3.12"
    system_python.write_text("#!/bin/sh\n")
    executable.symlink_to(system_python)

    selected, relative = _graphkit_python_identity(
        executable,
        checkout,
    )

    assert selected == executable.absolute()
    assert selected.resolve() == system_python.resolve()
    assert relative == ".venv/bin/python"


def test_identity_helpers_are_deterministic() -> None:
    assert stable_uri("prediction", "case 14/0") == (
        "https://genco.example/id/prediction/case%2014%2F0"
    )
    assert canonical_json_hash({"b": 2, "a": 1}) == canonical_json_hash({"a": 1, "b": 2})


def test_directory_hash_includes_relative_paths_and_contents(tmp_path) -> None:
    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "a.txt").write_text("a")
    first = sha256_path(tmp_path)
    (tmp_path / "a.txt").write_text("changed")
    assert sha256_path(tmp_path) != first


def test_authoritative_metadata_adapters_and_mismatch_detection(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "seed": 42,
                "task": {"task_name": "OptimalPowerFlow"},
                "data": {"networks": ["case14_ieee"], "scenarios": [300000]},
                "model": {"hidden_size": 48, "type": "GNS_heterogeneous"},
            },
        ),
    )
    run = tmp_path / "run"
    (run / "tags").mkdir(parents=True)
    for relative, value in {
        "params/args/seed": "42",
        "params/args/get_embeddings": "True",
        "params/args/data/networks": "['case14_ieee']",
        "params/args/data/scenarios": "[300000]",
        "params/args/task/task_name": "OptimalPowerFlow",
        "params/args/model/hidden_size": "48",
    }.items():
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    (run / "meta.yaml").write_text(
        yaml.safe_dump(
            {
                "run_id": "run-1",
                "run_name": "embedding-run",
                "status": 3,
                "start_time": 1,
                "end_time": 2,
            },
        ),
    )
    (run / "tags" / "mlflow.source.git.commit").write_text("a" * 40)
    config = load_config_metadata(config_path)
    metadata = load_mlflow_run_metadata(run)
    verify_run_matches_config(metadata, config)
    assert metadata.run_id == "run-1"
    bad_config = config.__class__(
        seed=14,
        grid_name=config.grid_name,
        task=config.task,
        corpus_scenario_count=config.corpus_scenario_count,
        hidden_dimension=config.hidden_dimension,
        model_variant=config.model_variant,
    )
    with pytest.raises(ValueError, match="seed"):
        verify_run_matches_config(metadata, bad_config)

    revision = "1" * 40
    tree = tmp_path / "raw" / ".cache" / "huggingface" / "trees"
    tree.mkdir(parents=True)
    (tree / f"{revision}.json").write_text("{}")
    assert extract_huggingface_revision(tmp_path / "raw") == revision

    source_files = {}
    for name in ("checkpoint.pt", "normalizer.pt"):
        source_files[name] = tmp_path / name
        source_files[name].write_text(name)
    source_files["split.pt"] = tmp_path / "split.pt"
    torch.save([torch.tensor(1)], source_files["split.pt"])
    artifacts = {
        "predictions.parquet": tmp_path / "predictions.parquet",
        "gen_predictions.parquet": tmp_path / "gen_predictions.parquet",
        "keyed.parquet": tmp_path / "keyed.parquet",
    }
    pd.DataFrame(
        {"scenario": [1], "bus": [0]},
    ).to_parquet(artifacts["predictions.parquet"])
    pd.DataFrame(
        {"scenario": [1], "idx": [0]},
    ).to_parquet(artifacts["gen_predictions.parquet"])
    pd.DataFrame(
        {
            "scenario_id": [1, 1],
            "entity_type": ["bus", "generator"],
            "entity_id": [0, 0],
            "embedding": [[0.1, 0.2], [0.3, 0.4]],
        },
    ).to_parquet(artifacts["keyed.parquet"])

    artifacts["effective_config.yaml"] = tmp_path / "effective_config.yaml"
    artifacts["effective_config.yaml"].write_text(config_path.read_text())
    artifacts["runtime_environment.txt"] = tmp_path / "runtime_environment.txt"
    artifacts["graphkit-requirements.lock"] = (
        tmp_path / "graphkit-requirements.lock"
    )
    artifacts["graphkit-source.txt"] = tmp_path / "graphkit-source.txt"
    artifacts["runtime_environment.txt"].write_text("torch==2.11.0\n")
    artifacts["graphkit-requirements.lock"].write_text("torch==2.11.0\n")
    artifacts["graphkit-source.txt"].write_text(
        "repository=https://github.com/gridfm/gridfm-graphkit.git\n"
        f"commit={metadata.graphkit_commit}\n",
    )
    runtime = {
        "python_version": "3.12.10",
        "platform": "macOS-14.8.9-arm64-arm-64bit",
        "graphkit": "0.9.0.3",
        "torch": "2.11.0",
        "torchvision": "0.26.0",
        "torchaudio": "2.11.0",
        "torch_scatter": "2.1.2",
        "python_executable": ".venv/bin/python",
        "environment_path": "runtime_environment.txt",
        "environment_sha256": sha256_file(
            artifacts["runtime_environment.txt"],
        ),
        "requirements_lock_path": "graphkit-requirements.lock",
        "requirements_lock_sha256": sha256_file(
            artifacts["graphkit-requirements.lock"],
        ),
        "source_record_path": "graphkit-source.txt",
        "source_record_sha256": sha256_file(
            artifacts["graphkit-source.txt"],
        ),
        "source_repository": "https://github.com/gridfm/gridfm-graphkit.git",
        "source_commit": metadata.graphkit_commit,
    }
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "run_id": metadata.run_id,
                "graphkit_commit": metadata.graphkit_commit,
                "dataset_revision": revision,
                "embedding_dimension": 2,
                "mlflow_run_path": "run",
                "mlflow_run_sha256": sha256_path(run),
                "checkpoint_sha256": sha256_file(source_files["checkpoint.pt"]),
                "normalizer_sha256": sha256_file(source_files["normalizer.pt"]),
                "config_sha256": sha256_file(config_path),
                "effective_config_sha256": sha256_file(
                    artifacts["effective_config.yaml"],
                ),
                "split_sha256": sha256_file(source_files["split.pt"]),
                "raw_dataset_sha256": sha256_path(tmp_path / "raw"),
                "runtime": runtime,
                "artifacts": [
                    {"path": path.name, "sha256": sha256_file(path)}
                    for path in artifacts.values()
                ],
            },
        ),
    )
    verified = verify_run_manifest(
        manifest_path,
        prediction_path=artifacts["predictions.parquet"],
        generator_prediction_path=artifacts["gen_predictions.parquet"],
        target_embedding_path=artifacts["keyed.parquet"],
        checkpoint_path=source_files["checkpoint.pt"],
        normalizer_path=source_files["normalizer.pt"],
        config_path=config_path,
        split_path=source_files["split.pt"],
        raw_dataset_path=tmp_path / "raw",
    )
    assert verified.run_id == "run-1"
    assert verified.embedding_dimension == 2
    assert verified.runtime["torch"] == "2.11.0"
    assert verified.runtime["source_commit"] == metadata.graphkit_commit

    manifest_value = json.loads(manifest_path.read_text())
    manifest_value["embedding_dimension"] = 3
    manifest_path.write_text(json.dumps(manifest_value))
    with pytest.raises(ValueError, match="Embedding dimension disagrees"):
        verify_run_manifest(
            manifest_path,
            prediction_path=artifacts["predictions.parquet"],
            generator_prediction_path=artifacts["gen_predictions.parquet"],
            target_embedding_path=artifacts["keyed.parquet"],
            checkpoint_path=source_files["checkpoint.pt"],
            normalizer_path=source_files["normalizer.pt"],
            config_path=config_path,
            split_path=source_files["split.pt"],
            raw_dataset_path=tmp_path / "raw",
        )
    manifest_value["embedding_dimension"] = 2
    manifest_path.write_text(json.dumps(manifest_value))

    original_effective_config = artifacts["effective_config.yaml"].read_text()
    changed_effective_config = yaml.safe_load(original_effective_config)
    changed_effective_config["seed"] = 14
    artifacts["effective_config.yaml"].write_text(
        yaml.safe_dump(changed_effective_config),
    )
    changed_hash = sha256_file(artifacts["effective_config.yaml"])
    manifest_value["effective_config_sha256"] = changed_hash
    for record in manifest_value["artifacts"]:
        if record["path"] == "effective_config.yaml":
            record["sha256"] = changed_hash
            break
    manifest_path.write_text(json.dumps(manifest_value))

    with pytest.raises(ValueError, match="seed"):
        verify_run_manifest(
            manifest_path,
            prediction_path=artifacts["predictions.parquet"],
            generator_prediction_path=artifacts["gen_predictions.parquet"],
            target_embedding_path=artifacts["keyed.parquet"],
            checkpoint_path=source_files["checkpoint.pt"],
            normalizer_path=source_files["normalizer.pt"],
            config_path=config_path,
            split_path=source_files["split.pt"],
            raw_dataset_path=tmp_path / "raw",
        )

    artifacts["effective_config.yaml"].write_text(original_effective_config)
    restored_hash = sha256_file(artifacts["effective_config.yaml"])
    manifest_value["effective_config_sha256"] = restored_hash
    for record in manifest_value["artifacts"]:
        if record["path"] == "effective_config.yaml":
            record["sha256"] = restored_hash
            break
    manifest_path.write_text(json.dumps(manifest_value))

    unexpected = tmp_path / "unexpected.txt"
    unexpected.write_text("unexpected")
    with pytest.raises(ValueError, match="unmanifested files"):
        verify_run_manifest(
            manifest_path,
            prediction_path=artifacts["predictions.parquet"],
            generator_prediction_path=artifacts["gen_predictions.parquet"],
            target_embedding_path=artifacts["keyed.parquet"],
            checkpoint_path=source_files["checkpoint.pt"],
            normalizer_path=source_files["normalizer.pt"],
            config_path=config_path,
            split_path=source_files["split.pt"],
            raw_dataset_path=tmp_path / "raw",
        )
    unexpected.unlink()
    original_config = config_path.read_text()
    config_path.write_text(original_config + "\nchanged: true\n")
    with pytest.raises(ValueError, match="config_sha256"):
        verify_run_manifest(
            manifest_path,
            prediction_path=artifacts["predictions.parquet"],
            generator_prediction_path=artifacts["gen_predictions.parquet"],
            target_embedding_path=artifacts["keyed.parquet"],
            checkpoint_path=source_files["checkpoint.pt"],
            normalizer_path=source_files["normalizer.pt"],
            config_path=config_path,
            split_path=source_files["split.pt"],
            raw_dataset_path=tmp_path / "raw",
        )
    config_path.write_text(original_config)
    artifacts["predictions.parquet"].write_text("changed")
    with pytest.raises(ValueError, match="hash disagrees"):
        verify_run_manifest(
            manifest_path,
            prediction_path=artifacts["predictions.parquet"],
            generator_prediction_path=artifacts["gen_predictions.parquet"],
            target_embedding_path=artifacts["keyed.parquet"],
            checkpoint_path=source_files["checkpoint.pt"],
            normalizer_path=source_files["normalizer.pt"],
            config_path=config_path,
            split_path=source_files["split.pt"],
            raw_dataset_path=tmp_path / "raw",
        )


def test_mlflow_run_discovery_skips_unrelated_incomplete_runs(tmp_path) -> None:
    incomplete = tmp_path / "0" / "failed"
    incomplete.mkdir(parents=True)
    (incomplete / "meta.yaml").write_text(
        yaml.safe_dump({"run_name": "unrelated", "status": 4}),
    )

    completed = tmp_path / "0" / "completed"
    (completed / "tags").mkdir(parents=True)
    for relative, value in {
        "params/args/seed": "42",
        "params/args/get_embeddings": "True",
        "params/args/data/networks": "['case14_ieee']",
        "params/args/data/scenarios": "[300000]",
        "params/args/task/task_name": "OptimalPowerFlow",
        "params/args/model/hidden_size": "48",
    }.items():
        path = completed / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    (completed / "meta.yaml").write_text(
        yaml.safe_dump(
            {
                "run_id": "run-1",
                "run_name": "target",
                "status": 3,
                "start_time": 1,
                "end_time": 2,
            },
        ),
    )
    (completed / "tags" / "mlflow.source.git.commit").write_text("a" * 40)

    assert _find_mlflow_run(tmp_path, "target") == completed


def test_reproduction_manifest_binds_runtime_and_exact_split(tmp_path) -> None:
    run = tmp_path / "run"
    (run / "tags").mkdir(parents=True)
    for relative, value in {
        "params/args/seed": "42",
        "params/args/get_embeddings": "False",
        "params/args/data/networks": "['case14_ieee']",
        "params/args/data/scenarios": "[300000]",
        "params/args/task/task_name": "OptimalPowerFlow",
        "params/args/model/hidden_size": "48",
    }.items():
        target = run / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(value)
    (run / "meta.yaml").write_text(
        yaml.safe_dump(
            {
                "run_id": "clean-run",
                "run_name": "clean-runtime",
                "status": 3,
                "start_time": 10,
                "end_time": 20,
            },
        ),
    )
    (run / "tags/mlflow.source.git.commit").write_text("a" * 40)
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "data.parquet").write_text("data")
    checkpoint = tmp_path / "checkpoint.pt"
    normalizer = tmp_path / "normalizer.pt"
    config = tmp_path / "config.yaml"
    split = tmp_path / "test.pt"
    checkpoint.write_text("checkpoint")
    normalizer.write_text("normalizer")
    config.write_text("seed: 42\n")
    torch.save([torch.tensor(1)], split)
    bus = tmp_path / "case14_ieee_predictions.parquet"
    generator = tmp_path / "case14_ieee_gen_predictions.parquet"
    pd.DataFrame({"scenario": [1]}).to_parquet(bus)
    pd.DataFrame({"scenario": [1]}).to_parquet(generator)
    environment = tmp_path / "runtime_environment.txt"
    environment.write_text("torch==2.11.0\n")
    runtime = {
        "python_version": "3.12.10",
        "platform": "macOS-14.8.9-arm-64bit",
        "torch": "2.11.0",
        "torchvision": "0.26.0",
        "torchaudio": "2.11.0",
        "torch_scatter": "2.1.2",
        "environment_path": environment.name,
        "environment_sha256": sha256_file(environment),
    }
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format_version": 1,
                "run_id": "clean-run",
                "run_status": 3,
                "graphkit_commit": "a" * 40,
                "dataset_revision": "b" * 40,
                "dataset_split": "test",
                "export_embeddings": False,
                "mlflow_run_path": run.name,
                "mlflow_run_sha256": sha256_path(run),
                "checkpoint_sha256": sha256_file(checkpoint),
                "normalizer_sha256": sha256_file(normalizer),
                "config_sha256": sha256_file(config),
                "split_sha256": sha256_file(split),
                "raw_dataset_sha256": sha256_path(raw),
                "runtime": runtime,
                "artifacts": [
                    {"path": path.name, "sha256": sha256_file(path)}
                    for path in (bus, generator, environment)
                ],
            },
        ),
    )
    verified = verify_reproduction_manifest(
        manifest,
        prediction_path=bus,
        generator_prediction_path=generator,
        checkpoint_path=checkpoint,
        normalizer_path=normalizer,
        config_path=config,
        split_path=split,
        raw_dataset_path=raw,
    )
    assert verified.runtime_environment_sha256 == sha256_file(environment)
    environment.write_text("torch==2.10.0\n")
    with pytest.raises(ValueError, match="environment hash"):
        verify_reproduction_manifest(
            manifest,
            prediction_path=bus,
            generator_prediction_path=generator,
            checkpoint_path=checkpoint,
            normalizer_path=normalizer,
            config_path=config,
            split_path=split,
            raw_dataset_path=raw,
        )


def test_baseline_residual_comparison_enforces_declared_tolerance(tmp_path) -> None:
    predictions = tmp_path / "predictions.parquet"
    pd.DataFrame(
        {
            "active res. (MW)": [-0.1, 0.3],
            "reactive res. (MVar)": [0.2, -0.4],
        },
    ).to_parquet(predictions)
    metrics = tmp_path / "metrics.csv"
    pd.DataFrame(
        {
            "Metric": [
                "Avg. active res. (MW)",
                "Avg. reactive res. (MVar)",
            ],
            "Value": [0.2, 0.3],
        },
    ).to_csv(metrics, index=False)

    result = compare_baseline_residuals(predictions, metrics, 1e-12)
    assert result.active_residual_actual == pytest.approx(0.2)
    pd.DataFrame(
        {
            "Metric": [
                "Avg. active res. (MW)",
                "Avg. reactive res. (MVar)",
            ],
            "Value": [0.21, 0.3],
        },
    ).to_csv(metrics, index=False)
    with pytest.raises(ValueError, match="exceeded absolute tolerance"):
        compare_baseline_residuals(predictions, metrics, 1e-3)


def test_authoritative_split_membership_rejects_substitution(tmp_path) -> None:
    split = tmp_path / "train.pt"
    torch.save([torch.tensor(1), torch.tensor(2)], split)
    assert verify_split_membership(["2", "1"], split, "reference") == ("1", "2")
    with pytest.raises(ValueError, match="missing=.*2.*extra=.*3"):
        verify_split_membership(["1", "3"], split, "reference")


def test_list_scenarios_cli_prints_validated_split_ids(tmp_path, capsys) -> None:
    split = tmp_path / "test.pt"
    torch.save([torch.tensor(215429), torch.tensor(4873)], split)

    assert main(["list-scenarios", "--split-artifact", str(split)]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "scenario_id",
        "215429",
        "4873",
    ]

    assert main(
        [
            "list-scenarios",
            "--split-artifact",
            str(split),
            "--no-header",
        ],
    ) == 0
    assert capsys.readouterr().out.splitlines() == ["215429", "4873"]


def test_reference_split_preparation_is_hash_preserving(tmp_path) -> None:
    sources = {}
    for name, values in {
        "train": [1, 2],
        "val": [3],
        "test": [4],
    }.items():
        sources[name] = tmp_path / f"{name}.pt"
        torch.save([torch.tensor(value) for value in values], sources[name])
    output = prepare_reference_prediction_splits(
        train_split=sources["train"],
        validation_split=sources["val"],
        test_split=sources["test"],
        output_dir=tmp_path / "view",
    )
    assert sha256_file(output / "test.pt") == sha256_file(sources["train"])
    assert load_split_ids(output / "test.pt") == ("1", "2")


def _write_comparison_manifest(
    path,
    artifacts,
    **identity_overrides,
) -> None:
    effective_config = path.parent / "effective_config.yaml"
    effective_config.write_text("seed: 42\n")
    identity = {
        "checkpoint_sha256": "a" * 64,
        "normalizer_sha256": "b" * 64,
        "config_sha256": "c" * 64,
        "effective_config_sha256": sha256_file(effective_config),
        "split_sha256": "d" * 64,
        "raw_dataset_sha256": "e" * 64,
        "dataset_revision": "f" * 40,
        "graphkit_commit": "1" * 40,
        "dataset_split": "test",
        "grid_name": "case14_ieee",
        "task": "OptimalPowerFlow",
        "seed": 42,
        "corpus_scenario_count": 300000,
        "run_status": 3,
        "export_embeddings": path.stem.startswith("expected"),
        "embedding_dimension": (
            384 if path.stem.startswith("expected") else None
        ),
    }
    identity.update(identity_overrides)

    environment = path.parent / f"{path.stem}-environment.txt"
    requirements_lock = path.parent / f"{path.stem}-requirements.lock"
    source_record = path.parent / f"{path.stem}-source.txt"
    environment.write_text("torch==2.11.0\n")
    requirements_lock.write_text("torch==2.11.0\n")
    source_record.write_text(
        "repository=https://github.com/gridfm/gridfm-graphkit.git\n"
        f"commit={identity['graphkit_commit']}\n",
    )

    runtime = {
        "python_version": "3.12.10",
        "platform": "macOS-14.8.9-arm-64bit",
        "graphkit": "0.9.0.3",
        "torch": "2.11.0",
        "torchvision": "0.26.0",
        "torchaudio": "2.11.0",
        "torch_scatter": "2.1.2",
        "python_executable": ".venv/bin/python",
        "environment_path": environment.name,
        "environment_sha256": sha256_file(environment),
        "requirements_lock_path": requirements_lock.name,
        "requirements_lock_sha256": sha256_file(requirements_lock),
        "source_record_path": source_record.name,
        "source_record_sha256": sha256_file(source_record),
        "source_repository": "https://github.com/gridfm/gridfm-graphkit.git",
        "source_commit": identity["graphkit_commit"],
    }
    all_artifacts = (
        *artifacts,
        effective_config,
        environment,
        requirements_lock,
        source_record,
    )
    path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "run_id": path.stem,
                **identity,
                "runtime": runtime,
                "artifacts": [
                    {
                        "path": artifact.relative_to(path.parent).as_posix(),
                        "sha256": sha256_file(artifact),
                    }
                    for artifact in all_artifacts
                ],
            },
        ),
    )


def test_keyed_baseline_comparison_checks_every_prediction_row(tmp_path) -> None:
    bus_columns = {
        "scenario": [1, 1],
        "bus": [0, 1],
        "Vm_pred": [1.0, 1.1],
        "Va_pred": [0.0, 0.1],
        "Pg_pred": [2.0, 3.0],
        "Qg_pred": [0.2, 0.3],
        "active res. (MW)": [0.01, 0.02],
        "reactive res. (MVar)": [0.03, 0.04],
        "PBE": [0.04, 0.06],
        "Vm_target": [1.0, 1.0],
    }
    generator_columns = {
        "scenario": [1],
        "idx": [0],
        "p_mw_pred": [2.0],
        "p_mw_target": [2.1],
    }
    paths = {}
    for name, table in {
        "actual_bus": bus_columns,
        "expected_bus": bus_columns,
        "actual_gen": generator_columns,
        "expected_gen": generator_columns,
    }.items():
        paths[name] = tmp_path / f"{name}.parquet"
        pd.DataFrame(table).to_parquet(paths[name])

    actual_manifest = tmp_path / "actual_manifest.json"
    expected_manifest = tmp_path / "expected_manifest.json"
    _write_comparison_manifest(
        actual_manifest,
        (paths["actual_bus"], paths["actual_gen"]),
    )
    _write_comparison_manifest(
        expected_manifest,
        (paths["expected_bus"], paths["expected_gen"]),
    )

    result = compare_keyed_baseline(
        prediction_path=paths["actual_bus"],
        generator_prediction_path=paths["actual_gen"],
        expected_prediction_path=paths["expected_bus"],
        expected_generator_prediction_path=paths["expected_gen"],
        actual_run_manifest=actual_manifest,
        expected_run_manifest=expected_manifest,
    )
    assert result.bus_row_count == 2

    substituted = tmp_path / "substituted_bus.parquet"
    pd.DataFrame(bus_columns).to_parquet(substituted)
    with pytest.raises(ValueError, match="absent from run manifest"):
        compare_keyed_baseline(
            prediction_path=substituted,
            generator_prediction_path=paths["actual_gen"],
            expected_prediction_path=paths["expected_bus"],
            expected_generator_prediction_path=paths["expected_gen"],
            actual_run_manifest=actual_manifest,
            expected_run_manifest=expected_manifest,
        )

    changed = pd.DataFrame(bus_columns)
    changed.loc[1, "PBE"] += 0.01
    changed.to_parquet(paths["actual_bus"])
    _write_comparison_manifest(
        actual_manifest,
        (paths["actual_bus"], paths["actual_gen"]),
    )
    with pytest.raises(ValueError, match="PBE.*exceeded"):
        compare_keyed_baseline(
            prediction_path=paths["actual_bus"],
            generator_prediction_path=paths["actual_gen"],
            expected_prediction_path=paths["expected_bus"],
            expected_generator_prediction_path=paths["expected_gen"],
            actual_run_manifest=actual_manifest,
            expected_run_manifest=expected_manifest,
        )


def test_keyed_baseline_rejects_incompatible_run_identity(tmp_path) -> None:
    actual_bus = tmp_path / "actual_bus.parquet"
    expected_bus = tmp_path / "expected_bus.parquet"
    actual_generator = tmp_path / "actual_generator.parquet"
    expected_generator = tmp_path / "expected_generator.parquet"

    pd.DataFrame(
        {"scenario": [1], "bus": [0], "Vm_pred": [1.0]},
    ).to_parquet(actual_bus)
    pd.DataFrame(
        {"scenario": [1], "bus": [0], "Vm_pred": [1.0]},
    ).to_parquet(expected_bus)
    pd.DataFrame(
        {"scenario": [1], "idx": [0], "p_mw_pred": [1.0]},
    ).to_parquet(actual_generator)
    pd.DataFrame(
        {"scenario": [1], "idx": [0], "p_mw_pred": [1.0]},
    ).to_parquet(expected_generator)

    actual_manifest = tmp_path / "actual_manifest.json"
    expected_manifest = tmp_path / "expected_manifest.json"
    _write_comparison_manifest(
        actual_manifest,
        (actual_bus, actual_generator),
    )
    _write_comparison_manifest(
        expected_manifest,
        (expected_bus, expected_generator),
        checkpoint_sha256="9" * 64,
    )

    with pytest.raises(
        ValueError,
        match="checkpoint_sha256.*disagrees",
    ):
        compare_keyed_baseline(
            prediction_path=actual_bus,
            generator_prediction_path=actual_generator,
            expected_prediction_path=expected_bus,
            expected_generator_prediction_path=expected_generator,
            actual_run_manifest=actual_manifest,
            expected_run_manifest=expected_manifest,
        )


def test_physical_evidence_counts_declared_limit_violations(tmp_path) -> None:
    bus_path = tmp_path / "bus.parquet"
    gen_path = tmp_path / "gen.parquet"
    pd.DataFrame(
        {
            "scenario": [7, 7],
            "bus": [0, 1],
            "Vm_pred": [1.0, 1.1],
            "Vm_min": [0.95, 0.95],
            "Vm_max": [1.05, 1.05],
            "Qg_pred": [0.0, 2.0],
            "Qg_min": [-1.0, -1.0],
            "Qg_max": [1.0, 1.0],
            "active res. (MW)": [0.1, 0.2],
            "reactive res. (MVar)": [0.2, 0.3],
            "PBE": [0.3, 0.5],
        },
    ).to_parquet(bus_path)
    pd.DataFrame(
        {
            "scenario": [7],
            "idx": [0],
            "p_mw_pred": [2.0],
            "min_p_mw": [0.0],
            "max_p_mw": [3.0],
        },
    ).to_parquet(gen_path)
    evidence = _physical_evidence(bus_path, "7", gen_path)
    assert evidence.violation_count == 2
    assert evidence.extra["reactive_violation_count"] == 1
    assert evidence.extra["max_pbe_bus"] == 1


def _write_keyed_embeddings(path, *, split="train", stage="final", step=None) -> None:
    pd.DataFrame(
        {
            "scenario_id": [1, 1, 1],
            "entity_type": ["bus", "bus", "generator"],
            "entity_id": [0, 1, 0],
            "correction_step": [step, step, step],
            "embedding_stage": [stage, stage, stage],
            "embedding": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            "dataset_split": [split, split, split],
        },
    ).to_parquet(path)


def test_keyed_embedding_loader_enforces_final_train_semantics(tmp_path) -> None:
    path = tmp_path / "train.parquet"
    _write_keyed_embeddings(path)
    scenario_ids, embeddings = _keyed_embedding_table(path, "train")
    assert scenario_ids == ["1"]
    np.testing.assert_allclose(embeddings, [[2.0, 3.0, 5.0, 6.0]])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"split": "test"}, "dataset_split"),
        ({"stage": "intermediate"}, "embedding_stage"),
        ({"step": 0}, "null correction_step"),
    ],
)
def test_keyed_embedding_loader_rejects_wrong_semantics(tmp_path, change, message) -> None:
    path = tmp_path / "invalid.parquet"
    _write_keyed_embeddings(path, **change)
    with pytest.raises(ValueError, match=message):
        _keyed_embedding_table(path, "train")


def test_keyed_embedding_loader_rejects_duplicate_entity_keys(tmp_path) -> None:
    path = tmp_path / "duplicate.parquet"
    _write_keyed_embeddings(path)
    table = pd.read_parquet(path)
    pd.concat([table.iloc[:1], table], ignore_index=True).to_parquet(path)
    with pytest.raises(ValueError, match="duplicate entity key"):
        _keyed_embedding_table(path, "train")


def test_graphkit_embeddings_are_canonicalized_with_final_semantics(tmp_path) -> None:
    pd.DataFrame(
        {
            "scenario": [7, 7],
            "bus": [0, 1],
            "emb_000": [1.0, 3.0],
            "emb_001": [2.0, 4.0],
        },
    ).to_parquet(tmp_path / "case14_ieee_bus_embeddings.parquet")
    pd.DataFrame(
        {
            "scenario": [7],
            "idx": [0],
            "bus": [0],
            "emb_000": [5.0],
            "emb_001": [6.0],
        },
    ).to_parquet(tmp_path / "case14_ieee_gen_embeddings.parquet")
    path = canonicalize_graphkit_embeddings(tmp_path, "test")
    table = pd.read_parquet(path)
    assert set(table["embedding_stage"]) == {"final"}
    assert table["correction_step"].isna().all()
    assert set(table["dataset_split"]) == {"test"}
    scenario_ids, embeddings = _keyed_embedding_table(path, "test")
    assert scenario_ids == ["7"]
    np.testing.assert_allclose(embeddings, [[2.0, 3.0, 5.0, 6.0]])


def test_native_embeddings_are_compacted_without_removing_canonical_table(
    tmp_path,
) -> None:
    canonical = tmp_path / "case14_ieee_keyed_embeddings.parquet"
    canonical.write_text("canonical")
    native_paths = (
        tmp_path / "case14_ieee_bus_embeddings.parquet",
        tmp_path / "case14_ieee_gen_embeddings.parquet",
    )
    for native in native_paths:
        native.write_text("native")

    _compact_native_embedding_artifacts(tmp_path)

    assert canonical.read_text() == "canonical"
    assert all(not native.exists() for native in native_paths)


def test_rdf_graph_contains_lineage_and_physical_evidence() -> None:
    graph = build_evidence_graph(
        DatasetProvenance("dataset", "case14_ieee", "OptimalPowerFlow"),
        ModelProvenance("run", "checkpoint", "dataset"),
        PredictionEvidence(
            "scenario-0",
            {
                "artifact_records": (
                    {"path": "predictions.parquet", "sha256": "abc123"},
                ),
            },
            ({"scenario_id": "scenario-1", "distance": 1.5, "rank": 1},),
            1.5,
            None,
            PhysicalEvidence(active_residual=0.1),
        ),
    )
    assert (None, GENCO.hasPhysicalEvidence, None) in graph
    assert (None, GENCO.sha256, None) in graph
    assert len(graph) > 10


def _valid_shacl_graph(
    *,
    authoritative_training_attestation: VerifiedTrainingAttestation | None = None,
    checkpoint_hash: str = "a" * 64,
) -> Graph:
    semantics = tuple(
        MetricSemantics(name, unit, treatment, aggregation, "bus rows")
        for name, unit, treatment, aggregation in (
            ("mean_pbe", "MVA", "magnitude", "mean"),
            ("max_pbe", "MVA", "magnitude", "maximum"),
            ("mean_active_residual", "MW", "absolute", "mean"),
            ("mean_reactive_residual", "MVar", "absolute", "mean"),
            ("total_violation_count", "count", "count", "sum"),
            ("voltage_violation_count", "count", "count", "sum"),
            ("reactive_power_violation_count", "count", "count", "sum"),
            ("active_power_violation_count", "count", "count", "sum"),
        )
    )
    return build_evidence_graph(
        DatasetProvenance(
            "dataset",
            "case14_ieee",
            "OptimalPowerFlow",
            corpus_scenario_count=300_000,
            dataset_split="test",
            split_scenario_count=15_000,
            dataset_revision="b" * 40,
            config_hash="f" * 64,
            software_version="gridfm-datakit:test",
            generation_method="deterministic conversion",
            generation_seed_status="not applicable",
            perturbations=("published scenarios",),
            provenance_limitations=("upstream seed unavailable",),
        ),
        ModelProvenance(
            "run",
            "checkpoint",
            "dataset",
            checkpoint_hash=checkpoint_hash,
            mlflow_run_id="training-run",
            training_dataset_id="dataset:train",
            training_split_scenario_count=270_000,
            training_start_time_ms=1,
            training_end_time_ms=2,
            training_seed=42,
            training_epochs=200,
            training_batch_size=64,
            training_learning_rate=0.0005,
            training_optimizer="AdamW",
            training_selected_metrics={
                "training_loss": 6.0e-6,
                "validation_loss": 5.9e-6,
                "validation_bus_mse_loss": 4.1e-7,
                "validation_generator_mse_loss": 1.6e-6,
                "validation_physics_loss": 4.8e-5,
                "validation_qg_violation_penalty_loss": 6.1e-4,
            },
            training_software_revision_status="unavailable upstream",
            checkpoint_binding_status=(
                "synthetic authoritative checkpoint-to-run attestation"
                if authoritative_training_attestation is not None
                else "co-published at pinned repository revision"
            ),
            training_params_hash="c" * 64,
            training_metrics_hash="d" * 64,
            checkpoint_repository_path=(
                "case14_ieee/seed42/best_model_state_dict.pt"
            ),
            training_evidence_association=(
                None
                if authoritative_training_attestation is not None
                else _unverified_training_association()
            ),
            authoritative_training_attestation=authoritative_training_attestation,
        ),
        PredictionEvidence(
            "scenario-0",
            {},
            ({"scenario_id": "scenario-1", "distance": 1.5, "rank": 1},),
            1.5,
            None,
            PhysicalEvidence(
                mean_residual=0.1,
                max_residual=0.2,
                active_residual=0.1,
                reactive_residual=0.1,
                violation_count=0,
                metric_semantics=semantics,
                extra={"max_pbe_bus": 0},
            ),
            inference_run_id="inference-run",
            inference_start_time_ms=3,
            inference_end_time_ms=4,
            inference_graphkit_commit="e" * 40,
            reference_extraction_run_id="reference-run",
            reference_extraction_start_time_ms=5,
            reference_extraction_end_time_ms=6,
            reference_extraction_graphkit_commit="e" * 40,
            reproduction_run_id="reproduction-run",
            reproduction_start_time_ms=7,
            reproduction_end_time_ms=8,
            reproduction_graphkit_commit="e" * 40,
            reproduction_environment_hash="f" * 64,
            reproduction_runtime={
                "python_version": "3.12.10",
                "platform": "macOS-14.8.9-arm-64bit",
                "torch": "2.11.0",
                "torchvision": "0.26.0",
                "torchaudio": "2.11.0",
                "torch_scatter": "2.1.2",
            },
            reproduction_comparison_status="passed",
            reproduction_tolerance_policy_id=(
                APPROVED_TOLERANCE_POLICY["policy_id"]
            ),
            reproduction_tolerance_policy_hash=(
                APPROVED_TOLERANCE_POLICY_SHA256
            ),
            reproduction_maximum_voltage_difference=1.5e-6,
            reproduction_maximum_power_difference=2.34e-3,
        ),
    )


def test_shacl_rejects_missing_training_dataset_lineage() -> None:
    graph = _valid_shacl_graph()
    training = next(graph.subjects(RDF.type, GENCO.TrainingRun))
    dataset = next(graph.objects(training, PROV.used))
    graph.remove((training, PROV.used, dataset))
    with pytest.raises(ValueError, match="qualifiedMinCount"):
        validate_evidence_graph(graph, "ontology/shapes.ttl")


def test_published_association_is_non_causal_and_queryable() -> None:
    graph = _valid_shacl_graph()
    checkpoint = next(graph.subjects(RDF.type, GENCO.ModelCheckpoint))
    training = next(graph.subjects(RDF.type, GENCO.TrainingRun))
    assert (checkpoint, PROV.wasGeneratedBy, training) not in graph
    assert (checkpoint, PROV.wasDerivedFrom, training) not in graph
    rows = list(graph.query(ASSOCIATED_TRAINING_EVIDENCE))
    assert rows
    assert str(rows[0].productionIdentityVerified) == "false"
    assert str(rows[0].split) == "train"


@pytest.mark.parametrize(
    ("predicate", "value", "message"),
    (
        (GENCO.productionIdentityVerified, True, "hasValue"),
        (GENCO.configurationCompatible, False, "hasValue"),
    ),
)
def test_shacl_rejects_invalid_association_flags(predicate, value, message) -> None:
    graph = _valid_shacl_graph()
    association = next(
        graph.subjects(RDF.type, GENCO.TrainingEvidenceAssociation),
    )
    graph.remove((association, predicate, None))
    graph.add((association, predicate, Literal(value)))
    with pytest.raises(ValueError, match=message):
        validate_evidence_graph(graph, "ontology/shapes.ttl")


@pytest.mark.parametrize(
    "predicate",
    (GENCO.evidenceBasis, GENCO.provenanceLimitation),
)
def test_shacl_rejects_missing_association_explanation(predicate) -> None:
    graph = _valid_shacl_graph()
    association = next(
        graph.subjects(RDF.type, GENCO.TrainingEvidenceAssociation),
    )
    graph.remove((association, predicate, None))
    with pytest.raises(ValueError, match=str(predicate).split("#")[-1]):
        validate_evidence_graph(graph, "ontology/shapes.ttl")


def test_shacl_rejects_association_to_inference_run() -> None:
    graph = _valid_shacl_graph()
    association = next(
        graph.subjects(RDF.type, GENCO.TrainingEvidenceAssociation),
    )
    training = next(graph.objects(association, GENCO.associatedTrainingRun))
    inference = next(graph.subjects(RDF.type, GENCO.InferenceRun))
    graph.remove((association, GENCO.associatedTrainingRun, training))
    graph.add((association, GENCO.associatedTrainingRun, inference))
    with pytest.raises(ValueError, match="associatedTrainingRun|inference"):
        validate_evidence_graph(graph, "ontology/shapes.ttl")


def test_shacl_rejects_association_to_run_using_test_split() -> None:
    graph = _valid_shacl_graph()
    association = next(
        graph.subjects(RDF.type, GENCO.TrainingEvidenceAssociation),
    )
    training = next(graph.objects(association, GENCO.associatedTrainingRun))
    train_dataset = next(graph.objects(training, PROV.used))
    test_dataset = next(
        dataset
        for dataset in graph.subjects(GENCO.datasetSplit, Literal("test"))
    )
    graph.remove((training, PROV.used, train_dataset))
    graph.add((training, PROV.used, test_dataset))
    with pytest.raises(ValueError, match="test|qualifiedMinCount"):
        validate_evidence_graph(graph, "ontology/shapes.ttl")


def test_shacl_rejects_checkpoint_training_claim_without_attestation() -> None:
    graph = _valid_shacl_graph()
    checkpoint = next(graph.subjects(RDF.type, GENCO.ModelCheckpoint))
    training = next(graph.subjects(RDF.type, GENCO.TrainingRun))
    graph.add((checkpoint, PROV.wasGeneratedBy, training))
    with pytest.raises(ValueError, match="attestation|generation lineage"):
        validate_evidence_graph(graph, "ontology/shapes.ttl")


def test_shacl_rejects_checkpoint_training_derivation_for_association() -> None:
    graph = _valid_shacl_graph()
    checkpoint = next(graph.subjects(RDF.type, GENCO.ModelCheckpoint))
    training = next(graph.subjects(RDF.type, GENCO.TrainingRun))
    graph.add((checkpoint, PROV.wasDerivedFrom, training))
    with pytest.raises(ValueError, match="direct checkpoint-to-training PROV"):
        validate_evidence_graph(graph, "ontology/shapes.ttl")


@pytest.mark.parametrize(
    ("qualified_predicate", "role"),
    (
        (PROV.qualifiedGeneration, PROV.activity),
        (PROV.qualifiedDerivation, PROV.entity),
        (PROV.qualifiedInfluence, PROV.influencer),
    ),
)
def test_shacl_rejects_qualified_checkpoint_training_claim_for_association(
    qualified_predicate,
    role,
) -> None:
    graph = _valid_shacl_graph()
    checkpoint = next(graph.subjects(RDF.type, GENCO.ModelCheckpoint))
    training = next(graph.subjects(RDF.type, GENCO.TrainingRun))
    qualification = BNode()
    graph.add((checkpoint, qualified_predicate, qualification))
    graph.add((qualification, role, training))
    with pytest.raises(ValueError, match="qualified checkpoint-to-training PROV"):
        validate_evidence_graph(graph, "ontology/shapes.ttl")


@pytest.mark.parametrize("inverse_predicate", (PROV.generated, PROV.influenced))
def test_shacl_rejects_inverse_training_checkpoint_claim_for_association(
    inverse_predicate,
) -> None:
    graph = _valid_shacl_graph()
    checkpoint = next(graph.subjects(RDF.type, GENCO.ModelCheckpoint))
    training = next(graph.subjects(RDF.type, GENCO.TrainingRun))
    graph.add((training, inverse_predicate, checkpoint))
    with pytest.raises(ValueError, match="inverse training-to-checkpoint PROV"):
        validate_evidence_graph(graph, "ontology/shapes.ttl")


@pytest.mark.parametrize("inverse_second_hop", (False, True))
def test_shacl_rejects_multihop_checkpoint_training_claim_for_association(
    inverse_second_hop,
) -> None:
    graph = _valid_shacl_graph()
    checkpoint = next(graph.subjects(RDF.type, GENCO.ModelCheckpoint))
    training = next(graph.subjects(RDF.type, GENCO.TrainingRun))
    intermediate = BNode()
    graph.add((checkpoint, PROV.wasDerivedFrom, intermediate))
    if inverse_second_hop:
        graph.add((training, PROV.generated, intermediate))
    else:
        graph.add((intermediate, PROV.wasGeneratedBy, training))
    with pytest.raises(ValueError, match="multi-hop causal PROV path"):
        validate_evidence_graph(graph, "ontology/shapes.ttl")


def test_shacl_rejects_multihop_qualified_checkpoint_training_claim() -> None:
    graph = _valid_shacl_graph()
    checkpoint = next(graph.subjects(RDF.type, GENCO.ModelCheckpoint))
    training = next(graph.subjects(RDF.type, GENCO.TrainingRun))
    intermediate = BNode()
    qualification = BNode()
    graph.add((checkpoint, PROV.wasDerivedFrom, intermediate))
    graph.add((intermediate, PROV.qualifiedGeneration, qualification))
    graph.add((qualification, PROV.activity, training))
    with pytest.raises(ValueError, match="multi-hop causal PROV path"):
        validate_evidence_graph(graph, "ontology/shapes.ttl")


@pytest.fixture
def synthetic_authoritative_attestation_graph() -> tuple[Graph, dict[str, bytes]]:
    checkpoint_hash = (
        "e31cdaafdffccae5c557e7194153fc52bf7bf0fe5f143765c681350bba4e37dc"
    )
    attestation, trusted_keys = _synthetic_training_attestation(checkpoint_hash)
    return (
        _valid_shacl_graph(
            authoritative_training_attestation=attestation,
            checkpoint_hash=checkpoint_hash,
        ),
        trusted_keys,
    )


def test_synthetic_attestation_constructs_authoritative_training_lineage(
    synthetic_authoritative_attestation_graph,
) -> None:
    graph, trusted_keys = synthetic_authoritative_attestation_graph
    validate_synthetic_evidence_graph(
        graph,
        "ontology/shapes.ttl",
        synthetic_trusted_attestation_keys=trusted_keys,
    )
    checkpoint = next(graph.subjects(RDF.type, GENCO.ModelCheckpoint))
    training = next(graph.subjects(RDF.type, GENCO.TrainingRun))
    assert (checkpoint, PROV.wasGeneratedBy, training) in graph
    assert (
        checkpoint,
        GENCO.hasAssociatedTrainingEvidence,
        None,
    ) not in graph


def test_production_policy_rejects_self_signed_seed42_attestation(
    synthetic_authoritative_attestation_graph,
) -> None:
    graph, _ = synthetic_authoritative_attestation_graph
    with pytest.raises(
        ValueError,
        match="prohibits authoritative lineage for checkpoint",
    ):
        validate_evidence_graph(graph, "ontology/shapes.ttl")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("checkpoint_id", "wrong-checkpoint", "checkpoint ID"),
        ("training_run_id", "wrong-run", "training-run ID"),
    ],
)
def test_production_attestation_policy_pins_resource_ids(
    monkeypatch,
    field,
    value,
    message,
) -> None:
    checkpoint_hash = "a" * 64
    document = {
        "format_version": 1,
        "attestation_id": "attestation",
        "issuer": "approved-issuer",
        "source_uri": "https://example.invalid/attestation.json",
        "checkpoint_id": "approved-checkpoint",
        "checkpoint_sha256": checkpoint_hash,
        "training_run_id": "approved-run",
        "content_sha256": "b" * 64,
        "verification_status": "verified",
    }
    document[field] = value
    monkeypatch.setitem(
        PRODUCTION_ATTESTATION_TRUST_POLICY["checkpoints"],
        checkpoint_hash,
        {
            "status": "approved",
            "checkpoint_id": "approved-checkpoint",
            "training_run_id": "approved-run",
            "issuers": {},
        },
    )

    with pytest.raises(ValueError, match=message):
        verify_production_training_attestation(document, "unused")


def test_authoritative_attestation_rejects_tampered_signature(
    synthetic_authoritative_attestation_graph,
) -> None:
    graph, trusted_keys = synthetic_authoritative_attestation_graph
    attestation = next(
        graph.subjects(RDF.type, GENCO.AuthoritativeTrainingAttestation),
    )
    graph.set((attestation, GENCO.attestationSignature, Literal("Zm9yZ2Vk")))
    with pytest.raises(ValueError, match="signature verification failed"):
        validate_synthetic_evidence_graph(
            graph,
            "ontology/shapes.ttl",
            synthetic_trusted_attestation_keys=trusted_keys,
        )


def test_shacl_rejects_free_text_authoritative_attestation() -> None:
    graph = _valid_shacl_graph()
    checkpoint = next(graph.subjects(RDF.type, GENCO.ModelCheckpoint))
    training = next(graph.subjects(RDF.type, GENCO.TrainingRun))
    association = next(
        graph.objects(checkpoint, GENCO.hasAssociatedTrainingEvidence),
    )
    graph.remove((checkpoint, GENCO.hasAssociatedTrainingEvidence, association))
    graph.add(
        (
            checkpoint,
            GENCO.authoritativeTrainingAttestation,
            Literal("forged local assertion"),
        ),
    )
    graph.add((checkpoint, PROV.wasGeneratedBy, training))
    with pytest.raises(ValueError, match="AuthoritativeTrainingAttestation"):
        validate_evidence_graph(graph, "ontology/shapes.ttl")


def test_published_artifact_terminology_and_relations_are_non_causal() -> None:
    artifact = Graph().parse(
        "artifacts/case14_scenario_215429_evidence.ttl",
        format="turtle",
    )
    checkpoint = next(artifact.subjects(RDF.type, GENCO.ModelCheckpoint))
    training = next(artifact.subjects(RDF.type, GENCO.TrainingRun))
    assert (checkpoint, PROV.wasGeneratedBy, training) not in artifact
    assert (checkpoint, PROV.wasDerivedFrom, training) not in artifact

    documentation = "\n".join(
        Path(path).read_text().lower()
        for path in (
            "readme.md",
            "reproduction_graphkit.md",
        )
    )
    for unsupported_claim in (
        "authoritative checkpoint training run",
        "checkpoint training lineage from authoritative run",
        "run 1927fe80558a48539ee37be478706781 produced",
        "1927fe80558a48539ee37be478706781 trained",
    ):
        assert unsupported_claim not in documentation


def test_shacl_rejects_missing_inference_checkpoint_lineage() -> None:
    graph = _valid_shacl_graph()
    inference = next(graph.subjects(RDF.type, GENCO.InferenceRun))
    checkpoint = next(graph.subjects(RDF.type, GENCO.ModelCheckpoint))
    graph.remove((inference, PROV.used, checkpoint))
    with pytest.raises(ValueError, match="qualifiedMinCount"):
        validate_evidence_graph(graph, "ontology/shapes.ttl")


def test_shacl_rejects_missing_reproduction_environment_hash() -> None:
    graph = _valid_shacl_graph()
    reproduction = next(graph.subjects(RDF.type, GENCO.ReproductionRun))
    graph.remove((reproduction, GENCO.environmentHash, None))
    with pytest.raises(ValueError, match="environmentHash"):
        validate_evidence_graph(graph, "ontology/shapes.ttl")


def test_shacl_rejects_missing_reproduction_tolerance_policy() -> None:
    graph = _valid_shacl_graph()
    reproduction = next(graph.subjects(RDF.type, GENCO.ReproductionRun))
    graph.remove((reproduction, GENCO.tolerancePolicyHash, None))
    with pytest.raises(ValueError, match="tolerancePolicyHash"):
        validate_evidence_graph(graph, "ontology/shapes.ttl")


def test_shacl_rejects_unapproved_reproduction_tolerance_policy_hash() -> None:
    graph = _valid_shacl_graph()
    reproduction = next(graph.subjects(RDF.type, GENCO.ReproductionRun))
    graph.set((reproduction, GENCO.tolerancePolicyHash, Literal("0" * 64)))
    with pytest.raises(ValueError, match="tolerancePolicyHash"):
        validate_evidence_graph(graph, "ontology/shapes.ttl")


def test_reproduction_query_returns_runtime_and_comparison() -> None:
    rows = list(_valid_shacl_graph().query(REPRODUCTION))
    assert len(rows) == 1
    assert str(rows[0].runId) == "reproduction-run"
    assert str(rows[0].environmentHash) == "f" * 64
    assert (
        str(rows[0].tolerancePolicyId)
        == APPROVED_TOLERANCE_POLICY["policy_id"]
    )
    assert str(rows[0].tolerancePolicyHash) == APPROVED_TOLERANCE_POLICY_SHA256
    assert float(rows[0].maximumPowerDifference) == pytest.approx(2.34e-3)


def test_shacl_rejects_missing_physical_metric_unit() -> None:
    graph = _valid_shacl_graph()
    metric = next(graph.subjects(RDF.type, GENCO.MetricSemantics))
    graph.remove((metric, GENCO.unit, None))
    with pytest.raises(ValueError, match="unit"):
        validate_evidence_graph(graph, "ontology/shapes.ttl")


def test_shacl_rejects_missing_violation_comparison_tolerance() -> None:
    graph = _valid_shacl_graph()
    metric = next(
        subject
        for subject in graph.subjects(RDF.type, GENCO.MetricSemantics)
        if str(graph.value(subject, GENCO.metricName)).endswith("violation_count")
    )
    graph.remove((metric, GENCO.comparisonTolerance, None))
    with pytest.raises(ValueError, match="comparisonTolerance"):
        validate_evidence_graph(graph, "ontology/shapes.ttl")


def test_six_acceptance_queries_answer_from_one_graph() -> None:
    graph = build_evidence_graph(
        DatasetProvenance("dataset", "case14_ieee", "OptimalPowerFlow"),
        ModelProvenance(
            "run",
            "checkpoint",
            "dataset",
            checkpoint_hash="abc123",
            training_evidence_association=_unverified_training_association(),
        ),
        PredictionEvidence(
            "scenario-0",
            {},
            ({"scenario_id": "scenario-1", "distance": 1.5, "rank": 1},),
            1.5,
            None,
            PhysicalEvidence(
                mean_residual=0.2,
                max_residual=0.3,
                violation_count=1,
                extra={
                    "voltage_violation_count": 0,
                    "reactive_violation_count": 1,
                    "active_power_violation_count": 0,
                    "available_residual_types": "active_residual,reactive_residual",
                    "available_violation_types": "voltage_limits,reactive_limits",
                },
            ),
        ),
    )
    queries = (
        ASSOCIATED_TRAINING_EVIDENCE,
        CHECKPOINT,
        REFERENCE_CASES,
        EMBEDDING_DISTANCE,
        RESIDUAL,
        high_residual_and_distance(0.1, 1.0),
    )
    assert [len(list(graph.query(query))) for query in queries] == [1, 1, 1, 1, 1, 1]
    training_row = next(iter(graph.query(ASSOCIATED_TRAINING_EVIDENCE)))
    assert str(training_row.dataset).endswith("dataset%3Atrain")
    assert str(training_row.productionIdentityVerified) == "false"
    assert len(list(graph.query(VIOLATIONS))) == 1
    assert len(list(graph.query(AVAILABLE_VIOLATION_TYPES))) == 1
    assert len(list(graph.query(AVAILABLE_RESIDUAL_TYPES))) == 1


def test_high_evidence_query_requires_nonnegative_thresholds() -> None:
    with pytest.raises(ValueError, match="negative"):
        high_residual_and_distance(-1.0, 0.0)


def test_named_query_prints_descriptive_headers(tmp_path, capsys) -> None:
    graph = build_evidence_graph(
        DatasetProvenance("dataset", "case14_ieee", "OptimalPowerFlow"),
        ModelProvenance("run", "checkpoint", "dataset"),
        PredictionEvidence(
            "scenario-0",
            {},
            (),
            1.5,
            None,
            PhysicalEvidence(),
        ),
    )
    graph_path = tmp_path / "evidence.ttl"
    graph.serialize(graph_path, format="turtle")
    assert main(
        [
            "query",
            "--graph",
            str(graph_path),
            "--named",
            "embedding-distance",
        ],
    ) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "prediction\tnearest_embedding_distance"
    assert lines[1].endswith("\t1.5")


def test_public_query_and_validation_apis_use_serialized_graph(tmp_path) -> None:
    graph = _valid_shacl_graph()
    graph_path = tmp_path / "evidence.ttl"
    graph.serialize(graph_path, format="turtle")

    result = query_evidence(graph_path, named="embedding-distance")
    assert result.headers == ("prediction", "nearest_embedding_distance")
    assert result.rows[0][1] == "1.5"
    validate_evidence(graph_path, "ontology/shapes.ttl")


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_keyed_baseline_rejects_non_finite_values(
    tmp_path,
    invalid_value,
) -> None:
    actual_bus = tmp_path / "actual_bus.parquet"
    expected_bus = tmp_path / "expected_bus.parquet"
    actual_generator = tmp_path / "actual_generator.parquet"
    expected_generator = tmp_path / "expected_generator.parquet"
    tolerances = tmp_path / "tolerances.yaml"

    pd.DataFrame(
        {
            "scenario": [1, 1],
            "bus": [0, 1],
            "Vm_pred": [invalid_value, 1.0],
            "Va_pred": [0.0, 0.0],
            "Pg_pred": [0.0, 0.0],
            "Qg_pred": [0.0, 0.0],
            "active res. (MW)": [0.0, 0.0],
            "reactive res. (MVar)": [0.0, 0.0],
            "PBE": [0.0, 0.0],
            "Vm_target": [1.0, 1.0],
        },
    ).to_parquet(actual_bus)
    pd.DataFrame(
        {
            "scenario": [1, 1],
            "bus": [0, 1],
            "Vm_pred": [0.5, 1.0],
            "Va_pred": [0.0, 0.0],
            "Pg_pred": [0.0, 0.0],
            "Qg_pred": [0.0, 0.0],
            "active res. (MW)": [0.0, 0.0],
            "reactive res. (MVar)": [0.0, 0.0],
            "PBE": [0.0, 0.0],
            "Vm_target": [1.0, 1.0],
        },
    ).to_parquet(expected_bus)

    generator_table = pd.DataFrame(
        {
            "scenario": [1],
            "idx": [0],
            "p_mw_pred": [1.0],
        },
    )
    generator_table.to_parquet(actual_generator)
    generator_table.to_parquet(expected_generator)

    tolerances.write_text(
        yaml.safe_dump(APPROVED_TOLERANCE_POLICY, sort_keys=False),
    )

    actual_manifest = tmp_path / "actual_manifest.json"
    expected_manifest = tmp_path / "expected_manifest.json"
    _write_comparison_manifest(
        actual_manifest,
        (actual_bus, actual_generator),
    )
    _write_comparison_manifest(
        expected_manifest,
        (expected_bus, expected_generator),
    )

    with pytest.raises(
        ValueError,
        match=r"Non-finite values in actual baseline field 'Vm_pred'.*scenario.*1.*bus.*0",
    ):
        compare_keyed_baseline(
            prediction_path=actual_bus,
            generator_prediction_path=actual_generator,
            expected_prediction_path=expected_bus,
            expected_generator_prediction_path=expected_generator,
            actual_run_manifest=actual_manifest,
            expected_run_manifest=expected_manifest,
            tolerance_config=tolerances,
        )


def test_keyed_baseline_rejects_missing_runtime_provenance(tmp_path) -> None:
    actual_bus = tmp_path / "actual_bus.parquet"
    expected_bus = tmp_path / "expected_bus.parquet"
    actual_generator = tmp_path / "actual_generator.parquet"
    expected_generator = tmp_path / "expected_generator.parquet"

    pd.DataFrame(
        {"scenario": [1], "bus": [0], "Vm_pred": [1.0]},
    ).to_parquet(actual_bus)
    pd.DataFrame(
        {"scenario": [1], "bus": [0], "Vm_pred": [1.0]},
    ).to_parquet(expected_bus)
    pd.DataFrame(
        {"scenario": [1], "idx": [0], "p_mw_pred": [1.0]},
    ).to_parquet(actual_generator)
    pd.DataFrame(
        {"scenario": [1], "idx": [0], "p_mw_pred": [1.0]},
    ).to_parquet(expected_generator)

    actual_manifest = tmp_path / "actual_manifest.json"
    expected_manifest = tmp_path / "expected_manifest.json"
    _write_comparison_manifest(
        actual_manifest,
        (actual_bus, actual_generator),
    )
    _write_comparison_manifest(
        expected_manifest,
        (expected_bus, expected_generator),
    )

    value = json.loads(actual_manifest.read_text())
    del value["runtime"]
    actual_manifest.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="Actual run manifest lacks runtime"):
        compare_keyed_baseline(
            prediction_path=actual_bus,
            generator_prediction_path=actual_generator,
            expected_prediction_path=expected_bus,
            expected_generator_prediction_path=expected_generator,
            actual_run_manifest=actual_manifest,
            expected_run_manifest=expected_manifest,
        )


def test_reproduction_maxima_are_derived_from_comparison_result() -> None:
    differences = {
        "bus.Vm_pred": 1.5e-6,
        "bus.Va_pred": 1.1e-6,
        "bus.PBE": 2.3e-3,
        "bus.Pg_pred": 2.0e-3,
        "generator.p_mw_pred": 2.1e-3,
    }

    voltage, power = _comparison_maxima(differences)

    assert voltage == pytest.approx(1.5e-6)
    assert power == pytest.approx(2.3e-3)


def test_keyed_comparison_records_approved_tolerance_policy(tmp_path) -> None:
    actual_bus = tmp_path / "actual_bus.parquet"
    expected_bus = tmp_path / "expected_bus.parquet"
    actual_generator = tmp_path / "actual_generator.parquet"
    expected_generator = tmp_path / "expected_generator.parquet"
    tolerance_config = tmp_path / "tolerances.yaml"

    pd.DataFrame(
        {
            "scenario": [1],
            "bus": [0],
            "Vm_pred": [1.000001],
            "Va_pred": [0.0],
            "Pg_pred": [0.0],
            "Qg_pred": [0.0],
            "active res. (MW)": [0.0],
            "reactive res. (MVar)": [0.0],
            "PBE": [0.002],
        },
    ).to_parquet(actual_bus)
    pd.DataFrame(
        {
            "scenario": [1],
            "bus": [0],
            "Vm_pred": [1.0],
            "Va_pred": [0.0],
            "Pg_pred": [0.0],
            "Qg_pred": [0.0],
            "active res. (MW)": [0.0],
            "reactive res. (MVar)": [0.0],
            "PBE": [0.0],
        },
    ).to_parquet(expected_bus)
    pd.DataFrame(
        {
            "scenario": [1],
            "idx": [0],
            "p_mw_pred": [1.001],
        },
    ).to_parquet(actual_generator)
    pd.DataFrame(
        {
            "scenario": [1],
            "idx": [0],
            "p_mw_pred": [1.0],
        },
    ).to_parquet(expected_generator)

    actual_manifest = tmp_path / "actual_manifest.json"
    expected_manifest = tmp_path / "expected_manifest.json"
    _write_comparison_manifest(
        actual_manifest,
        (actual_bus, actual_generator),
    )
    _write_comparison_manifest(
        expected_manifest,
        (expected_bus, expected_generator),
    )

    tolerance_config.write_text(
        yaml.safe_dump(APPROVED_TOLERANCE_POLICY, sort_keys=False),
    )

    result = compare_keyed_baseline(
        prediction_path=actual_bus,
        generator_prediction_path=actual_generator,
        expected_prediction_path=expected_bus,
        expected_generator_prediction_path=expected_generator,
        actual_run_manifest=actual_manifest,
        expected_run_manifest=expected_manifest,
        tolerance_config=tolerance_config,
    )
    voltage, power = _comparison_maxima(
        result.maximum_absolute_differences,
    )

    assert voltage == pytest.approx(1e-6)
    assert power == pytest.approx(0.002)
    assert result.tolerance_policy_id == APPROVED_TOLERANCE_POLICY["policy_id"]
    assert result.tolerance_policy_sha256 == APPROVED_TOLERANCE_POLICY_SHA256


def test_keyed_comparison_rejects_missing_approved_field(tmp_path) -> None:
    actual_bus = tmp_path / "actual_bus.parquet"
    expected_bus = tmp_path / "expected_bus.parquet"
    actual_generator = tmp_path / "actual_generator.parquet"
    expected_generator = tmp_path / "expected_generator.parquet"
    incomplete_bus = pd.DataFrame(
        {
            "scenario": [1],
            "bus": [0],
            "Vm_pred": [1.0],
            "Va_pred": [0.0],
            "Pg_pred": [0.0],
            "active res. (MW)": [0.0],
            "reactive res. (MVar)": [0.0],
            "PBE": [0.0],
        },
    )
    incomplete_bus.to_parquet(actual_bus)
    incomplete_bus.to_parquet(expected_bus)
    generator = pd.DataFrame(
        {"scenario": [1], "idx": [0], "p_mw_pred": [0.0]},
    )
    generator.to_parquet(actual_generator)
    generator.to_parquet(expected_generator)
    actual_manifest = tmp_path / "actual_manifest.json"
    expected_manifest = tmp_path / "expected_manifest.json"
    _write_comparison_manifest(
        actual_manifest,
        (actual_bus, actual_generator),
    )
    _write_comparison_manifest(
        expected_manifest,
        (expected_bus, expected_generator),
    )

    with pytest.raises(ValueError, match="missing required.*Qg_pred"):
        compare_keyed_baseline(
            prediction_path=actual_bus,
            generator_prediction_path=actual_generator,
            expected_prediction_path=expected_bus,
            expected_generator_prediction_path=expected_generator,
            actual_run_manifest=actual_manifest,
            expected_run_manifest=expected_manifest,
        )


def test_keyed_comparison_rejects_mutable_huge_tolerance(tmp_path) -> None:
    actual_bus = tmp_path / "actual_bus.parquet"
    expected_bus = tmp_path / "expected_bus.parquet"
    actual_generator = tmp_path / "actual_generator.parquet"
    expected_generator = tmp_path / "expected_generator.parquet"
    tolerance_config = tmp_path / "tolerances.yaml"

    pd.DataFrame(
        {"scenario": [1], "bus": [0], "PBE": [999.0]},
    ).to_parquet(actual_bus)
    pd.DataFrame(
        {"scenario": [1], "bus": [0], "PBE": [1.0]},
    ).to_parquet(expected_bus)
    for path in (actual_generator, expected_generator):
        pd.DataFrame(
            {"scenario": [1], "idx": [0], "p_mw_pred": [1.0]},
        ).to_parquet(path)

    actual_manifest = tmp_path / "actual_manifest.json"
    expected_manifest = tmp_path / "expected_manifest.json"
    _write_comparison_manifest(
        actual_manifest,
        (actual_bus, actual_generator),
    )
    _write_comparison_manifest(
        expected_manifest,
        (expected_bus, expected_generator),
    )
    mutable_policy = json.loads(json.dumps(APPROVED_TOLERANCE_POLICY))
    mutable_policy["absolute_tolerances"]["bus"]["PBE"] = 1e9
    tolerance_config.write_text(yaml.safe_dump(mutable_policy))

    with pytest.raises(ValueError, match="approved immutable policy"):
        compare_keyed_baseline(
            prediction_path=actual_bus,
            generator_prediction_path=actual_generator,
            expected_prediction_path=expected_bus,
            expected_generator_prediction_path=expected_generator,
            actual_run_manifest=actual_manifest,
            expected_run_manifest=expected_manifest,
            tolerance_config=tolerance_config,
        )


@pytest.mark.parametrize(
    ("target", "field", "value", "message"),
    (
        ("runtime", "torch", "9.9.9", "runtime identity 'torch' disagrees"),
        (
            "manifest",
            "effective_config_sha256",
            "9" * 64,
            "effective configuration hash disagrees",
        ),
        (
            "manifest",
            "embedding_dimension",
            999,
            "Actual uninstrumented reproduction must not declare",
        ),
        (
            "manifest",
            "export_embeddings",
            True,
            "Actual run must be the uninstrumented reproduction",
        ),
    ),
)
def test_keyed_comparison_rejects_incompatible_runs_before_numeric_reads(
    tmp_path,
    monkeypatch,
    target,
    field,
    value,
    message,
) -> None:
    actual_bus = tmp_path / "actual_bus.parquet"
    expected_bus = tmp_path / "expected_bus.parquet"
    actual_generator = tmp_path / "actual_generator.parquet"
    expected_generator = tmp_path / "expected_generator.parquet"
    for path, frame in (
        (
            actual_bus,
            pd.DataFrame({"scenario": [1], "bus": [0], "Vm_pred": [1.0]}),
        ),
        (
            expected_bus,
            pd.DataFrame({"scenario": [1], "bus": [0], "Vm_pred": [1.0]}),
        ),
        (
            actual_generator,
            pd.DataFrame({"scenario": [1], "idx": [0], "p_mw_pred": [1.0]}),
        ),
        (
            expected_generator,
            pd.DataFrame({"scenario": [1], "idx": [0], "p_mw_pred": [1.0]}),
        ),
    ):
        frame.to_parquet(path)

    actual_manifest = tmp_path / "actual_manifest.json"
    expected_manifest = tmp_path / "expected_manifest.json"
    _write_comparison_manifest(
        actual_manifest,
        (actual_bus, actual_generator),
    )
    _write_comparison_manifest(
        expected_manifest,
        (expected_bus, expected_generator),
    )
    actual_value = json.loads(actual_manifest.read_text())
    if target == "runtime":
        actual_value["runtime"][field] = value
    else:
        actual_value[field] = value
    actual_manifest.write_text(json.dumps(actual_value))
    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda *args, **kwargs: pytest.fail("numeric tables were read"),
    )

    with pytest.raises(ValueError, match=message):
        compare_keyed_baseline(
            prediction_path=actual_bus,
            generator_prediction_path=actual_generator,
            expected_prediction_path=expected_bus,
            expected_generator_prediction_path=expected_generator,
            actual_run_manifest=actual_manifest,
            expected_run_manifest=expected_manifest,
        )


def test_target_manifest_rejects_missing_runtime(tmp_path) -> None:
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format_version": 1,
                "artifacts": [],
            },
        ),
    )

    with pytest.raises(ValueError, match="Target run manifest lacks runtime"):
        verify_run_manifest(
            manifest,
            prediction_path=tmp_path / "predictions.parquet",
            generator_prediction_path=tmp_path / "generators.parquet",
            target_embedding_path=tmp_path / "embeddings.parquet",
            checkpoint_path=tmp_path / "checkpoint.pt",
            normalizer_path=tmp_path / "normalizer.pt",
            config_path=tmp_path / "config.yaml",
            split_path=tmp_path / "split.pt",
            raw_dataset_path=tmp_path / "raw",
        )


@pytest.mark.parametrize(
    ("reuse_flag", "expected"),
    [
        ("--reuse-verified-run", True),
        ("--resume", True),
        (None, False),
    ],
)
def test_run_baseline_reuse_flags(reuse_flag, expected) -> None:
    arguments = [
        "run-baseline",
        "--graphkit-python", "/graphkit/.venv/bin/python",
        "--graphkit-workdir", "/graphkit",
        "--config", "config.yaml",
        "--checkpoint", "checkpoint.pt",
        "--normalizer", "normalizer.pt",
        "--data-path", "data",
        "--split-artifact", "test.pt",
        "--output-dir", "output",
        "--run-name", "test-run",
        "--graphkit-requirements-lock", "requirements.lock",
        "--graphkit-source-record", "source.txt",
    ]
    if reuse_flag is not None:
        arguments.append(reuse_flag)

    args = build_parser().parse_args(arguments)

    assert args.reuse_verified_run is expected
    assert not hasattr(args, "resume")


def test_selected_runtime_must_match_reused_run_manifest(
    tmp_path,
    monkeypatch,
) -> None:
    workdir = tmp_path / "graphkit"
    executable = workdir / ".venv/bin/python"
    executable.parent.mkdir(parents=True)
    executable.write_text("")
    requirements_lock = tmp_path / "graphkit-requirements.lock"
    source_record = tmp_path / "graphkit-source.txt"
    requirements_lock.write_text("torch==2.11.0\n")
    source_record.write_text(
        "repository=https://github.com/gridfm/gridfm-graphkit.git\n"
        "commit=" + "a" * 40 + "\n",
    )

    runtime = {
        "python_version": "3.12.10",
        "platform": "macOS-14.8.9-arm-64bit",
        "graphkit": "0.9.0.3",
        "torch": "2.11.0",
        "torchvision": "0.26.0",
        "torchaudio": "2.11.0",
        "torch_scatter": "2.1.2",
        "python_executable": ".venv/bin/python",
        "requirements_lock_sha256": sha256_file(requirements_lock),
        "source_record_sha256": sha256_file(source_record),
        "source_repository": "https://github.com/gridfm/gridfm-graphkit.git",
        "source_commit": "a" * 40,
    }

    active_runtime = {
        key: runtime[key]
        for key in (
            "python_version",
            "platform",
            "graphkit",
            "torch",
            "torchvision",
            "torchaudio",
            "torch_scatter",
        )
    }

    def fake_run(command, **kwargs):
        if command[1:4] == ["-m", "pip", "freeze"]:
            stdout = requirements_lock.read_text()
        elif command[:4] == ["git", "remote", "get-url", "origin"]:
            stdout = runtime["source_repository"] + "\n"
        elif command[:3] == ["git", "rev-parse", "HEAD"]:
            stdout = runtime["source_commit"] + "\n"
        elif command[1] == "-c":
            stdout = json.dumps(active_runtime) + "\n"
        else:
            raise AssertionError(f"Unexpected command: {command}")
        return type("Result", (), {"stdout": stdout})()

    monkeypatch.setattr(
        "genco_evidence_graph.baseline.subprocess.run",
        fake_run,
    )

    _verify_selected_runtime(
        graphkit_python=executable,
        graphkit_workdir=workdir,
        requirements_lock=requirements_lock,
        source_record=source_record,
        recorded_runtime=runtime,
    )

    incompatible = dict(runtime)
    incompatible["torch"] = "2.10.0"
    with pytest.raises(ValueError, match="runtime 'torch' disagrees"):
        _verify_selected_runtime(
            graphkit_python=executable,
            graphkit_workdir=workdir,
            requirements_lock=requirements_lock,
            source_record=source_record,
            recorded_runtime=incompatible,
        )


def test_selected_runtime_rejects_different_lock(tmp_path) -> None:
    requirements_lock = tmp_path / "graphkit-requirements.lock"
    source_record = tmp_path / "graphkit-source.txt"
    requirements_lock.write_text("torch==2.11.0\n")
    source_record.write_text(
        "repository=https://github.com/gridfm/gridfm-graphkit.git\n"
        "commit=" + "a" * 40 + "\n",
    )
    runtime = {
        "requirements_lock_sha256": "0" * 64,
        "source_record_sha256": sha256_file(source_record),
    }

    with pytest.raises(ValueError, match="requirements lock disagrees"):
        _verify_selected_runtime(
            graphkit_python=tmp_path / "graphkit/.venv/bin/python",
            graphkit_workdir=tmp_path / "graphkit",
            requirements_lock=requirements_lock,
            source_record=source_record,
            recorded_runtime=runtime,
        )


def test_embedding_entity_coverage_requires_exact_prediction_keys(
    tmp_path,
) -> None:
    embeddings = tmp_path / "embeddings.parquet"
    buses = tmp_path / "buses.parquet"
    generators = tmp_path / "generators.parquet"

    pd.DataFrame(
        {
            "scenario": [1, 1],
            "bus": [0, 1],
        },
    ).to_parquet(buses)
    pd.DataFrame(
        {
            "scenario": [1],
            "idx": [0],
        },
    ).to_parquet(generators)
    pd.DataFrame(
        {
            "scenario_id": [1, 1, 1],
            "entity_type": ["bus", "bus", "generator"],
            "entity_id": [0, 1, 0],
        },
    ).to_parquet(embeddings)

    verify_embedding_entity_coverage(
        embedding_path=embeddings,
        prediction_path=buses,
        generator_prediction_path=generators,
        label="test embeddings",
    )

    incomplete = pd.read_parquet(embeddings)
    incomplete = incomplete[
        ~(
            (incomplete["entity_type"] == "bus")
            & (incomplete["entity_id"] == 1)
        )
    ]
    incomplete.to_parquet(embeddings)

    with pytest.raises(
        ValueError,
        match=r"entity coverage disagrees.*missing=.*bus.*1",
    ):
        verify_embedding_entity_coverage(
            embedding_path=embeddings,
            prediction_path=buses,
            generator_prediction_path=generators,
            label="test embeddings",
        )


def test_embedding_entity_coverage_rejects_extra_entity(tmp_path) -> None:
    embeddings = tmp_path / "embeddings.parquet"
    buses = tmp_path / "buses.parquet"
    generators = tmp_path / "generators.parquet"

    pd.DataFrame(
        {"scenario": [1], "bus": [0]},
    ).to_parquet(buses)
    pd.DataFrame(
        {"scenario": [1], "idx": [0]},
    ).to_parquet(generators)
    pd.DataFrame(
        {
            "scenario_id": [1, 1, 1],
            "entity_type": ["bus", "generator", "generator"],
            "entity_id": [0, 0, 1],
        },
    ).to_parquet(embeddings)

    with pytest.raises(
        ValueError,
        match=r"entity coverage disagrees.*extra=.*generator.*1",
    ):
        verify_embedding_entity_coverage(
            embedding_path=embeddings,
            prediction_path=buses,
            generator_prediction_path=generators,
            label="test embeddings",
        )


@pytest.mark.parametrize(
    ("artifact", "column"),
    [
        ("bus", "Vm_min"),
        ("bus", "Qg_max"),
        ("generator", "max_p_mw"),
    ],
)
def test_physical_evidence_rejects_missing_limit_columns(
    tmp_path,
    artifact,
    column,
) -> None:
    bus_path = tmp_path / "bus.parquet"
    generator_path = tmp_path / "generator.parquet"

    bus = pd.DataFrame(
        {
            "scenario": [7],
            "bus": [0],
            "Vm_pred": [1.0],
            "Vm_min": [0.95],
            "Vm_max": [1.05],
            "Qg_pred": [0.0],
            "Qg_min": [-1.0],
            "Qg_max": [1.0],
            "active res. (MW)": [0.1],
            "reactive res. (MVar)": [0.2],
            "PBE": [0.3],
        },
    )
    generator = pd.DataFrame(
        {
            "scenario": [7],
            "idx": [0],
            "p_mw_pred": [2.0],
            "min_p_mw": [0.0],
            "max_p_mw": [3.0],
        },
    )
    if artifact == "bus":
        bus = bus.drop(columns=[column])
    else:
        generator = generator.drop(columns=[column])
    bus.to_parquet(bus_path)
    generator.to_parquet(generator_path)

    with pytest.raises(
        ValueError,
        match=rf"lacks required physical-evidence columns.*{column}",
    ):
        _physical_evidence(bus_path, "7", generator_path)


def test_physical_evidence_requires_generator_predictions(tmp_path) -> None:
    bus_path = tmp_path / "bus.parquet"
    pd.DataFrame(
        {
            "scenario": [7],
            "bus": [0],
            "Vm_pred": [1.0],
            "Vm_min": [0.95],
            "Vm_max": [1.05],
            "Qg_pred": [0.0],
            "Qg_min": [-1.0],
            "Qg_max": [1.0],
            "active res. (MW)": [0.1],
            "reactive res. (MVar)": [0.2],
            "PBE": [0.3],
        },
    ).to_parquet(bus_path)

    with pytest.raises(
        ValueError,
        match="Generator predictions are required",
    ):
        _physical_evidence(bus_path, "7")


def test_authoritative_training_metadata_extracts_parameters_and_metrics() -> None:
    run = (
        "genco-opfdata/mlflow/train/case14_ieee/seed42/"
        "1927fe80558a48539ee37be478706781"
    )
    metadata = load_training_run_metadata(run)

    assert metadata.epochs == 200
    assert metadata.batch_size == 64
    assert metadata.learning_rate == pytest.approx(0.0005)
    assert metadata.optimizer == "AdamW"
    assert metadata.selected_metrics["training_loss"] == pytest.approx(
        6.062971806386486e-06,
    )
    assert metadata.selected_metrics["validation_loss"] == pytest.approx(
        5.9444350881676655e-06,
    )
    assert metadata.selected_metrics[
        "validation_bus_mse_loss"
    ] == pytest.approx(4.1562995534150105e-07)
    assert metadata.selected_metrics[
        "validation_generator_mse_loss"
    ] == pytest.approx(1.6805921632112586e-06)
    assert metadata.selected_metrics[
        "validation_physics_loss"
    ] == pytest.approx(4.85103482787963e-05)
    assert metadata.selected_metrics[
        "validation_qg_violation_penalty_loss"
    ] == pytest.approx(0.0006136162555776536)
    assert "unavailable upstream" in metadata.software_revision_status
    assert "does not record a checkpoint hash" in metadata.checkpoint_binding_status


def test_training_parameters_metrics_and_limitations_are_in_rdf() -> None:
    graph = _valid_shacl_graph()
    training = next(graph.subjects(RDF.type, GENCO.TrainingRun))
    checkpoint = next(graph.subjects(RDF.type, GENCO.ModelCheckpoint))

    assert int(graph.value(training, GENCO.trainingEpochs)) == 200
    assert int(graph.value(training, GENCO.trainingBatchSize)) == 64
    assert float(graph.value(training, GENCO.trainingLearningRate)) == pytest.approx(
        0.0005,
    )
    assert str(graph.value(training, GENCO.trainingOptimizer)) == "AdamW"
    assert float(graph.value(training, GENCO.validationLoss)) == pytest.approx(
        5.9e-6,
    )
    assert "unavailable upstream" in str(
        graph.value(training, GENCO.trainingSoftwareRevisionStatus),
    )
    assert "co-published" in str(
        graph.value(checkpoint, GENCO.checkpointBindingStatus),
    )

def _training_upstream_fixture(tmp_path):
    revision = "a" * 40
    run_id = "training-run"
    run_relative = (
        Path("mlflow/train/case14_ieee/seed42") / run_id
    )
    normalized_run = tmp_path / "normalized" / run_relative
    original_root = tmp_path / "original-upstream"
    original_run = original_root / run_relative
    cache_run = (
        original_root
        / ".cache/huggingface/download"
        / run_relative
    )

    original_run.mkdir(parents=True)
    cache_run.mkdir(parents=True)
    model_repository = tmp_path / "model-repository"
    tree_dir = model_repository / ".cache/huggingface/trees"
    tree_dir.mkdir(parents=True)
    (tree_dir / f"{revision}.json").write_text("{}\n")

    upstream = {
        "meta.yaml": (
            b"artifact_uri: file:///private/artifacts\n"
            b"user_id: private-user\n"
        ),
        "params/args/data/split_from_existing_files": b"/private/splits",
        "tags/mlflow.source.name": b"/private/bin/gridfm_graphkit",
        "tags/mlflow.user": b"private-user",
    }
    for declared_path in sorted(EXPECTED_TRAINING_METADATA_PATHS):
        upstream_bytes = upstream.get(
            declared_path,
            f"upstream value for {declared_path}\n".encode(),
        )
        original_file = original_run / declared_path
        metadata_file = cache_run / f"{declared_path}.metadata"
        original_file.parent.mkdir(parents=True, exist_ok=True)
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        original_file.write_bytes(upstream_bytes)
        etag = hashlib.sha1(
            f"blob {len(upstream_bytes)}\0".encode() + upstream_bytes,
        ).hexdigest()
        metadata_file.write_text(f"{revision}\n{etag}\n0\n")

    manifest_path = prepare_normalized_training_metadata(
        original_repository_root=original_root,
        model_repository_path=model_repository,
        run_relative_path=run_relative,
        output_run_path=normalized_run,
    )

    return normalized_run, original_root, revision, manifest_path


def test_training_metadata_manifest_model_round_trips(tmp_path) -> None:
    _, _, _, manifest_path = _training_upstream_fixture(tmp_path)
    manifest = load_training_metadata_manifest(manifest_path)
    assert len(manifest.files) == 79
    assert training_metadata_manifest_from_dict(
        training_metadata_manifest_to_dict(manifest),
    ) == manifest
    unsorted = type(manifest)(
        manifest.format_version,
        manifest.source_repository,
        manifest.source_revision,
        manifest.source_run_id,
        manifest.normalization,
        tuple(reversed(manifest.files)),
    )
    assert [item["path"] for item in training_metadata_manifest_to_dict(unsorted)["files"]] == sorted(
        record.path for record in manifest.files
    )


def test_training_metadata_manifest_loader_rejects_duplicate_json_keys(
    tmp_path,
) -> None:
    malformed = tmp_path / "duplicate.json"
    malformed.write_text('{"format_version": 2, "format_version": 2}')
    with pytest.raises(ValueError, match="Duplicate JSON object key"):
        load_training_metadata_manifest(malformed)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("unknown", "unknown"),
        ("missing", "missing"),
        ("wrong_type", "source run ID"),
        ("boolean_version", "integer"),
        ("unsupported_version", "Unsupported"),
        ("revision", "revision"),
        ("hash", "SHA-256"),
        ("duplicate", "Duplicate"),
        ("absolute", "Unsafe"),
        ("parent", "Unsafe"),
        ("backslash", "Unsafe"),
        ("unsorted", "canonical path order"),
        ("normalization", "normalization policy"),
    ],
)
def test_training_metadata_manifest_parser_rejects_malformed_values(
    tmp_path,
    mutation,
    message,
) -> None:
    _, _, _, manifest_path = _training_upstream_fixture(tmp_path)
    value = json.loads(manifest_path.read_text())
    if mutation == "unknown":
        value["extra"] = 1
    elif mutation == "missing":
        del value["source_run_id"]
    elif mutation == "wrong_type":
        value["source_run_id"] = 42
    elif mutation == "boolean_version":
        value["format_version"] = True
    elif mutation == "unsupported_version":
        value["format_version"] = 3
    elif mutation == "revision":
        value["source_revision"] = "not-a-revision"
    elif mutation == "hash":
        value["files"][0]["upstream_sha256"] = "z" * 64
    elif mutation == "duplicate":
        value["files"].append(dict(value["files"][0]))
    elif mutation == "absolute":
        value["files"][0]["path"] = "/tmp/unsafe"
    elif mutation == "parent":
        value["files"][0]["path"] = "../unsafe"
    elif mutation == "backslash":
        value["files"][0]["path"] = "artifacts\\unsafe"
    elif mutation == "unsorted":
        value["files"].reverse()
    elif mutation == "normalization":
        value["normalization"]["transforms"]["extra"] = "not-approved"
    with pytest.raises(ValueError, match=message):
        training_metadata_manifest_from_dict(value)


def test_normalized_training_metadata_generation_is_complete_and_deterministic(
    tmp_path,
) -> None:
    normalized, original_root, revision, manifest_path = _training_upstream_fixture(tmp_path)
    manifest = load_training_metadata_manifest(manifest_path)
    assert manifest.source_revision == revision
    assert {record.path for record in manifest.files} == EXPECTED_TRAINING_METADATA_PATHS
    verify_normalized_upstream_metadata(normalized, original_root, revision)

    run_relative = Path("mlflow/train/case14_ieee/seed42/training-run")
    second_output = tmp_path / "second" / run_relative
    second_manifest = prepare_normalized_training_metadata(
        original_repository_root=original_root,
        model_repository_path=tmp_path / "model-repository",
        run_relative_path=run_relative,
        output_run_path=second_output,
    )
    assert manifest_path.read_bytes() == second_manifest.read_bytes()
    original_run = original_root / run_relative
    changed = {
        path
        for path in EXPECTED_TRAINING_METADATA_PATHS
        if (original_run / path).read_bytes() != (normalized / path).read_bytes()
    }
    assert changed == set(TRAINING_METADATA_NORMALIZATION["transforms"])
    for path in EXPECTED_TRAINING_METADATA_PATHS - changed:
        assert (original_run / path).read_bytes() == (normalized / path).read_bytes()


def test_training_metadata_generator_rejects_etag_failure_and_cleans_temporary(
    tmp_path,
) -> None:
    normalized, original, revision, _ = _training_upstream_fixture(tmp_path)
    output = tmp_path / "failed" / normalized.relative_to(tmp_path / "normalized")
    cache_record = (
        original
        / ".cache/huggingface/download/mlflow/train/case14_ieee/seed42/training-run/meta.yaml.metadata"
    )
    cache_record.write_text(f"{revision}\n{'0' * 40}\n")
    with pytest.raises(ValueError, match="cache ETag"):
        prepare_normalized_training_metadata(
            original_repository_root=original,
            model_repository_path=tmp_path / "model-repository",
            run_relative_path="mlflow/train/case14_ieee/seed42/training-run",
            output_run_path=output,
        )
    assert not output.exists()
    assert not list(output.parent.glob(".training-run.tmp-*"))

def test_training_metadata_generator_accepts_sha256_cache_etag(tmp_path) -> None:
    _, original, revision, _ = _training_upstream_fixture(tmp_path)
    run_relative = Path("mlflow/train/case14_ieee/seed42/training-run")
    declared_path = (
        "artifacts/test_plots/case14_ieee/"
        "case14_ieee_correlation_PQ.png"
    )

    original_file = original / run_relative / declared_path
    cache_record = (
        original
        / ".cache/huggingface/download"
        / run_relative
        / f"{declared_path}.metadata"
    )
    sha256_etag = hashlib.sha256(original_file.read_bytes()).hexdigest()
    cache_record.write_text(f"{revision}\n{sha256_etag}\n0\n")

    output = tmp_path / "sha256-output" / run_relative
    manifest_path = prepare_normalized_training_metadata(
        original_repository_root=original,
        model_repository_path=tmp_path / "model-repository",
        run_relative_path=run_relative,
        output_run_path=output,
    )

    manifest = load_training_metadata_manifest(manifest_path)
    record = next(item for item in manifest.files if item.path == declared_path)
    assert record.upstream_sha256 == sha256_etag
    assert (output / declared_path).read_bytes() == original_file.read_bytes()


def test_training_metadata_generator_rejects_mismatched_sha256_cache_etag(
    tmp_path,
) -> None:
    _, original, revision, _ = _training_upstream_fixture(tmp_path)
    run_relative = Path("mlflow/train/case14_ieee/seed42/training-run")
    declared_path = (
        "artifacts/test_plots/case14_ieee/"
        "case14_ieee_correlation_PQ.png"
    )

    cache_record = (
        original
        / ".cache/huggingface/download"
        / run_relative
        / f"{declared_path}.metadata"
    )
    wrong_etag = hashlib.sha256(b"not the source file").hexdigest()
    cache_record.write_text(f"{revision}\n{wrong_etag}\n0\n")

    output = tmp_path / "sha256-failure" / run_relative
    with pytest.raises(ValueError, match="disagrees with cache ETag"):
        prepare_normalized_training_metadata(
            original_repository_root=original,
            model_repository_path=tmp_path / "model-repository",
            run_relative_path=run_relative,
            output_run_path=output,
        )

    assert not output.exists()
    assert not list(output.parent.glob(".training-run.tmp-*"))

def test_training_metadata_generator_rejects_nonempty_and_overlapping_output(
    tmp_path,
) -> None:
    normalized, original, _, _ = _training_upstream_fixture(tmp_path)
    nonempty = tmp_path / "nonempty" / "training-run"
    nonempty.mkdir(parents=True)
    (nonempty / "sentinel").write_text("keep")
    args = {
        "original_repository_root": original,
        "model_repository_path": tmp_path / "model-repository",
        "run_relative_path": "mlflow/train/case14_ieee/seed42/training-run",
    }
    with pytest.raises(ValueError, match="not empty"):
        prepare_normalized_training_metadata(**args, output_run_path=nonempty)
    with pytest.raises(ValueError, match="overlap"):
        prepare_normalized_training_metadata(**args, output_run_path=original)
    assert normalized.is_dir()


@pytest.mark.parametrize("mutation", ["missing", "unexpected", "symlink", "cache"])
def test_training_metadata_generator_rejects_unsafe_or_incomplete_inputs(
    tmp_path,
    mutation,
) -> None:
    normalized, original, _, _ = _training_upstream_fixture(tmp_path)
    run_relative = Path("mlflow/train/case14_ieee/seed42/training-run")
    original_run = original / run_relative
    first = original_run / "meta.yaml"
    if mutation == "missing":
        first.unlink()
    elif mutation == "unexpected":
        (original_run / "unexpected.txt").write_text("extra")
    elif mutation == "symlink":
        first.unlink()
        first.symlink_to(original_run / "tags/mlflow.user")
    elif mutation == "cache":
        (
            original
            / ".cache/huggingface/download"
            / run_relative
            / "meta.yaml.metadata"
        ).unlink()

    output = tmp_path / f"failed-{mutation}" / run_relative
    with pytest.raises((FileNotFoundError, ValueError)):
        prepare_normalized_training_metadata(
            original_repository_root=original,
            model_repository_path=tmp_path / "model-repository",
            run_relative_path=run_relative,
            output_run_path=output,
        )
    assert not output.exists()
    assert normalized.is_dir()


def test_prepare_training_metadata_cli_generates_manifest(tmp_path, capsys) -> None:
    normalized, original, revision, _ = _training_upstream_fixture(tmp_path)
    output = tmp_path / "cli-output" / normalized.relative_to(tmp_path / "normalized")
    result = main(
        [
            "prepare-training-metadata",
            "--original-upstream-root",
            str(original),
            "--model-repository-path",
            str(tmp_path / "model-repository"),
            "--run-relative-path",
            "mlflow/train/case14_ieee/seed42/training-run",
            "--output-run-path",
            str(output),
        ],
    )
    captured = capsys.readouterr()
    assert result == 0
    assert "Normalized training metadata prepared" in captured.out
    assert str(output / "upstream_source_manifest.json") in captured.out
    assert f"Source revision: {revision}" in captured.out
    assert "Files: 79" in captured.out


def test_prepare_training_metadata_cli_rejects_nonempty_output(tmp_path, capsys) -> None:
    normalized, original, _, _ = _training_upstream_fixture(tmp_path)
    output = tmp_path / "cli-nonempty" / "training-run"
    output.mkdir(parents=True)
    (output / "keep.txt").write_text("keep")
    result = main(
        [
            "prepare-training-metadata",
            "--original-upstream-root",
            str(original),
            "--model-repository-path",
            str(tmp_path / "model-repository"),
            "--run-relative-path",
            "mlflow/train/case14_ieee/seed42/training-run",
            "--output-run-path",
            str(output),
        ],
    )
    captured = capsys.readouterr()
    assert result == 2
    assert captured.err.startswith("error: ")
    assert (output / "keep.txt").read_text() == "keep"
    assert normalized.is_dir()


def test_prepare_training_metadata_cli_failure_has_no_partial_output(
    tmp_path,
    capsys,
) -> None:
    normalized, original, revision, _ = _training_upstream_fixture(tmp_path)
    cache_record = (
        original
        / ".cache/huggingface/download/mlflow/train/case14_ieee/seed42/training-run/meta.yaml.metadata"
    )
    cache_record.write_text(f"{'b' * 40}\n{'0' * 40}\n")
    output = tmp_path / "cli-failed" / normalized.relative_to(tmp_path / "normalized")
    result = main(
        [
            "prepare-training-metadata",
            "--original-upstream-root",
            str(original),
            "--model-repository-path",
            str(tmp_path / "model-repository"),
            "--run-relative-path",
            "mlflow/train/case14_ieee/seed42/training-run",
            "--output-run-path",
            str(output),
        ],
    )
    captured = capsys.readouterr()
    assert result == 2
    assert captured.err.startswith("error: ")
    assert not output.exists()


def test_normalized_upstream_metadata_verifies_original_upstream_chain(
    tmp_path,
) -> None:
    normalized, original, revision, _ = _training_upstream_fixture(tmp_path)

    verify_normalized_upstream_metadata(
        normalized,
        original,
        revision,
    )


def test_normalized_upstream_metadata_rejects_changed_original_upstream_file(
    tmp_path,
) -> None:
    normalized, original, revision, _ = _training_upstream_fixture(tmp_path)
    original_file = (
        original
        / "mlflow/train/case14_ieee/seed42/training-run/meta.yaml"
    )
    original_file.write_bytes(b"tampered upstream metadata\n")

    with pytest.raises(ValueError, match="cache ETag"):
        verify_normalized_upstream_metadata(
            normalized,
            original,
            revision,
        )


def test_normalized_upstream_metadata_rejects_changed_normalized_file(
    tmp_path,
) -> None:
    normalized, original, revision, _ = _training_upstream_fixture(tmp_path)
    (normalized / "meta.yaml").write_bytes(b"tampered normalized metadata\n")

    with pytest.raises(ValueError, match="deterministic transform"):
        verify_normalized_upstream_metadata(
            normalized,
            original,
            revision,
        )


def test_normalized_upstream_metadata_rejects_locally_rehashed_forgery(
    tmp_path,
) -> None:
    normalized, original, revision, manifest_path = (
        _training_upstream_fixture(tmp_path)
    )
    forged = b"arbitrary-forged-user"
    (normalized / "tags/mlflow.user").write_bytes(forged)
    manifest = json.loads(manifest_path.read_text())
    user_record = next(
        record
        for record in manifest["files"]
        if record["path"] == "tags/mlflow.user"
    )
    user_record["normalized_sha256"] = hashlib.sha256(forged).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="deterministic transform"):
        verify_normalized_upstream_metadata(
            normalized,
            original,
            revision,
        )


def test_normalized_upstream_metadata_rejects_omitted_forged_file(
    tmp_path,
) -> None:
    normalized, original, revision, manifest_path = (
        _training_upstream_fixture(tmp_path)
    )
    (normalized / "tags/mlflow.user").write_bytes(b"arbitrary-forged-user")
    manifest = json.loads(manifest_path.read_text())
    manifest["files"] = [
        record
        for record in manifest["files"]
        if record["path"] != "tags/mlflow.user"
    ]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="approved inventory"):
        verify_normalized_upstream_metadata(
            normalized,
            original,
            revision,
        )


def test_normalized_upstream_metadata_rejects_added_undeclared_file(
    tmp_path,
) -> None:
    normalized, original, revision, _ = _training_upstream_fixture(tmp_path)
    (normalized / "extra.txt").write_text("undeclared")
    with pytest.raises(
        ValueError,
        match="does not match the approved training metadata inventory",
    ):
        verify_normalized_upstream_metadata(normalized, original, revision)


def test_training_metadata_generator_requires_exact_transform_path_set(
    tmp_path,
) -> None:
    normalized, original, revision, _ = _training_upstream_fixture(tmp_path)
    run_relative = Path("mlflow/train/case14_ieee/seed42/training-run")
    original_meta = original / run_relative / "meta.yaml"
    original_meta.write_bytes((normalized / "meta.yaml").read_bytes())
    metadata = (
        original
        / ".cache/huggingface/download"
        / run_relative
        / "meta.yaml.metadata"
    )
    data = original_meta.read_bytes()
    etag = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    metadata.write_text(f"{revision}\n{etag}\n0\n")

    output = tmp_path / "transform-mismatch" / run_relative
    with pytest.raises(ValueError, match="transformed path set"):
        prepare_normalized_training_metadata(
            original_repository_root=original,
            model_repository_path=tmp_path / "model-repository",
            run_relative_path=run_relative,
            output_run_path=output,
        )
    assert not output.exists()
    assert not list(output.parent.glob(".training-run.tmp-*"))


def test_training_metadata_is_not_parsed_before_authentication(
    tmp_path,
    monkeypatch,
) -> None:
    loader_called = False

    monkeypatch.setattr(
        "genco_evidence_graph.cli.extract_huggingface_revision",
        lambda path: "a" * 40,
    )
    monkeypatch.setattr(
        "genco_evidence_graph.cli.verify_normalized_upstream_metadata",
        lambda *args: (_ for _ in ()).throw(ValueError("authentication failed")),
    )

    def record_loader_call(path):
        nonlocal loader_called
        loader_called = True
        pytest.fail("training metadata loader was called before authentication")

    monkeypatch.setattr(
        "genco_evidence_graph.cli.load_training_run_metadata",
        record_loader_call,
    )

    with pytest.raises(ValueError, match="authentication failed"):
        _load_verified_training_run(
            tmp_path / "training-run",
            tmp_path / "original-upstream",
            tmp_path / "model-repository",
        )
    assert not loader_called


def test_normalized_upstream_metadata_rejects_wrong_cache_revision(
    tmp_path,
) -> None:
    normalized, original, revision, _ = _training_upstream_fixture(tmp_path)
    metadata = (
        original
        / ".cache/huggingface/download"
        / "mlflow/train/case14_ieee/seed42/training-run/meta.yaml.metadata"
    )
    lines = metadata.read_text().splitlines()
    lines[0] = "b" * 40
    metadata.write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError, match="Hugging Face revision"):
        verify_normalized_upstream_metadata(
            normalized,
            original,
            revision,
        )


def test_normalized_upstream_metadata_rejects_duplicate_manifest_path(
    tmp_path,
) -> None:
    normalized, original, revision, manifest_path = (
        _training_upstream_fixture(tmp_path)
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["files"].append(dict(manifest["files"][0]))
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="Duplicate training metadata"):
        verify_normalized_upstream_metadata(
            normalized,
            original,
            revision,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "source_repository",
            "attacker/substituted-repository",
            "source repository",
        ),
        (
            "source_run_id",
            "substituted-run",
            "source run ID",
        ),
        (
            "source_revision",
            "b" * 40,
            "source revision",
        ),
    ],
)
def test_normalized_upstream_metadata_rejects_wrong_source_identity(
    tmp_path,
    field,
    value,
    message,
) -> None:
    normalized, original, revision, manifest_path = (
        _training_upstream_fixture(tmp_path)
    )
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=message):
        verify_normalized_upstream_metadata(
            normalized,
            original,
            revision,
        )
