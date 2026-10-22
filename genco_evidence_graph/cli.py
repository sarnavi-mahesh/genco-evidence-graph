"""Command-line interface for GENCO evidence graph."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .adapters import (
    extract_huggingface_revision,
    load_config_metadata,
    load_mlflow_run_metadata,
    load_training_run_metadata,
    verify_normalized_upstream_metadata,
    verify_run_matches_config,
)
from .baseline import (
    BaselineInferenceSpec,
    compare_baseline_residuals,
    compare_keyed_baseline,
    prepare_reference_prediction_splits,
    run_baseline_inference,
    verify_reproduction_manifest,
    verify_run_manifest,
)
from .contracts import (
    DatasetProvenance,
    MetricSemantics,
    ModelProvenance,
    PhysicalEvidence,
    PredictionEvidence,
    TrainingEvidenceAssociation,
)
from .embeddings import ReferenceIndex
from .provenance import sha256_file, sha256_path
from .rdf import build_evidence_graph, serialize_graph
from .splits import load_split_ids, verify_split_membership
from .validation import validate_evidence_graph
from .training_metadata import (
    load_training_metadata_manifest,
    prepare_normalized_training_metadata,
)


def _load_verified_training_run(
    training_run_path: Path,
    original_upstream_training_metadata_root: str | Path,
    model_repository_path: str | Path,
):
    model_repository_revision = extract_huggingface_revision(
        model_repository_path,
    )
    verify_normalized_upstream_metadata(
        training_run_path,
        original_upstream_training_metadata_root,
        model_repository_revision,
    )
    return (
        load_training_run_metadata(training_run_path),
        model_repository_revision,
    )


def _keyed_embedding_table(
    path: Path,
    expected_split: str,
) -> tuple[list[str], np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Embedding artifact does not exist: {path}")
    parquet = pq.ParquetFile(path)
    required = {
        "scenario_id",
        "entity_type",
        "entity_id",
        "correction_step",
        "embedding_stage",
        "embedding",
        "dataset_split",
    }
    if not required.issubset(parquet.schema_arrow.names):
        raise ValueError(
            f"{path} must contain keyed embedding columns: "
            f"{', '.join(sorted(required))}",
        )

    sums: dict[str, dict[str, np.ndarray]] = {"bus": {}, "generator": {}}
    counts: dict[str, dict[str, int]] = {"bus": {}, "generator": {}}
    completed_groups: set[tuple[str, str]] = set()
    current_group: tuple[str, str] | None = None
    current_entity_ids: set[int] = set()
    embedding_dimension: int | None = None

    def complete_group() -> None:
        nonlocal current_group, current_entity_ids
        if current_group is None:
            return
        if current_entity_ids != set(range(len(current_entity_ids))):
            raise ValueError(
                f"{path} has incomplete or non-contiguous entity IDs for "
                f"{current_group[0]} scenario {current_group[1]}",
            )
        completed_groups.add(current_group)

    for batch in parquet.iter_batches(
        columns=[
            "scenario_id",
            "entity_type",
            "entity_id",
            "correction_step",
            "embedding_stage",
            "embedding",
            "dataset_split",
        ],
        batch_size=50_000,
    ):
        scenario_ids = np.asarray(batch.column(0))
        entity_types = np.asarray(batch.column(1))
        entity_ids = np.asarray(batch.column(2))
        correction_steps = batch.column(3)
        embedding_stages = np.asarray(batch.column(4))
        embeddings = np.asarray(batch.column(5).to_pylist(), dtype=float)
        dataset_splits = np.asarray(batch.column(6))
        if embeddings.ndim != 2 or embeddings.shape[1] == 0:
            raise ValueError(f"{path} contains invalid embedding vectors")
        if embedding_dimension is None:
            embedding_dimension = embeddings.shape[1]
        elif embeddings.shape[1] != embedding_dimension:
            raise ValueError(f"{path} contains mixed embedding dimensions")
        if not np.isfinite(embeddings).all():
            raise ValueError(f"{path} contains non-finite embeddings")
        if set(dataset_splits) != {expected_split}:
            raise ValueError(
                f"{path} must contain only dataset_split={expected_split!r}",
            )
        if set(embedding_stages) != {"final"}:
            raise ValueError(f"{path} must contain only embedding_stage='final'")
        if correction_steps.null_count != batch.num_rows:
            raise ValueError(
                f"{path} final embeddings must have null correction_step",
            )
        invalid_types = set(entity_types) - {"bus", "generator"}
        if invalid_types:
            raise ValueError(f"{path} contains unsupported entity types: {invalid_types}")

        for scenario_id, entity_type, entity_id in zip(
            scenario_ids,
            entity_types,
            entity_ids,
        ):
            group = (str(entity_type), str(scenario_id))
            if group != current_group:
                complete_group()
                if group in completed_groups:
                    raise ValueError(
                        f"{path} contains non-contiguous duplicate entity group {group}",
                    )
                current_group = group
                current_entity_ids = set()
            numeric_entity_id = int(entity_id)
            if numeric_entity_id in current_entity_ids:
                raise ValueError(
                    f"{path} contains duplicate entity key "
                    f"({scenario_id}, {entity_type}, {numeric_entity_id})",
                )
            current_entity_ids.add(numeric_entity_id)

        for entity_type in ("bus", "generator"):
            mask = entity_types == entity_type
            for scenario_id in np.unique(scenario_ids[mask]):
                key = str(scenario_id)
                values = embeddings[mask & (scenario_ids == scenario_id)]
                if key in sums[entity_type]:
                    sums[entity_type][key] += values.sum(axis=0)
                    counts[entity_type][key] += len(values)
                else:
                    sums[entity_type][key] = values.sum(axis=0)
                    counts[entity_type][key] = len(values)
    complete_group()

    scenario_ids = sorted(set(sums["bus"]) | set(sums["generator"]))
    if set(sums["bus"]) != set(sums["generator"]):
        raise ValueError(f"Keyed embedding artifact has unaligned entity scenario IDs: {path}")
    pooled = np.stack(
        [
            np.concatenate(
                [
                    sums["bus"][scenario_id] / counts["bus"][scenario_id],
                    sums["generator"][scenario_id] / counts["generator"][scenario_id],
                ],
            )
            for scenario_id in scenario_ids
        ],
    )
    return scenario_ids, pooled


def _physical_evidence(
    prediction_path: Path,
    scenario_id: str,
    generator_prediction_path: Path | None = None,
) -> PhysicalEvidence:
    table = pd.read_parquet(prediction_path)
    required_bus_columns = {
        "scenario",
        "bus",
        "active res. (MW)",
        "reactive res. (MVar)",
        "PBE",
        "Vm_pred",
        "Vm_min",
        "Vm_max",
        "Qg_pred",
        "Qg_min",
        "Qg_max",
    }
    missing_bus_columns = sorted(required_bus_columns - set(table.columns))
    if missing_bus_columns:
        raise ValueError(
            f"{prediction_path} lacks required physical-evidence columns: "
            f"{missing_bus_columns}",
        )
    selected = table[table["scenario"].astype(str) == scenario_id]
    if selected.empty:
        raise ValueError(f"Scenario {scenario_id!r} is absent from {prediction_path}")
    active = selected["active res. (MW)"].abs()
    reactive = selected["reactive res. (MVar)"].abs()
    pbe = selected["PBE"].abs()
    max_pbe_bus = int(selected.loc[pbe.idxmax(), "bus"])
    voltage_violations = (
        (selected["Vm_pred"] < selected["Vm_min"] - 1e-6)
        | (selected["Vm_pred"] > selected["Vm_max"] + 1e-6)
    ).sum()
    reactive_violations = (
        (selected["Qg_pred"] < selected["Qg_min"] - 1e-6)
        | (selected["Qg_pred"] > selected["Qg_max"] + 1e-6)
    ).sum()
    if generator_prediction_path is None:
        raise ValueError(
            "Generator predictions are required for active-power violation evidence",
        )
    generator_table = pd.read_parquet(generator_prediction_path)
    required_generator_columns = {
        "scenario",
        "idx",
        "p_mw_pred",
        "min_p_mw",
        "max_p_mw",
    }
    missing_generator_columns = sorted(
        required_generator_columns - set(generator_table.columns),
    )
    if missing_generator_columns:
        raise ValueError(
            f"{generator_prediction_path} lacks required physical-evidence "
            f"columns: {missing_generator_columns}",
        )
    generator_selected = generator_table[
        generator_table["scenario"].astype(str) == scenario_id
    ]
    if generator_selected.empty:
        raise ValueError(
            f"Scenario {scenario_id!r} is absent from {generator_prediction_path}",
        )
    active_limit_violations = (
        (
            generator_selected["p_mw_pred"]
            < generator_selected["min_p_mw"] - 1e-6
        )
        | (
            generator_selected["p_mw_pred"]
            > generator_selected["max_p_mw"] + 1e-6
        )
    ).sum()
    available_violations = [
        "voltage_limits",
        "reactive_limits",
        "active_power_limits",
    ]
    return PhysicalEvidence(
        mean_residual=float(pbe.mean()),
        max_residual=float(pbe.max()),
        active_residual=float(active.mean()),
        reactive_residual=float(reactive.mean()),
        violation_count=int(
            voltage_violations + reactive_violations + active_limit_violations,
        ),
        metric_semantics=(
            MetricSemantics(
                "mean_pbe",
                "MVA",
                "magnitude",
                "mean",
                "bus rows within one scenario",
            ),
            MetricSemantics(
                "max_pbe",
                "MVA",
                "magnitude",
                "maximum",
                "bus rows within one scenario",
            ),
            MetricSemantics(
                "mean_active_residual",
                "MW",
                "absolute",
                "mean",
                "bus rows within one scenario",
            ),
            MetricSemantics(
                "mean_reactive_residual",
                "MVar",
                "absolute",
                "mean",
                "bus rows within one scenario",
            ),
            MetricSemantics(
                "total_violation_count",
                "count",
                "count rows outside declared limits",
                "sum",
                "all supported bus and generator limit checks within one scenario",
                "component-specific: 1e-6 p.u., MW, or MVar",
            ),
            MetricSemantics(
                "voltage_violation_count",
                "count",
                "count rows outside inclusive voltage limits",
                "sum",
                "bus rows within one scenario",
                "1e-6 p.u.",
            ),
            MetricSemantics(
                "reactive_power_violation_count",
                "count",
                "count rows outside inclusive reactive-power limits",
                "sum",
                "bus rows within one scenario",
                "1e-6 MVar",
            ),
            MetricSemantics(
                "active_power_violation_count",
                "count",
                "count rows outside inclusive active-power limits",
                "sum",
                "generator rows within one scenario",
                "1e-6 MW",
            ),
        ),
        extra={
            "max_pbe_bus": max_pbe_bus,
            "voltage_violation_count": int(voltage_violations),
            "reactive_violation_count": int(reactive_violations),
            "active_power_violation_count": int(active_limit_violations),
            "available_residual_types": "active_residual,reactive_residual",
            "available_violation_types": ",".join(available_violations),
        },
    )



def _comparison_maxima(
    differences: dict[str, float],
) -> tuple[float, float]:
    voltage_fields = {"bus.Vm_pred", "bus.Va_pred"}
    voltage = [
        value
        for field, value in differences.items()
        if field in voltage_fields
    ]
    power = [
        value
        for field, value in differences.items()
        if field not in voltage_fields
    ]
    if not voltage:
        raise ValueError("Baseline comparison produced no voltage differences")
    if not power:
        raise ValueError("Baseline comparison produced no power differences")
    return max(voltage), max(power)


def _build_evidence(args: argparse.Namespace) -> int:
    working_directory = Path.cwd().resolve()

    def portable_path(path: str | Path) -> str:
        source = Path(path)
        resolved = source.resolve()
        try:
            return resolved.relative_to(working_directory).as_posix()
        except ValueError:
            return source.as_posix()

    prediction_path = Path(args.prediction_parquet)
    generator_prediction_path = Path(args.generator_prediction_parquet)
    target_path = Path(args.target_keyed_embeddings)
    config_path = Path(args.config_artifact)
    raw_dataset_path = Path(args.raw_dataset_path)
    manifest = verify_run_manifest(
        args.run_manifest,
        prediction_path=prediction_path,
        generator_prediction_path=generator_prediction_path,
        target_embedding_path=target_path,
        checkpoint_path=args.checkpoint,
        normalizer_path=args.normalizer_artifact,
        config_path=config_path,
        split_path=args.split_artifact,
        raw_dataset_path=raw_dataset_path,
    )
    reproduction_manifest_path = Path(args.reproduction_run_manifest)
    reproduction_value = json.loads(reproduction_manifest_path.read_text())
    reproduction_artifacts = {
        record["path"]: reproduction_manifest_path.parent / record["path"]
        for record in reproduction_value.get("artifacts", ())
    }
    try:
        reproduction_prediction_path = next(
            path
            for relative, path in reproduction_artifacts.items()
            if Path(relative).name == "case14_ieee_predictions.parquet"
        )
        reproduction_generator_path = next(
            path
            for relative, path in reproduction_artifacts.items()
            if Path(relative).name == "case14_ieee_gen_predictions.parquet"
        )
    except StopIteration as error:
        raise ValueError(
            "Reproduction manifest lacks bus or generator predictions",
        ) from error
    reproduction_manifest = verify_reproduction_manifest(
        reproduction_manifest_path,
        prediction_path=reproduction_prediction_path,
        generator_prediction_path=reproduction_generator_path,
        checkpoint_path=args.checkpoint,
        normalizer_path=args.normalizer_artifact,
        config_path=config_path,
        split_path=args.split_artifact,
        raw_dataset_path=raw_dataset_path,
    )
    mlflow_run_path = manifest.mlflow_run_dir
    config_metadata = load_config_metadata(config_path)
    run_metadata = load_mlflow_run_metadata(mlflow_run_path)
    verify_run_matches_config(run_metadata, config_metadata)
    dataset_revision = extract_huggingface_revision(raw_dataset_path)
    if dataset_revision != manifest.dataset_revision:
        raise ValueError("Dataset revision disagrees with run manifest")
    if reproduction_manifest.dataset_revision != dataset_revision:
        raise ValueError("Reproduction dataset revision disagrees with target run")
    expected_values = {
        "dataset revision": (dataset_revision, args.expected_dataset_revision),
        "graphkit commit": (
            run_metadata.graphkit_commit,
            args.expected_graphkit_commit,
        ),
        "MLflow run ID": (run_metadata.run_id, args.expected_mlflow_run_id),
    }
    for label, (actual, expected) in expected_values.items():
        if expected is not None and actual != expected:
            raise ValueError(
                f"Extracted {label} {actual!r} does not match expected {expected!r}",
            )
    if reproduction_manifest.graphkit_commit != run_metadata.graphkit_commit:
        raise ValueError("Reproduction graphkit commit disagrees with target run")
    tolerance_config_path = Path(args.baseline_tolerance_config)
    comparison = compare_keyed_baseline(
        prediction_path=reproduction_prediction_path,
        generator_prediction_path=reproduction_generator_path,
        expected_prediction_path=prediction_path,
        expected_generator_prediction_path=generator_prediction_path,
        actual_run_manifest=reproduction_manifest_path,
        expected_run_manifest=args.run_manifest,
        tolerance_config=tolerance_config_path,
    )
    maximum_voltage_difference, maximum_power_difference = _comparison_maxima(
        comparison.maximum_absolute_differences,
    )

    target_ids, target_matrix = _keyed_embedding_table(target_path, "test")
    verify_split_membership(
        target_ids,
        args.split_artifact,
        "Target embeddings",
    )
    if args.prediction_id not in target_ids:
        raise ValueError(
            f"Scenario {args.prediction_id!r} is absent from {target_path}",
        )
    query_vector = target_matrix[target_ids.index(args.prediction_id)]
    reference_path = Path(args.reference_keyed_embeddings)
    reference_manifest_path = Path(args.reference_run_manifest)
    reference_manifest_value = json.loads(reference_manifest_path.read_text())
    if (
        reference_manifest_value.get("dataset_split") != "train"
        or reference_manifest_value.get("export_embeddings") is not True
    ):
        raise ValueError(
            "Reference run manifest must describe a train-split embedding run",
        )
    reference_manifest = verify_run_manifest(
        reference_manifest_path,
        prediction_path=args.reference_prediction_parquet,
        generator_prediction_path=args.reference_generator_prediction_parquet,
        target_embedding_path=reference_path,
        checkpoint_path=args.checkpoint,
        normalizer_path=args.normalizer_artifact,
        config_path=args.reference_config_artifact,
        split_path=args.reference_split_artifact,
        raw_dataset_path=args.raw_dataset_path,
    )
    if reference_manifest.graphkit_commit != run_metadata.graphkit_commit:
        raise ValueError(
            "Target and reference runs used different graphkit commits",
        )
    if reference_manifest.dataset_revision != dataset_revision:
        raise ValueError(
            "Target and reference runs used different dataset revisions",
        )
    training_run_path = Path(args.training_run_metadata)
    training_run, model_repository_revision = _load_verified_training_run(
        training_run_path,
        args.original_upstream_training_metadata_root,
        args.model_repository_path,
    )
    if (
        training_run.seed != config_metadata.seed
        or training_run.grid_name != config_metadata.grid_name
        or training_run.task != config_metadata.task
        or training_run.corpus_scenario_count
        != config_metadata.corpus_scenario_count
    ):
        raise ValueError("Authoritative training run disagrees with model configuration")
    training_split_ids = verify_split_membership(
        json.loads(
            (
                training_run_path
                / "artifacts/stats/case14_ieee_scenario_splits.json"
            ).read_text(),
        )["train"],
        args.reference_split_artifact,
        "Checkpoint training split",
    )
    reference_ids, reference_matrix = _keyed_embedding_table(reference_path, "train")
    verify_split_membership(
        reference_ids,
        args.reference_split_artifact,
        "Reference embeddings",
    )
    overlap = set(target_ids) & set(reference_ids)
    if overlap:
        raise ValueError(
            f"Target and reference embedding artifacts overlap in {len(overlap)} scenarios",
        )
    index = ReferenceIndex(reference_ids, reference_matrix)
    reference_cases = index.query(query_vector, top_k=args.top_k)
    prediction_ids = {
        str(value)
        for value in pd.read_parquet(prediction_path, columns=["scenario"])[
            "scenario"
        ].unique()
    }
    if prediction_ids != set(target_ids):
        raise ValueError(
            "Prediction and target embedding artifacts contain different scenario IDs",
        )
    verify_split_membership(
        prediction_ids,
        args.split_artifact,
        "Bus predictions",
    )
    generator_prediction_ids = {
        str(value)
        for value in pd.read_parquet(
            generator_prediction_path,
            columns=["scenario"],
        )["scenario"].unique()
    }
    if generator_prediction_ids != set(target_ids):
        raise ValueError(
            "Generator prediction and target embedding artifacts contain "
            "different scenario IDs",
        )
    verify_split_membership(
        generator_prediction_ids,
        args.split_artifact,
        "Generator predictions",
    )
    physical = _physical_evidence(
        prediction_path,
        args.prediction_id,
        generator_prediction_path,
    )

    artifact_paths = [prediction_path, target_path, reference_path]
    artifact_paths.append(generator_prediction_path)
    prediction_paths = {prediction_path}
    prediction_paths.add(generator_prediction_path)
    artifact_records = []
    for path in dict.fromkeys(artifact_paths):
        if path in prediction_paths:
            kind = "prediction"
        elif path == target_path:
            kind = "target_embedding"
        elif path == reference_path:
            kind = "reference_embedding"
        else:
            raise ValueError(f"Unclassified evidence artifact: {path}")
        artifact_records.append(
            {"path": portable_path(path), "sha256": sha256_path(path), "kind": kind},
        )
    provenance_paths = [
        raw_dataset_path,
        Path(args.split_artifact),
        Path(args.normalizer_artifact),
        config_path,
        mlflow_run_path,
        Path(args.run_manifest),
        Path(args.reference_run_manifest),
        reference_manifest.mlflow_run_dir,
        training_run_path,
        reproduction_manifest_path,
        reproduction_manifest.mlflow_run_dir,
        reproduction_manifest_path.parent
        / reproduction_manifest.runtime["environment_path"],
        tolerance_config_path,
    ]
    provenance_records = [
        {
            "path": portable_path(path),
            "sha256": sha256_path(path),
            "kind": "provenance",
        }
        for path in dict.fromkeys(Path(path) for path in provenance_paths)
    ]
    checkpoint_hash = sha256_file(args.checkpoint)
    checkpoint_id = (
        f"{config_metadata.grid_name}_seed{config_metadata.seed}_"
        f"{checkpoint_hash[:12]}"
    )
    dataset = DatasetProvenance(
        dataset_id=f"{args.dataset_id}:test",
        grid_name=config_metadata.grid_name,
        task=config_metadata.task,
        corpus_scenario_count=config_metadata.corpus_scenario_count,
        dataset_split="test",
        split_scenario_count=len(target_ids),
        artifact_records=tuple(provenance_records),
        dataset_revision=dataset_revision,
        software_version=(
            "Julia + PowerModels.jl + Ipopt + MUMPS; converted with "
            "gridfm-datakit:genco-paper-repro"
        ),
        config_path=portable_path(raw_dataset_path / "README.md"),
        config_hash=sha256_file(raw_dataset_path / "README.md"),
        generation_method=(
            "OPFData converted with gridfm-datakit "
            "opf_data/batch_convert.py on genco-paper-repro"
        ),
        generation_seed_status=(
            "upstream OPFData scenario-generation seed and seed schedule "
            "were not published"
        ),
        perturbations=(
            "independently multiply every active and reactive load by "
            "Uniform[0.8,1.2]",
            "with probability 0.5 remove one uniformly selected eligible "
            "generator; otherwise remove one eligible line or transformer",
            "exclude reference-bus generators and removals that disconnect "
            "the graph; remove selected components entirely",
            "discard infeasible AC-OPF instances and retain 300000 solved "
            "N-1 scenarios",
        ),
        provenance_limitations=(
            "original OPFData generator repository and commit were not published",
            "Julia, PowerModels.jl, Ipopt, MUMPS, and PGLib versions were not published",
            "exact gridfm-datakit conversion commit was not published; only "
            "branch genco-paper-repro is identified",
        ),
    )
    model = ModelProvenance(
        training_run_id=training_run.run_id,
        checkpoint_id=checkpoint_id,
        dataset_id=f"{args.dataset_id}:test",
        model_variant=config_metadata.model_variant,
        hidden_dim=config_metadata.hidden_dimension,
        correction_steps=None,
        checkpoint_hash=checkpoint_hash,
        normalizer_hash=(
            sha256_file(args.normalizer_artifact)
            if args.normalizer_artifact
            else None
        ),
        config_hash=(
            sha256_file(args.config_artifact)
            if args.config_artifact
            else None
        ),
        graphkit_commit=None,
        mlflow_run_id=training_run.run_id,
        training_dataset_id=f"{args.dataset_id}:train",
        training_split_scenario_count=len(training_split_ids),
        training_split_hash=sha256_file(args.reference_split_artifact),
        training_start_time_ms=training_run.start_time_ms,
        training_end_time_ms=training_run.end_time_ms,
        training_seed=training_run.seed,
        training_epochs=training_run.epochs,
        training_batch_size=training_run.batch_size,
        training_learning_rate=training_run.learning_rate,
        training_optimizer=training_run.optimizer,
        training_selected_metrics=training_run.selected_metrics,
        training_software_revision_status=(
            training_run.software_revision_status
        ),
        checkpoint_binding_status=training_run.checkpoint_binding_status,
        training_params_hash=training_run.parameters_hash,
        training_metrics_hash=training_run.metrics_hash,
        model_repository_revision=model_repository_revision,
        checkpoint_repository_path=portable_path(args.checkpoint),
        training_evidence_association=TrainingEvidenceAssociation(
            association_status=(
                "co-published-compatible-unverified-production"
            ),
            evidence_basis=(
                "checkpoint and training metadata were co-published under "
                "case14_ieee/seed42 at the pinned model-repository revision",
                "training configuration is compatible with the published "
                "checkpoint configuration: grid, task, seed, and architecture",
                "MLflow parameters and metrics are authenticated independently "
                "by hashes of the published training-run metadata",
            ),
            production_identity_verified=False,
            configuration_compatible=True,
            provenance_limitations=(
                "the MLflow training run does not record the published "
                "checkpoint SHA-256",
                "the historical training software commit is unavailable",
            ),
        ),
    )
    evidence = PredictionEvidence(
        prediction_id=args.prediction_id,
        provenance_summary={
            "source": "gridfm_graphkit",
            "artifact_records": tuple(artifact_records + provenance_records),
        },
        reference_cases=tuple(
            {
                "scenario_id": match.scenario_id,
                "distance": match.distance,
                "rank": match.rank,
            }
            for match in reference_cases
        ),
        reference_distance=reference_cases[0].distance if reference_cases else None,
        reference_distance_percentile=None,
        physical_evidence=physical,
        inference_run_id=run_metadata.run_id,
        inference_start_time_ms=run_metadata.start_time_ms,
        inference_end_time_ms=run_metadata.end_time_ms,
        inference_graphkit_commit=run_metadata.graphkit_commit,
        reference_extraction_run_id=load_mlflow_run_metadata(
            reference_manifest.mlflow_run_dir,
        ).run_id,
        reference_extraction_start_time_ms=load_mlflow_run_metadata(
            reference_manifest.mlflow_run_dir,
        ).start_time_ms,
        reference_extraction_end_time_ms=load_mlflow_run_metadata(
            reference_manifest.mlflow_run_dir,
        ).end_time_ms,
        reference_extraction_graphkit_commit=reference_manifest.graphkit_commit,
        reproduction_run_id=reproduction_manifest.run_id,
        reproduction_start_time_ms=reproduction_manifest.start_time_ms,
        reproduction_end_time_ms=reproduction_manifest.end_time_ms,
        reproduction_graphkit_commit=reproduction_manifest.graphkit_commit,
        reproduction_environment_hash=(
            reproduction_manifest.runtime_environment_sha256
        ),
        reproduction_runtime=reproduction_manifest.runtime,
        reproduction_comparison_status="passed",
        reproduction_tolerance_policy_id=comparison.tolerance_policy_id,
        reproduction_tolerance_policy_hash=comparison.tolerance_policy_sha256,
        reproduction_maximum_voltage_difference=maximum_voltage_difference,
        reproduction_maximum_power_difference=maximum_power_difference,
    )
    graph = build_evidence_graph(dataset, model, evidence)
    serialize_graph(graph, args.output)
    print(f"Serialized {len(graph)} triples to {args.output}")
    return 0


def _query(args: argparse.Namespace) -> int:
    from rdflib import Graph
    from .queries import NAMED_QUERIES, QUERY_HEADERS, high_residual_and_distance

    graph = Graph()
    graph.parse(args.graph, format="turtle")
    if args.named == "high-residual-distance":
        if args.residual_threshold is None or args.distance_threshold is None:
            raise ValueError(
                "high-residual-distance requires --residual-threshold "
                "and --distance-threshold",
            )
        sparql = high_residual_and_distance(
            args.residual_threshold,
            args.distance_threshold,
        )
    elif args.named:
        sparql = NAMED_QUERIES[args.named]
    else:
        sparql = args.sparql
    results = graph.query(sparql)
    if not args.no_header:
        headers = (
            QUERY_HEADERS[args.named]
            if args.named
            else tuple(str(variable) for variable in results.vars)
        )
        print("\t".join(headers))
    for row in results:
        print("\t".join("" if value is None else str(value) for value in row))
    return 0


def _validate(args: argparse.Namespace) -> int:
    from rdflib import Graph

    graph = Graph()
    graph.parse(args.graph, format="turtle")
    validate_evidence_graph(graph, args.shapes)
    print(f"SHACL validation passed for {args.graph}")
    return 0


def _run_baseline(args: argparse.Namespace) -> int:
    result = run_baseline_inference(
        BaselineInferenceSpec(
            graphkit_python=Path(args.graphkit_python),
            graphkit_workdir=Path(args.graphkit_workdir),
            config=Path(args.config),
            checkpoint=Path(args.checkpoint),
            normalizer=Path(args.normalizer),
            data_path=Path(args.data_path),
            split_artifact=Path(args.split_artifact),
            output_dir=Path(args.output_dir),
            run_name=args.run_name,
            graphkit_requirements_lock=Path(args.graphkit_requirements_lock),
            graphkit_source_record=Path(args.graphkit_source_record),
            experiment_name=args.experiment_name,
            num_workers=args.num_workers,
            reuse_verified_run=args.reuse_verified_run,
            dataset_split=args.dataset_split,
            export_embeddings=not args.no_embeddings,
        ),
    )
    print(f"Baseline run {result.run_id} completed")
    print(f"Manifest: {result.manifest_path}")
    return 0


def _compare_baseline(args: argparse.Namespace) -> int:
    keyed = compare_keyed_baseline(
        prediction_path=args.prediction_parquet,
        generator_prediction_path=args.generator_prediction_parquet,
        expected_prediction_path=args.expected_prediction_parquet,
        expected_generator_prediction_path=(
            args.expected_generator_prediction_parquet
        ),
        actual_run_manifest=args.actual_run_manifest,
        expected_run_manifest=args.expected_run_manifest,
        tolerance_config=args.tolerance_config,
    )
    print(
        "Keyed baseline comparison passed for "
        f"{keyed.bus_row_count} bus rows and "
        f"{keyed.generator_row_count} generator rows",
    )
    print(
        "Tolerance policy "
        f"{keyed.tolerance_policy_id}\tsha256={keyed.tolerance_policy_sha256}",
    )
    for field, difference in sorted(keyed.maximum_absolute_differences.items()):
        print(f"{field}\tmax_abs_difference={difference:.10g}")
    if args.expected_metrics:
        result = compare_baseline_residuals(
            args.prediction_parquet,
            args.expected_metrics,
            args.absolute_tolerance,
        )
        print(
            "Supplementary aggregate residual comparison passed at absolute "
            f"tolerance {result.absolute_tolerance:g}",
        )
    return 0


def _prepare_reference_split(args: argparse.Namespace) -> int:
    output = prepare_reference_prediction_splits(
        train_split=args.train_split,
        validation_split=args.validation_split,
        test_split=args.test_split,
        output_dir=args.output_dir,
    )
    print(f"Reference prediction split view: {output}")
    return 0


def _list_scenarios(args: argparse.Namespace) -> int:
    scenario_ids = load_split_ids(args.split_artifact)
    if not args.no_header:
        print("scenario_id")
    print("\n".join(scenario_ids))
    return 0


def _prepare_training_metadata(args: argparse.Namespace) -> int:
    manifest_path = prepare_normalized_training_metadata(
        original_repository_root=args.original_upstream_root,
        model_repository_path=args.model_repository_path,
        run_relative_path=args.run_relative_path,
        output_run_path=args.output_run_path,
    )
    manifest = load_training_metadata_manifest(manifest_path)
    print("Normalized training metadata prepared")
    print(f"Manifest: {manifest_path}")
    print(f"Source revision: {manifest.source_revision}")
    print(f"Files: {len(manifest.files)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local GENCO Evidence Graph CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evidence = subparsers.add_parser("build-evidence", help="Build one prediction evidence graph")
    evidence.add_argument("--dataset-id", required=True)
    evidence.add_argument("--checkpoint", required=True)
    evidence.add_argument("--prediction-id", required=True)
    evidence.add_argument("--prediction-parquet", required=True)
    evidence.add_argument("--generator-prediction-parquet", required=True)
    evidence.add_argument("--target-keyed-embeddings", required=True)
    evidence.add_argument("--reference-keyed-embeddings", required=True)
    evidence.add_argument("--reference-prediction-parquet", required=True)
    evidence.add_argument(
        "--reference-generator-prediction-parquet",
        required=True,
    )
    evidence.add_argument("--reference-run-manifest", required=True)
    evidence.add_argument("--reference-split-artifact", required=True)
    evidence.add_argument("--reference-config-artifact", required=True)
    evidence.add_argument("--training-run-metadata", required=True)
    evidence.add_argument(
        "--original-upstream-training-metadata-root",
        "--pristine-training-metadata-root",
        dest="original_upstream_training_metadata_root",
        required=True,
        help=(
            "Root of the untouched original upstream training metadata from "
            "the pinned model-repository revision."
        ),
    )
    evidence.add_argument("--model-repository-path", required=True)
    evidence.add_argument("--raw-dataset-path", required=True)
    evidence.add_argument("--split-artifact", required=True)
    evidence.add_argument("--normalizer-artifact", required=True)
    evidence.add_argument("--config-artifact", required=True)
    evidence.add_argument("--run-manifest", required=True)
    evidence.add_argument("--reproduction-run-manifest", required=True)
    evidence.add_argument("--baseline-tolerance-config", required=True)
    evidence.add_argument("--expected-dataset-revision")
    evidence.add_argument("--expected-graphkit-commit")
    evidence.add_argument("--expected-mlflow-run-id")
    evidence.add_argument("--top-k", type=int, default=5)
    evidence.add_argument("--output", required=True)
    evidence.set_defaults(handler=_build_evidence)

    baseline = subparsers.add_parser(
        "run-baseline",
        help="Run graphkit prediction, export final embeddings, and write a manifest",
    )
    baseline.add_argument("--graphkit-python", required=True)
    baseline.add_argument("--graphkit-workdir", required=True)
    baseline.add_argument("--config", required=True)
    baseline.add_argument("--checkpoint", required=True)
    baseline.add_argument("--normalizer", required=True)
    baseline.add_argument("--data-path", required=True)
    baseline.add_argument("--split-artifact", required=True)
    baseline.add_argument("--output-dir", required=True)
    baseline.add_argument("--run-name", required=True)
    baseline.add_argument("--graphkit-requirements-lock", required=True)
    baseline.add_argument("--graphkit-source-record", required=True)
    baseline.add_argument(
        "--experiment-name",
        default="genco-evidence-baseline",
    )
    baseline.add_argument("--num-workers", type=int, default=0)
    baseline.add_argument(
        "--dataset-split",
        choices=("test", "train"),
        default="test",
    )
    baseline.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Generate the frozen uninstrumented prediction baseline only.",
    )
    baseline.add_argument(
        "--reuse-verified-run",
        "--resume",
        dest="reuse_verified_run",
        action="store_true",
        help=(
            "Verify and reuse an existing completed embedding run without "
            "rerunning GraphKit inference. --resume is a compatibility alias."
        ),
    )
    baseline.set_defaults(handler=_run_baseline)

    comparison = subparsers.add_parser(
        "compare-baseline",
        help="Compare coherent predictions with published aggregate residuals.",
    )
    comparison.add_argument("--prediction-parquet", required=True)
    comparison.add_argument("--generator-prediction-parquet", required=True)
    comparison.add_argument("--expected-prediction-parquet", required=True)
    comparison.add_argument(
        "--expected-generator-prediction-parquet",
        required=True,
    )
    comparison.add_argument("--actual-run-manifest", required=True)
    comparison.add_argument("--expected-run-manifest", required=True)
    comparison.add_argument("--tolerance-config", required=True)
    comparison.add_argument("--expected-metrics")
    comparison.add_argument("--absolute-tolerance", type=float, default=1e-6)
    comparison.set_defaults(handler=_compare_baseline)

    split_view = subparsers.add_parser(
        "prepare-reference-split",
        help="Create a graphkit split view that predicts the published train split.",
    )
    split_view.add_argument("--train-split", required=True)
    split_view.add_argument("--validation-split", required=True)
    split_view.add_argument("--test-split", required=True)
    split_view.add_argument("--output-dir", required=True)
    split_view.set_defaults(handler=_prepare_reference_split)

    scenario_list = subparsers.add_parser(
        "list-scenarios",
        help="List authoritative scenario IDs from a split artifact",
    )
    scenario_list.add_argument("--split-artifact", required=True)
    scenario_list.add_argument(
        "--no-header",
        action="store_true",
        help="Omit the scenario_id header for scripting.",
    )
    scenario_list.set_defaults(handler=_list_scenarios)

    training_metadata = subparsers.add_parser(
        "prepare-training-metadata",
        help="Authenticate and normalize pinned upstream MLflow training metadata",
    )
    training_metadata.add_argument("--original-upstream-root", required=True)
    training_metadata.add_argument("--model-repository-path", required=True)
    training_metadata.add_argument("--run-relative-path", required=True)
    training_metadata.add_argument("--output-run-path", required=True)
    training_metadata.set_defaults(handler=_prepare_training_metadata)

    query = subparsers.add_parser("query", help="Run a SPARQL query against local Turtle")
    query.add_argument("--graph", required=True)
    query_source = query.add_mutually_exclusive_group(required=True)
    query_source.add_argument("--sparql")
    query_source.add_argument(
        "--named",
        choices=[
            "associated-training-evidence",
            "checkpoint",
            "reference-cases",
            "embedding-distance",
            "residuals",
            "violations",
            "available-violation-types",
            "available-residual-types",
            "reproduction",
            "high-residual-distance",
        ],
    )
    query.add_argument("--residual-threshold", type=float)
    query.add_argument("--distance-threshold", type=float)
    query.add_argument(
        "--no-header",
        action="store_true",
        help="Omit the descriptive TSV header row for scripting.",
    )
    query.set_defaults(handler=_query)

    validation = subparsers.add_parser(
        "validate",
        help="Validate a local Turtle graph with SHACL",
    )
    validation.add_argument("--graph", required=True)
    validation.add_argument("--shapes", required=True)
    validation.set_defaults(handler=_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except BrokenPipeError:
        return 0
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
