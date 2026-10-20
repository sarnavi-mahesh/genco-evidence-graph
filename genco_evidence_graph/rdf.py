"""RDF/PROV-O graph construction for normalized GENCO evidence."""

from pathlib import Path
from typing import Any

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import PROV, RDF, XSD

from .contracts import (
    DatasetProvenance,
    ModelProvenance,
    PhysicalEvidence,
    PredictionEvidence,
)
from .provenance import stable_uri

GENCO = Namespace("https://genco.example/ontology#")


def _resource(kind: str, identifier: str) -> URIRef:
    return URIRef(stable_uri(kind, identifier))


def _add_literal(
    graph: Graph,
    subject: URIRef,
    predicate: URIRef,
    value: Any,
) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        literal = Literal(value, datatype=XSD.boolean)
    elif isinstance(value, int):
        literal = Literal(value, datatype=XSD.integer)
    elif isinstance(value, float):
        literal = Literal(value, datatype=XSD.double)
    else:
        literal = Literal(str(value))
    graph.add((subject, predicate, literal))


def _add_physical_evidence(
    graph: Graph,
    prediction: URIRef,
    evidence_id: str,
    evidence: PhysicalEvidence,
) -> URIRef:
    resource = _resource("physical-evidence", evidence_id)
    graph.add((resource, RDF.type, GENCO.PhysicalEvidence))
    graph.add((prediction, GENCO.hasPhysicalEvidence, resource))
    _add_literal(graph, resource, GENCO.meanResidual, evidence.mean_residual)
    _add_literal(graph, resource, GENCO.maxResidual, evidence.max_residual)
    _add_literal(graph, resource, GENCO.activeResidual, evidence.active_residual)
    _add_literal(graph, resource, GENCO.reactiveResidual, evidence.reactive_residual)
    _add_literal(graph, resource, GENCO.violationCount, evidence.violation_count)
    for key, value in sorted(evidence.extra.items()):
        _add_literal(graph, resource, GENCO[key], value)
    for semantics in evidence.metric_semantics:
        metric = _resource(
            "metric-semantics",
            f"{evidence_id}:{semantics.metric_name}",
        )
        graph.add((metric, RDF.type, GENCO.MetricSemantics))
        graph.add((resource, GENCO.hasMetricSemantics, metric))
        _add_literal(graph, metric, GENCO.metricName, semantics.metric_name)
        _add_literal(graph, metric, GENCO.unit, semantics.unit)
        _add_literal(graph, metric, GENCO.treatment, semantics.treatment)
        _add_literal(graph, metric, GENCO.aggregation, semantics.aggregation)
        _add_literal(graph, metric, GENCO.population, semantics.population)
        _add_literal(
            graph,
            metric,
            GENCO.comparisonTolerance,
            semantics.comparison_tolerance,
        )
    return resource


def build_evidence_graph(
    dataset: DatasetProvenance,
    model: ModelProvenance,
    prediction: PredictionEvidence,
) -> Graph:
    """Build a deterministic graph for one normalized prediction."""
    graph = Graph()
    graph.bind("genco", GENCO)
    graph.bind("prov", PROV)

    grid = _resource("grid-model", dataset.grid_name)
    dataset_uri = _resource("dataset", dataset.dataset_id)
    training_dataset = _resource(
        "dataset",
        model.training_dataset_id or f"{model.dataset_id}:train",
    )
    data_generation = _resource("data-generation-run", dataset.dataset_id)
    training = _resource("training-run", model.training_run_id)
    checkpoint = _resource("model-checkpoint", model.checkpoint_id)
    inference = _resource(
        "inference-run",
        prediction.inference_run_id or prediction.prediction_id,
    )
    reference_extraction = _resource(
        "reference-extraction-run",
        prediction.reference_extraction_run_id or prediction.prediction_id,
    )
    reproduction = (
        _resource("reproduction-run", prediction.reproduction_run_id)
        if prediction.reproduction_run_id is not None
        else None
    )
    scenario = _resource("scenario", prediction.prediction_id)
    prediction_uri = _resource("prediction", prediction.prediction_id)
    training_association = (
        _resource(
            "training-evidence-association",
            f"{model.checkpoint_id}:{model.training_run_id}",
        )
        if model.training_evidence_association is not None
        else None
    )
    training_attestation = (
        _resource(
            "training-attestation",
            model.authoritative_training_attestation.attestation_id,
        )
        if model.authoritative_training_attestation is not None
        else None
    )

    graph.add((grid, RDF.type, GENCO.GridModel))
    graph.add((dataset_uri, RDF.type, PROV.Entity))
    graph.add((dataset_uri, RDF.type, GENCO.Dataset))
    graph.add((training_dataset, RDF.type, PROV.Entity))
    graph.add((training_dataset, RDF.type, GENCO.Dataset))
    graph.add((data_generation, RDF.type, PROV.Activity))
    graph.add((data_generation, RDF.type, GENCO.DataGenerationRun))
    graph.add((training, RDF.type, PROV.Activity))
    graph.add((training, RDF.type, GENCO.TrainingRun))
    if training_association is not None:
        graph.add(
            (training_association, RDF.type, GENCO.TrainingEvidenceAssociation),
        )
    if training_attestation is not None:
        graph.add((training_attestation, RDF.type, PROV.Entity))
        graph.add(
            (training_attestation, RDF.type, GENCO.AuthoritativeTrainingAttestation),
        )
    graph.add((checkpoint, RDF.type, PROV.Entity))
    graph.add((checkpoint, RDF.type, GENCO.ModelCheckpoint))
    graph.add((inference, RDF.type, PROV.Activity))
    graph.add((inference, RDF.type, GENCO.InferenceRun))
    graph.add((reference_extraction, RDF.type, PROV.Activity))
    graph.add((reference_extraction, RDF.type, GENCO.ReferenceExtractionRun))
    if reproduction is not None:
        graph.add((reproduction, RDF.type, PROV.Activity))
        graph.add((reproduction, RDF.type, GENCO.ReproductionRun))
    graph.add((scenario, RDF.type, PROV.Entity))
    graph.add((scenario, RDF.type, GENCO.Scenario))
    graph.add((prediction_uri, RDF.type, PROV.Entity))
    graph.add((prediction_uri, RDF.type, GENCO.Prediction))

    graph.add((dataset_uri, PROV.wasGeneratedBy, data_generation))
    graph.add((training_dataset, PROV.wasGeneratedBy, data_generation))
    graph.add((dataset_uri, PROV.wasDerivedFrom, grid))
    graph.add((training_dataset, PROV.wasDerivedFrom, grid))
    graph.add((data_generation, PROV.used, grid))
    graph.add((training, PROV.used, training_dataset))
    if training_attestation is not None:
        graph.add((checkpoint, PROV.wasGeneratedBy, training))
        graph.add(
            (
                checkpoint,
                GENCO.authoritativeTrainingAttestation,
                training_attestation,
            ),
        )
        graph.add(
            (training_attestation, GENCO.attestsCheckpoint, checkpoint),
        )
        graph.add(
            (training_attestation, GENCO.attestsTrainingRun, training),
        )
    if training_association is not None:
        graph.add(
            (checkpoint, GENCO.hasAssociatedTrainingEvidence, training_association),
        )
        graph.add(
            (training_association, GENCO.associatedTrainingRun, training),
        )
    graph.add((inference, PROV.used, checkpoint))
    graph.add((inference, PROV.used, scenario))
    graph.add((inference, PROV.used, dataset_uri))
    graph.add((reference_extraction, PROV.used, checkpoint))
    graph.add((reference_extraction, PROV.used, training_dataset))
    if reproduction is not None:
        graph.add((reproduction, PROV.used, checkpoint))
        graph.add((reproduction, PROV.used, dataset_uri))
        graph.add((reproduction, GENCO.reproducesRun, inference))
    graph.add((prediction_uri, PROV.wasGeneratedBy, inference))

    _add_literal(graph, grid, GENCO.gridName, dataset.grid_name)
    _add_literal(graph, dataset_uri, GENCO.datasetId, dataset.dataset_id)
    _add_literal(graph, dataset_uri, GENCO.task, dataset.task)
    _add_literal(
        graph,
        dataset_uri,
        GENCO.corpusScenarioCount,
        dataset.corpus_scenario_count,
    )
    _add_literal(graph, dataset_uri, GENCO.datasetSplit, dataset.dataset_split)
    _add_literal(
        graph,
        dataset_uri,
        GENCO.splitScenarioCount,
        dataset.split_scenario_count,
    )
    _add_literal(graph, dataset_uri, GENCO.datasetRevision, dataset.dataset_revision)
    _add_literal(
        graph,
        training_dataset,
        GENCO.datasetId,
        model.training_dataset_id or f"{model.dataset_id}:train",
    )
    _add_literal(graph, training_dataset, GENCO.task, dataset.task)
    _add_literal(
        graph,
        training_dataset,
        GENCO.corpusScenarioCount,
        dataset.corpus_scenario_count,
    )
    _add_literal(graph, training_dataset, GENCO.datasetSplit, "train")
    _add_literal(
        graph,
        training_dataset,
        GENCO.splitScenarioCount,
        model.training_split_scenario_count,
    )
    _add_literal(
        graph,
        training_dataset,
        GENCO.datasetRevision,
        dataset.dataset_revision,
    )
    _add_literal(
        graph,
        data_generation,
        GENCO.generationMethod,
        dataset.generation_method,
    )
    _add_literal(
        graph,
        data_generation,
        GENCO.generationSeed,
        dataset.generation_seed,
    )
    _add_literal(
        graph,
        data_generation,
        GENCO.generationSeedStatus,
        dataset.generation_seed_status,
    )
    _add_literal(
        graph,
        data_generation,
        GENCO.softwareVersion,
        dataset.software_version,
    )
    _add_literal(
        graph,
        data_generation,
        GENCO.configHash,
        dataset.config_hash,
    )
    for perturbation in dataset.perturbations:
        _add_literal(graph, data_generation, GENCO.perturbation, perturbation)
    for limitation in dataset.provenance_limitations:
        _add_literal(
            graph,
            data_generation,
            GENCO.provenanceLimitation,
            limitation,
        )
    _add_literal(graph, checkpoint, GENCO.checkpointId, model.checkpoint_id)
    _add_literal(graph, checkpoint, GENCO.checkpointHash, model.checkpoint_hash)
    _add_literal(
        graph,
        checkpoint,
        GENCO.checkpointRepositoryPath,
        model.checkpoint_repository_path,
    )
    _add_literal(
        graph,
        checkpoint,
        GENCO.modelRepositoryRevision,
        model.model_repository_revision,
    )
    if training_attestation is not None:
        attestation = model.authoritative_training_attestation
        for predicate, value in (
            (GENCO.attestationId, attestation.attestation_id),
            (GENCO.attestationIssuer, attestation.issuer),
            (GENCO.attestationSource, attestation.source_uri),
            (GENCO.attestedCheckpointHash, attestation.checkpoint_sha256),
            (GENCO.attestedTrainingRunId, attestation.training_run_id),
            (GENCO.attestationContentHash, attestation.content_sha256),
            (GENCO.attestationSignature, attestation.signature),
            (GENCO.attestationVerificationStatus, attestation.verification_status),
        ):
            _add_literal(graph, training_attestation, predicate, value)
    _add_literal(graph, checkpoint, GENCO.normalizerHash, model.normalizer_hash)
    _add_literal(graph, checkpoint, GENCO.configHash, model.config_hash)
    _add_literal(graph, training, GENCO.trainingRunId, model.training_run_id)
    _add_literal(graph, training, GENCO.graphkitCommit, model.graphkit_commit)
    _add_literal(graph, training, GENCO.mlflowRunId, model.mlflow_run_id)
    _add_literal(graph, training, GENCO.startTimeMs, model.training_start_time_ms)
    _add_literal(graph, training, GENCO.endTimeMs, model.training_end_time_ms)
    _add_literal(graph, training, GENCO.randomSeed, model.training_seed)
    _add_literal(graph, training, GENCO.trainingEpochs, model.training_epochs)
    _add_literal(
        graph,
        training,
        GENCO.trainingBatchSize,
        model.training_batch_size,
    )
    _add_literal(
        graph,
        training,
        GENCO.trainingLearningRate,
        model.training_learning_rate,
    )
    _add_literal(
        graph,
        training,
        GENCO.trainingOptimizer,
        model.training_optimizer,
    )
    _add_literal(
        graph,
        training,
        GENCO.trainingSoftwareRevisionStatus,
        model.training_software_revision_status,
    )
    _add_literal(
        graph,
        checkpoint,
        GENCO.checkpointBindingStatus,
        model.checkpoint_binding_status,
    )
    if training_association is not None:
        association = model.training_evidence_association
        _add_literal(
            graph,
            training_association,
            GENCO.associationStatus,
            association.association_status,
        )
        _add_literal(
            graph,
            training_association,
            GENCO.productionIdentityVerified,
            association.production_identity_verified,
        )
        _add_literal(
            graph,
            training_association,
            GENCO.configurationCompatible,
            association.configuration_compatible,
        )
        for basis in association.evidence_basis:
            _add_literal(
                graph,
                training_association,
                GENCO.evidenceBasis,
                basis,
            )
        for limitation in association.provenance_limitations:
            _add_literal(
                graph,
                training_association,
                GENCO.provenanceLimitation,
                limitation,
            )
    training_metric_predicates = {
        "training_loss": GENCO.trainingLoss,
        "validation_loss": GENCO.validationLoss,
        "validation_bus_mse_loss": GENCO.validationBusMseLoss,
        "validation_generator_mse_loss": GENCO.validationGeneratorMseLoss,
        "validation_physics_loss": GENCO.validationPhysicsLoss,
        "validation_qg_violation_penalty_loss": (
            GENCO.validationQgViolationPenaltyLoss
        ),
    }
    for name, predicate in training_metric_predicates.items():
        _add_literal(
            graph,
            training,
            predicate,
            model.training_selected_metrics.get(name),
        )
    _add_literal(graph, training, GENCO.parametersHash, model.training_params_hash)
    _add_literal(graph, training, GENCO.metricsHash, model.training_metrics_hash)
    _add_literal(
        graph,
        training,
        GENCO.modelRepositoryRevision,
        model.model_repository_revision,
    )
    _add_literal(
        graph,
        training_dataset,
        GENCO.splitArtifactHash,
        model.training_split_hash,
    )
    _add_literal(graph, checkpoint, GENCO.modelVariant, model.model_variant)
    _add_literal(graph, checkpoint, GENCO.hiddenDimension, model.hidden_dim)
    _add_literal(graph, checkpoint, GENCO.correctionSteps, model.correction_steps)
    _add_literal(graph, scenario, GENCO.scenarioId, prediction.prediction_id)
    _add_literal(graph, prediction_uri, GENCO.predictionId, prediction.prediction_id)
    _add_literal(
        graph,
        inference,
        GENCO.startTimeMs,
        prediction.inference_start_time_ms,
    )
    _add_literal(
        graph,
        inference,
        GENCO.endTimeMs,
        prediction.inference_end_time_ms,
    )
    _add_literal(
        graph,
        inference,
        GENCO.graphkitCommit,
        prediction.inference_graphkit_commit,
    )
    _add_literal(
        graph,
        reference_extraction,
        GENCO.mlflowRunId,
        prediction.reference_extraction_run_id,
    )
    _add_literal(
        graph,
        reference_extraction,
        GENCO.startTimeMs,
        prediction.reference_extraction_start_time_ms,
    )
    _add_literal(
        graph,
        reference_extraction,
        GENCO.endTimeMs,
        prediction.reference_extraction_end_time_ms,
    )
    _add_literal(
        graph,
        reference_extraction,
        GENCO.graphkitCommit,
        prediction.reference_extraction_graphkit_commit,
    )
    if reproduction is not None:
        _add_literal(
            graph,
            reproduction,
            GENCO.mlflowRunId,
            prediction.reproduction_run_id,
        )
        _add_literal(
            graph,
            reproduction,
            GENCO.startTimeMs,
            prediction.reproduction_start_time_ms,
        )
        _add_literal(
            graph,
            reproduction,
            GENCO.endTimeMs,
            prediction.reproduction_end_time_ms,
        )
        _add_literal(
            graph,
            reproduction,
            GENCO.graphkitCommit,
            prediction.reproduction_graphkit_commit,
        )
        _add_literal(
            graph,
            reproduction,
            GENCO.environmentHash,
            prediction.reproduction_environment_hash,
        )
        for key, predicate in (
            ("python_version", GENCO.pythonVersion),
            ("platform", GENCO.runtimePlatform),
            ("torch", GENCO.torchVersion),
            ("torchvision", GENCO.torchvisionVersion),
            ("torchaudio", GENCO.torchaudioVersion),
            ("torch_scatter", GENCO.torchScatterVersion),
        ):
            _add_literal(
                graph,
                reproduction,
                predicate,
                prediction.reproduction_runtime.get(key),
            )
        _add_literal(
            graph,
            reproduction,
            GENCO.comparisonStatus,
            prediction.reproduction_comparison_status,
        )
        _add_literal(
            graph,
            reproduction,
            GENCO.tolerancePolicyId,
            prediction.reproduction_tolerance_policy_id,
        )
        _add_literal(
            graph,
            reproduction,
            GENCO.tolerancePolicyHash,
            prediction.reproduction_tolerance_policy_hash,
        )
        _add_literal(
            graph,
            reproduction,
            GENCO.maximumVoltageDifference,
            prediction.reproduction_maximum_voltage_difference,
        )
        _add_literal(
            graph,
            reproduction,
            GENCO.maximumPowerDifference,
            prediction.reproduction_maximum_power_difference,
        )

    for key, value in sorted(model.metrics.items()):
        _add_literal(graph, checkpoint, GENCO[key], value)

    evidence = _add_physical_evidence(
        graph,
        prediction_uri,
        prediction.prediction_id,
        prediction.physical_evidence,
    )

    for reference in prediction.reference_cases:
        reference_id = str(reference["scenario_id"])
        reference_uri = _resource("reference-case", reference_id)
        graph.add((reference_uri, RDF.type, GENCO.ReferenceCase))
        graph.add((prediction_uri, GENCO.hasReferenceCase, reference_uri))
        _add_literal(graph, reference_uri, GENCO.scenarioId, reference_id)
        _add_literal(graph, reference_uri, GENCO.distance, reference.get("distance"))
        _add_literal(graph, reference_uri, GENCO.rank, reference.get("rank"))

    _add_literal(
        graph,
        prediction_uri,
        GENCO.referenceDistance,
        prediction.reference_distance,
    )
    _add_literal(
        graph,
        prediction_uri,
        GENCO.referenceDistancePercentile,
        prediction.reference_distance_percentile,
    )
    graph.add((prediction_uri, GENCO.hasPhysicalEvidence, evidence))
    artifact_records = (
        list(dataset.artifact_records)
        + list(prediction.provenance_summary.get("artifact_records", ()))
    )
    if not artifact_records:
        artifact_records = (
            {"path": path}
            for path in prediction.provenance_summary.get("embedding_artifacts", ())
        )
    for record in artifact_records:
        artifact_path = str(record["path"])
        artifact = _resource("artifact", artifact_path)
        kind = record.get("kind")
        graph.add((artifact, RDF.type, PROV.Entity))
        graph.add((artifact, RDF.type, GENCO.Artifact))
        artifact_class = {
            "prediction": GENCO.PredictionArtifact,
            "target_embedding": GENCO.EmbeddingArtifact,
            "reference_embedding": GENCO.ReferenceEmbeddingArtifact,
            "provenance": GENCO.ProvenanceArtifact,
        }.get(kind)
        if artifact_class is not None:
            graph.add((artifact, RDF.type, artifact_class))
        graph.add((prediction_uri, GENCO.hasProvenanceArtifact, artifact))
        if kind in {"target_embedding", "reference_embedding"}:
            graph.add((prediction_uri, GENCO.hasEmbeddingArtifact, artifact))
        if kind == "reference_embedding":
            graph.add((artifact, PROV.wasGeneratedBy, reference_extraction))
        if kind in {"prediction", "target_embedding"}:
            graph.add((artifact, PROV.wasGeneratedBy, inference))
        _add_literal(graph, artifact, GENCO.artifactPath, artifact_path)
        _add_literal(graph, artifact, GENCO.sha256, record.get("sha256"))
    return graph


def serialize_graph(graph: Graph, output_path: str | Path) -> None:
    """Serialize Turtle deterministically enough for stable review artifacts."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    graph.serialize(destination=path, format="turtle")
