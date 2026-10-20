"""Curated local SPARQL queries for standard evidence graph questions."""

ASSOCIATED_TRAINING_EVIDENCE = """
PREFIX prov: <http://www.w3.org/ns/prov#>
PREFIX genco: <https://genco.example/ontology#>
SELECT ?checkpoint ?trainingRun ?dataset ?split ?status ?productionIdentityVerified
       ?evidenceBasis ?limitation
WHERE {
  ?checkpoint a genco:ModelCheckpoint ;
              genco:hasAssociatedTrainingEvidence ?association .
  ?association a genco:TrainingEvidenceAssociation ;
               genco:associatedTrainingRun ?trainingRun ;
               genco:associationStatus ?status ;
               genco:productionIdentityVerified ?productionIdentityVerified ;
               genco:evidenceBasis ?evidenceBasis ;
               genco:provenanceLimitation ?limitation .
  ?trainingRun a genco:TrainingRun ;
               prov:used ?dataset .
  ?dataset a genco:Dataset ;
           genco:datasetSplit ?split .
}
ORDER BY ?checkpoint ?evidenceBasis ?limitation
"""

CHECKPOINT = """
PREFIX prov: <http://www.w3.org/ns/prov#>
PREFIX genco: <https://genco.example/ontology#>
SELECT ?prediction ?checkpoint ?checkpointId ?checkpointHash
WHERE {
  ?prediction a genco:Prediction ;
              prov:wasGeneratedBy ?inference .
  ?inference a genco:InferenceRun ;
             prov:used ?checkpoint .
  ?checkpoint a genco:ModelCheckpoint ;
              genco:checkpointId ?checkpointId .
  OPTIONAL { ?checkpoint genco:checkpointHash ?checkpointHash }
}
"""

REFERENCE_CASES = """
PREFIX genco: <https://genco.example/ontology#>
SELECT ?prediction ?reference ?scenarioId ?distance ?rank
WHERE {
  ?prediction a genco:Prediction ;
              genco:hasReferenceCase ?reference .
  ?reference genco:scenarioId ?scenarioId ;
             genco:distance ?distance ;
             genco:rank ?rank .
}
ORDER BY ?prediction ?rank
"""

EMBEDDING_DISTANCE = """
PREFIX genco: <https://genco.example/ontology#>
SELECT ?prediction ?distance
WHERE {
  ?prediction a genco:Prediction ;
              genco:referenceDistance ?distance .
}
"""

RESIDUAL = """
PREFIX genco: <https://genco.example/ontology#>
SELECT ?prediction ?meanResidual ?maxResidual ?maxPbeBus ?activeResidual ?reactiveResidual
WHERE {
  ?prediction a genco:Prediction ;
              genco:hasPhysicalEvidence ?evidence .
  OPTIONAL { ?evidence genco:meanResidual ?meanResidual }
  OPTIONAL { ?evidence genco:maxResidual ?maxResidual }
  OPTIONAL { ?evidence genco:max_pbe_bus ?maxPbeBus }
  OPTIONAL { ?evidence genco:activeResidual ?activeResidual }
  OPTIONAL { ?evidence genco:reactiveResidual ?reactiveResidual }
}
"""

VIOLATIONS = """
PREFIX genco: <https://genco.example/ontology#>
SELECT ?prediction ?total ?voltage ?reactive ?activePower
WHERE {
  ?prediction a genco:Prediction ;
              genco:hasPhysicalEvidence ?evidence .
  ?evidence genco:violationCount ?total ;
            genco:voltage_violation_count ?voltage ;
            genco:reactive_violation_count ?reactive ;
            genco:active_power_violation_count ?activePower .
}
"""

AVAILABLE_VIOLATION_TYPES = """
PREFIX genco: <https://genco.example/ontology#>
SELECT ?prediction ?availableTypes
WHERE {
  ?prediction a genco:Prediction ;
              genco:hasPhysicalEvidence ?evidence .
  ?evidence genco:available_violation_types ?availableTypes .
}
"""

AVAILABLE_RESIDUAL_TYPES = """
PREFIX genco: <https://genco.example/ontology#>
SELECT ?prediction ?availableTypes
WHERE {
  ?prediction a genco:Prediction ;
              genco:hasPhysicalEvidence ?evidence .
  ?evidence genco:available_residual_types ?availableTypes .
}
"""

REPRODUCTION = """
PREFIX genco: <https://genco.example/ontology#>
SELECT ?reproduction ?targetRun ?runId ?environmentHash ?platform
       ?pythonVersion ?torchVersion ?torchScatterVersion
       ?tolerancePolicyId ?tolerancePolicyHash
       ?maximumVoltageDifference ?maximumPowerDifference
WHERE {
  ?reproduction a genco:ReproductionRun ;
                genco:reproducesRun ?targetRun ;
                genco:mlflowRunId ?runId ;
                genco:environmentHash ?environmentHash ;
                genco:runtimePlatform ?platform ;
                genco:pythonVersion ?pythonVersion ;
                genco:torchVersion ?torchVersion ;
                genco:torchScatterVersion ?torchScatterVersion ;
                genco:comparisonStatus "passed" ;
                genco:tolerancePolicyId ?tolerancePolicyId ;
                genco:tolerancePolicyHash ?tolerancePolicyHash ;
                genco:maximumVoltageDifference ?maximumVoltageDifference ;
                genco:maximumPowerDifference ?maximumPowerDifference .
}
"""


def high_residual_and_distance(
    residual_threshold: float,
    distance_threshold: float,
) -> str:
    """Return the thresholded cross-evidence acceptance query."""
    if residual_threshold < 0 or distance_threshold < 0:
        raise ValueError("acceptance thresholds cannot be negative")
    return f"""
PREFIX genco: <https://genco.example/ontology#>
SELECT ?prediction ?residual ?distance
WHERE {{
  ?prediction a genco:Prediction ;
              genco:referenceDistance ?distance ;
              genco:hasPhysicalEvidence ?evidence .
  ?evidence genco:maxResidual ?residual .
  FILTER (
    ?residual > {residual_threshold!r} &&
    ?distance > {distance_threshold!r}
  )
}}
ORDER BY DESC(?residual) DESC(?distance)
"""


NAMED_QUERIES = {
    "associated-training-evidence": ASSOCIATED_TRAINING_EVIDENCE,
    "checkpoint": CHECKPOINT,
    "reference-cases": REFERENCE_CASES,
    "embedding-distance": EMBEDDING_DISTANCE,
    "residuals": RESIDUAL,
    "violations": VIOLATIONS,
    "available-violation-types": AVAILABLE_VIOLATION_TYPES,
    "available-residual-types": AVAILABLE_RESIDUAL_TYPES,
    "reproduction": REPRODUCTION,
}

QUERY_HEADERS = {
    "associated-training-evidence": (
        "checkpoint_resource",
        "associated_training_run",
        "training_dataset",
        "dataset_split",
        "association_status",
        "production_identity_verified",
        "evidence_basis",
        "provenance_limitation",
    ),
    "checkpoint": (
        "prediction",
        "checkpoint_resource",
        "checkpoint_id",
        "checkpoint_sha256",
    ),
    "reference-cases": (
        "prediction",
        "reference_resource",
        "reference_scenario_id",
        "embedding_distance",
        "rank",
    ),
    "embedding-distance": ("prediction", "nearest_embedding_distance"),
    "residuals": (
        "prediction",
        "mean_pbe",
        "max_pbe",
        "max_pbe_bus_index",
        "mean_active_residual_mw",
        "mean_reactive_residual_mvar",
    ),
    "violations": (
        "prediction",
        "total_violations",
        "voltage_violations",
        "reactive_power_violations",
        "active_power_violations",
    ),
    "available-violation-types": ("prediction", "available_violation_types"),
    "available-residual-types": ("prediction", "available_residual_types"),
    "reproduction": (
        "reproduction_run",
        "target_inference_run",
        "mlflow_run_id",
        "environment_sha256",
        "platform",
        "python_version",
        "torch_version",
        "torch_scatter_version",
        "tolerance_policy_id",
        "tolerance_policy_sha256",
        "maximum_voltage_difference",
        "maximum_power_or_residual_difference",
    ),
    "high-residual-distance": (
        "prediction",
        "max_pbe",
        "nearest_embedding_distance",
    ),
}
