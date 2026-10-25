"""Verified view model for evidence showcase figures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import PROV, RDF
import yaml

from ..provenance import sha256_file
from ..rdf import GENCO
from ..validation import validate_evidence_graph


@dataclass(frozen=True)
class ReferenceCaseView:
    scenario_id: str
    distance: float
    rank: int


@dataclass(frozen=True)
class EvidenceShowcaseSpec:
    graph: Path
    shapes: Path
    prediction_parquet: Path
    generator_prediction_parquet: Path
    checkpoint: Path
    tolerance_config: Path
    output_dir: Path
    project_root: Path = Path(".")


@dataclass(frozen=True)
class EvidenceShowcaseView:
    scenario_id: str
    graph_sha256: str
    prediction_sha256: str
    generator_prediction_sha256: str
    checkpoint_sha256: str
    checkpoint_id: str
    checkpoint_verified: bool
    shacl_validated: bool
    inference_run_id: str
    dataset_id: str
    dataset_revision: str
    dataset_split: str
    references: tuple[ReferenceCaseView, ...]
    nearest_reference_distance: float
    mean_pbe: float
    max_pbe: float
    max_pbe_bus: int
    mean_active_residual: float
    mean_reactive_residual: float
    total_violations: int
    voltage_violations: int
    reactive_violations: int
    active_power_violations: int
    reproduction_passed: bool
    maximum_voltage_difference: float
    maximum_power_difference: float
    voltage_tolerance: float
    power_tolerance: float
    tolerance_policy_id: str
    training_association_status: str
    training_production_identity_verified: bool
    bus_predictions: pd.DataFrame
    generator_predictions: pd.DataFrame
    artifact_hashes: tuple[tuple[str, str], ...]


def _single_subject(
    graph: Graph,
    rdf_type: URIRef,
    label: str,
) -> URIRef:
    subjects = tuple(
        subject
        for subject in graph.subjects(RDF.type, rdf_type)
        if isinstance(subject, URIRef)
    )
    if len(subjects) != 1:
        raise ValueError(
            f"Evidence graph must contain exactly one {label}; "
            f"found {len(subjects)}"
        )
    return subjects[0]


def _single_object(
    graph: Graph,
    subject: URIRef,
    predicate: URIRef,
    label: str,
) -> Any:
    values = tuple(graph.objects(subject, predicate))
    if len(values) != 1:
        raise ValueError(
            f"{label} must have exactly one value; found {len(values)}"
        )
    value = values[0]
    return value.toPython() if isinstance(value, Literal) else value


def _artifact_hash(
    graph: Graph,
    path: Path,
    project_root: Path,
) -> str:
    matches: list[str] = []
    for artifact, recorded_path in graph.subject_objects(
        GENCO.artifactPath
    ):
        candidate = Path(str(recorded_path))
        resolved = (
            candidate
            if candidate.is_absolute()
            else project_root / candidate
        ).resolve()
        if resolved == path.resolve():
            matches.extend(
                str(value)
                for value in graph.objects(
                    artifact,
                    GENCO.sha256,
                )
            )
    if len(matches) != 1:
        raise ValueError(
            "Evidence graph must contain exactly one artifact record "
            f"for {path}; found {len(matches)}"
        )
    actual = sha256_file(path)
    if matches[0] != actual:
        raise ValueError(
            f"Evidence graph hash disagrees for {path}: "
            f"{matches[0]} != {actual}"
        )
    return actual


def _checkpoint(
    graph: Graph,
    checkpoint_path: Path,
    project_root: Path,
) -> tuple[URIRef, str, str]:
    resource = _single_subject(
        graph,
        GENCO.ModelCheckpoint,
        "model checkpoint",
    )
    recorded_path = Path(
        str(
            _single_object(
                graph,
                resource,
                GENCO.checkpointRepositoryPath,
                "Model checkpoint",
            )
        )
    )
    resolved = (
        recorded_path
        if recorded_path.is_absolute()
        else project_root / recorded_path
    ).resolve()
    if resolved != checkpoint_path.resolve():
        raise ValueError(
            "Evidence graph checkpoint path disagrees with "
            f"the explicit input: {recorded_path} != {checkpoint_path}"
        )
    recorded_hash = str(
        _single_object(
            graph,
            resource,
            GENCO.checkpointHash,
            "Model checkpoint",
        )
    )
    actual_hash = sha256_file(checkpoint_path)
    if recorded_hash != actual_hash:
        raise ValueError(
            "Evidence graph checkpoint hash disagrees with "
            f"{checkpoint_path}: {recorded_hash} != {actual_hash}"
        )
    checkpoint_id = str(
        _single_object(
            graph,
            resource,
            GENCO.checkpointId,
            "Model checkpoint",
        )
    )
    return resource, checkpoint_id, actual_hash


def _prediction_tables(
    scenario_id: str,
    prediction_path: Path,
    generator_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenario = int(scenario_id)
    buses = pd.read_parquet(
        prediction_path,
        filters=[("scenario", "==", scenario)],
    ).sort_values("bus").reset_index(drop=True)
    generators = pd.read_parquet(
        generator_path,
        filters=[("scenario", "==", scenario)],
    ).sort_values("idx").reset_index(drop=True)
    if len(buses) != 14 or tuple(buses["bus"]) != tuple(range(14)):
        raise ValueError(
            "Showcase prediction must contain buses 0-13 exactly once"
        )
    if generators.empty or generators["idx"].duplicated().any():
        raise ValueError(
            "Showcase generator predictions must have unique entities"
        )
    numeric_columns = (
        "Vm_target",
        "Vm_pred",
        "PBE",
        "active res. (MW)",
        "reactive res. (MVar)",
    )
    if not np.isfinite(
        buses[list(numeric_columns)].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Showcase bus prediction values must be finite"
        )
    if not np.isfinite(
        generators[
            ["p_mw_target", "p_mw_pred"]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Showcase generator prediction values must be finite"
        )
    return buses, generators


def _assert_close(
    actual: float,
    expected: float,
    label: str,
) -> None:
    if not np.isclose(
        actual,
        expected,
        rtol=1e-5,
        atol=1e-9,
    ):
        raise ValueError(
            f"RDF {label} disagrees with prediction table: "
            f"{actual} != {expected}"
        )


def _portable_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root).as_posix()
    except ValueError:
        return str(path.resolve())


def load_showcase_view(
    spec: EvidenceShowcaseSpec,
) -> EvidenceShowcaseView:
    """Load and verify one read-only evidence showcase view."""
    project_root = spec.project_root.resolve()
    graph_path = spec.graph.resolve()
    shapes_path = spec.shapes.resolve()
    prediction_path = spec.prediction_parquet.resolve()
    generator_path = spec.generator_prediction_parquet.resolve()
    checkpoint_path = spec.checkpoint.resolve()
    tolerance_path = spec.tolerance_config.resolve()
    for path, label in (
        (graph_path, "Evidence graph"),
        (shapes_path, "SHACL shapes"),
        (prediction_path, "Bus prediction artifact"),
        (generator_path, "Generator prediction artifact"),
        (checkpoint_path, "Model checkpoint"),
        (tolerance_path, "Tolerance policy"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    graph = Graph()
    graph.parse(graph_path, format="turtle")
    validate_evidence_graph(graph, shapes_path)
    prediction_hash = _artifact_hash(
        graph,
        prediction_path,
        project_root,
    )
    generator_hash = _artifact_hash(
        graph,
        generator_path,
        project_root,
    )
    checkpoint_resource, checkpoint_id, checkpoint_hash = (
        _checkpoint(
            graph,
            checkpoint_path,
            project_root,
        )
    )
    tolerance_hash = _artifact_hash(
        graph,
        tolerance_path,
        project_root,
    )

    prediction = _single_subject(
        graph,
        GENCO.Prediction,
        "prediction",
    )
    scenario_id = str(
        _single_object(
            graph,
            prediction,
            GENCO.predictionId,
            "Prediction",
        )
    )
    inference = _single_object(
        graph,
        prediction,
        PROV.wasGeneratedBy,
        "Prediction",
    )
    if not isinstance(inference, URIRef):
        raise ValueError("Prediction inference resource must be a URI")
    inference_run_id = str(inference).rsplit("/", 1)[-1]
    datasets = tuple(
        value
        for value in graph.objects(inference, PROV.used)
        if (value, RDF.type, GENCO.Dataset) in graph
    )
    if len(datasets) != 1:
        raise ValueError(
            "Inference must use exactly one dataset resource"
        )
    dataset = datasets[0]
    dataset_id = str(
        _single_object(
            graph,
            dataset,
            GENCO.datasetId,
            "Inference dataset",
        )
    )
    dataset_revision = str(
        _single_object(
            graph,
            dataset,
            GENCO.datasetRevision,
            "Inference dataset",
        )
    )
    dataset_split = str(
        _single_object(
            graph,
            dataset,
            GENCO.datasetSplit,
            "Inference dataset",
        )
    )

    references = []
    for reference in graph.objects(
        prediction,
        GENCO.hasReferenceCase,
    ):
        references.append(
            ReferenceCaseView(
                scenario_id=str(
                    _single_object(
                        graph,
                        reference,
                        GENCO.scenarioId,
                        "Reference case",
                    )
                ),
                distance=float(
                    _single_object(
                        graph,
                        reference,
                        GENCO.distance,
                        "Reference case",
                    )
                ),
                rank=int(
                    _single_object(
                        graph,
                        reference,
                        GENCO.rank,
                        "Reference case",
                    )
                ),
            )
        )
    references.sort(key=lambda value: value.rank)
    if [value.rank for value in references] != list(
        range(1, len(references) + 1)
    ):
        raise ValueError(
            "Reference case ranks must be contiguous from one"
        )
    nearest_distance = float(
        _single_object(
            graph,
            prediction,
            GENCO.referenceDistance,
            "Prediction",
        )
    )
    _assert_close(
        nearest_distance,
        references[0].distance,
        "nearest reference distance",
    )

    evidence = _single_object(
        graph,
        prediction,
        GENCO.hasPhysicalEvidence,
        "Prediction",
    )
    if not isinstance(evidence, URIRef):
        raise ValueError("Physical evidence resource must be a URI")
    mean_pbe = float(
        _single_object(
            graph,
            evidence,
            GENCO.meanResidual,
            "Physical evidence",
        )
    )
    max_pbe = float(
        _single_object(
            graph,
            evidence,
            GENCO.maxResidual,
            "Physical evidence",
        )
    )
    max_pbe_bus = int(
        _single_object(
            graph,
            evidence,
            GENCO.max_pbe_bus,
            "Physical evidence",
        )
    )
    mean_active = float(
        _single_object(
            graph,
            evidence,
            GENCO.activeResidual,
            "Physical evidence",
        )
    )
    mean_reactive = float(
        _single_object(
            graph,
            evidence,
            GENCO.reactiveResidual,
            "Physical evidence",
        )
    )
    total_violations = int(
        _single_object(
            graph,
            evidence,
            GENCO.violationCount,
            "Physical evidence",
        )
    )
    voltage_violations = int(
        _single_object(
            graph,
            evidence,
            GENCO.voltage_violation_count,
            "Physical evidence",
        )
    )
    reactive_violations = int(
        _single_object(
            graph,
            evidence,
            GENCO.reactive_violation_count,
            "Physical evidence",
        )
    )
    active_violations = int(
        _single_object(
            graph,
            evidence,
            GENCO.active_power_violation_count,
            "Physical evidence",
        )
    )

    reproduction = _single_subject(
        graph,
        GENCO.ReproductionRun,
        "reproduction run",
    )
    comparison_status = str(
        _single_object(
            graph,
            reproduction,
            GENCO.comparisonStatus,
            "Reproduction run",
        )
    )
    maximum_voltage_difference = float(
        _single_object(
            graph,
            reproduction,
            GENCO.maximumVoltageDifference,
            "Reproduction run",
        )
    )
    maximum_power_difference = float(
        _single_object(
            graph,
            reproduction,
            GENCO.maximumPowerDifference,
            "Reproduction run",
        )
    )
    tolerance_policy_id = str(
        _single_object(
            graph,
            reproduction,
            GENCO.tolerancePolicyId,
            "Reproduction run",
        )
    )
    try:
        tolerance_policy = yaml.safe_load(
            tolerance_path.read_text()
        )
    except yaml.YAMLError as error:
        raise ValueError(
            f"Tolerance policy is not valid YAML: {tolerance_path}"
        ) from error
    if not isinstance(tolerance_policy, dict):
        raise ValueError("Tolerance policy must be a YAML mapping")
    if tolerance_policy.get("policy_id") != tolerance_policy_id:
        raise ValueError(
            "RDF tolerance policy ID disagrees with the "
            "hash-verified tolerance policy"
        )
    try:
        bus_tolerances = tolerance_policy[
            "absolute_tolerances"
        ]["bus"]
        generator_tolerances = tolerance_policy[
            "absolute_tolerances"
        ]["generator"]
        voltage_tolerance = max(
            float(bus_tolerances["Vm_pred"]),
            float(bus_tolerances["Va_pred"]),
        )
        power_tolerance = max(
            *(
                float(value)
                for field, value in bus_tolerances.items()
                if field not in {"Vm_pred", "Va_pred"}
            ),
            *(
                float(value)
                for value in generator_tolerances.values()
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Tolerance policy does not contain the required "
            "bus and generator absolute tolerances"
        ) from error

    association = _single_object(
        graph,
        checkpoint_resource,
        GENCO.hasAssociatedTrainingEvidence,
        "Model checkpoint",
    )
    if not isinstance(association, URIRef):
        raise ValueError(
            "Training evidence association must be a URI"
        )
    association_status = str(
        _single_object(
            graph,
            association,
            GENCO.associationStatus,
            "Training evidence association",
        )
    )
    production_verified = bool(
        _single_object(
            graph,
            association,
            GENCO.productionIdentityVerified,
            "Training evidence association",
        )
    )

    buses, generators = _prediction_tables(
        scenario_id,
        prediction_path,
        generator_path,
    )
    _assert_close(
        mean_pbe,
        float(buses["PBE"].mean()),
        "mean PBE",
    )
    _assert_close(
        max_pbe,
        float(buses["PBE"].max()),
        "maximum PBE",
    )
    if max_pbe_bus != int(
        buses.loc[buses["PBE"].idxmax(), "bus"]
    ):
        raise ValueError(
            "RDF maximum-PBE bus disagrees with prediction table"
        )
    _assert_close(
        mean_active,
        float(buses["active res. (MW)"].abs().mean()),
        "mean active residual",
    )
    _assert_close(
        mean_reactive,
        float(
            buses["reactive res. (MVar)"].abs().mean()
        ),
        "mean reactive residual",
    )

    artifact_hashes = (
        (
            _portable_path(graph_path, project_root),
            sha256_file(graph_path),
        ),
        (
            _portable_path(prediction_path, project_root),
            prediction_hash,
        ),
        (
            _portable_path(generator_path, project_root),
            generator_hash,
        ),
        (
            _portable_path(checkpoint_path, project_root),
            checkpoint_hash,
        ),
        (
            _portable_path(tolerance_path, project_root),
            tolerance_hash,
        ),
    )
    return EvidenceShowcaseView(
        scenario_id=scenario_id,
        graph_sha256=sha256_file(graph_path),
        prediction_sha256=prediction_hash,
        generator_prediction_sha256=generator_hash,
        checkpoint_sha256=checkpoint_hash,
        checkpoint_id=checkpoint_id,
        checkpoint_verified=True,
        shacl_validated=True,
        inference_run_id=inference_run_id,
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        dataset_split=dataset_split,
        references=tuple(references),
        nearest_reference_distance=nearest_distance,
        mean_pbe=mean_pbe,
        max_pbe=max_pbe,
        max_pbe_bus=max_pbe_bus,
        mean_active_residual=mean_active,
        mean_reactive_residual=mean_reactive,
        total_violations=total_violations,
        voltage_violations=voltage_violations,
        reactive_violations=reactive_violations,
        active_power_violations=active_violations,
        reproduction_passed=comparison_status == "passed",
        maximum_voltage_difference=maximum_voltage_difference,
        maximum_power_difference=maximum_power_difference,
        voltage_tolerance=voltage_tolerance,
        power_tolerance=power_tolerance,
        tolerance_policy_id=tolerance_policy_id,
        training_association_status=association_status,
        training_production_identity_verified=production_verified,
        bus_predictions=buses,
        generator_predictions=generators,
        artifact_hashes=artifact_hashes,
    )
