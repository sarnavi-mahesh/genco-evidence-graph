"""SHACL validation for GENCO evidence graph."""

from pathlib import Path
from typing import Mapping

from pyshacl import validate
from rdflib import Graph, Namespace
from rdflib.namespace import RDF

from .attestation import (
    verify_production_training_attestation,
    verify_training_attestation,
)


GENCO = Namespace("https://genco.example/ontology#")


def _single_text(graph: Graph, subject, predicate) -> str:
    values = list(graph.objects(subject, predicate))
    if len(values) != 1:
        raise ValueError(
            f"Authoritative attestation requires one {predicate}, found {len(values)}",
        )
    return str(values[0])


def _single_object(graph: Graph, subject, predicate):
    values = list(graph.objects(subject, predicate))
    if len(values) != 1:
        raise ValueError(
            f"Authoritative attestation requires one {predicate}, found {len(values)}",
        )
    return values[0]


def _verify_authoritative_attestations(
    graph: Graph,
    trusted_attestation_keys: Mapping[str, bytes] | None = None,
) -> None:
    attestations = list(
        graph.subjects(RDF.type, GENCO.AuthoritativeTrainingAttestation),
    )
    if not attestations:
        return
    for attestation in attestations:
        document = {
            "format_version": 1,
            "attestation_id": _single_text(
                graph,
                attestation,
                GENCO.attestationId,
            ),
            "issuer": _single_text(graph, attestation, GENCO.attestationIssuer),
            "source_uri": _single_text(graph, attestation, GENCO.attestationSource),
            "checkpoint_id": _single_text(
                graph,
                _single_object(
                    graph,
                    attestation,
                    GENCO.attestsCheckpoint,
                ),
                GENCO.checkpointId,
            ),
            "checkpoint_sha256": _single_text(
                graph,
                attestation,
                GENCO.attestedCheckpointHash,
            ),
            "training_run_id": _single_text(
                graph,
                attestation,
                GENCO.attestedTrainingRunId,
            ),
            "content_sha256": _single_text(
                graph,
                attestation,
                GENCO.attestationContentHash,
            ),
            "verification_status": _single_text(
                graph,
                attestation,
                GENCO.attestationVerificationStatus,
            ),
        }
        signature = _single_text(
            graph,
            attestation,
            GENCO.attestationSignature,
        )
        if trusted_attestation_keys is None:
            verify_production_training_attestation(document, signature)
        else:
            verify_training_attestation(
                document,
                signature,
                trusted_attestation_keys,
            )


def validate_evidence_graph(
    graph: Graph,
    shapes_path: str | Path,
) -> None:
    """Raise ValueError when the graph violates the project SHACL shapes."""
    _verify_authoritative_attestations(graph)
    _validate_shacl(graph, shapes_path)


def validate_synthetic_evidence_graph(
    graph: Graph,
    shapes_path: str | Path,
    *,
    synthetic_trusted_attestation_keys: Mapping[str, bytes],
) -> None:
    """Validate an explicitly synthetic graph with test-only issuer keys."""
    if not synthetic_trusted_attestation_keys:
        raise ValueError("Synthetic attestation validation requires test-only keys")
    _verify_authoritative_attestations(
        graph,
        synthetic_trusted_attestation_keys,
    )
    _validate_shacl(graph, shapes_path)


def _validate_shacl(graph: Graph, shapes_path: str | Path) -> None:
    shapes = Graph()
    shapes.parse(shapes_path, format="turtle")
    conforms, _, report_text = validate(
        graph,
        shacl_graph=shapes,
        inference="rdfs",
        abort_on_first=False,
    )
    if not conforms:
        raise ValueError(f"Evidence graph failed SHACL validation:\n{report_text}")
