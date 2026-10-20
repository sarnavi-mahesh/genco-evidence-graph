"""Cryptographic verification for authoritative checkpoint-training attestations."""

from dataclasses import dataclass
import base64
import binascii
import json
import re
from typing import Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


ATTESTATION_FIELDS = {
    "format_version",
    "attestation_id",
    "issuer",
    "source_uri",
    "checkpoint_id",
    "checkpoint_sha256",
    "training_run_id",
    "content_sha256",
    "verification_status",
}

PRODUCTION_ATTESTATION_TRUST_POLICY = {
    "format_version": 1,
    "policy_id": "genco-authoritative-training-attestation-trust-v1",
    "checkpoints": {
        "e31cdaafdffccae5c557e7194153fc52bf7bf0fe5f143765c681350bba4e37dc": {
            "status": "prohibited",
            "checkpoint_id": "case14_ieee_seed42_e31cdaafdffc",
            "training_run_id": "1927fe80558a48539ee37be478706781",
            "reason": (
                "No independently authenticated upstream checkpoint-training "
                "attestation is available for the published seed-42 checkpoint"
            ),
            "issuers": {},
        },
    },
}


def canonical_attestation_payload(document: Mapping[str, object]) -> bytes:
    """Return the canonical signed bytes for one attestation document."""
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


@dataclass(frozen=True, init=False)
class VerifiedTrainingAttestation:
    attestation_id: str
    issuer: str
    source_uri: str
    checkpoint_id: str
    checkpoint_sha256: str
    training_run_id: str
    content_sha256: str
    signature: str
    verification_status: str


def verify_training_attestation(
    document: Mapping[str, object],
    signature: str,
    trusted_issuer_keys: Mapping[str, bytes],
) -> VerifiedTrainingAttestation:
    """Verify and construct a trusted Ed25519 training attestation."""
    if set(document) != ATTESTATION_FIELDS:
        raise ValueError("Authoritative training attestation fields are incomplete")
    if document.get("format_version") != 1:
        raise ValueError("Unsupported authoritative training attestation format")
    if document.get("verification_status") != "verified":
        raise ValueError("Authoritative training attestation is not verified")

    values = {
        name: str(document[name])
        for name in ATTESTATION_FIELDS - {"format_version"}
    }
    for name in (
        "attestation_id",
        "issuer",
        "source_uri",
        "checkpoint_id",
        "training_run_id",
    ):
        if not values[name].strip():
            raise ValueError(f"Authoritative training attestation {name} is empty")
    if not values["source_uri"].startswith("https://"):
        raise ValueError("Authoritative training attestation source must use HTTPS")
    for name in ("checkpoint_sha256", "content_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", values[name]) is None:
            raise ValueError(
                f"Authoritative training attestation {name} is not SHA-256",
            )

    public_key = trusted_issuer_keys.get(values["issuer"])
    if public_key is None:
        raise ValueError(
            f"Authoritative training attestation issuer is not trusted: "
            f"{values['issuer']}",
        )
    try:
        signature_bytes = base64.b64decode(signature, validate=True)
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            signature_bytes,
            canonical_attestation_payload(document),
        )
    except (binascii.Error, InvalidSignature, ValueError) as error:
        raise ValueError(
            "Authoritative training attestation signature verification failed",
        ) from error

    attestation = object.__new__(VerifiedTrainingAttestation)
    for name, value in values.items():
        object.__setattr__(attestation, name, value)
    object.__setattr__(attestation, "signature", signature)
    return attestation


def verify_production_training_attestation(
    document: Mapping[str, object],
    signature: str,
) -> VerifiedTrainingAttestation:
    """Verify against the release-governed checkpoint-scoped trust policy."""
    checkpoint_hash = str(document.get("checkpoint_sha256", ""))
    checkpoint_policy = PRODUCTION_ATTESTATION_TRUST_POLICY["checkpoints"].get(
        checkpoint_hash,
    )
    if checkpoint_policy is None:
        raise ValueError(
            "No production attestation trust policy exists for checkpoint "
            f"{checkpoint_hash!r}",
        )
    if checkpoint_policy["status"] != "approved":
        raise ValueError(
            "Production trust policy prohibits authoritative lineage for "
            f"checkpoint {checkpoint_hash}: {checkpoint_policy['reason']}",
        )
    if document.get("checkpoint_id") != checkpoint_policy["checkpoint_id"]:
        raise ValueError(
            "Production attestation checkpoint ID is not policy-approved",
        )
    if document.get("training_run_id") != checkpoint_policy["training_run_id"]:
        raise ValueError(
            "Production attestation training-run ID is not policy-approved",
        )

    issuer = str(document.get("issuer", ""))
    issuer_policy = checkpoint_policy["issuers"].get(issuer)
    if issuer_policy is None:
        raise ValueError(
            f"Production attestation issuer is not approved for checkpoint: "
            f"{issuer}",
        )
    if document.get("source_uri") != issuer_policy["source_uri"]:
        raise ValueError("Production attestation source is not policy-approved")
    if document.get("content_sha256") != issuer_policy["content_sha256"]:
        raise ValueError(
            "Production attestation source content hash is not policy-approved",
        )
    return verify_training_attestation(
        document,
        signature,
        {issuer: bytes.fromhex(issuer_policy["public_key_hex"])},
    )
