"""Stable identity and artifact hashing helpers."""

from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import quote


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Artifact does not exist: {file_path}")
    digest = sha256()
    with file_path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file or directory using stable relative paths for directories."""
    resolved = Path(path)
    if resolved.is_file():
        return sha256_file(resolved, chunk_size)
    if not resolved.is_dir():
        raise FileNotFoundError(f"Artifact does not exist: {resolved}")
    digest = sha256()
    for child in sorted(
        item
        for item in resolved.rglob("*")
        if item.is_file() and item.name != ".DS_Store"
    ):
        relative = child.relative_to(resolved).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        with child.open("rb") as stream:
            while chunk := stream.read(chunk_size):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()
def canonical_json_hash(value: Any) -> str:
    """Hash JSON-compatible metadata deterministically."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(payload.encode("utf-8")).hexdigest()


def stable_uri(kind: str, identifier: str) -> str:
    """Build a deterministic URI for a GENCO resource."""
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", kind):
        raise ValueError(f"Invalid URI kind: {kind!r}")
    if not identifier:
        raise ValueError("URI identifier cannot be empty")
    return f"https://genco.example/id/{kind}/{quote(identifier, safe='._~-')}"
