"""Closed frames and transport records for the Julia process boundary.

These validators preserve launch, progress, and checkpoint identities. They
never publish workspace state or acknowledge a checkpoint themselves."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from ..errors import BackendProtocolError, EvidenceIntegrityError
from ..canonical import float64_from_hex


_SHA256_LENGTH = 64


@dataclass(frozen=True)
class BootstrapReady:
    """Validated child bootstrap evidence, before the attempt is sealed."""

    request_sha256: str
    attempt_ordinal: int
    julia_version: str
    julia_threads: int
    blas_threads: int
    blas_vendor: str
    fftw_threads: int | None = None


@dataclass(frozen=True)
class TerminalOutcome:
    """Transport facts returned only after a successful child exit/outcome pair."""

    outcome: Mapping[str, object]
    stdout_log: tuple[str, ...]
    stderr_log: tuple[str, ...]


def _canonical_json_line(value: Mapping[str, object]) -> str:
    """Return the protocol's one permitted canonical JSONL representation."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ) + "\n"
    except (TypeError, ValueError) as error:
        raise BackendProtocolError(
            "protocol frame cannot be represented as canonical JSON",
            stage="protocol_frame",
            evidence={"error": str(error)},
        ) from error


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _decode_canonical_line(raw: str, *, stage: str) -> Mapping[str, object]:
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as error:
        raise BackendProtocolError(
            "child emitted malformed JSONL protocol framing",
            stage=stage,
            evidence={"line": raw.rstrip("\n"), "error": str(error)},
        ) from error
    if not isinstance(decoded, dict) or raw != _canonical_json_line(decoded):
        raise BackendProtocolError(
            "child protocol JSONL is not the required canonical frame",
            stage=stage,
            evidence={"line": raw.rstrip("\n")},
        )
    return decoded


def _validate_bootstrap(
    raw: str,
    *,
    request_sha256: str,
    attempt_ordinal: int,
    expected_version: str,
    hb_operation: bool,
) -> BootstrapReady:
    frame = _decode_canonical_line(raw, stage="bootstrap")
    required = {
        "schema": "scnsim.bootstrap_ready",
        "schema_version": 1,
        "request_sha256": request_sha256,
        "attempt_ordinal": attempt_ordinal,
        "julia_version": expected_version,
        "julia_threads": 1,
        "blas_threads": 1,
    }
    expected_fields = {*required, "blas_vendor"}
    if hb_operation:
        expected_fields.add("fftw_threads")
    if set(frame) != expected_fields or any(
        frame.get(key) != value for key, value in required.items()
    ) or not isinstance(frame.get("blas_vendor"), str) or not frame["blas_vendor"] or (
        hb_operation and frame.get("fftw_threads") != 1
    ):
        raise BackendProtocolError(
            "child bootstrap evidence does not match the sealed launch",
            stage="bootstrap",
            evidence={"frame": frame},
        )
    return BootstrapReady(
        request_sha256=request_sha256,
        attempt_ordinal=attempt_ordinal,
        julia_version=expected_version,
        julia_threads=1,
        blas_threads=1,
        blas_vendor=str(frame["blas_vendor"]),
        fftw_threads=1 if hb_operation else None,
    )


def _validate_progress(
    raw: str,
    *,
    request_sha256: str,
    attempt_sha256: str,
) -> Mapping[str, object] | None:
    try:
        frame = _decode_canonical_line(raw, stage="progress")
    except BackendProtocolError:
        return None
    if frame.get("schema") != "scnsim.progress":
        return None
    required = {
        "schema": "scnsim.progress",
        "schema_version": 1,
        "request_sha256": request_sha256,
        "attempt_sha256": attempt_sha256,
        "event": frame.get("event"),
    }
    phase = frame.get("event")
    fields = {"completed_generations", "total_generations", "evaluated_count",
              "requested_budget", "achievable_evaluations"}
    if any(frame.get(key) != value for key, value in required.items()) or set(frame) != {
        *required,
        *fields, "reused", "best_cost_f64",
    } or phase not in {"initial", "resume", "generation", "complete"} or not isinstance(frame.get("reused"), bool) or any(
        not isinstance(frame.get(key), int) or isinstance(frame.get(key), bool) or frame[key] < (0 if key == "completed_generations" else 1)
        for key in fields
    ):
        raise BackendProtocolError(
            "child progress frame does not bind the authorized request and attempt",
            stage="progress",
            evidence={"frame": frame},
        )
    try:
        best = float64_from_hex(frame["best_cost_f64"])
    except (TypeError, ValueError, KeyError, EvidenceIntegrityError) as error:
        raise BackendProtocolError("optimization progress cost is malformed", stage="progress") from error
    if (not math.isfinite(best) or best < 0.0 or
        frame["completed_generations"] > frame["total_generations"] or
        frame["achievable_evaluations"] > frame["requested_budget"] or
        (phase == "initial" and (frame["completed_generations"] != 0 or frame["evaluated_count"] != 1)) or
        (phase == "generation" and frame["reused"]) or
        (phase == "resume" and not frame["reused"])):
        raise BackendProtocolError("optimization progress counts are inconsistent", stage="progress")
    return frame


def _validate_checkpoint_ready(
    raw: str, *, request_sha256: str, attempt_sha256: str
) -> Mapping[str, object]:
    frame = _decode_canonical_line(raw, stage="optimization_checkpoint")
    expected = {
        "schema": "scnsim.optimization_checkpoint_ready",
        "schema_version": 1,
        "event": "baseline_checkpoint_ready",
        "request_sha256": request_sha256,
        "attempt_sha256": attempt_sha256,
    }
    if (
        set(frame) != {*expected, "checkpoint_sha256", "byte_length"}
        or any(frame.get(key) != value for key, value in expected.items())
        or not _is_sha256(frame.get("checkpoint_sha256"))
        or not isinstance(frame.get("byte_length"), int)
        or isinstance(frame.get("byte_length"), bool)
        or frame["byte_length"] < 1
    ):
        raise BackendProtocolError(
            "optimization checkpoint-ready frame is open or mismatched",
            stage="optimization_checkpoint",
            evidence={"frame": frame},
        )
    return frame


def _validate_point_ready(raw: str, *, request_sha256: str,
        attempt_sha256: str) -> Mapping[str, object]:
    frame = _decode_canonical_line(raw, stage="point_checkpoint")
    if (set(frame) != {"schema", "schema_version", "request_sha256", "attempt_sha256",
            "ordinal", "record_sha256", "byte_length"} or
        frame.get("schema") != "scnsim.point_checkpoint_ready" or frame.get("schema_version") != 1 or
        frame.get("request_sha256") != request_sha256 or frame.get("attempt_sha256") != attempt_sha256 or
        not isinstance(frame.get("ordinal"), int) or isinstance(frame.get("ordinal"), bool) or frame["ordinal"] < 0 or
        not _is_sha256(frame.get("record_sha256")) or
        not isinstance(frame.get("byte_length"), int) or frame["byte_length"] < 1):
        raise BackendProtocolError("point checkpoint ready frame is malformed", stage="point_checkpoint")
    return frame


@dataclass(frozen=True)
class _JSONObjectPairs:
    """Decoded JSON object retaining every key before canonical validation."""

    pairs: tuple[tuple[str, object], ...]


def _decoded_json(raw: str) -> object:
    return json.loads(
        raw,
        object_pairs_hook=lambda pairs: _JSONObjectPairs(tuple(pairs)),
    )


def _reserved_optimization_frame(raw: str) -> bool:
    reserved_events = {
        "baseline_checkpoint_ready",
        "baseline_checkpoint_committed",
    }
    try:
        decoded = _decoded_json(raw)
    except json.JSONDecodeError:
        return re.search(
            r'"schema"\s*:\s*"scnsim\.optimization_checkpoint', raw
        ) is not None or any(
            re.search(rf'"event"\s*:\s*"{event}"', raw) is not None
            for event in reserved_events
        )
    if not isinstance(decoded, _JSONObjectPairs):
        return False
    return any(
        (key == "schema" and isinstance(value, str)
         and value.startswith("scnsim.optimization_checkpoint"))
        or (key == "event" and value in reserved_events)
        for key, value in decoded.pairs
    )


def _read_outcome(
    staging_directory: Path,
    *,
    request_sha256: str,
    attempt_sha256: str,
) -> Mapping[str, object]:
    path = staging_directory / "outcome.json"
    try:
        if (
            staging_directory.parent.is_symlink()
            or staging_directory.is_symlink()
            or not staging_directory.is_dir()
            or path.is_symlink()
            or not path.is_file()
        ):
            raise OSError("outcome.json is not a regular file")
        raw = path.read_text(encoding="utf-8")
        outcome = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise BackendProtocolError(
            "a completed child did not produce a readable outcome envelope",
            stage="outcome",
            evidence={"path": str(path), "error": str(error)},
        ) from error
    if not isinstance(outcome, dict) or outcome.get("schema") != "scnsim.outcome" or outcome.get(
        "schema_version"
    ) != 1 or outcome.get("request_sha256") != request_sha256 or outcome.get(
        "attempt_sha256"
    ) != attempt_sha256:
        raise BackendProtocolError(
            "child outcome envelope does not bind the authorized request and attempt",
            stage="outcome",
            evidence={"path": str(path)},
        )
    return outcome
