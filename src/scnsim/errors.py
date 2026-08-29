"""Stable public failures for the CONVERGING SCNSim V1 contract."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import cast


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return value


class SCNSimError(Exception):
    """Base class for typed SCNSim failures.

    ``kind`` is a stable machine discriminator. ``stage`` names the exact
    validation, execution, or evidence stage, and ``evidence`` is an immutable
    recursively frozen mapping suitable for receipt reconstruction.
    """

    category = "error"
    kind = "scnsim_error"

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        evidence: Mapping[str, object] | None = None,
    ) -> None:
        if not isinstance(message, str) or not message:
            raise ValueError("SCNSim error message must be a nonempty string")
        if not isinstance(stage, str) or not stage:
            raise ValueError("SCNSim error stage must be a nonempty string")
        super().__init__(message)
        self._stage = stage
        self._evidence = cast(Mapping[str, object], _freeze(dict(evidence or ())))

    @property
    def stage(self) -> str:
        return self._stage

    @property
    def evidence(self) -> Mapping[str, object]:
        return self._evidence


class SCNSimValidationError(SCNSimError):
    """A public request is malformed before an attempt starts."""

    category = "validation"
    kind = "validation_error"


class SCNSimStateError(SCNSimError):
    """The requested operation conflicts with sealed or durable state."""

    category = "state"
    kind = "state_error"


class SCNSimCapabilityError(SCNSimError):
    """The request is valid but outside the implemented capability surface."""

    category = "capability"
    kind = "capability_error"


class SCNSimExecutionError(SCNSimError):
    """An attempted numerical or backend operation did not complete."""

    category = "execution"
    kind = "execution_error"


class SCNSimEvidenceError(SCNSimError):
    """Durable evidence is missing, stale, or internally inconsistent."""

    category = "evidence"
    kind = "evidence_error"


class PlanSealedError(SCNSimStateError):
    """A mutating call targeted an authoring Plan sealed by build or Run."""

    kind = "plan_sealed"


class WorkspacePlanReplacedError(SCNSimStateError):
    """A handle's bound leaf was replaced by another Plan workspace."""

    kind = "workspace_plan_replaced"


class WorkspaceVersioningDowngradeForbidden(SCNSimStateError):
    """A versioned workspace was reopened in destructive replacement mode."""

    kind = "workspace_versioning_downgrade_forbidden"


class UnsupportedRuntimePlatformError(SCNSimCapabilityError):
    """The requested backend operation is unsupported on this platform."""

    kind = "unsupported_runtime_platform"


class PortRealizabilityError(SCNSimValidationError):
    """The selected response View cannot be realized as the required Port network."""

    kind = "port_realizability"


class InvalidDiagonalRootHint(SCNSimValidationError):
    """A diagonal-root hint is missing, nonfinite, or dimensionally invalid."""

    kind = "invalid_diagonal_root_hint"


class InvalidOptimizationSpec(SCNSimValidationError):
    """Optimization variables, objectives, bounds, or controls are invalid."""

    kind = "invalid_optimization_spec"


class DirectResponseFormationError(SCNSimExecutionError):
    """A complete finite Direct S/Y/Z response could not be formed."""

    kind = "direct_response_formation"


class InvalidCandidatePhysicalParameter(SCNSimExecutionError):
    """One optimization candidate produced an invalid primitive parameter."""

    kind = "invalid_candidate_physical_parameter"


class CompilerInvariantError(SCNSimExecutionError):
    """Candidate compilation violated a sealed SCNSim compiler invariant."""

    kind = "compiler_invariant"


class UnsupportedSingularCapacitanceForDiagonalRootV1(SCNSimCapabilityError):
    """V1 diagonal-root evaluation received a singular capacitance matrix."""

    kind = "unsupported_singular_capacitance_for_diagonal_root_v1"


class EliminatedBlockSolveFailure(SCNSimExecutionError):
    """The eliminated coordinate block could not be solved with required residual."""

    kind = "eliminated_block_solve_failure"


class RootSlopeUnresolved(SCNSimExecutionError):
    """The resolved root lacked required local nonzero-slope evidence."""

    kind = "root_slope_unresolved"


class NumericalResolutionUnresolved(SCNSimExecutionError):
    """The numerical procedure ended without its required resolution evidence."""

    kind = "numerical_resolution_unresolved"


class HBCaseFailure(SCNSimExecutionError):
    """One declared HB case failed numerically inside a completed batch.

    The failure is receipt-backed and belongs to exactly one case outcome.  It
    does not turn a valid completed batch into a retryable request failure.
    ``str(failure)`` returns its human-readable message.
    """

    kind = "hb_case_failure"


class RuntimePreparationError(SCNSimExecutionError):
    """The exact declared backend runtime could not be prepared."""

    kind = "runtime_preparation"


class BackendProtocolError(SCNSimExecutionError):
    """The Julia process violated the declared request/outcome protocol."""

    kind = "backend_protocol"


class ResultUnavailableError(SCNSimEvidenceError):
    """No verified success exists for the exact request being resolved."""

    kind = "result_unavailable"


class EvidenceIntegrityError(SCNSimEvidenceError):
    """Stored evidence failed its schema, hash-chain, or artifact checks."""

    kind = "evidence_integrity"


class ScaffoldUnavailableError(SCNSimCapabilityError, NotImplementedError):
    """A declared surface has no candidate implementation in this checkpoint."""

    kind = "scaffold_unavailable"
