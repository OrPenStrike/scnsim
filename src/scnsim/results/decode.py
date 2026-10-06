"""Decode typed Results only from independently verified workspace evidence."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import numpy as np

from .. import units
from ..canonical import canonical_json_bytes, float64_from_hex, quantity_from_envelope, sha256_hex
from ..authoring import CircuitPlan, CoordinateRef, ElectricNodeRef, ParameterRef, ParameterSet
from ..authoring.physical_values import RLGC, RLGCParameterSpec
from ..errors import EvidenceIntegrityError
from ..workspace import VerifiedSuccess
from ..workspace.artifacts import _VerifiedEvidenceLease
from .base import ParameterPointIdentity, ResultIdentity
from .factory import _verified_result
from .decode_hb import _decode_hb_batch_operation
from .decode_result import _decode_result_operation
from .decode_sweep import _decode_parameter_sweep_operation
from .hb import HBBatchResult
from .sweep import ParameterSweepResult


class VerifiedResultDecoder:
    """Captured identity lookups for decoding one verified success chain."""

    __slots__ = (
        "_plan_sha256",
        "_plan",
        "_parameter_lookup",
        "_coordinate_lookup",
    )

    def __init__(
        self,
        *,
        plan_sha256: str,
        plan: CircuitPlan,
        parameter_lookup: Mapping[tuple[str, str], ParameterRef],
        coordinate_lookup: Mapping[str, str | None],
    ) -> None:
        self._plan_sha256 = plan_sha256
        self._plan = plan
        self._parameter_lookup = MappingProxyType(dict(parameter_lookup))
        self._coordinate_lookup = MappingProxyType(dict(coordinate_lookup))

    def _coordinate_id(
        self,
        value: str | ElectricNodeRef | CoordinateRef,
    ) -> str:
        if isinstance(value, ElectricNodeRef) and value.plan is not self._plan:
            raise ValueError("coordinate belongs to another Plan")
        if isinstance(value, CoordinateRef):
            if value.scope.root is not self._plan:
                raise ValueError("coordinate belongs to another Plan")
            key = canonical_json_bytes(
                {"scope": list(value.scope.path()), "id": value.id}
            ).decode("utf-8")
            resolved = self._coordinate_lookup.get(key)
        else:
            identifier = value if isinstance(value, str) else getattr(value, "id", None)
            resolved = self._coordinate_lookup.get(identifier)
        if resolved is None:
            raise ValueError("coordinate is not a public alias in this Plan")
        return resolved

    def _decode_success(
        self,
        success: VerifiedSuccess,
        *,
        bound_spec: object | None = None,
        evidence_lease: _VerifiedEvidenceLease,
    ):
        attempt_sha = sha256_hex(canonical_json_bytes(success.attempt))
        result_sha = success.receipt["result_sha256"]
        identity = _verified_result(
            ResultIdentity,
            plan_sha256=self._plan_sha256,
            request_sha256=str(success.attempt["request_sha256"]),
            attempt_sha256=attempt_sha,
            result_sha256=result_sha,
        )
        result = success.result
        kind = result["result_kind"]
        if kind == "parameter_sweep":
            return self._decode_parameter_sweep(
                identity,
                result,
                success.request,
                success.directory,
                bound_spec=bound_spec,
                evidence_lease=evidence_lease,
            )
        return self._decode_result(
            identity,
            result,
            success.request,
            success.directory,
        )

    def _decode_parameter_ref(self, record: object) -> ParameterRef:
        if not isinstance(record, Mapping) or set(record) != {
            "definitions_id",
            "parameter_id",
        }:
            raise EvidenceIntegrityError(
                "parameter identity is malformed",
                stage="result_decode",
            )
        parameter = self._parameter_lookup.get(
            (record["definitions_id"], record["parameter_id"])
        )
        if parameter is None:
            raise EvidenceIntegrityError(
                "parameter is absent from sealed Plan",
                stage="result_decode",
            )
        return parameter

    def _decode_parameter_value(
        self,
        parameter: ParameterRef,
        record: object,
    ) -> object:
        if not isinstance(record, Mapping):
            raise EvidenceIntegrityError(
                "parameter value is malformed",
                stage="result_decode",
            )
        if not isinstance(parameter.spec, RLGCParameterSpec):
            return quantity_from_envelope(record, registry=units.registry)
        if record.get("type") != "rlgc":
            raise EvidenceIntegrityError(
                "RLGC parameter value is malformed",
                stage="result_decode",
            )

        def matrix(name: str, unit: str) -> object:
            value = record.get(name)
            if not isinstance(value, Mapping):
                raise EvidenceIntegrityError(
                    "RLGC parameter matrix is malformed",
                    stage="result_decode",
                )
            shape = value.get("shape")
            values = value.get("values_f64")
            if (
                not isinstance(shape, list)
                or len(shape) != 2
                or not all(
                    isinstance(item, int) and not isinstance(item, bool)
                    for item in shape
                )
                or not isinstance(values, list)
            ):
                raise EvidenceIntegrityError(
                    "RLGC parameter matrix is malformed",
                    stage="result_decode",
                )
            try:
                decoded = np.asarray(
                    [float64_from_hex(item) for item in values], dtype=np.float64
                ).reshape(tuple(shape))
            except (TypeError, ValueError) as error:
                raise EvidenceIntegrityError(
                    "RLGC parameter matrix is malformed",
                    stage="result_decode",
                ) from error
            return units.registry.Quantity(decoded, unit)

        extraction = record.get("extraction_frequency")
        source = record.get("source")
        if not isinstance(source, Mapping):
            raise EvidenceIntegrityError(
                "RLGC parameter source is malformed",
                stage="result_decode",
            )
        value = RLGC._from_source(
            conductors=tuple(record.get("conductors", ())),
            reference_conductor=record.get("reference_conductor"),
            resistance_per_length=matrix("resistance_per_length", "ohm / meter"),
            inductance_per_length=matrix("inductance_per_length", "henry / meter"),
            conductance_per_length=matrix("conductance_per_length", "siemens / meter"),
            capacitance_per_length=matrix("capacitance_per_length", "farad / meter"),
            extraction_frequency=(
                None
                if extraction is None
                else quantity_from_envelope(extraction, registry=units.registry)
            ),
            source=source,
        )
        if value._record() != record:
            raise EvidenceIntegrityError(
                "decoded RLGC parameter differs from its verified record",
                stage="result_decode",
            )
        return value

    def _decode_parameter_set(self, record: Mapping[str, object]) -> ParameterSet:
        if set(record) != {"type", "bindings", "allow_extrapolation"}:
            raise EvidenceIntegrityError(
                "parameter set is malformed",
                stage="result_decode",
            )
        bindings = record.get("bindings")
        authorizations = record.get("allow_extrapolation")
        if not isinstance(bindings, list) or not isinstance(authorizations, list):
            raise EvidenceIntegrityError(
                "parameter set bindings are malformed",
                stage="result_decode",
            )
        values: dict[ParameterRef, object] = {}
        for binding in bindings:
            if not isinstance(binding, Mapping) or set(binding) != {
                "parameter",
                "value",
            }:
                raise EvidenceIntegrityError(
                    "parameter binding is malformed",
                    stage="result_decode",
                )
            parameter = self._decode_parameter_ref(binding["parameter"])
            values[parameter] = self._decode_parameter_value(
                parameter, binding["value"]
            )
        allowed = tuple(self._decode_parameter_ref(value) for value in authorizations)
        parameters = ParameterSet(values, allow_extrapolation=allowed)
        if parameters._record() != record:
            raise EvidenceIntegrityError(
                "decoded parameter set differs from its verified record",
                stage="result_decode",
            )
        return parameters

    def _decode_result(
        self,
        identity: ResultIdentity | ParameterPointIdentity,
        result: Mapping[str, object],
        request: Mapping[str, object],
        directory: Path,
    ):
        return _decode_result_operation(self, identity, result, request, directory)

    def _decode_parameter_sweep(
        self,
        identity: ResultIdentity,
        result: Mapping[str, object],
        request: Mapping[str, object],
        directory: Path,
        *,
        bound_spec: object | None,
        evidence_lease: _VerifiedEvidenceLease,
    ) -> ParameterSweepResult:
        return _decode_parameter_sweep_operation(self, identity, result, request, directory, bound_spec=bound_spec, evidence_lease=evidence_lease)

    def _decode_hb_batch(
        self,
        identity: ResultIdentity | ParameterPointIdentity,
        result: Mapping[str, object],
        request: Mapping[str, object],
        directory: Path,
    ) -> HBBatchResult:
        return _decode_hb_batch_operation(self, identity, result, request, directory)
