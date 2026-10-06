"""Reconstruct harmonic-balance Results and their sealed evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .. import units
from ..canonical import complex_quantity_from_envelope, float64_from_hex
from ..errors import EvidenceIntegrityError, HBCaseFailure
from ..workspace.artifacts import _direct_request_frequencies, _is_sha256_text, _read_zarr
from .base import BiasState, ParameterPointIdentity, PumpState, ResultIdentity
from .discretization import _decode_discretization
from .factory import _verified_result
from .hb import HBBatchResult, HBCaseOutcome
from .matrix import HBScatteringMatrixResult, MatrixFamilyResult, MatrixView, ReconciliationEvidence
from .derived import TraceResult

def _decode_hb_batch_operation(
    decoder,
    identity: ResultIdentity | ParameterPointIdentity,
    result: Mapping[str, object],
    request: Mapping[str, object],
    directory: Path,
) -> HBBatchResult:
    """Reconstruct one fully verified ordered HB case batch."""

    raw_cases = result.get("cases")
    topology_evidence = result.get("topology_evidence")
    request_spec = request.get("spec")
    declared = (
        request_spec.get("cases") if isinstance(request_spec, Mapping) else None
    )
    trace_declarations = (
        request_spec.get("traces") if isinstance(request_spec, Mapping) else None
    )
    if (
        not isinstance(raw_cases, list)
        or not isinstance(declared, list)
        or len(raw_cases) != len(declared)
        or not isinstance(trace_declarations, list)
        or not isinstance(topology_evidence, Mapping)
    ):
        raise EvidenceIntegrityError(
            "HB batch cases disagree with the request", stage="result_decode"
        )
    declared_traces: dict[str, dict[str, object]] = {}
    for declaration in trace_declarations:
        if not isinstance(declaration, Mapping):
            raise EvidenceIntegrityError(
                "HB request trace declaration is malformed", stage="result_decode"
            )
        identifier = declaration.get("id")
        input_port = declaration.get("input_port")
        output_port = declaration.get("output_port")
        input_mode = declaration.get("input_mode")
        output_mode = declaration.get("output_mode")
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in declared_traces
            or not isinstance(input_port, str)
            or not input_port
            or not isinstance(output_port, str)
            or not output_port
            or not isinstance(input_mode, list)
            or any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in input_mode
            )
            or not isinstance(output_mode, list)
            or any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in output_mode
            )
        ):
            raise EvidenceIntegrityError(
                "HB request trace declaration is malformed", stage="result_decode"
            )
        declared_traces[identifier] = {
            "input_channel": {
                "coordinate": input_port,
                "mode": list(input_mode),
            },
            "output_channel": {
                "coordinate": output_port,
                "mode": list(output_mode),
            },
        }
    outcomes: dict[str, HBCaseOutcome] = {}
    for ordinal, (raw, declaration) in enumerate(zip(raw_cases, declared), 1):
        if not isinstance(raw, Mapping) or not isinstance(declaration, Mapping):
            raise EvidenceIntegrityError(
                "HB case outcome is malformed", stage="result_decode"
            )
        case_id = declaration.get("id")
        if (
            raw.get("case_ordinal") != ordinal
            or raw.get("case_id") != case_id
            or not isinstance(case_id, str)
            or case_id in outcomes
        ):
            raise EvidenceIntegrityError(
                "HB case ordering or identity is malformed", stage="result_decode"
            )
        effective_sources = _decode_hb_effective_sources(
            raw.get("effective_sources")
        )
        status = raw.get("status")
        if status == "failure":
            failure = raw.get("failure")
            if (
                not isinstance(failure, Mapping)
                or set(failure) != {"kind", "stage", "message", "evidence_sha256"}
                or failure.get("kind") != "hb_case_failure"
                or failure.get("stage")
                not in {"operating_point", "linearization", "response_formation"}
                or not isinstance(failure.get("message"), str)
                or not failure["message"]
                or not _is_sha256_text(failure.get("evidence_sha256"))
            ):
                raise EvidenceIntegrityError(
                    "HB case failure is malformed", stage="result_decode"
                )
            outcome_failure = HBCaseFailure(
                failure["message"],
                stage=failure["stage"],
                evidence={"evidence_sha256": failure["evidence_sha256"]},
            )
            outcomes[case_id] = _verified_result(
                HBCaseOutcome,
                id=case_id,
                failure=outcome_failure,
                effective_sources=effective_sources,
                operating_point_closure=None,
                bias_state=None,
                pump_state=None,
                s=None,
                y=None,
                z=None,
                traces=None,
                states=None,
                state_node_map=None,
            )
            continue
        if status != "success":
            raise EvidenceIntegrityError(
                "HB case status is malformed", stage="result_decode"
            )
        artifacts = raw.get("artifacts")
        trace_artifacts = raw.get("traces")
        reconciliation = raw.get("reconciliation")
        state_node_map = raw.get("state_node_map")
        operating_point_closure = raw.get("operating_point_closure")
        if (
            not isinstance(artifacts, Mapping)
            or set(artifacts)
            != {
                "s",
                "y",
                "z",
                "backend_native_s",
                "backend_native_z",
                "states",
                "effective_source_vectors",
            }
            or not isinstance(trace_artifacts, list)
            or not isinstance(reconciliation, Mapping)
            or not isinstance(operating_point_closure, Mapping)
            or not isinstance(state_node_map, list)
            or not state_node_map
        ):
            raise EvidenceIntegrityError(
                "successful HB case evidence is malformed", stage="result_decode"
            )

        frequency = _direct_request_frequencies(request)
        frequencies = units.registry.Quantity(frequency, "hertz")
        decoded_arrays = {
            name: _read_zarr(directory, artifacts[name], complex_values=True)
            for name in (
                "s",
                "y",
                "z",
                "backend_native_s",
                "backend_native_z",
                "states",
                "effective_source_vectors",
            )
        }
        if any(
            not np.all(np.isfinite(values)) for values in decoded_arrays.values()
        ):
            raise EvidenceIntegrityError(
                "HB artifacts contain non-finite values", stage="result_decode"
            )

        def matrix_view(name: str, unit: str) -> MatrixView:
            artifact = artifacts[name]
            if not isinstance(artifact, Mapping):
                raise EvidenceIntegrityError(
                    "HB matrix catalog is malformed", stage="result_decode"
                )
            output_channels = _decode_hb_channel_axis(
                artifact, index=1, kind="output_channel"
            )
            input_channels = _decode_hb_channel_axis(
                artifact, index=2, kind="input_channel"
            )
            values = decoded_arrays[name]
            if values.shape != (
                frequency.size,
                len(output_channels),
                len(input_channels),
            ):
                raise EvidenceIntegrityError(
                    "HB matrix shape disagrees with its channel axes",
                    stage="result_decode",
                )
            coordinates = tuple(artifact.get("coordinate_ids", ()))
            if (
                not coordinates
                or any(
                    not isinstance(item, str) or not item for item in coordinates
                )
                or len(set(coordinates)) != len(coordinates)
            ):
                raise EvidenceIntegrityError(
                    "HB matrix coordinate identity is malformed",
                    stage="result_decode",
                )
            loads = artifact.get("probe_load_state")
            if not isinstance(loads, list):
                raise EvidenceIntegrityError(
                    "HB probe-load evidence is malformed", stage="result_decode"
                )
            probe_loads: dict[str, str] = {}
            for item in loads:
                if (
                    not isinstance(item, Mapping)
                    or set(item) != {"port_id", "state"}
                    or item.get("state") not in {"raw", "compensated"}
                ):
                    raise EvidenceIntegrityError(
                        "HB probe-load evidence is malformed", stage="result_decode"
                    )
                port_id = item.get("port_id")
                if (
                    not isinstance(port_id, str)
                    or not port_id
                    or port_id in probe_loads
                ):
                    raise EvidenceIntegrityError(
                        "HB probe-load identity is malformed", stage="result_decode"
                    )
                probe_loads[port_id] = item["state"]
            return _verified_result(
                MatrixView,
                matrix=units.registry.Quantity(values, unit),
                frequencies=frequencies,
                coordinates=coordinates,
                input_channels=input_channels,
                output_channels=output_channels,
                probe_loads=probe_loads,
            )

        selected_s = matrix_view("s", "dimensionless")
        selected_y = matrix_view("y", "siemens")
        selected_z = matrix_view("z", "ohm")
        native_s = matrix_view("backend_native_s", "dimensionless")
        # Native Z is durable evidence even though the public S surface owns
        # only the native scattering view.
        matrix_view("backend_native_z", "ohm")
        recon = _decode_hb_reconciliation(reconciliation)
        _verify_hb_reconciliation_shape(
            reconciliation,
            selected=np.asarray(selected_s.matrix.magnitude),
            native=np.asarray(native_s.matrix.magnitude),
        )
        traces: dict[str, TraceResult] = {}
        if len(trace_artifacts) != len(trace_declarations):
            raise EvidenceIntegrityError(
                "HB trace catalog disagrees with its request", stage="result_decode"
            )
        selected_matrix = np.asarray(selected_s.matrix.magnitude)
        for artifact, declaration in zip(trace_artifacts, trace_declarations):
            if not isinstance(artifact, Mapping):
                raise EvidenceIntegrityError(
                    "HB trace catalog is malformed", stage="result_decode"
                )
            identifier = artifact.get("id")
            values = _read_zarr(directory, artifact, complex_values=True)
            if (
                not isinstance(declaration, Mapping)
                or not isinstance(identifier, str)
                or not identifier
                or identifier != declaration.get("id")
                or identifier in traces
                or values.shape != (frequency.size,)
                or not np.all(np.isfinite(values))
            ):
                raise EvidenceIntegrityError(
                    "HB trace artifact is malformed", stage="result_decode"
                )
            input_channel = (
                declaration.get("input_port"),
                tuple(declaration.get("input_mode", ())),
            )
            output_channel = (
                declaration.get("output_port"),
                tuple(declaration.get("output_mode", ())),
            )
            try:
                input_index = selected_s.input_channels.index(input_channel)
                output_index = selected_s.output_channels.index(output_channel)
            except ValueError as error:
                raise EvidenceIntegrityError(
                    "HB trace declaration is absent from the selected S basis",
                    stage="result_decode",
                ) from error
            projected = selected_matrix[:, output_index, input_index]
            values_bits = np.ascontiguousarray(values).view(np.uint64)
            projected_bits = np.ascontiguousarray(projected).view(np.uint64)
            if not np.array_equal(values_bits, projected_bits):
                raise EvidenceIntegrityError(
                    "HB trace artifact is not the bit-exact declared projection of selected S",
                    stage="result_decode",
                )
            traces[identifier] = _verified_result(
                TraceResult,
                frequencies=frequencies,
                value=units.registry.Quantity(values, "dimensionless"),
                _parent_identity=identity,
                _presentation={
                    "id": identifier,
                    "family": "S",
                    "case_id": case_id,
                    "input_channel": {
                        "coordinate": input_channel[0],
                        "mode": list(input_channel[1]),
                    },
                    "output_channel": {
                        "coordinate": output_channel[0],
                        "mode": list(output_channel[1]),
                    },
                    "view": request.get("view"),
                    "ref_lineage": result.get("ref_lineage"),
                },
            )
        states = decoded_arrays["states"]
        if states.ndim != 2 or states.shape[1] != len(state_node_map):
            raise EvidenceIntegrityError(
                "HB state evidence disagrees with state_node_map",
                stage="result_decode",
            )
        source_modes = _decode_hb_mode_axis(
            artifacts["effective_source_vectors"], kind="pump_mode"
        )
        source_vectors = decoded_arrays["effective_source_vectors"]
        if source_vectors.ndim != 2 or source_vectors.shape[0] != len(source_modes):
            raise EvidenceIntegrityError(
                "HB effective-source vectors disagree with their mode axis",
                stage="result_decode",
            )
        active_rows = np.any(source_vectors != 0.0, axis=1)
        derived_bias = any(
            active and not any(mode)
            for active, mode in zip(active_rows, source_modes)
        )
        derived_pump = any(
            active and any(mode) for active, mode in zip(active_rows, source_modes)
        )
        if raw.get("bias_state") != ("on" if derived_bias else "off") or raw.get(
            "pump_state"
        ) != ("on" if derived_pump else "off"):
            raise EvidenceIntegrityError(
                "HB BiasState/PumpState disagrees with effective source vectors",
                stage="result_decode",
            )
        outcomes[case_id] = _verified_result(
            HBCaseOutcome,
            id=case_id,
            failure=None,
            effective_sources=effective_sources,
            operating_point_closure=operating_point_closure,
            bias_state=BiasState(raw["bias_state"]),
            pump_state=PumpState(raw["pump_state"]),
            s=_verified_result(
                HBScatteringMatrixResult,
                view=selected_s,
                backend_native=native_s,
                reconciliation=recon,
                _parent_identity=identity,
                _presentation={
                    "family": "S",
                    "case_id": case_id,
                    "view": request.get("view"),
                    "ref_lineage": result.get("ref_lineage"),
                },
            ),
            y=_verified_result(
                MatrixFamilyResult,
                view=selected_y,
                _parent_identity=identity,
                _presentation={
                    "family": "Y",
                    "case_id": case_id,
                    "view": request.get("view"),
                    "ref_lineage": result.get("ref_lineage"),
                },
            ),
            z=_verified_result(
                MatrixFamilyResult,
                view=selected_z,
                _parent_identity=identity,
                _presentation={
                    "family": "Z",
                    "case_id": case_id,
                    "view": request.get("view"),
                    "ref_lineage": result.get("ref_lineage"),
                },
            ),
            traces=traces,
            states=units.registry.Quantity(states, "weber"),
            state_node_map=tuple(state_node_map),
        )
    return _verified_result(
        HBBatchResult,
        identity=identity,
        discretization=_decode_discretization(result.get("discretization")),
        cases=outcomes,
        topology_evidence=topology_evidence,
        _presentation={"declared_traces": declared_traces},
    )


def _decode_hb_effective_sources(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list):
        raise EvidenceIntegrityError(
            "HB effective-source evidence is malformed", stage="result_decode"
        )
    decoded: list[Mapping[str, object]] = []
    identities: set[tuple[str, tuple[int, ...]]] = set()
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {
            "drive_id",
            "mode",
            "coefficient",
            "generated_conjugate",
            "backend_binding",
            "injection_map_sha256",
        }:
            raise EvidenceIntegrityError(
                "HB effective-source evidence is malformed", stage="result_decode"
            )
        drive_id = item.get("drive_id")
        mode = item.get("mode")
        conjugate = item.get("generated_conjugate")
        backend = item.get("backend_binding")
        if (
            not isinstance(drive_id, str)
            or not drive_id
            or not isinstance(mode, list)
            or any(
                not isinstance(entry, int) or isinstance(entry, bool) for entry in mode
            )
            or not isinstance(conjugate, Mapping)
            or set(conjugate) != {"mode", "coefficient"}
            or not isinstance(conjugate.get("mode"), list)
            or any(
                not isinstance(entry, int) or isinstance(entry, bool)
                for entry in conjugate["mode"]
            )
            or not isinstance(backend, Mapping)
            or set(backend)
            != {
                "representative_mode",
                "representative_index",
                "coefficient",
                "coefficient_convention",
            }
            or not isinstance(backend.get("representative_mode"), list)
            or any(
                not isinstance(entry, int) or isinstance(entry, bool)
                for entry in backend["representative_mode"]
            )
            or not isinstance(backend.get("representative_index"), int)
            or isinstance(backend.get("representative_index"), bool)
            or backend["representative_index"] < 0
            or backend.get("coefficient_convention")
            != "exp_plus_i_m_dot_omega_t_josephsoncircuits_source"
            or not _is_sha256_text(item.get("injection_map_sha256"))
        ):
            raise EvidenceIntegrityError(
                "HB effective-source identity is malformed", stage="result_decode"
            )
        key = (drive_id, tuple(mode))
        if key in identities:
            raise EvidenceIntegrityError(
                "HB effective-source identity is duplicated", stage="result_decode"
            )
        identities.add(key)
        decoded.append(
            {
                "drive_id": drive_id,
                "mode": tuple(mode),
                "coefficient": complex_quantity_from_envelope(
                    item["coefficient"], registry=units.registry
                ),
                "generated_conjugate": {
                    "mode": tuple(conjugate["mode"]),
                    "coefficient": complex_quantity_from_envelope(
                        conjugate["coefficient"], registry=units.registry
                    ),
                },
                "backend_binding": {
                    "representative_mode": tuple(backend["representative_mode"]),
                    "representative_index": backend["representative_index"],
                    "coefficient": complex_quantity_from_envelope(
                        backend["coefficient"], registry=units.registry
                    ),
                    "coefficient_convention": backend["coefficient_convention"],
                },
                "injection_map_sha256": item["injection_map_sha256"],
            }
        )
    return tuple(decoded)

def _decode_hb_channel_axis(
    artifact: Mapping[str, object],
    *,
    index: int,
    kind: str,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    axes = artifact.get("axes")
    if (
        not isinstance(axes, list)
        or len(axes) != 3
        or not isinstance(axes[index], Mapping)
    ):
        raise EvidenceIntegrityError(
            "HB matrix axes are malformed", stage="result_decode"
        )
    axis = axes[index]
    values = axis.get("values")
    if axis.get("kind") != kind or not isinstance(values, list) or not values:
        raise EvidenceIntegrityError(
            "HB matrix channel axis is malformed", stage="result_decode"
        )
    channels: list[tuple[str, tuple[int, ...]]] = []
    for value in values:
        if not isinstance(value, Mapping) or set(value) != {"coordinate", "mode"}:
            raise EvidenceIntegrityError(
                "HB matrix channel label is malformed", stage="result_decode"
            )
        coordinate = value.get("coordinate")
        mode = value.get("mode")
        if (
            not isinstance(coordinate, str)
            or not coordinate
            or not isinstance(mode, list)
            or any(
                not isinstance(entry, int) or isinstance(entry, bool) for entry in mode
            )
        ):
            raise EvidenceIntegrityError(
                "HB matrix channel label is malformed", stage="result_decode"
            )
        channels.append((coordinate, tuple(mode)))
    if len(set(channels)) != len(channels):
        raise EvidenceIntegrityError(
            "HB matrix channel labels are duplicated", stage="result_decode"
        )
    return tuple(channels)

def _decode_hb_mode_axis(artifact: object, *, kind: str) -> tuple[tuple[int, ...], ...]:
    if not isinstance(artifact, Mapping):
        raise EvidenceIntegrityError(
            "HB mode artifact is malformed", stage="result_decode"
        )
    axes = artifact.get("axes")
    if not isinstance(axes, list) or not axes or not isinstance(axes[0], Mapping):
        raise EvidenceIntegrityError("HB mode axis is malformed", stage="result_decode")
    axis = axes[0]
    values = axis.get("values")
    if (
        axis.get("kind") != kind
        or not isinstance(values, list)
        or (not values and kind != "pump_mode")
    ):
        raise EvidenceIntegrityError("HB mode axis is malformed", stage="result_decode")
    modes: list[tuple[int, ...]] = []
    for value in values:
        if not isinstance(value, list) or any(
            not isinstance(entry, int) or isinstance(entry, bool) for entry in value
        ):
            raise EvidenceIntegrityError(
                "HB mode-axis tuple is malformed", stage="result_decode"
            )
        modes.append(tuple(value))
    if len(set(modes)) != len(modes):
        raise EvidenceIntegrityError(
            "HB mode axis repeats a tuple", stage="result_decode"
        )
    return tuple(modes)

def _decode_hb_reconciliation(value: Mapping[str, object]) -> ReconciliationEvidence:
    comparable = value.get("comparable")
    expected = {
        "comparable",
        "reason",
        "last_comparable_ancestor",
        "normalization",
        "evidence_sha256",
        *(("residual_f64", "coordinate_projection") if comparable is True else ()),
    }
    if (
        isinstance(comparable, bool)
        and set(value) == expected
        and value.get("normalization") == "backend_photon_flux_to_scnsim_power_wave"
        and _is_sha256_text(value.get("last_comparable_ancestor"))
        and _is_sha256_text(value.get("evidence_sha256"))
        and (
            (comparable and value.get("reason") is None)
            or (
                not comparable
                and value.get("reason")
                in {
                    "topology",
                    "load_or_ptc",
                    "reference_plane",
                    "reference_matrix",
                    "signed_frequency_grid",
                    "channel_basis",
                    "normalization",
                }
            )
        )
    ):
        try:
            residual = float64_from_hex(value["residual_f64"]) if comparable else None
        except (TypeError, ValueError) as error:
            raise EvidenceIntegrityError(
                "HB reconciliation residual is malformed", stage="result_decode"
            ) from error
        if residual is None or (math.isfinite(residual) and residual >= 0.0):
            return _verified_result(
                ReconciliationEvidence,
                comparable=comparable,
                reason=value.get("reason"),
                last_comparable_ancestor=value["last_comparable_ancestor"],
                residual=residual,
                evidence_sha256=value["evidence_sha256"],
            )
    raise EvidenceIntegrityError(
        "HB reconciliation evidence is malformed", stage="result_decode"
    )

def _verify_hb_reconciliation_shape(
    evidence: Mapping[str, object],
    *,
    selected: np.ndarray,
    native: np.ndarray,
) -> None:
    """Check comparable HB evidence against decoded matrix dimensions."""

    if evidence.get("comparable") is not True:
        return
    projection = evidence.get("coordinate_projection")
    if not isinstance(projection, Mapping) or set(projection) != {
        "shape",
        "values_f64",
    }:
        raise EvidenceIntegrityError(
            "HB reconciliation coordinate projection is malformed",
            stage="result_decode",
        )
    shape = projection.get("shape")
    values = projection.get("values_f64")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in shape
        )
        or not isinstance(values, list)
        or len(values) != shape[0] * shape[1]
    ):
        raise EvidenceIntegrityError(
            "HB reconciliation coordinate projection is malformed",
            stage="result_decode",
        )
    try:
        coordinates = [float64_from_hex(value) for value in values]
    except (TypeError, ValueError) as error:
        raise EvidenceIntegrityError(
            "HB reconciliation coordinate projection is malformed",
            stage="result_decode",
        ) from error
    if any(not math.isfinite(value) for value in coordinates):
        raise EvidenceIntegrityError(
            "HB reconciliation coordinate projection is malformed",
            stage="result_decode",
        )
    rows, columns = shape
    if (
        selected.ndim != 3
        or native.ndim != 3
        or selected.shape[0] != native.shape[0]
        or selected.shape[1] != selected.shape[2]
        or native.shape[1] != native.shape[2]
        or selected.shape[1] % rows != 0
        or native.shape[1] % columns != 0
        or selected.shape[1] // rows != native.shape[1] // columns
    ):
        raise EvidenceIntegrityError(
            "HB reconciliation matrices disagree with their coordinate projection",
            stage="result_decode",
        )
