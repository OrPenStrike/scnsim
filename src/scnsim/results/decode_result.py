"""Reconstruct ordinary scientific Result families from verified payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import numpy as np

from .. import units
from ..canonical import complex_quantity_from_envelope, float64_from_hex, quantity_from_envelope
from ..errors import EvidenceIntegrityError
from ..workspace.artifacts import (
    _direct_request_frequencies, _operator_request_frequencies, _read_json_artifact,
    _read_zarr, _validate_direct_values,
)
from .base import ParameterPointIdentity, ResultIdentity
from .discretization import _decode_discretization
from .factory import _verified_result
from .derived import TraceResult
from .matrix import (
    DiagonalRootResult, DirectQuantityResult, DirectSolveResult, MatrixFamilyResult,
    MatrixView, OperatorElementRootResult, OperatorPointResult, OperatorResult,
    ScatteringMatrixResult,
)
from .optimization import OptimizationBest, OptimizationResult

def _decode_result_operation(
    decoder,
    identity: ResultIdentity | ParameterPointIdentity,
    result: Mapping[str, object],
    request: Mapping[str, object],
    directory: Path,
):
    """Decode one already-verified ordinary scientific payload."""

    kind = result["result_kind"]
    discretization = _decode_discretization(result.get("discretization"))
    if kind == "hb_batch":
        return decoder._decode_hb_batch(identity, result, request, directory)
    if kind == "direct_response":
        arrays = result["array_catalog"]
        frequency = _read_zarr(
            directory, arrays["frequencies"], complex_values=False
        )
        s = _read_zarr(directory, arrays["s"], complex_values=True)
        y = _read_zarr(directory, arrays["y"], complex_values=True)
        z = _read_zarr(directory, arrays["z"], complex_values=True)
        _validate_direct_values(
            frequency,
            s,
            y,
            z,
            expected_frequency=_direct_request_frequencies(request),
            stage="result_decode",
        )
        frequencies = units.registry.Quantity(frequency, "hertz")
        coordinates = tuple(arrays["s"]["coordinate_ids"])
        expected_shape = (frequency.size, len(coordinates), len(coordinates))
        if (
            not coordinates
            or len(set(coordinates)) != len(coordinates)
            or any(
                not isinstance(coordinate, str) or not coordinate
                for coordinate in coordinates
            )
            or any(values.shape != expected_shape for values in (s, y, z))
            or tuple(arrays["y"].get("coordinate_ids", ())) != coordinates
            or tuple(arrays["z"].get("coordinate_ids", ())) != coordinates
        ):
            raise EvidenceIntegrityError(
                "Direct arrays disagree with the selected N-port basis",
                stage="result_decode",
            )
        channels = tuple((coordinate, ()) for coordinate in coordinates)
        loads = {
            item["port_id"]: item["state"]
            for item in arrays["s"]["probe_load_state"]
        }

        def view(values: np.ndarray, unit: str) -> MatrixView:
            return _verified_result(
                MatrixView,
                matrix=units.registry.Quantity(values, unit),
                frequencies=frequencies,
                coordinates=coordinates,
                input_channels=channels,
                output_channels=channels,
                probe_loads=loads,
            )

        trace_spec = request.get("spec")
        declared_traces = (
            trace_spec.get("traces") if isinstance(trace_spec, Mapping) else None
        )
        if not isinstance(declared_traces, list):
            raise EvidenceIntegrityError(
                "Direct request trace declarations are malformed",
                stage="result_decode",
            )
        trace_results: dict[str, TraceResult] = {}
        for trace in declared_traces:
            if not isinstance(trace, Mapping):
                raise EvidenceIntegrityError(
                    "Direct trace declaration is malformed", stage="result_decode"
                )
            identifier = trace.get("id")
            input_coordinate = trace.get("input_port")
            output_coordinate = trace.get("output_port")
            input_mode = trace.get("input_mode")
            output_mode = trace.get("output_mode")
            if (
                not isinstance(identifier, str)
                or not isinstance(input_coordinate, str)
                or not isinstance(output_coordinate, str)
                or input_mode != []
                or output_mode != []
                or identifier in trace_results
                or input_coordinate not in coordinates
                or output_coordinate not in coordinates
            ):
                raise EvidenceIntegrityError(
                    "Direct trace does not bind the selected S basis",
                    stage="result_decode",
                )
            trace_results[identifier] = _verified_result(
                TraceResult,
                frequencies=frequencies,
                value=units.registry.Quantity(
                    s[
                        :,
                        coordinates.index(output_coordinate),
                        coordinates.index(input_coordinate),
                    ],
                    "dimensionless",
                ),
                _parent_identity=identity,
                _presentation={
                    "id": identifier,
                    "family": "S",
                    "input_channel": {"coordinate": input_coordinate, "mode": []},
                    "output_channel": {"coordinate": output_coordinate, "mode": []},
                    "view": request.get("view"),
                    "ref_lineage": result.get("ref_lineage"),
                },
            )

        return _verified_result(
            DirectSolveResult,
            identity=identity,
            discretization=discretization,
            frequencies=frequencies,
            s=_verified_result(
                ScatteringMatrixResult,
                view=view(s, "dimensionless"),
                _parent_identity=identity,
                _presentation={
                    "family": "S",
                    "view": request.get("view"),
                    "ref_lineage": result.get("ref_lineage"),
                },
            ),
            y=_verified_result(
                MatrixFamilyResult,
                view=view(y, "siemens"),
                _parent_identity=identity,
                _presentation={
                    "family": "Y",
                    "view": request.get("view"),
                    "ref_lineage": result.get("ref_lineage"),
                },
            ),
            z=_verified_result(
                MatrixFamilyResult,
                view=view(z, "ohm"),
                _parent_identity=identity,
                _presentation={
                    "family": "Z",
                    "view": request.get("view"),
                    "ref_lineage": result.get("ref_lineage"),
                },
            ),
            traces=trace_results,
        )
    if kind in {"diagonal_root", "operator_element_root"}:
        scalars = result["scalar_catalog"]
        return _verified_result(
            DiagonalRootResult if kind == "diagonal_root" else OperatorElementRootResult,
            identity=identity,
            discretization=discretization,
            root=complex_quantity_from_envelope(
                scalars["root"], registry=units.registry
            ),
            frequency=quantity_from_envelope(
                scalars["frequency"], registry=units.registry
            ),
            linewidth=quantity_from_envelope(scalars["linewidth"], registry=units.registry) if kind == "diagonal_root" else None,
            slope=complex_quantity_from_envelope(
                scalars["slope"], registry=units.registry
            ),
            value=None,
            magnitude=None,
            real=None,
            imag=None,
            _presentation={
                "view": request.get("view"),
                "ref_lineage": result.get("ref_lineage"),
                "spec": request.get("spec"),
            },
        )
    if kind == "hybridized_pole":
        scalars = result["scalar_catalog"]
        return _verified_result(
            DirectQuantityResult,
            identity=identity,
            discretization=discretization,
            root=complex_quantity_from_envelope(
                scalars["root"], registry=units.registry
            ),
            frequency=quantity_from_envelope(
                scalars["frequency"], registry=units.registry
            ),
            linewidth=quantity_from_envelope(
                scalars["linewidth"], registry=units.registry
            ),
            slope=complex_quantity_from_envelope(
                scalars["slope"], registry=units.registry
            ),
            _presentation={
                "view": request.get("view"),
                "ref_lineage": result.get("ref_lineage"),
                "spec": request.get("spec"),
            },
        )
    if kind == "transfer_zero":
        scalars = result["scalar_catalog"]
        return _verified_result(
            DirectQuantityResult,
            identity=identity,
            discretization=discretization,
            zero=complex_quantity_from_envelope(
                scalars["zero"], registry=units.registry
            ),
            frequency=quantity_from_envelope(
                scalars["frequency"], registry=units.registry
            ),
            numerator_slope=complex_quantity_from_envelope(
                scalars["numerator_slope"], registry=units.registry
            ),
            denominator=complex_quantity_from_envelope(
                scalars["denominator"], registry=units.registry
            ),
            _presentation={
                "view": request.get("view"),
                "ref_lineage": result.get("ref_lineage"),
                "spec": request.get("spec"),
            },
        )
    if kind == "residue_normalized_coupling":
        scalars = result["scalar_catalog"]
        coupling = complex_quantity_from_envelope(
            scalars["coupling"], registry=units.registry
        )
        return _verified_result(
            DirectQuantityResult,
            identity=identity,
            discretization=discretization,
            coupling=coupling,
            magnitude=quantity_from_envelope(
                scalars["magnitude"], registry=units.registry
            ),
            real=units.registry.Quantity(
                complex(coupling.magnitude).real, coupling.units
            ),
            imag=units.registry.Quantity(
                complex(coupling.magnitude).imag, coupling.units
            ),
            branch_a_residue=complex_quantity_from_envelope(
                scalars["branch_a_residue"], registry=units.registry
            ),
            branch_b_residue=complex_quantity_from_envelope(
                scalars["branch_b_residue"], registry=units.registry
            ),
            evaluation_omega=complex_quantity_from_envelope(
                scalars["evaluation_omega"], registry=units.registry
            ),
            _presentation={
                "view": request.get("view"),
                "ref_lineage": result.get("ref_lineage"),
                "spec": request.get("spec"),
            },
        )
    if kind == "response_element":
        scalars = result["scalar_catalog"]
        return _verified_result(
            DirectQuantityResult,
            identity=identity,
            discretization=discretization,
            family=scalars["family"],
            value=complex_quantity_from_envelope(
                scalars["value"], registry=units.registry
            ),
            magnitude=quantity_from_envelope(
                scalars["magnitude"], registry=units.registry
            ),
            real=quantity_from_envelope(scalars["real"], registry=units.registry),
            imag=quantity_from_envelope(scalars["imag"], registry=units.registry),
            _presentation={
                "view": request.get("view"),
                "ref_lineage": result.get("ref_lineage"),
                "spec": request.get("spec"),
            },
        )
    if kind == "operator":
        arrays = result["array_catalog"]
        frequency = _read_zarr(
            directory, arrays["frequencies"], complex_values=False
        )
        matrix = _read_zarr(directory, arrays["operator"], complex_values=True)
        coordinates = tuple(arrays["operator"].get("coordinate_ids", ()))
        expected = _operator_request_frequencies(request)
        if (
            frequency.shape != expected.shape
            or not np.array_equal(
                frequency.view(np.uint64), expected.view(np.uint64)
            )
            or matrix.shape != (frequency.size, len(coordinates), len(coordinates))
            or not coordinates
            or len(set(coordinates)) != len(coordinates)
            or any(not isinstance(value, str) or not value for value in coordinates)
            or not np.all(np.isfinite(matrix))
        ):
            raise EvidenceIntegrityError(
                "operator artifacts disagree with the request basis",
                stage="result_decode",
            )
        frequencies = units.registry.Quantity(frequency, "hertz")
        points = tuple(
            _verified_result(
                OperatorPointResult,
                frequency=units.registry.Quantity(float(value), "hertz"),
                matrix=units.registry.Quantity(matrix[index], "siemens / second"),
                coordinates=coordinates,
            )
            for index, value in enumerate(frequency)
        )
        return _verified_result(OperatorResult, identity=identity, points=points, discretization=discretization)
    if kind == "optimization":
        best = result["best"]
        parameters = decoder._decode_parameter_set(best["parameters"])
        baseline = result["baseline"]
        initial_parameters = decoder._decode_parameter_set(baseline["parameters"])
        ledger = tuple(
            _read_json_artifact(directory, artifact)
            for artifact in result["ledger_artifacts"]
        )
        return _verified_result(
            OptimizationResult,
            identity=identity,
            discretization=discretization,
            best=_verified_result(
                OptimizationBest,
                parameters=parameters,
                cost=float64_from_hex(best["cost_f64"]),
                discretization=_decode_discretization(best.get("discretization")),
            ),
            ledger=ledger,
            candidate_discretization=tuple(
                _decode_discretization(candidate.get("discretization"))
                for generation in ledger for candidate in cast(Sequence[Mapping[str, object]], generation["candidates"])
            ),
            _presentation={
                "initial_parameters": initial_parameters,
                "initial_candidate": baseline,
                "objectives": tuple(request["spec"]["objectives"]),
                "variables": tuple(request["spec"]["variables"]),
                "best_evaluation_ordinal": best["evaluation_ordinal"],
            },
        )
    raise EvidenceIntegrityError(
        "verified Result kind is outside the runtime",
        stage="result_decode",
        evidence={"result_kind": kind},
    )
