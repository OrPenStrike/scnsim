"""Scalar scientific result payload verification."""

from __future__ import annotations

from collections.abc import Mapping

from ...canonical import canonical_json_bytes as _canonical_bytes, sha256_hex as _sha256
from .common import _integrity, _valid_sha, _verify_quantity_role
from .requests import _verify_v1_lineage
from .result_artifacts import (
    _verify_null_vector_artifact,
    _verify_residue_coupling_evidence,
    _verify_root_like_result,
)

def _verify_scalar_result_payload(result: Mapping[str, object], request: Mapping[str, object], plan: Mapping[str, object], kind: object, common: set[str]) -> None:
    if kind in {"diagonal_root", "operator_element_root"}:
        if set(result) != common | {"scalar_catalog", "array_catalog"} or result.get("array_catalog") != {}:
            raise _integrity("Element-root Result envelope is open or has array payloads.")
        scalars = result.get("scalar_catalog")
        expected = {"root", "frequency", "slope"} | ({"linewidth"} if kind == "diagonal_root" else set())
        if not isinstance(scalars, dict) or set(scalars) != expected:
            raise _integrity("Element-root scalar catalog is incomplete.")
        _verify_quantity_role(scalars["root"], complex_value=True, unit="radian / second", dimensionality="inverse_time")
        _verify_quantity_role(scalars["frequency"], complex_value=False, unit="hertz", dimensionality="inverse_time")
        if kind == "diagonal_root":
            _verify_quantity_role(scalars["linewidth"], complex_value=False, unit="hertz", dimensionality="inverse_time")
        _verify_quantity_role(scalars["slope"], complex_value=True, unit="siemens", dimensionality="conductance")
    elif kind == "hybridized_pole":
        _verify_root_like_result(result, {"root", "frequency", "linewidth", "slope", "evidence_sha256"})
        arrays = result.get("array_catalog")
        if not isinstance(arrays, dict) or set(arrays) != {"null_vector"}:
            raise _integrity("Hybridized-pole artifact catalog is incomplete.")
        terminal, _ = _verify_v1_lineage(request.get("ref_lineage"), plan)
        _verify_null_vector_artifact(arrays["null_vector"], terminal)
    elif kind == "transfer_zero":
        if set(result) != common | {"scalar_catalog", "array_catalog"} or result.get("array_catalog") != {}:
            raise _integrity("Transfer-zero Result envelope is malformed.")
        scalars = result.get("scalar_catalog")
        if not isinstance(scalars, dict) or set(scalars) != {"zero", "frequency", "numerator_slope", "denominator", "evidence_sha256"}:
            raise _integrity("Transfer-zero scalar catalog is incomplete.")
        _verify_quantity_role(scalars["zero"], complex_value=True, unit="radian / second", dimensionality="inverse_time")
        _verify_quantity_role(scalars["frequency"], complex_value=False, unit="hertz", dimensionality="inverse_time")
        for field in ("numerator_slope", "denominator"):
            _verify_quantity_role(scalars[field], complex_value=True, unit="dimensionless", dimensionality="dimensionless")
        _valid_sha(scalars["evidence_sha256"])
    elif kind == "residue_normalized_coupling":
        if set(result) != common | {"scalar_catalog", "array_catalog"} or result.get("array_catalog") != {}:
            raise _integrity("Residue coupling Result envelope is malformed.")
        scalars = result.get("scalar_catalog")
        expected_scalars = {
            "coupling", "magnitude", "branch_a_residue", "branch_b_residue",
            "branch_a_root", "branch_b_root", "evaluation_omega", "evidence_sha256",
        }
        if not isinstance(scalars, dict) or set(scalars) != expected_scalars:
            raise _integrity("Residue coupling scalar catalog is incomplete.")
        coupling_evidence = {
            name: scalars[name]
            for name in ("branch_a_root", "branch_b_root", "evaluation_omega", "coupling")
        }
        request_spec = request.get("spec")
        if not isinstance(request_spec, Mapping):
            raise _integrity("Residue coupling request Spec is malformed.")
        _verify_residue_coupling_evidence(
            coupling_evidence,
            request_spec,
            expected_projection="magnitude",
            projected_value=scalars["magnitude"],
        )
        for field in ("branch_a_residue", "branch_b_residue"):
            _verify_quantity_role(scalars[field], complex_value=True, unit="ohm", dimensionality="resistance")
        expected_evidence = {
            "schema": "scnsim.residue_normalized_coupling_evidence",
            "schema_version": 2,
            **coupling_evidence,
        }
        if scalars.get("evidence_sha256") != _sha256(_canonical_bytes(expected_evidence)):
            raise _integrity("Residue coupling evidence digest disagrees with its scalar catalog.")
    elif kind == "response_element":
        if set(result) != common | {"scalar_catalog", "array_catalog"} or result.get("array_catalog") != {}:
            raise _integrity("Response-element Result envelope is malformed.")
        scalars = result.get("scalar_catalog")
        if not isinstance(scalars, dict) or set(scalars) != {"family", "value", "magnitude", "real", "imag", "evidence_sha256"}:
            raise _integrity("Response-element scalar catalog is incomplete.")
        role = {"S": ("dimensionless", "dimensionless"), "Y": ("siemens", "conductance"), "Z": ("ohm", "resistance")}.get(scalars.get("family"))
        if role is None:
            raise _integrity("Response-element family is malformed.")
        _verify_quantity_role(scalars["value"], complex_value=True, unit=role[0], dimensionality=role[1])
        for field in ("magnitude", "real", "imag"):
            _verify_quantity_role(scalars[field], complex_value=False, unit=role[0], dimensionality=role[1])
        _valid_sha(scalars["evidence_sha256"])
    else:
        raise _integrity("Result operation is outside the supported runtime.")
