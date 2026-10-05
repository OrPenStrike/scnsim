"""Direct response and operator payload verification."""

from __future__ import annotations

from collections.abc import Mapping

from .common import _integrity
from .requests import _verify_v1_direct_spec, _verify_v1_lineage
from .result_artifacts import (
    _expected_probe_load_state,
    _verify_direct_artifact,
    _verify_operator_artifact,
)

def _verify_direct_response_result(result: Mapping[str, object], request: Mapping[str, object], plan: Mapping[str, object], common: set[str]) -> None:
    if set(result) != common | {"scalar_catalog", "array_catalog"} or result.get("scalar_catalog") != {}:
        raise _integrity("Direct Result envelope is open or has scalar payloads.")
    catalog = result.get("array_catalog")
    if not isinstance(catalog, dict) or set(catalog) != {"frequencies", "s", "y", "z"}:
        raise _integrity("Direct Result array catalog is incomplete.")
    terminal, port_realizable = _verify_v1_lineage(request.get("ref_lineage"), plan)
    expected_probes = _expected_probe_load_state(request.get("ref_lineage"))
    _verify_v1_direct_spec(request.get("spec"), terminal, port_realizable)
    frequencies = request["spec"]["frequencies"]
    expected_frequency_count = len(frequencies)
    frequency_count = _verify_direct_artifact(catalog["frequencies"], "frequencies")
    if frequency_count != expected_frequency_count:
        raise _integrity("Direct artifacts disagree with the requested frequency grid length.")
    for role in ("s", "y", "z"):
        if _verify_direct_artifact(catalog[role], role) != frequency_count:
            raise _integrity("Direct artifacts disagree on frequency-axis length.")
        if (
            catalog[role].get("coordinate_ids") != terminal
            or catalog[role].get("coordinate_ids") != catalog["s"].get("coordinate_ids")
            or catalog[role].get("probe_load_state") != expected_probes
        ):
            raise _integrity("Direct artifacts disagree with the request View or each other.")

def _verify_operator_result(result: Mapping[str, object], request: Mapping[str, object], plan: Mapping[str, object], common: set[str]) -> None:
    if set(result) != common | {"scalar_catalog", "array_catalog"} or result.get("scalar_catalog") != {}:
        raise _integrity("Operator Result envelope is malformed.")
    catalog = result.get("array_catalog")
    if not isinstance(catalog, dict) or set(catalog) != {"frequencies", "operator"}:
        raise _integrity("Operator artifact catalog is incomplete.")
    count = _verify_direct_artifact(catalog["frequencies"], "frequencies")
    spec_frequencies = request.get("spec", {}).get("frequencies") if isinstance(request.get("spec"), dict) else None
    terminal, _ = _verify_v1_lineage(request.get("ref_lineage"), plan)
    if not isinstance(spec_frequencies, list) or count != len(spec_frequencies):
        raise _integrity("Operator frequency artifact disagrees with its request grid.")
    _verify_operator_artifact(
        catalog["operator"], count, terminal,
        _expected_probe_load_state(request.get("ref_lineage")),
    )
