"""Independent A/B witnesses for one emitted structured diagram scene.

Expected facts come from the immutable resolved point/compiler evidence.
Observed facts come from visible neutral-scene geometry only; renderer
correlation, placement, semantic tags, and captured net names are never read.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from .._authoring_snapshot import ResolvedPlanPoint
from .._canonical import canonical_json_bytes, sha256_hex
from ..errors import SCNSimValidationError
from .expected import build_point_expected
from .reconstruction import reconstruct_authoring
from .scene import NeutralScene


def _fail(message: str, **evidence: object) -> SCNSimValidationError:
    return SCNSimValidationError(message, stage="schematic_audit", evidence=evidence)


def _compare(
    name: str,
    expected: Mapping[str, object],
    observed: Mapping[str, object],
) -> None:
    expected_bytes = canonical_json_bytes(expected)
    observed_bytes = canonical_json_bytes(observed)
    if expected_bytes != observed_bytes:
        raise _fail(
            f"observed scene {name} differs from its independent expected manifest",
            expected_sha256=sha256_hex(expected_bytes),
            observed_sha256=sha256_hex(observed_bytes),
        )


def _value_rows(rows: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    result: dict[str, object] = {}
    for row in rows:
        if row.get("kind") not in {"displayed_parameter_value", "displayed_coupling_polarity", "port_impedance"}:
            continue
        identity = row.get("identity")
        value = row.get("comparison_value", row.get("value"))
        if not isinstance(identity, str) or not isinstance(value, Mapping):
            raise _fail("visible scalar row is malformed")
        if identity in result:
            raise _fail(
                "visible scalar evidence duplicates one canonical identity",
                identity=identity,
            )
        result[identity] = value
    return result


def _port_value_id(port_id: str) -> str:
    return canonical_json_bytes(
        {"kind": "port_impedance", "port_id": port_id}
    ).decode("utf-8")


def witness_authoring(
    point: ResolvedPlanPoint,
    scene: NeutralScene,
    *,
    show_values: bool,
) -> tuple[Mapping[str, object], Mapping[str, object], tuple[Mapping[str, object], ...]]:
    """Reconstruct and compare exact authoring A/B visible evidence."""

    if not isinstance(point, ResolvedPlanPoint) or not isinstance(scene, NeutralScene):
        raise TypeError("visible authoring witness requires point and NeutralScene")
    if not isinstance(show_values, bool):
        raise TypeError("show_values must be boolean")
    expected = build_point_expected(point, representation="authoring")
    electrical, structural, rows = reconstruct_authoring(
        scene, show_values=show_values
    )
    _compare("authoring electrical reconstruction", expected.electrical, electrical)
    _compare("authoring structural reconstruction", expected.structural, structural)

    expected_values = dict(expected.expected_values)
    if not show_values:
        from .equivalence import coupling_sign

        ports = cast(
            Sequence[Mapping[str, object]], expected.electrical.get("ports", ())
        )
        required = {
            identity: expected_values[identity]
            for identity in (
                _port_value_id(cast(str, port["port_id"])) for port in ports
            )
        }
        for coupling in expected.electrical["couplings"]:
            identity = canonical_json_bytes({
                "kind": "coupling_coefficient", "coupling_id": coupling["coupling_id"],
            }).decode()
            required[identity] = coupling_sign(expected_values[identity])
    else:
        required = expected_values
    observed_values = _value_rows(rows)
    if canonical_json_bytes(required) != canonical_json_bytes(observed_values):
        raise _fail(
            "visible scalar labels do not completely reconstruct the resolved point",
            expected_sha256=sha256_hex(required),
            observed_sha256=sha256_hex(observed_values),
        )
    return electrical, structural, rows


def witness_compiled(
    point: ResolvedPlanPoint,
    scene: NeutralScene,
    compiled_evidence: Mapping[str, object],
    *,
    show_values: bool,
) -> tuple[Mapping[str, object], Mapping[str, object], tuple[Mapping[str, object], ...]]:
    """Reconstruct and compare exact same-point compiled visible evidence."""

    if not isinstance(point, ResolvedPlanPoint) or not isinstance(scene, NeutralScene):
        raise TypeError("visible compiled witness requires point and NeutralScene")
    if not isinstance(compiled_evidence, Mapping):
        raise TypeError("compiled witness requires compiler evidence mapping")
    if not isinstance(show_values, bool):
        raise TypeError("show_values must be boolean")
    expected = build_point_expected(
        point,
        representation="compiled",
        compiled_evidence=compiled_evidence,
    )
    from .audit import (
        _compiled_manifests,
        _verify_compiled_geometry,
        _visible_compiled_rows,
    )

    node_order, visible_rows = _visible_compiled_rows(scene)
    detail_rows, ports = _verify_compiled_geometry(scene, node_order, visible_rows)
    ordered_ports = sorted(ports, key=lambda row: cast(str, row["port_id"]))
    electrical, structural = _compiled_manifests(
        node_order, visible_rows, ordered_ports
    )
    _compare("compiled electrical reconstruction", expected.electrical, electrical)
    _compare("compiled structural reconstruction", expected.structural, structural)
    if any((symbol.value is not None) != show_values for symbol in scene.symbols):
        raise _fail(
            "compiled coefficient glyph value visibility disagrees with the diagram request"
        )
    return electrical, structural, tuple(detail_rows)


__all__ = ["witness_authoring", "witness_compiled"]
