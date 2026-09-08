"""Capture-bound orchestration for one audited frozen structured drawing."""

from __future__ import annotations

from collections.abc import Mapping

from .._authoring_snapshot import ResolvedPlanPoint
from .._canonical import canonical_diagram_digests, canonical_parameters_sha256
from ..errors import SCNSimValidationError
from ..results import CircuitDiagramAudit, CircuitDiagramResult, _verified_result
from ..specs import CircuitDiagramSpec
from .audit import certify_scene
from .drawing import _svg_certificate_from_audit, freeze_drawing
from .scene import NeutralScene, append_provenance_band


def _fail(message: str, **evidence: object) -> SCNSimValidationError:
    return SCNSimValidationError(message, stage="schematic_audit", evidence=evidence)


def _provenance_lines(*, representation: str, plan_id: str, digests: Mapping[str, str], parameters_sha256: str) -> tuple[str, str]:
    """Keep requested source digests in a compact, separate footer."""

    return (
        f"{representation.upper()} · CircuitPlan {plan_id} · parameters {parameters_sha256[:12]}",
        f"plan {digests['plan_sha256'][:12]} · connectivity {digests['connectivity_sha256'][:12]} · semantic {digests['semantic_sha256'][:12]}",
    )


def _point_identity(point: ResolvedPlanPoint, *, representation: str) -> tuple[str, Mapping[str, str], str]:
    """Read one immutable point identity without consulting its live Plan."""
    semantic = point.snapshot.semantic_record
    plan_id = semantic.get("plan_id") if isinstance(semantic, Mapping) else None
    if not isinstance(plan_id, str) or not plan_id:
        raise _fail("captured authoring point has no plan identity")
    digests = canonical_diagram_digests(point.snapshot, representation=representation)
    if not isinstance(digests, Mapping) or any(
        not isinstance(digests.get(name), str)
        for name in ("plan_sha256", "connectivity_sha256", "semantic_sha256")
    ):
        raise _fail("captured authoring point has incomplete diagram identities")
    parameters_sha256 = canonical_parameters_sha256(point.parameter_record)
    if not isinstance(parameters_sha256, str):
        raise _fail("captured authoring point has no parameter identity")
    return plan_id, digests, parameters_sha256


def _with_provenance(scene: NeutralScene, *, point: ResolvedPlanPoint, spec: CircuitDiagramSpec) -> NeutralScene:
    if not spec.show_provenance:
        return scene
    plan_id, digests, parameters_sha256 = _point_identity(point, representation=spec.representation)
    return append_provenance_band(
        scene,
        _provenance_lines(
            representation=spec.representation,
            plan_id=plan_id,
            digests=digests,
            parameters_sha256=parameters_sha256,
        ),
    )


def _finish(
    point: ResolvedPlanPoint,
    spec: CircuitDiagramSpec,
    scene: NeutralScene,
    *,
    compiled_evidence: Mapping[str, object] | None,
    composition: object = None,
) -> CircuitDiagramResult:
    """Certify before a scene is exposed through a Drawing or Result facade."""
    audit_data = certify_scene(
        point,
        scene,
        representation=spec.representation,
        show_values=spec.show_parameter_values,
        compiled_evidence=compiled_evidence,
    )
    audit = CircuitDiagramAudit._from_data(audit_data)
    drawing = freeze_drawing(
        scene,
        theme=spec.theme,
        certificate=_svg_certificate_from_audit(audit),
    )
    return _verified_result(
        CircuitDiagramResult, drawing=drawing, audit=audit, composition=composition,
    )


def _authoring_result(point: ResolvedPlanPoint, spec: CircuitDiagramSpec, layout: object) -> CircuitDiagramResult:
    from ..composition import detached_composition
    from .composition_constraints import verify_composition
    from .composition_planner import plan_composition

    planned = plan_composition(point, layout, show_values=spec.show_parameter_values)
    scene = planned.scene
    verify_composition(scene, planned.composition, point=point)
    return _finish(
        point,
        spec,
        _with_provenance(scene, point=point, spec=spec),
        compiled_evidence=None,
        composition=detached_composition(planned.composition),
    )


def _compiled_result(point: ResolvedPlanPoint, spec: CircuitDiagramSpec) -> CircuitDiagramResult:
    from ..runtime import _compiled_schematic_evidence
    from .compiled import layout_compiled

    evidence = _compiled_schematic_evidence(point)
    if not isinstance(evidence, Mapping):
        raise _fail("compiler audit did not return an immutable evidence mapping")
    scene = layout_compiled(point, evidence, show_values=spec.show_parameter_values)
    return _finish(
        point,
        spec,
        _with_provenance(scene, point=point, spec=spec),
        compiled_evidence=evidence,
    )


def render_schematic(point: ResolvedPlanPoint, spec: CircuitDiagramSpec, *, layout: object = None) -> CircuitDiagramResult:
    """Render one resolved point; no Run, View, or mutable Plan is consulted."""
    if not isinstance(point, ResolvedPlanPoint):
        raise TypeError("diagram rendering requires ResolvedPlanPoint")
    if not isinstance(spec, CircuitDiagramSpec):
        raise TypeError("diagram rendering requires CircuitDiagramSpec")
    if spec.representation == "authoring":
        return _authoring_result(point, spec, layout)
    return _compiled_result(point, spec)


__all__ = ["render_schematic"]
