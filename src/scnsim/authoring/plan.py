"""Root physical Plan, Ports, and Run preparation/sealing lifecycle.

The root owns assembly until validated publication seals it. Capture, parameter
resolution, and rendering remain delegated consumers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from ..errors import PlanSealedError, SCNSimValidationError
from ..units import Quantity, require_positive_quantity
from .assembly import _Scope
from .handles import _Net, BusRef, PortRef, TapRef
from .parameters import ParameterSet
from .physical_values import identifier
from .snapshot import AuthoringSnapshot, ResolvedPlanPoint


class CircuitPlan(_Scope):
    """Root authoring scope for a structured circuit and its physical connections.

    Create named buses, add catalog components, and assemble them with
    ``series()``, ``parallel()``, ``branch()``, and ``link()``. Inline
    subsystems retain their own ownership boundaries; Ports belong to the root.
    The Plan also owns grounding and physical parameter bindings.

    Parameter resolution, calculation, and schematic rendering consume captured
    Plan snapshots. Rendering neither edits nor seals the authoring graph;
    constructing a ``CircuitRun`` validates and seals it against further edits.
    """

    def __init__(self, *, id: str) -> None:
        self.ground_net = _Net("ground", True)
        self.sealed = False
        self.ports = []
        self._capture_net_ids = {self.ground_net: "ground"}
        self.init(id, None, self)

    def prepare_schematic(
        self, spec: object = None, *, parameters: ParameterSet | None = None
    ) -> object:
        """Capture readonly diagram inventory and measurements without layout."""
        from ..specs import CircuitDiagramSpec
        from ..schematic import _prepare_schematic

        checked_spec = CircuitDiagramSpec() if spec is None else spec
        if not isinstance(checked_spec, CircuitDiagramSpec):
            raise TypeError("prepare_schematic() requires CircuitDiagramSpec")
        if checked_spec.representation != "authoring" or checked_spec.layout is not None:
            raise SCNSimValidationError(
                "prepare_schematic() requires an authoring request without layout",
                stage="schematic_layout",
            )
        if parameters is not None and not isinstance(parameters, ParameterSet):
            raise TypeError("prepare_schematic() parameters must be a ParameterSet or None")
        return _prepare_schematic(self, checked_spec, parameters=parameters)

    def render_schematic(
        self, spec: object = None, *, parameters: ParameterSet | None = None
    ) -> object:
        """Delegate a captured authoring point to the diagram-owned renderer."""
        from ..specs import CircuitDiagramSpec
        from ..schematic import _render_schematic

        checked_spec = CircuitDiagramSpec() if spec is None else spec
        if not isinstance(checked_spec, CircuitDiagramSpec):
            raise TypeError("render_schematic() requires CircuitDiagramSpec")
        if parameters is not None and not isinstance(parameters, ParameterSet):
            raise TypeError(
                "render_schematic() parameters must be a ParameterSet or None"
            )
        return _render_schematic(self, checked_spec, parameters=parameters)

    def add_port(
        self,
        *,
        id: str,
        at: BusRef | TapRef,
        role: str,
        reference_impedance: Quantity,
    ) -> PortRef:
        self.check()
        id = identifier(id, field="port id")
        if not isinstance(at, (BusRef, TapRef)) or at.scope is not self:
            raise TypeError("Port needs root BusRef/TapRef")
        n = self.getnet(at)
        if n.root().ground or any(
            p.id == id or p.net.root() is n.root() for p in self.ports
        ):
            raise SCNSimValidationError("invalid/duplicate Port", stage="authoring")
        if role not in ("terminated", "nonloading_probe"):
            raise ValueError("invalid Port role")
        x = PortRef(
            self,
            id,
            n,
            role,
            require_positive_quantity(
                reference_impedance, "ohm", name="reference_impedance"
            ),
        )
        self.ports.append(x)
        return x

    @contextmanager
    def _run_seal_preparation(self) -> Iterator[object | None]:
        """Temporarily own editable state while one Run is prepared."""

        if self.sealed:
            yield None
            return
        if getattr(self, "_run_seal_token", None) is not None:
            raise PlanSealedError(
                "Plan already has an active Run preparation",
                stage="plan_mutation",
            )
        token = object()
        self._run_seal_token = token
        try:
            yield token
        finally:
            if getattr(self, "_run_seal_token", None) is token:
                del self._run_seal_token

    def _seal_validated(self, snapshot: AuthoringSnapshot, token: object | None) -> None:
        """Commit a Plan after Runtime has validated this captured state.

        The caller owns the exact snapshot and invokes this non-failing step
        only at the workspace publication boundary.  Completion must not run
        again after that logical commit.
        """

        if not isinstance(snapshot, AuthoringSnapshot):
            raise TypeError("validated seal requires AuthoringSnapshot")
        if self.sealed:
            if token is not None:
                raise RuntimeError("sealed Plan retained an editable preparation token")
            return
        if token is None or getattr(self, "_run_seal_token", None) is not token:
            raise RuntimeError("validated seal lost its preparation ownership")
        self.sealed = True
        del self._run_seal_token

    def _walk(self):
        out = []

        def f(s, path):
            out.append((s, path))
            for x in s.children.values():
                f(x, (*path, x.id))
            for c in s.components.values():
                if c.body:
                    f(c.body, (*path, c.id))

        f(self, ())
        return out



    def _resolve_parameter_point(
        self,
        supplied: ParameterSet | None = None,
        *,
        snapshot: AuthoringSnapshot | None = None,
    ) -> ResolvedPlanPoint:
        from .resolution import resolve_parameter_point

        snap = self._capture_authoring_snapshot() if snapshot is None else snapshot
        return resolve_parameter_point(snap, supplied)

    def _capture_authoring_snapshot(self) -> AuthoringSnapshot:
        """Capture this root through the authoring-owned normalized handoff."""
        from .capture import capture_authoring_snapshot

        return capture_authoring_snapshot(self)


# Preserve public class identity through the authoring facade.
CircuitPlan.__module__ = "scnsim.authoring"
