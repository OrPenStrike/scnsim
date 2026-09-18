"""Structured schematic layout declarations and the capture-bound entry.

Public layout values retain the actual typed authoring handles supplied by the
caller.  At render capture they are checked against that exact Plan and
compiled into private immutable snapshot keys; no renderer consults mutable
authoring state afterwards.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, TypeAlias

from .authoring import BranchRef, BusRef, CircuitPlan, ComponentInstance, ParallelRef, PinRef, PortRef, SeriesRef, SubsystemPlan, TapRef
from .errors import SCNSimValidationError

if TYPE_CHECKING:
    from .specs import DiagramSide


# These aliases deliberately exclude CompositePlan: factory templates and
# cloned implementation bodies are not caller-addressable layout scopes.
ScopeHandle: TypeAlias = CircuitPlan | SubsystemPlan
AxisHandle: TypeAlias = ScopeHandle | SeriesRef | ParallelRef | BranchRef
OrderHandle: TypeAlias = ScopeHandle | ParallelRef
OrderMember: TypeAlias = ComponentInstance | SubsystemPlan | SeriesRef
TerminalHandle: TypeAlias = PinRef | TapRef


def _fail(message: str, **evidence: object) -> SCNSimValidationError:
    return SCNSimValidationError(message, stage="schematic_layout", evidence=evidence)


class DiagramAxis(str, Enum):
    """Local reading axis for one eligible rendered scope or structure."""

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


def _side(value: object) -> object:
    # specs imports this module, so retain the accepted DiagramSide lazily.
    from .specs import DiagramSide
    if not isinstance(value, DiagramSide):
        raise TypeError("diagram side must be DiagramSide")
    return value


def _tuple(value: object, *, field: str) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{field} values must be tuples")
    if len({id(item) for item in value}) != len(value):
        raise _fail(f"{field} cannot repeat a handle")
    return value


def _handles(mapping: Mapping[object, object], *, field: str, allowed: tuple[type[object], ...]) -> None:
    if any(not isinstance(handle, allowed) for handle in mapping):
        raise TypeError(f"{field} keys must be eligible typed authoring handles")


def _members(mapping: Mapping[object, tuple[object, ...]], *, field: str, allowed: tuple[type[object], ...]) -> None:
    for values in mapping.values():
        if any(not isinstance(value, allowed) for value in values):
            raise TypeError(f"{field} values must be eligible typed authoring handles")


def _scope_path(scope: object) -> tuple[str, ...]:
    path = scope.path() if callable(getattr(scope, "path", None)) else None
    if not isinstance(path, tuple) or not all(isinstance(item, str) and item for item in path):
        raise _fail("layout handle has no attached portable scope path")
    return path


def _owned_scope(scope: object, plan: CircuitPlan) -> None:
    if not isinstance(scope, (CircuitPlan, SubsystemPlan)) or getattr(scope, "root", None) is not plan:
        raise _fail("layout handle belongs to another CircuitPlan")
    if scope is not plan and not any(candidate is scope for candidate, _ in plan._walk()):
        raise _fail("layout handle refers to stale scope")


def _key(value: object, plan: CircuitPlan) -> tuple[object, ...]:
    """Validate one current public handle and make a private snapshot key."""
    if isinstance(value, CircuitPlan):
        if value is not plan:
            raise _fail("layout CircuitPlan belongs to another Plan")
        return ("scope", ())
    if isinstance(value, SubsystemPlan):
        _owned_scope(value, plan)
        return ("scope", _scope_path(value))
    if isinstance(value, ComponentInstance):
        owner = value.owner
        _owned_scope(owner, plan)
        if owner.components.get(value.id) is not value:
            raise _fail("layout component is detached or stale")
        return ("component", _scope_path(owner), value.id)
    if isinstance(value, (SeriesRef, ParallelRef, BranchRef)):
        _owned_scope(value.scope, plan)
        kind = "series" if isinstance(value, SeriesRef) else "parallel" if isinstance(value, ParallelRef) else "branch"
        direct = any(row.get("kind") == kind and row.get("id") == value.id for row in value.scope.structures)
        parallel_branch = isinstance(value, SeriesRef) and any(
            row.get("kind") == "parallel"
            and any(branch.get("id") == value.id for branch in row.get("branches", ()))
            for row in value.scope.structures
        )
        if not direct and not parallel_branch:
            raise _fail("layout structure handle is stale")
        return (kind, _scope_path(value.scope), value.id)
    if isinstance(value, PinRef):
        _owned_scope(value.scope, plan)
        if value.component is None:
            if not value.public or value.scope.exposed_pins.get(value.name) is not value:
                raise _fail("layout public pin is stale")
        else:
            # Complete native bodies (notably CPW/MTL) deliberately expose
            # their outer terminals without a Composite body.  Ordinary
            # R/L/C/JJ leaf pins remain private implementation detail.
            if value.component.body is None and value.component.kind == "ordinary":
                raise _fail("layout terminal side cannot target a hidden physical leaf pin")
            if value.scope.components.get(value.component.id) is not value.component or value.component.pins.get(value.name) is not value:
                raise _fail("layout component pin is stale")
        return ("pin", _scope_path(value.scope), value.component_id, value.name, value.public)
    if isinstance(value, TapRef):
        _owned_scope(value.scope, plan)
        if value.scope.buses.get(value.bus.id) is not value.bus or value.bus.taps.get(value.id) is not value:
            raise _fail("layout tap is stale")
        return ("tap", _scope_path(value.scope), value.bus.id, value.id)
    if isinstance(value, BusRef):
        _owned_scope(value.scope, plan)
        if value.scope.buses.get(value.id) is not value:
            raise _fail("layout bus is stale")
        return ("bus", _scope_path(value.scope), value.id)
    if isinstance(value, PortRef):
        if value.plan is not plan or not any(port is value for port in plan.ports):
            raise _fail("layout Port belongs to another CircuitPlan or is stale")
        return ("port", value.id)
    raise TypeError("layout key must be an eligible typed authoring handle")


@dataclass(frozen=True, slots=True)
class SchematicLayout:
    """Frozen typed-handle constraints; it owns no topology or catalog API."""

    axes: Mapping[AxisHandle, DiagramAxis]
    order: Mapping[OrderHandle, tuple[OrderMember, ...]]
    terminal_sides: Mapping[TerminalHandle, DiagramSide]
    port_sides: Mapping[PortRef, DiagramSide]
    tap_order: Mapping[BusRef, tuple[TapRef, ...]]

    def __init__(self, *, axes: Mapping[AxisHandle, DiagramAxis] = MappingProxyType({}), order: Mapping[OrderHandle, tuple[OrderMember, ...]] = MappingProxyType({}), terminal_sides: Mapping[TerminalHandle, DiagramSide] = MappingProxyType({}), port_sides: Mapping[PortRef, DiagramSide] = MappingProxyType({}), tap_order: Mapping[BusRef, tuple[TapRef, ...]] = MappingProxyType({})) -> None:
        if not all(isinstance(value, Mapping) for value in (axes, order, terminal_sides, port_sides, tap_order)):
            raise TypeError("SchematicLayout arguments must be mappings")
        _handles(axes, field="axes", allowed=(CircuitPlan, SubsystemPlan, SeriesRef, ParallelRef, BranchRef))
        _handles(order, field="order", allowed=(CircuitPlan, SubsystemPlan, ParallelRef))
        _handles(terminal_sides, field="terminal_sides", allowed=(PinRef, TapRef))
        _handles(port_sides, field="port_sides", allowed=(PortRef,))
        _handles(tap_order, field="tap_order", allowed=(BusRef,))
        if any(not isinstance(axis, DiagramAxis) for axis in axes.values()):
            raise TypeError("layout axes values must be DiagramAxis")
        for side in (*terminal_sides.values(), *port_sides.values()):
            _side(side)
        order_values = {handle: _tuple(value, field="order") for handle, value in order.items()}
        tap_values = {handle: _tuple(value, field="tap_order") for handle, value in tap_order.items()}
        _members(order_values, field="order", allowed=(ComponentInstance, SubsystemPlan, SeriesRef))
        _members(tap_values, field="tap_order", allowed=(TapRef,))
        object.__setattr__(self, "axes", MappingProxyType(dict(axes)))
        object.__setattr__(self, "order", MappingProxyType(order_values))
        object.__setattr__(self, "terminal_sides", MappingProxyType(dict(terminal_sides)))
        object.__setattr__(self, "port_sides", MappingProxyType(dict(port_sides)))
        object.__setattr__(self, "tap_order", MappingProxyType(tap_values))

    def _capture_for(self, plan: CircuitPlan) -> "_CapturedLayout":
        """Compile validated owned handles into renderer-only immutable keys."""
        if not isinstance(plan, CircuitPlan):
            raise TypeError("layout capture requires CircuitPlan")
        axes: dict[tuple[object, ...], str] = {}
        for handle, axis in self.axes.items():
            key = _key(handle, plan)
            if key[0] not in {"scope", "series", "parallel", "branch"}:
                raise _fail("axis target is not a rendered scope or structured group")
            if key in axes:
                raise _fail("layout axes duplicate one target")
            axes[key] = axis.value
        order: dict[tuple[object, ...], tuple[tuple[object, ...], ...]] = {}
        for handle, members in self.order.items():
            key = _key(handle, plan)
            if key[0] not in {"scope", "parallel"}:
                raise _fail("order target is not a scope or parallel group")
            values = tuple(_key(member, plan) for member in members)
            if len(set(values)) != len(values):
                raise _fail("order cannot repeat one logical handle")
            if key[0] == "scope":
                scope = plan if isinstance(handle, CircuitPlan) else handle
                expected = {
                    ("component", _scope_path(scope), component.id)
                    for component in scope.components.values()
                    if component.kind != "ordinary" and not component.used
                } | {
                    ("scope", _scope_path(child))
                    for child in scope.children.values()
                }
                if set(values) != expected or len(values) != len(expected):
                    raise _fail("scope order must name exactly its independent complete peers and child regions")
            else:
                parallel = next(
                    (row for row in handle.scope.structures if row.get("kind") == "parallel" and row.get("id") == handle.id),
                    None,
                )
                if parallel is None:
                    raise _fail("layout parallel order is stale")
                expected = {
                    ("series", _scope_path(handle.scope), branch.get("id"))
                    for branch in parallel.get("branches", ())
                }
                if set(values) != expected or len(values) != len(expected):
                    raise _fail("parallel order must name exactly its declared branch chains")
            order[key] = values
        terminals: dict[tuple[object, ...], str] = {}
        for handle, side in self.terminal_sides.items():
            key = _key(handle, plan)
            if key[0] not in {"pin", "tap"}:
                raise _fail("terminal_sides requires PinRef or TapRef")
            terminals[key] = side.value
        ports: dict[tuple[object, ...], str] = {}
        for handle, side in self.port_sides.items():
            key = _key(handle, plan)
            if key[0] != "port":
                raise _fail("port_sides requires PortRef")
            ports[key] = side.value
        taps: dict[tuple[object, ...], tuple[tuple[object, ...], ...]] = {}
        for handle, members in self.tap_order.items():
            key = _key(handle, plan)
            if key[0] != "bus":
                raise _fail("tap_order requires BusRef keys")
            values = tuple(_key(member, plan) for member in members)
            expected = {("tap", _scope_path(handle.scope), handle.id, tap.id) for tap in handle.taps.values()}
            if set(values) != expected or len(values) != len(expected):
                raise _fail("tap_order must name exactly the BusRef's current taps")
            taps[key] = values
        return _CapturedLayout(axes, order, terminals, ports, taps)


@dataclass(frozen=True, slots=True)
class _CapturedLayout:
    axes: Mapping[tuple[object, ...], str]
    order: Mapping[tuple[object, ...], tuple[tuple[object, ...], ...]]
    terminal_sides: Mapping[tuple[object, ...], str]
    port_sides: Mapping[tuple[object, ...], str]
    tap_order: Mapping[tuple[object, ...], tuple[tuple[object, ...], ...]]

    def __init__(self, axes: Mapping[tuple[object, ...], str], order: Mapping[tuple[object, ...], tuple[tuple[object, ...], ...]], terminal_sides: Mapping[tuple[object, ...], str], port_sides: Mapping[tuple[object, ...], str], tap_order: Mapping[tuple[object, ...], tuple[tuple[object, ...], ...]]) -> None:
        object.__setattr__(self, "axes", MappingProxyType(dict(axes)))
        object.__setattr__(self, "order", MappingProxyType(dict(order)))
        object.__setattr__(self, "terminal_sides", MappingProxyType(dict(terminal_sides)))
        object.__setattr__(self, "port_sides", MappingProxyType(dict(port_sides)))
        object.__setattr__(self, "tap_order", MappingProxyType(dict(tap_order)))


def _render_schematic(plan: CircuitPlan, spec: object, *, parameters: object | None = None) -> object:
    """Capture one immutable target point before diagram lowering starts."""
    if not isinstance(plan, CircuitPlan):
        raise TypeError("diagram rendering requires CircuitPlan")
    snapshot = plan._capture_authoring_snapshot()
    point = plan._resolve_parameter_point(parameters, snapshot=snapshot)
    from .composition import SchematicComposition

    requested = getattr(spec, "layout", None)
    if getattr(spec, "representation", None) == "compiled":
        layout = None
    else:
        composition = requested if isinstance(requested, SchematicComposition) else SchematicComposition._automatic_from_snapshot(plan=plan, snapshot=snapshot, hints=requested)
        layout = composition._capture_intent_for(plan, snapshot=snapshot)
    from ._diagram.pipeline import render_schematic
    return render_schematic(point, spec, layout=layout)


__all__ = ["DiagramAxis", "SchematicLayout"]
