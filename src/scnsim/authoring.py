"""Structured authoring declarations and immutable normalized capture.

This module owns mutable assembly only until capture.  Canonical identity,
numeric lowering, and schematic realization consume its snapshot elsewhere.
"""

from __future__ import annotations
from dataclasses import dataclass
from types import MappingProxyType
from copy import deepcopy
from functools import wraps
import inspect
import json
import numpy as np
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from ._authoring_snapshot import AuthoringSnapshot, ResolvedPlanPoint, freeze
from ._immutable_values import immutable_quantity
from ._physical_values import (
    AffineMap,
    ParameterSpec,
    RLGC,
    RLGCParameterSpec,
    _checked_field_baseline,
    _retained_literal_field,
    identifier,
    quantity_record,
)
from ._parameters import (
    ParameterDefinitions,
    ParameterRef,
    ParameterSet,
    ParameterSpace,
)
from .errors import PlanSealedError, SCNSimValidationError
from .units import Quantity, registry, require_positive_quantity, require_quantity

_identifier = identifier
_factory_context: ContextVar[tuple[object, str] | None] = ContextVar(
    "scnsim_library_factory", default=None
)
_component_creation_token = object()
_two_terminal_use_token = object()


def _source_unit_identity(
    *,
    scope: str,
    component_path: tuple[str, ...] = (),
    parameter_id: str,
    field: str,
) -> str:
    return json.dumps(
        {
            "scope": scope,
            "component_path": list(component_path),
            "parameter_id": parameter_id,
            "field": field,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _source_unit_record(
    identity: str, value: Quantity, si_unit: str
) -> dict[str, object]:
    magnitude = np.asarray(value.magnitude)
    probe = (
        value
        if magnitude.ndim == 0
        else registry.Quantity(float(magnitude.flat[0]), value.units)
    )
    envelope = quantity_record(probe, si_unit)
    return {
        "identity": identity,
        "source_unit": str(value.units),
        "canonical_si_unit": envelope["si_unit"],
        "canonical_dimensionality": envelope["dimensionality"],
    }


class _Net:
    """Mutable authoring-time wire equivalence, never a physical branch."""

    def __init__(self, label, ground=False):
        self.parent = self
        self.label = label
        self.ground = ground

    def root(self):
        if self.parent is not self:
            self.parent = self.parent.root()
        return self.parent

    def join(self, other):
        left, right = self.root(), other.root()
        if left is right:
            return left
        if right.ground:
            left.parent = right
            return right
        right.parent = left
        return left


class _H:
    def __setattr__(self, name, value):
        raise AttributeError("authoring handles are immutable")


class GroundRef(_H):
    def __init__(self, scope):
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "net", scope.root.ground_net)


class BusRef(_H):
    def __init__(self, scope, id, anonymous=False):
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "net", _Net("/".join((*scope.path(), id))))
        object.__setattr__(self, "taps", {})
        object.__setattr__(self, "anonymous", anonymous)

    def tap(self, *, id: str) -> "TapRef":
        self.scope.check()
        id = identifier(id, field="tap id")
        if id in self.taps:
            raise SCNSimValidationError("tap IDs must be unique", stage="authoring")
        tap = TapRef(self, id)
        self.taps[id] = tap
        return tap

    @property
    def node(self) -> "ElectricNodeRef":
        if self.scope is not self.scope.root:
            raise SCNSimValidationError(
                "child bus needs coordinate exposure", stage="authoring"
            )
        if self.net.root().ground:
            raise SCNSimValidationError(
                "grounded bus has no ElectricNodeRef", stage="authoring"
            )
        if self.anonymous:
            port = next(
                (p for p in self.scope.ports if p.net.root() is self.net.root()), None
            )
            if port is None:
                raise SCNSimValidationError(
                    "anonymous root bus needs Port promotion", stage="authoring"
                )
            return ElectricNodeRef(self.scope.root, port.id)
        return ElectricNodeRef(self.scope.root, self.id)


class TapRef(_H):
    def __init__(self, bus, id):
        object.__setattr__(self, "bus", bus)
        object.__setattr__(self, "scope", bus.scope)
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "net", bus.net)

    @property
    def node(self) -> "ElectricNodeRef":
        return self.bus.node


class PinRef(_H):
    def __init__(self, component, scope, name, net=None, public=False):
        object.__setattr__(self, "component", component)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "net", net)
        object.__setattr__(self, "public", public)
        object.__setattr__(self, "bound", False)

    @property
    def component_id(self) -> str:
        return self.component.id if self.component else self.name


class CoordinateRef(_H):
    def __init__(self, scope, id, net):
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "net", net)

    @property
    def name(self) -> str:
        return self.id

    @property
    def component_id(self) -> str:
        return self.scope.id


class ElectricNodeRef(_H):
    def __init__(self, plan, id):
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "id", id)

    @property
    def is_public(self) -> bool:
        return True


class InductiveBranchRef(_H):
    def __init__(self, component, id):
        object.__setattr__(self, "component", component)
        object.__setattr__(self, "id", id)

    @property
    def component_id(self) -> str:
        return self.component.id


def _physical_inductive_branch(branch: InductiveBranchRef) -> InductiveBranchRef:
    """Follow explicit Composite exposures to the original oriented leaf."""
    while branch.component.body is not None:
        branch = branch.component.body.exposed_branches[branch.id]
    return branch


class PortRef(_H):
    def __init__(self, plan, id, net, role, impedance):
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "net", net)
        object.__setattr__(self, "role", role)
        object.__setattr__(
            self, "reference_impedance", immutable_quantity(impedance)
        )

    @property
    def node(self) -> ElectricNodeRef:
        return ElectricNodeRef(self.plan, self.id)


@dataclass(frozen=True, init=False)
class TwoTerminalUse:
    component: object
    pin_1: PinRef
    pin_2: PinRef

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError(
            "TwoTerminalUse values are created by ComponentInstance.between()"
        )

    @classmethod
    def _create(
        cls,
        component: "ComponentInstance",
        pin_1: PinRef,
        pin_2: PinRef,
        *,
        _token: object,
    ) -> "TwoTerminalUse":
        if _token is not _two_terminal_use_token:
            raise TypeError(
                "TwoTerminalUse construction is reserved to complete bodies"
            )
        result = object.__new__(cls)
        object.__setattr__(result, "component", component)
        object.__setattr__(result, "pin_1", pin_1)
        object.__setattr__(result, "pin_2", pin_2)
        return result


@dataclass(frozen=True)
class SeriesRef:
    id: str
    scope: object


@dataclass(frozen=True)
class ParallelRef:
    id: str
    scope: object
    branches: tuple


@dataclass(frozen=True)
class BranchRef:
    id: str
    scope: object


@dataclass(frozen=True)
class LinkRef:
    id: str
    scope: object


@dataclass(frozen=True)
class CouplingRef:
    id: str
    scope: object


def _binding(v, u, name, positive=False, nonnegative=False):
    """Bind a physical field and validate its baseline domain immediately."""

    if isinstance(v, ParameterRef):
        if isinstance(v.baseline, RLGC):
            return v.baseline, {"kind": "ref", "parameter": v._key_record()}, v
        return (
            _checked_field_baseline(v.baseline, u, name, positive, nonnegative),
            {"kind": "ref", "parameter": v._key_record()},
            v,
        )
    if isinstance(v, AffineMap):
        if isinstance(v.input.spec, RLGCParameterSpec):
            raise TypeError(f"{name} cannot use structured AffineMap")
        # Capture the coefficient data at physical-field binding time.  The
        # public AffineMap remains mutable author input, but a registered body
        # must not observe later edits to that input object or its Quantities.
        binding = {
            **v._record(),
            "_affine_data": (
                freeze(v.slope),
                freeze(v.intercept),
                tuple(freeze(value) for value in v.support),
            ),
            "_affine_source": MappingProxyType(
                {name: freeze(value) for name, value in v._source_quantities.items()}
            ),
        }
        return (
            _checked_field_baseline(
                v.value_at(v.input.baseline, unit=u, name=name),
                u,
                name,
                positive,
                nonnegative,
            ),
            MappingProxyType(binding),
            v.input,
        )
    if isinstance(v, RLGC):
        if u != "rlgc":
            raise TypeError(f"{name} is not an RLGC field")
        return v, {"kind": "constant", "value": v._record()}, None
    x = _retained_literal_field(v, u, name, positive, nonnegative)
    return x, {"kind": "constant", "value": quantity_record(x, u)}, None


class ComponentInstance:
    """Catalog-created immutable body; scope assembly owns only overlay state."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("ComponentInstance values are created by Library factories")

    @classmethod
    def _create(
        cls,
        *,
        id,
        factory,
        pins,
        fields,
        kind,
        catalog_id="scnsim.components",
        catalog_source=None,
        branches=(),
        body=None,
        metadata=None,
        _token=None,
    ) -> "ComponentInstance":
        if _token is not _component_creation_token:
            raise TypeError("ComponentInstance construction is reserved to catalogs")
        component = object.__new__(cls)
        component._initialize(
            id=id,
            factory=factory,
            pins=pins,
            fields=fields,
            kind=kind,
            catalog_id=catalog_id,
            catalog_source=catalog_source,
            branches=branches,
            body=body,
            metadata=metadata,
        )
        return component

    def _initialize(
        self,
        *,
        id,
        factory,
        pins,
        fields,
        kind,
        catalog_id="scnsim.components",
        catalog_source=None,
        branches=(),
        body=None,
        metadata=None,
    ) -> None:
        self.id = identifier(id, field="component id")
        self.factory = factory
        self.catalog_id = catalog_id
        self.catalog_source = MappingProxyType(dict(catalog_source or {}))
        self.kind = kind
        self.body = body
        self.owner = None
        self.used = 0
        self.metadata = MappingProxyType(dict(metadata or {}))
        self.pins = MappingProxyType({name: PinRef(self, None, name) for name in pins})
        self._intrinsic_pin_classes = MappingProxyType(
            {name: (name,) for name in self.pins}
        )
        self.fields = MappingProxyType(
            {
                name: (
                    *_binding(value, unit, name, positive, nonnegative),
                    unit,
                    positive,
                    nonnegative,
                )
                for name, (value, unit, positive, *nonnegative_values) in fields.items()
                for nonnegative in (
                    nonnegative_values[0] if nonnegative_values else False,
                )
            }
        )
        self.branches = MappingProxyType(dict(branches))
        self._branch_refs = MappingProxyType(
            {name: InductiveBranchRef(self, name) for name in self.branches}
        )
        self.coordinates = MappingProxyType({})
        # Primitive fields expose the original consumed input reference, not
        # an affine field's transformed value. Composite build replaces this
        # surface with its explicitly exposed parameters only.
        self.parameters = MappingProxyType(
            {
                name: ref
                for name, (_, _, ref, _, _, _) in self.fields.items()
                if ref is not None
            }
        )
        self._frozen = True

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError("ComponentInstance templates are immutable")
        object.__setattr__(self, name, value)

    def pin(self, name: str, *, conductor: str | None = None) -> PinRef:
        if conductor is None and self.factory == "transmission_line":
            raise TypeError("transmission_line pins require conductor=")
        key = (
            f"{identifier(name, field='pin id')}.{identifier(conductor, field='conductor')}"
            if conductor is not None
            else name
        )
        if key not in self.pins:
            raise KeyError(key)
        return self.pins[key]

    def parameter(self, n: str) -> ParameterRef:
        if n not in self.parameters:
            raise KeyError(n)
        return self.parameters[n]

    def coordinate(self, n: str) -> CoordinateRef:
        if n not in self.coordinates:
            raise KeyError(n)
        return self.coordinates[n]

    def inductive_branch(self, n: str) -> InductiveBranchRef:
        if n not in self.branches:
            raise KeyError(n)
        return self._branch_refs[n]

    def between(self, a: PinRef, b: PinRef) -> TwoTerminalUse:
        if self.kind == "ordinary":
            raise SCNSimValidationError(
                "ordinary two-terminal body is direct ElementUse", stage="authoring"
            )
        if a.component is not self or b.component is not self or a is b:
            raise SCNSimValidationError(
                "between requires two pins of one complete body", stage="authoring"
            )
        self._check_intrinsic_pair(a, b)
        return TwoTerminalUse._create(
            self,
            a,
            b,
            _token=_two_terminal_use_token,
        )

    def _check_intrinsic_pair(self, a: PinRef, b: PinRef) -> None:
        # These classes predate every containing-Plan union. Current pin nets
        # cannot distinguish intrinsic aliases from externally commoned returns.
        if (
            self._intrinsic_pin_classes[a.name]
            == self._intrinsic_pin_classes[b.name]
        ):
            raise SCNSimValidationError(
                "between requires intrinsically distinct terminals",
                stage="authoring",
                evidence={"component": self.id, "pins": (a.name, b.name)},
            )


def _clone_composite_scope(template, parent, root, net_map):
    """Materialize one occurrence-local overlay from a sealed Composite body.

    The template remains untouched.  Every old zero-wire equivalence class is
    represented by one fresh overlay net (or the containing Plan ground), so a
    parent link never mutates a reusable factory graph.
    """
    clone = object.__new__(type(template))
    if isinstance(clone, CompositePlan):
        clone.ground_net = root.ground_net
        clone.sealed = True
        clone.built = True
        clone.library = template.library
        clone.factory_name = template.factory_name
    clone.init(template.id, parent, root)
    # Stored declarations retain their authored path frame. An inline child
    # inside a cloned Composite has a new live path but has not had its stored
    # endpoint records rewritten; capture must rebase those records once.
    clone._record_origin = getattr(template, "_record_origin", template.path())

    def overlay_net(old):
        old_root = old.root()
        if old_root.ground:
            return root.ground_net
        if old_root not in net_map:
            net_map[old_root] = _Net("overlay")
        return net_map[old_root]

    for bus in template.buses.values():
        copied = BusRef(clone, bus.id, bus.anonymous)
        object.__setattr__(copied, "net", overlay_net(bus.net))
        clone.buses[bus.id] = copied
        for tap in bus.taps.values():
            copied_tap = TapRef(copied, tap.id)
            object.__setattr__(copied_tap, "net", overlay_net(tap.net))
            copied.taps[tap.id] = copied_tap

    component_map = {}
    for component in template.components.values():
        copied = ComponentInstance._create(
            id=component.id,
            factory=component.factory,
            pins=tuple(component.pins),
            fields={},
            kind=component.kind,
            catalog_id=component.catalog_id,
            catalog_source=component.catalog_source,
            branches=component.branches,
            metadata=component.metadata,
            _token=_component_creation_token,
        )
        object.__setattr__(copied, "fields", MappingProxyType(dict(component.fields)))
        object.__setattr__(
            copied, "_intrinsic_pin_classes", component._intrinsic_pin_classes
        )
        object.__setattr__(copied, "used", component.used)
        object.__setattr__(copied, "owner", clone)
        for name, pin in component.pins.items():
            copied_pin = copied.pins[name]
            object.__setattr__(copied_pin, "scope", clone)
            object.__setattr__(copied_pin, "net", overlay_net(pin.net))
            object.__setattr__(copied_pin, "bound", pin.bound)
        clone.components[copied.id] = copied
        component_map[component] = copied

    for child in template.children.values():
        copied_child = _clone_composite_scope(child, clone, root, net_map)
        clone.children[copied_child.id] = copied_child

    for component, copied in component_map.items():
        if component.body is not None:
            object.__setattr__(
                copied,
                "body",
                _clone_composite_scope(component.body, clone, root, net_map),
            )
        object.__setattr__(
            copied,
            "coordinates",
            MappingProxyType(
                {
                    name: CoordinateRef(
                        clone, coordinate.id, overlay_net(coordinate.net)
                    )
                    for name, coordinate in component.coordinates.items()
                }
            ),
        )
        object.__setattr__(
            copied, "parameters", MappingProxyType(dict(component.parameters))
        )
        object.__setattr__(
            copied, "branches", MappingProxyType(dict(component.branches))
        )
        object.__setattr__(
            copied,
            "_branch_refs",
            MappingProxyType(
                {name: InductiveBranchRef(copied, name) for name in copied.branches}
            ),
        )

    clone._declarations = deepcopy(template._declarations)
    clone.structures = deepcopy(template.structures)
    clone.ids = set(template.ids)
    for name, pin in template.exposed_pins.items():
        copied = PinRef(None, clone, name, overlay_net(pin.net), True)
        object.__setattr__(
            copied, "intrinsic_endpoint", deepcopy(pin.intrinsic_endpoint)
        )
        clone.exposed_pins[name] = copied
    for name, coordinate in template.exposed_coordinates.items():
        copied = CoordinateRef(clone, name, overlay_net(coordinate.net))
        object.__setattr__(
            copied, "intrinsic_endpoint", deepcopy(coordinate.intrinsic_endpoint)
        )
        clone.exposed_coordinates[name] = copied
    for name, branch in template.exposed_branches.items():
        clone.exposed_branches[name] = component_map[branch.component].inductive_branch(
            branch.id
        )
    clone.exposed_parameters = dict(template.exposed_parameters)
    # Grouping remains provenance only, but the actual copied pins preserve
    # the owner-local parent-ground role needed by the authoring projection.
    def copied_ground_pin(pin):
        if pin.component is not None:
            return component_map[pin.component].pins[pin.name]
        owner = clone if pin.scope is template else clone.children[pin.scope.id]
        return owner.exposed_pins[pin.name]

    clone.ground_calls = [
        tuple(copied_ground_pin(pin) for pin in group)
        for group in template.ground_calls
    ]
    return clone


class _Scope:
    composite = False

    def init(self, id, parent, root):
        self.id = identifier(id, field="scope id")
        self.parent = parent
        self.root = root
        self.components = {}
        self.children = {}
        self._declarations = []
        self.buses = {}
        self.structures = []
        self.ids = set()
        self.exposed_pins = {}
        self.exposed_coordinates = {}
        self.exposed_branches = {}
        self.exposed_parameters = {}
        self.ground_calls = []
        self._coupled_pairs = set()

    def path(self) -> tuple[str, ...]:
        return () if self.parent is None else (*self.parent.path(), self.id)

    def _walk(self):
        result = []

        def visit(scope, path):
            result.append((scope, path))
            for child in scope.children.values():
                visit(child, (*path, child.id))
            for component in scope.components.values():
                if component.body is not None:
                    visit(component.body, (*path, component.id))

        visit(self, ())
        return result

    def _final_net_id(self, net):
        return self.root._capture_net_ids[net.root()]

    def check(self) -> None:
        if getattr(self.root, "_run_seal_token", None) is not None:
            raise PlanSealedError("Plan is preparing a Run", stage="plan_mutation")
        if self.root.sealed or getattr(self, "built", False):
            raise PlanSealedError("Plan is sealed", stage="plan_mutation")

    @property
    def ground(self) -> GroundRef:
        return GroundRef(self)

    def pin(self, id: str) -> PinRef:
        try:
            return self.exposed_pins[id]
        except KeyError as exc:
            raise KeyError(id) from exc

    def coordinate(self, id: str) -> CoordinateRef:
        try:
            return self.exposed_coordinates[id]
        except KeyError as exc:
            raise KeyError(id) from exc

    def inductive_branch(self, id: str) -> InductiveBranchRef:
        try:
            return self.exposed_branches[id]
        except KeyError as exc:
            raise KeyError(id) from exc

    def subsystem(self, *, id: str) -> "SubsystemPlan":
        self.check()
        id = identifier(id, field="subsystem id")
        if id in self.components or id in self.children:
            raise SCNSimValidationError("direct-part ID duplicate", stage="authoring")
        x = SubsystemPlan._attached(id=id, parent=self, root=self.root)
        self.children[id] = x
        self._declarations.append({"kind": "subsystem", "id": id})
        return x

    def add(self, c: ComponentInstance) -> ComponentInstance:
        self.check()
        if not isinstance(c, ComponentInstance):
            raise TypeError("add requires ComponentInstance")
        if c.id in self.components or c.id in self.children or c.owner is not None:
            raise SCNSimValidationError(
                "duplicate/foreign component occurrence", stage="authoring"
            )
        object.__setattr__(c, "owner", self)
        if c.body is not None:
            template = c.body
            overlay = _clone_composite_scope(template, self, self.root, {})
            object.__setattr__(c, "body", overlay)
            for name, public_pin in overlay.exposed_pins.items():
                pin = c.pins[name]
                object.__setattr__(pin, "net", public_pin.net)
                object.__setattr__(pin, "bound", False)
            object.__setattr__(
                c, "coordinates", MappingProxyType(dict(overlay.exposed_coordinates))
            )
            object.__setattr__(
                c, "parameters", MappingProxyType(dict(overlay.exposed_parameters))
            )
            object.__setattr__(
                c,
                "branches",
                MappingProxyType(
                    {name: ("", "", "") for name in overlay.exposed_branches}
                ),
            )
            object.__setattr__(
                c,
                "_branch_refs",
                MappingProxyType(
                    {name: InductiveBranchRef(c, name) for name in c.branches}
                ),
            )
        for p in c.pins.values():
            object.__setattr__(p, "scope", self)
        self.components[c.id] = c
        self._declarations.append({"kind": "component", "id": c.id})
        return c

    def bus(self, *, id: str | None = None) -> BusRef:
        self.check()
        anonymous = id is None
        if anonymous:
            index = 0
            while f"internal-{index}" in self.buses:
                index += 1
            id = f"internal-{index}"
        else:
            id = identifier(id, field="bus id")
        if id in self.buses:
            raise SCNSimValidationError("duplicate bus ID", stage="authoring")
        x = BusRef(self, id, anonymous)
        self.buses[id] = x
        return x

    def getnet(self, x: object, link: bool = False) -> _Net:
        if isinstance(x, GroundRef):
            if link:
                raise SCNSimValidationError("GroundRef cannot link", stage="authoring")
            if x.scope is not self:
                raise SCNSimValidationError("foreign ground", stage="authoring")
            return x.net
        if isinstance(x, (BusRef, TapRef)):
            if x.scope is not self:
                raise SCNSimValidationError("foreign bus", stage="authoring")
            return x.net
        if isinstance(x, PinRef):
            if x.scope is not self and not (x.public and x.scope.parent is self):
                raise SCNSimValidationError("private/foreign PinRef", stage="authoring")
            if x.net is None:
                object.__setattr__(
                    x, "net", _Net("pin:" + x.component_id + ":" + x.name)
                )
            if x.component is not None:
                object.__setattr__(x, "bound", True)
            return x.net
        raise TypeError("invalid endpoint")

    def ep(self, x: object) -> dict[str, object]:
        if isinstance(x, GroundRef):
            return {"kind": "ground"}
        if isinstance(x, BusRef):
            return {"kind": "bus", "scope": list(x.scope.path()), "id": x.id}
        if isinstance(x, TapRef):
            return {
                "kind": "tap",
                "scope": list(x.scope.path()),
                "bus": x.bus.id,
                "id": x.id,
            }
        if isinstance(x, PinRef):
            return {
                "kind": "pin",
                "scope": list(x.scope.path()),
                "component": x.component_id,
                "id": x.name,
                "public": x.public,
            }
        raise TypeError("invalid endpoint")

    def use(self, x: object) -> tuple[ComponentInstance, PinRef, PinRef, str]:
        if isinstance(x, ComponentInstance):
            if x.owner is not self or x.kind != "ordinary":
                raise SCNSimValidationError(
                    "ordinary local component required", stage="authoring"
                )
            a, b = tuple(x.pins.values())
            return x, a, b, "ordinary"
        if isinstance(x, TwoTerminalUse):
            component = x.component
            if (
                not isinstance(component, ComponentInstance)
                or component.kind == "ordinary"
                or component.owner is not self
                or self.components.get(component.id) is not component
                or x.pin_1 is x.pin_2
                or component.pins.get(x.pin_1.name) is not x.pin_1
                or component.pins.get(x.pin_2.name) is not x.pin_2
            ):
                raise SCNSimValidationError("foreign complete body", stage="authoring")
            component._check_intrinsic_pair(x.pin_1, x.pin_2)
            return component, x.pin_1, x.pin_2, "between"
        raise TypeError("invalid ElementUse")

    def _check_endpoint(self, x, *, link=False):
        """Validate endpoint ownership before any operation publishes state."""
        if isinstance(x, GroundRef):
            if link:
                raise SCNSimValidationError("GroundRef cannot link", stage="authoring")
            if x.scope is not self:
                raise SCNSimValidationError("foreign ground", stage="authoring")
            return
        if isinstance(x, (BusRef, TapRef)):
            if x.scope is not self:
                raise SCNSimValidationError("foreign bus", stage="authoring")
            if isinstance(x, BusRef) and self.buses.get(x.id) is not x:
                raise SCNSimValidationError("stale bus", stage="authoring")
            if isinstance(x, TapRef) and (
                self.buses.get(x.bus.id) is not x.bus or x.bus.taps.get(x.id) is not x
            ):
                raise SCNSimValidationError("stale tap", stage="authoring")
            return
        if isinstance(x, PinRef):
            if x.scope is not self and not (x.public and x.scope.parent is self):
                raise SCNSimValidationError("private/foreign PinRef", stage="authoring")
            if x.component is not None and x.component.owner is not x.scope:
                raise SCNSimValidationError("stale pin", stage="authoring")
            if (
                x.component is None
                and x.public
                and x.scope.exposed_pins.get(x.name) is not x
            ):
                raise SCNSimValidationError("stale public pin", stage="authoring")
            return
        raise TypeError("invalid endpoint")

    def _prepared_chain(self, e, start, end):
        if not isinstance(e, tuple) or not e:
            raise SCNSimValidationError(
                "structure needs nonempty tuple", stage="authoring"
            )
        self._check_endpoint(start)
        self._check_endpoint(end)
        if isinstance(start, PinRef) and start is end:
            raise SCNSimValidationError(
                "structure cannot short one physical terminal", stage="authoring"
            )
        if self._endpoint_root(start) is self._endpoint_root(end):
            raise SCNSimValidationError(
                "structure endpoints must be distinct nets", stage="authoring"
            )
        prepared = [self.use(x) for x in e]
        if any(
            c is other
            for index, (c, _, _, _) in enumerate(prepared)
            for other, _, _, _ in prepared[index + 1 :]
        ) or any(c.used for c, _, _, _ in prepared):
            raise SCNSimValidationError("body already consumed", stage="authoring")
        return prepared

    def _endpoint_root(self, endpoint):
        if isinstance(endpoint, GroundRef):
            return self.root.ground_net.root()
        if isinstance(endpoint, (BusRef, TapRef)):
            return endpoint.net.root()
        if isinstance(endpoint, PinRef):
            return endpoint.net.root() if endpoint.net is not None else endpoint
        raise TypeError("invalid endpoint")

    def _validate_proposed_union(self, endpoints, *, ground=False):
        """Validate a complete hypothetical wire union without touching authoring state."""
        roots = {self._endpoint_root(endpoint) for endpoint in endpoints}
        target = self.root.ground_net.root() if ground else object()

        def projected(net):
            root = net.root() if isinstance(net, _Net) else net
            return target if root in roots else root

        if not ground:
            local_buses = {
                bus for bus in self.buses.values() if projected(bus.net) is target
            }
            if len({bus.net.root() for bus in local_buses}) > 1:
                raise SCNSimValidationError(
                    "link merges distinct local buses", stage="authoring"
                )
        for scope, _ in self.root._walk():
            for component in scope.components.values():
                if component.kind != "ordinary":
                    continue
                pin_roots = [
                    projected(pin.net) if pin.net is not None else pin
                    for pin in component.pins.values()
                ]
                if any(
                    left is right
                    for index, left in enumerate(pin_roots)
                    for right in pin_roots[index + 1 :]
                ):
                    raise SCNSimValidationError(
                        "wire operation shorts physical body terminals",
                        stage="authoring",
                    )
        for port in getattr(self.root, "ports", ()):
            if projected(port.net) is self.root.ground_net.root() or (
                ground and port.net.root() in roots
            ):
                raise SCNSimValidationError(
                    "wire operation grounds a Port", stage="authoring"
                )

    def chain(self, e, start, end):
        prepared = self._prepared_chain(e, start, end)
        ns = [self.getnet(start), *[_Net("generated") for _ in e[1:]], self.getnet(end)]
        out = []
        for i, (c, a, b, k) in enumerate(prepared):
            object.__setattr__(c, "used", 1)
            self.getnet(a).join(ns[i])
            self.getnet(b).join(ns[i + 1])
            out.append(
                {
                    "path": [*self.path(), c.id],
                    "kind": k,
                    "pin_1": a.name,
                    "pin_2": b.name,
                }
            )
        return out

    def sid(self, id):
        id = identifier(id, field="structure id")
        if id in self.ids:
            raise SCNSimValidationError("structure ID duplicate", stage="authoring")
        self.ids.add(id)
        self._declarations.append({"kind": "structure", "id": id})
        return id

    def series(
        self, *, id: str, start: object, elements: tuple[object, ...], end: object
    ) -> SeriesRef:
        self.check()
        self._prepared_chain(elements, start, end)
        id = self.sid(id)
        self.structures.append(
            {
                "kind": "series",
                "id": id,
                "start": self.ep(start),
                "elements": self.chain(elements, start, end),
                "end": self.ep(end),
            }
        )
        return SeriesRef(id, self)

    def parallel(
        self,
        *,
        id: str,
        start: object,
        branches: tuple[tuple[object, ...], ...],
        end: object,
    ) -> ParallelRef:
        self.check()
        if (
            not isinstance(branches, tuple)
            or len(branches) < 2
            or any(not isinstance(x, tuple) or not x for x in branches)
        ):
            raise SCNSimValidationError(
                "parallel needs two nonempty branches", stage="authoring"
            )
        prepared = [self._prepared_chain(x, start, end) for x in branches]
        flattened = [c for branch in prepared for c, _, _, _ in branch]
        if any(
            c is other
            for index, c in enumerate(flattened)
            for other in flattened[index + 1 :]
        ):
            raise SCNSimValidationError("body already consumed", stage="authoring")
        id = self.sid(id)
        refs = tuple(
            SeriesRef(f"{id}.branch_{i + 1}", self) for i in range(len(branches))
        )
        self.structures.append(
            {
                "kind": "parallel",
                "id": id,
                "start": self.ep(start),
                "branches": [
                    {"id": r.id, "elements": self.chain(x, start, end)}
                    for r, x in zip(refs, branches)
                ],
                "end": self.ep(end),
            }
        )
        return ParallelRef(id, self, refs)

    def branch(
        self, *, id: str, at: BusRef | TapRef, elements: tuple[object, ...], end: object
    ) -> BranchRef:
        self.check()
        if not isinstance(at, (BusRef, TapRef)) or at.scope is not self:
            raise TypeError("branch needs local BusRef/TapRef")
        self._prepared_chain(elements, at, end)
        id = self.sid(id)
        self.structures.append(
            {
                "kind": "branch",
                "id": id,
                "at": self.ep(at),
                "elements": self.chain(elements, at, end),
                "end": self.ep(end),
            }
        )
        return BranchRef(id, self)

    def link(self, *, id: str, endpoints: tuple[object, ...]) -> LinkRef:
        self.check()
        if (
            not isinstance(endpoints, tuple)
            or len(endpoints) < 2
            or len(set(endpoints)) != len(endpoints)
        ):
            raise SCNSimValidationError(
                "link needs unique endpoints", stage="authoring"
            )
        for x in endpoints:
            self._check_endpoint(x, link=True)
        # Do not allocate a PinRef net or mark a body bound until the entire
        # N-link has passed preflight: rejected declarations are retry-safe.
        ns = [
            x.net
            if isinstance(x, PinRef) and x.net is not None
            else x.net
            if isinstance(x, (BusRef, TapRef))
            else _Net("pending-link")
            for x in endpoints
        ]
        roots = [x.root() for x in ns]
        if all(x is roots[0] for x in roots[1:]):
            raise SCNSimValidationError("redundant link", stage="authoring")
        self._validate_proposed_union(endpoints)
        id = self.sid(id)
        ns = [self.getnet(x, True) for x in endpoints]
        for n in ns[1:]:
            ns[0].join(n)
        self.structures.append(
            {"kind": "link", "id": id, "endpoints": [self.ep(x) for x in endpoints]}
        )
        return LinkRef(id, self)

    def ground_pins(self, *, pins: tuple[PinRef, ...]) -> None:
        self.check()
        if not isinstance(pins, tuple) or not pins or len(set(pins)) != len(pins):
            raise SCNSimValidationError(
                "ground_pins needs unique pins", stage="authoring"
            )
        if any(not isinstance(x, PinRef) for x in pins):
            raise TypeError("ground_pins needs PinRef")
        for x in pins:
            self._check_endpoint(x)
        ns = [x.net if x.net is not None else _Net("pending-ground") for x in pins]
        if any(x.root().ground for x in ns):
            raise SCNSimValidationError("pin already grounded", stage="authoring")
        self._validate_proposed_union(pins, ground=True)
        ns = [self.getnet(x) for x in pins]
        for x in ns:
            x.join(self.root.ground_net)
        self.ground_calls.append(pins)

    def couple_inductive(
        self,
        *,
        id: str,
        inductor_a: InductiveBranchRef,
        inductor_b: InductiveBranchRef,
        coupling_coefficient: Quantity,
    ) -> CouplingRef:
        self.check()
        if not isinstance(inductor_a, InductiveBranchRef) or not isinstance(
            inductor_b, InductiveBranchRef
        ):
            raise TypeError("couple_inductive needs InductiveBranchRef handles")
        if inductor_a is inductor_b:
            raise SCNSimValidationError(
                "mutual coupling requires distinct branches", stage="authoring"
            )

        def allowed(branch):
            return branch.component.owner is self or any(
                branch in child.exposed_branches.values()
                for child in self.children.values()
            )

        if not allowed(inductor_a) or not allowed(inductor_b):
            raise SCNSimValidationError(
                "private/foreign inductive branch", stage="authoring"
            )
        pair = frozenset((inductor_a, inductor_b))
        if pair in self._coupled_pairs:
            raise SCNSimValidationError(
                "inductive branch pair already coupled", stage="authoring"
            )
        coefficient = require_quantity(
            coupling_coefficient, "dimensionless", name="coupling_coefficient"
        )
        magnitude = float(coefficient.to("dimensionless").magnitude)
        if not -1.0 < magnitude < 1.0:
            raise SCNSimValidationError(
                "coupling_coefficient must lie strictly between -1 and 1",
                stage="authoring",
            )

        def physical(branch):
            branch = _physical_inductive_branch(branch)
            component = branch.component
            return {
                "path": [*component.owner.path(), component.id],
                "branch_id": branch.id,
            }

        id = self.sid(id)
        self._coupled_pairs.add(pair)
        self.structures.append(
            {
                "kind": "coupling",
                "id": id,
                "inductor_a": physical(inductor_a),
                "inductor_b": physical(inductor_b),
                "coupling_coefficient": quantity_record(coefficient, "dimensionless"),
            }
        )
        return CouplingRef(id, self)

    def expose_pin(self, *, id: str, at: BusRef | TapRef) -> PinRef:
        self.check()
        if self.parent is None and not self.composite:
            raise AttributeError("CircuitPlan has no public boundary exposure API")
        if not isinstance(at, (BusRef, TapRef)) or at.scope is not self:
            raise TypeError("expose_pin needs local BusRef or TapRef")
        id = identifier(id, field="public pin id")
        if id in self.exposed_pins:
            raise SCNSimValidationError("public pin duplicate", stage="authoring")
        endpoint = self.ep(at)
        x = PinRef(None, self, id, self.getnet(at), True)
        object.__setattr__(x, "intrinsic_endpoint", endpoint)
        self.exposed_pins[id] = x
        return x

    def expose_coordinate(self, *, id: str, at: BusRef | TapRef) -> CoordinateRef:
        self.check()
        if self.parent is None and not self.composite:
            raise AttributeError("CircuitPlan has no public boundary exposure API")
        if not isinstance(at, (BusRef, TapRef)) or at.scope is not self:
            raise TypeError("expose_coordinate needs local BusRef or TapRef")
        id = identifier(id, field="coordinate id")
        if id in self.exposed_coordinates:
            raise SCNSimValidationError("coordinate duplicate", stage="authoring")
        endpoint = self.ep(at)
        x = CoordinateRef(self, id, self.getnet(at))
        object.__setattr__(x, "intrinsic_endpoint", endpoint)
        self.exposed_coordinates[id] = x
        return x

    def expose_inductive_branch(
        self, *, id: str, branch: InductiveBranchRef
    ) -> InductiveBranchRef:
        self.check()
        id = identifier(id, field="branch id")
        if (
            id in self.exposed_branches
            or not isinstance(branch, InductiveBranchRef)
            or branch.component.owner is not self
        ):
            raise SCNSimValidationError("invalid branch exposure", stage="authoring")
        self.exposed_branches[id] = branch
        return branch

    def complete(self) -> None:
        for s in self.children.values():
            s.complete()
        for c in self.components.values():
            if c.kind == "composite":
                if c.body is None or not c.body.built:
                    raise SCNSimValidationError(
                        "Composite body is incomplete",
                        stage="authoring",
                        evidence={"component": c.id},
                    )
                # Public boundaries may remain open; the actual internal
                # physical graph must still satisfy all native invariants.
                c.body.complete()
            elif any(not p.bound for p in c.pins.values()):
                raise SCNSimValidationError(
                    "component terminal unbound",
                    stage="authoring",
                    evidence={"component": c.id},
                )
            roots = [p.net.root() for p in c.pins.values()]
            if c.kind == "ordinary" and any(
                root is other
                for index, root in enumerate(roots)
                for other in roots[index + 1 :]
            ):
                raise SCNSimValidationError(
                    "physical body terminals cannot be shorted",
                    stage="authoring",
                    evidence={"component": c.id},
                )
            if c.kind == "ordinary" and c.used != 1:
                raise SCNSimValidationError(
                    "ordinary native needs one structure", stage="authoring"
                )
            if c.used > 1:
                raise SCNSimValidationError("body multiply consumed", stage="authoring")

    def record(self, *, path=None):
        path = self.path() if path is None else path
        children = [x.record(path=(*path, x.id)) for x in self.children.values()]
        # A clone preserves the path frame of its stored declarations, including
        # recursively cloned inline children. Rebase from that frame rather than
        # its newly attached live path, without mutating the reusable template.
        origin = getattr(self, "_record_origin", self.path())
        prefix = path[: len(path) - len(origin)]

        def rebase(value):
            if isinstance(value, dict):
                return {
                    k: (
                        list((*prefix, *v))
                        if k in {"path", "scope"} and isinstance(v, list)
                        else rebase(v)
                    )
                    for k, v in value.items()
                }
            if isinstance(value, list):
                return [rebase(v) for v in value]
            return value

        structures = rebase(self.structures)
        for structure in structures:
            if structure["kind"] == "link":
                structure["endpoints"].sort(
                    key=lambda endpoint: repr(sorted(endpoint.items()))
                )
        bodies = [
            {
                "occurrence_id": component.id,
                "body": component.body.record(path=(*path, component.id)),
            }
            for component in self.components.values()
            if component.body
        ]
        final = self._final_net_id
        buses = [
            {
                "id": bus.id,
                "anonymous": bus.anonymous,
                "final_net": final(bus.net),
                "taps": [
                    {"id": tap.id, "final_net": final(tap.net)}
                    for tap in bus.taps.values()
                ],
            }
            for bus in self.buses.values()
        ]
        exposures = {
            "pins": [
                {
                    "id": key,
                    "intrinsic_endpoint": rebase(pin.intrinsic_endpoint),
                    "final_net": final(pin.net),
                    "ground_role": "ground" if pin.net.root().ground else "ungrounded",
                }
                for key, pin in self.exposed_pins.items()
            ],
            "coordinates": [
                {
                    "id": key,
                    "intrinsic_endpoint": rebase(coordinate.intrinsic_endpoint),
                    "final_net": final(coordinate.net),
                    "ground_role": "ground"
                    if coordinate.net.root().ground
                    else "ungrounded",
                }
                for key, coordinate in self.exposed_coordinates.items()
            ],
            "branches": [
                {
                    "id": key,
                    "intrinsic_branch": {
                        "path": [*branch.component.owner.path(), branch.component.id],
                        "id": branch.id,
                    },
                }
                for key, branch in self.exposed_branches.items()
            ],
            "parameters": [
                {"id": key, "parameter": value._key_record()}
                for key, value in self.exposed_parameters.items()
            ],
        }
        return {
            "kind": "composite_body"
            if self.composite
            else ("root" if self.parent is None else "subsystem"),
            "id": self.id,
            "path": list(path),
            "declaration_order": list(self._declarations),
            "buses": buses,
            "structures": structures,
            "exposures": exposures,
            "children": children,
            "component_bodies": bodies,
        }


class SubsystemPlan(_Scope):
    """A Plan-owned inline scope; callers obtain one through ``subsystem()``."""

    def __init__(self) -> None:
        raise TypeError("SubsystemPlan values are created by CircuitPlan.subsystem()")

    @classmethod
    def _attached(
        cls, *, id: str, parent: _Scope, root: "CircuitPlan"
    ) -> "SubsystemPlan":
        """Create one internally attached scope after its parent accepted its ID."""
        scope = object.__new__(cls)
        scope.init(id, parent, root)
        return scope


class CompositePlan(_Scope):
    composite = True

    def __init__(self, *, id: str, library: "Library") -> None:
        if not isinstance(library, Library):
            raise TypeError("library must be Library")
        context = _factory_context.get()
        if context is None or context[0] is not library:
            raise TypeError("CompositePlan is created only inside its Library factory")
        self.ground_net = _Net("ground", True)
        self.sealed = False
        self.built = False
        self.init(id, None, self)
        self.library = library
        self.factory_name = context[1]

    def expose_parameter(self, *, id: str, parameter: ParameterRef) -> ParameterRef:
        self.check()
        id = identifier(id, field="public parameter id")
        if not isinstance(parameter, ParameterRef) or id in self.exposed_parameters:
            raise TypeError("expose_parameter needs unique ParameterRef")

        def consumes(scope):
            for component in scope.components.values():
                if any(
                    ref is parameter for _, _, ref, _, _, _ in component.fields.values()
                ):
                    return True
                if component.body and consumes(component.body):
                    return True
            return any(consumes(child) for child in scope.children.values())

        if not consumes(self):
            raise SCNSimValidationError(
                "parameter has no physical consumer", stage="authoring"
            )
        self.exposed_parameters[id] = parameter
        return parameter

    def build(self) -> ComponentInstance:
        if self.built:
            raise PlanSealedError("CompositePlan sealed", stage="plan_mutation")
        self.complete()
        self.built = True
        src = _catalog_source(self.library)
        x = ComponentInstance._create(
            id=self.id,
            factory=self.factory_name,
            pins=tuple(self.exposed_pins),
            fields={},
            kind="composite",
            catalog_id=src["catalog_id"],
            catalog_source=src,
            body=self,
            _token=_component_creation_token,
        )
        for n, p in self.exposed_pins.items():
            object.__setattr__(x.pins[n], "net", p.net)
        # Freeze only names, not live union-find roots. Later parent links or
        # grounding cannot turn distinct intrinsic terminals into aliases here.
        intrinsic_groups = {}
        for name, pin in self.exposed_pins.items():
            intrinsic_groups.setdefault(pin.net.root(), []).append(name)
        object.__setattr__(
            x,
            "_intrinsic_pin_classes",
            MappingProxyType(
                {
                    name: tuple(names)
                    for names in intrinsic_groups.values()
                    for name in names
                }
            ),
        )
        object.__setattr__(
            x, "coordinates", MappingProxyType(dict(self.exposed_coordinates))
        )
        object.__setattr__(
            x, "parameters", MappingProxyType(dict(self.exposed_parameters))
        )
        object.__setattr__(
            x,
            "branches",
            MappingProxyType({n: ("", "", "") for n in self.exposed_branches}),
        )
        object.__setattr__(
            x,
            "_branch_refs",
            MappingProxyType({n: InductiveBranchRef(x, n) for n in x.branches}),
        )
        return x


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

    def render_schematic(
        self, spec: object = None, *, parameters: ParameterSet | None = None
    ) -> object:
        """Delegate a captured authoring point to the diagram-owned renderer."""
        from .specs import CircuitDiagramSpec
        from .schematic import _render_schematic

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

    def _seal(self) -> "CircuitPlan":
        self.complete()
        self.sealed = True
        return self

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

    def _net_identity_map(self):
        """Name final equivalence classes from their complete typed endpoint sets."""
        groups = {}

        def key(endpoint):
            return tuple(
                sorted(
                    (name, tuple(value) if isinstance(value, list) else value)
                    for name, value in endpoint.items()
                )
            )

        for scope, _ in self._walk():
            for bus in scope.buses.values():
                groups.setdefault(bus.net.root(), []).append(key(scope.ep(bus)))
                for tap in bus.taps.values():
                    groups.setdefault(tap.net.root(), []).append(key(scope.ep(tap)))
            for pin in scope.exposed_pins.values():
                groups.setdefault(pin.net.root(), []).append(key(scope.ep(pin)))
            for component in scope.components.values():
                for pin in component.pins.values():
                    groups.setdefault(pin.net.root(), []).append(key(scope.ep(pin)))
        for port in self.ports:
            groups.setdefault(port.net.root(), []).append(
                (("kind", "port"), ("id", port.id))
            )
        ordered = sorted(
            (tuple(sorted(keys)), root)
            for root, keys in groups.items()
            if not root.ground
        )
        result = {self.ground_net.root(): "ground"}
        result.update(
            {root: f"net-{index:04d}" for index, (_, root) in enumerate(ordered, 1)}
        )
        return result

    def _capture_authoring_snapshot(self) -> AuthoringSnapshot:
        self.complete()
        self._capture_net_ids = self._net_identity_map()
        occ = []
        leaves = []
        defs = {}
        edges = []
        eps = []
        endpoint_nets = []
        captured_fields = {}
        captured_refs = {}
        source_units = []
        source_identities = set()

        def source(identity, value, unit):
            if identity in source_identities:
                return
            source_identities.add(identity)
            source_units.append(_source_unit_record(identity, value, unit))

        couplings = []
        for s, scope_path in self._walk():
            final = self._final_net_id
            for bus in s.buses.values():
                endpoint_nets.append(
                    {"endpoint": s.ep(bus), "final_net": final(bus.net)}
                )
                endpoint_nets.extend(
                    {"endpoint": s.ep(tap), "final_net": final(tap.net)}
                    for tap in bus.taps.values()
                )
            endpoint_nets.extend(
                {"endpoint": s.ep(pin), "final_net": final(pin.net)}
                for pin in s.exposed_pins.values()
            )
            for c in s.components.values():
                path = (*scope_path, c.id)
                net = lambda pin: self._final_net_id(pin.net)
                boundary_pins = (
                    {
                        name: {
                            "backing_endpoint": {
                                "kind": "pin",
                                "scope": list(path),
                                "component": c.id,
                                "id": name,
                            },
                            "final_net": net(pin),
                        }
                        for name, pin in c.pins.items()
                    }
                    if c.body
                    else {}
                )
                boundary_coordinates = {
                    name: {
                        "backing_endpoint": {
                            **coordinate.intrinsic_endpoint,
                            "scope": list(path),
                        },
                        "final_net": net(coordinate),
                    }
                    for name, coordinate in c.coordinates.items()
                }
                boundary_branches = {
                    name: {
                        "oriented_physical_branch": {
                            "path": [
                                *branch.component.owner.path(), branch.component.id
                            ],
                            "id": branch.id,
                        }
                    }
                    for name, exposed in (
                        c.body.exposed_branches.items() if c.body else ()
                    )
                    for branch in (_physical_inductive_branch(exposed),)
                }
                occ.append(
                    {
                        "path": list(path),
                        "factory": c.factory,
                        "catalog_id": c.catalog_id,
                        "catalog_source": dict(c.catalog_source),
                        "body_kind": c.kind,
                        "public_pins": boundary_pins,
                        "public_coordinates": boundary_coordinates,
                        "public_inductive_branches": boundary_branches,
                        "public_parameters": [
                            {"id": name, "parameter": parameter._key_record()}
                            for name, parameter in c.parameters.items()
                        ],
                    }
                )
                if not c.body:
                    fs = []
                    for k, (base, b, r, u, positive, nonnegative) in c.fields.items():
                        captured_binding = dict(b)
                        if b["kind"] == "affine":
                            captured_binding["_affine_data"] = b["_affine_data"]
                        captured_fields[(path, k)] = (
                            freeze(base),
                            MappingProxyType(captured_binding),
                            r,
                            u,
                            positive,
                            nonnegative,
                        )
                        if r:
                            captured_refs[r._key()] = r
                        if isinstance(base, RLGC):
                            for (
                                rlgc_field,
                                rlgc_value,
                            ) in base._source_quantities.items():
                                source(
                                    _source_unit_identity(
                                        scope="plan_rlgc",
                                        component_path=path,
                                        parameter_id="rlgc",
                                        field=rlgc_field,
                                    ),
                                    rlgc_value,
                                    {
                                        "resistance_per_length": "ohm / meter",
                                        "inductance_per_length": "henry / meter",
                                        "conductance_per_length": "siemens / meter",
                                        "capacitance_per_length": "farad / meter",
                                        "extraction_frequency": "hertz",
                                    }[rlgc_field],
                                )
                        elif b["kind"] != "affine":
                            source_value = base
                            if r is not None and r._source_unit is not None:
                                source_value = registry.Quantity(
                                    base.to(r.spec.si_unit).magnitude, r._source_unit
                                )
                            source(
                                _source_unit_identity(
                                    scope="plan_parameter",
                                    component_path=path,
                                    parameter_id=k,
                                    field="baseline",
                                ),
                                source_value,
                                u,
                            )
                        if b["kind"] == "affine":
                            affine_source = b["_affine_source"]
                            input_baseline = r.baseline
                            if r._source_unit is not None:
                                input_baseline = registry.Quantity(
                                    r.baseline.to(r.spec.si_unit).magnitude,
                                    r._source_unit,
                                )
                            source(
                                _source_unit_identity(
                                    scope="plan_affine",
                                    component_path=path,
                                    parameter_id=k,
                                    field="input_baseline",
                                ),
                                input_baseline,
                                r.spec.si_unit,
                            )
                            source(
                                _source_unit_identity(
                                    scope="plan_affine",
                                    component_path=path,
                                    parameter_id=k,
                                    field="mapped_output_baseline",
                                ),
                                base.to(affine_source["intercept"].units),
                                u,
                            )
                            for affine_field, value, unit in (
                                (
                                    "slope",
                                    affine_source["slope"],
                                    b["slope"]["si_unit"],
                                ),
                                (
                                    "intercept",
                                    affine_source["intercept"],
                                    b["intercept"]["si_unit"],
                                ),
                                (
                                    "support_lower",
                                    affine_source["support_lower"],
                                    r.spec.si_unit,
                                ),
                                (
                                    "support_upper",
                                    affine_source["support_upper"],
                                    r.spec.si_unit,
                                ),
                            ):
                                source(
                                    _source_unit_identity(
                                        scope="plan_affine",
                                        component_path=path,
                                        parameter_id=k,
                                        field=affine_field,
                                    ),
                                    value,
                                    unit,
                                )
                        public_binding = {
                            x: y for x, y in b.items() if not x.startswith("_")
                        }
                        fs.append({"id": k, "unit": u, "binding": public_binding})
                        if r:
                            defs[r._key()] = r._definition_record()
                            edges.append(
                                {
                                    "path": list(path),
                                    "field": k,
                                    "parameter": r._key_record(),
                                }
                            )
                    leaves.append(
                        {
                            "path": list(path),
                            "model": c.factory,
                            "pin_order": list(c.pins),
                            "fields": fs,
                            "model_metadata": dict(c.metadata),
                            "oriented_branches": [
                                {
                                    "id": k,
                                    "positive_pin": v[0],
                                    "negative_pin": v[1],
                                    "value_field": v[2],
                                }
                                for k, v in c.branches.items()
                            ],
                        }
                    )
                if not c.body:
                    for p in c.pins.values():
                        eps.append(
                            {
                                "path": list(path),
                                "pin": p.name,
                                "net": self._final_net_id(p.net),
                            }
                        )
                    for p in c.pins.values():
                        endpoint_nets.append(
                            {"endpoint": s.ep(p), "final_net": final(p.net)}
                        )
                else:
                    # A registered Composite has two distinct public-boundary
                    # views: its containing-scope outer pins and the rebased
                    # exposed pins in its immutable body overlay.  Both are
                    # typed aliases of one final net and must be captured so
                    # Link/Port consumers never reconstruct topology from a
                    # live ComponentInstance.
                    endpoint_nets.extend(
                        {"endpoint": s.ep(pin), "final_net": final(pin.net)}
                        for pin in c.pins.values()
                    )
            couplings.extend(
                x
                for x in s.record(path=scope_path)["structures"]
                if x["kind"] == "coupling"
            )
        for port in self.ports:
            endpoint_nets.append(
                {
                    "endpoint": {"kind": "port", "id": port.id},
                    "final_net": final(port.net),
                }
            )
        aliases = {}
        for scope, scope_path in self._walk():
            if not scope_path:
                for bus in scope.buses.values():
                    aliases.setdefault(final(bus.net), []).append(
                        {"kind": "root_bus", "id": bus.id}
                    )
            for name, coordinate in scope.exposed_coordinates.items():
                aliases.setdefault(final(coordinate.net), []).append(
                    {
                        "kind": "exposed_coordinate",
                        "scope": list(scope_path),
                        "id": name,
                    }
                )
        for port in self.ports:
            aliases.setdefault(final(port.net), []).append(
                {"kind": "port", "id": port.id}
            )
        nodes = [
            {
                "final_net": net,
                "compiler_node_id": net,
                "visibility": "public" if aliases.get(net) else "internal",
                "public_aliases": aliases.get(net, []),
            }
            for net in sorted(
                {
                    row["final_net"]
                    for row in endpoint_nets
                    if row["final_net"] != "ground"
                }
            )
        ]
        semantic = {
            "schema": "scnsim.authoring_snapshot",
            "schema_version": 2,
            "plan_id": self.id,
            "scope_hierarchy": self.record(),
            "occurrences": occ,
            "physical_leaves": leaves,
            "connectivity": {
                "physical_endpoints": eps,
                "endpoint_nets": endpoint_nets,
                "node_coordinates": nodes,
                "canonical_ground": "ground",
                "ports": [
                    {
                        "id": p.id,
                        "net": self._final_net_id(p.net),
                        "role": p.role,
                        "reference_impedance": quantity_record(
                            p.reference_impedance, "ohm"
                        ),
                    }
                    for p in self.ports
                ],
                "couplings": couplings,
            },
            "parameter_closure": {
                "definitions": list(defs.values()),
                "field_bindings": edges,
            },
        }
        for port in self.ports:
            source(
                _source_unit_identity(
                    scope="plan_port", parameter_id=port.id, field="reference_impedance"
                ),
                port.reference_impedance,
                "ohm",
            )
        source_units.sort(key=lambda row: row["identity"])
        prov = {
            "source_units": source_units,
            "ground_pins_call_groups": [
                [s.ep(p) for p in g] for s, _ in self._walk() for g in s.ground_calls
            ],
        }
        return AuthoringSnapshot.create(
            semantic_record=semantic,
            source_provenance=prov,
            resolution_refs=captured_refs,
            resolution_fields=captured_fields,
        )

    def _resolve_parameter_point(
        self,
        supplied: ParameterSet | None = None,
        *,
        snapshot: AuthoringSnapshot | None = None,
    ) -> ResolvedPlanPoint:
        from ._parameter_resolution import resolve_parameter_point

        snap = self._capture_authoring_snapshot() if snapshot is None else snapshot
        return resolve_parameter_point(snap, supplied)


class _LibraryMeta(type):
    """Create storage-free, immutable catalog types before class creation ends."""

    def __new__(cls, name, bases, namespace, **kwargs):
        if bases and any(not isinstance(base, _LibraryMeta) for base in bases):
            raise TypeError("Library subclasses cannot use non-Library bases")
        if namespace.get("__slots__", ()) not in ((), []):
            raise TypeError("Library subclasses cannot declare instance storage")
        for method_name, method in tuple(namespace.items()):
            if method_name.startswith("_") or not inspect.isfunction(method):
                continue

            @wraps(method)
            def wrapped(
                self,
                *args,
                __factory=method,
                __name=method_name,
                **kw,
            ):
                token = _factory_context.set((self, __name))
                try:
                    result = __factory(self, *args, **kw)
                finally:
                    _factory_context.reset(token)
                if not isinstance(result, ComponentInstance):
                    raise TypeError("Library factories must return ComponentInstance")
                expected_source = _catalog_source(self)
                if (
                    result.factory != __name
                    or result.catalog_id != expected_source["catalog_id"]
                    or dict(result.catalog_source) != expected_source
                ):
                    raise TypeError(
                        "Library factory must return its own catalog component"
                    )
                return result

            namespace[method_name] = wrapped
        namespace["__slots__"] = ()
        return super().__new__(cls, name, bases, namespace, **kwargs)

    def __setattr__(self, name, value):
        raise AttributeError("Library catalog types are immutable")

    def __delattr__(self, name):
        raise AttributeError("Library catalog types are immutable")


class Library(metaclass=_LibraryMeta):
    """Immutable catalog base; factories return sealed ComponentInstances only."""

    __slots__ = ()

    def __setattr__(self, name, value):
        raise AttributeError("Library catalogs are immutable")

    def __delattr__(self, name):
        raise AttributeError("Library catalogs are immutable")


def _catalog_source(x):
    from ._canonical import catalog_source_record

    return catalog_source_record(x)


class _BuiltinComponents(Library):
    def _src(self) -> dict[str, object]:
        return _catalog_source(self)

    def _simple(self, id: str, f: str, k: str, v: object, u: str) -> ComponentInstance:
        return ComponentInstance._create(
            id=id,
            factory=f,
            pins=("terminal_1", "terminal_2"),
            fields={k: (v, u, True)},
            kind="ordinary",
            catalog_source=self._src(),
            branches={"self": ("terminal_1", "terminal_2", k)}
            if f == "inductor"
            else {},
            _token=_component_creation_token,
        )

    def resistor(self, *, id: str, resistance: object) -> ComponentInstance:
        return self._simple(id, "resistor", "resistance", resistance, "ohm")

    def capacitor(self, *, id: str, capacitance: object) -> ComponentInstance:
        return self._simple(id, "capacitor", "capacitance", capacitance, "farad")

    def inductor(self, *, id: str, inductance: object) -> ComponentInstance:
        return self._simple(id, "inductor", "inductance", inductance, "henry")

    def josephson_junction(
        self,
        *,
        id: str,
        josephson_inductance: object,
        junction_capacitance: object = 0 * registry.farad,
    ) -> ComponentInstance:
        return ComponentInstance._create(
            id=id,
            factory="josephson_junction",
            pins=("terminal_1", "terminal_2"),
            fields={
                "josephson_inductance": (josephson_inductance, "henry", True),
                "junction_capacitance": (
                    junction_capacitance,
                    "farad",
                    False,
                    True,
                ),
            },
            kind="ordinary",
            catalog_source=self._src(),
            branches={"self": ("terminal_1", "terminal_2", "josephson_inductance")},
            _token=_component_creation_token,
        )

    def transmission_line(
        self, *, id: str, length: object, rlgc: object, n_sections: int
    ) -> ComponentInstance:
        if (
            isinstance(n_sections, bool)
            or not isinstance(n_sections, int)
            or n_sections < 1
        ):
            raise ValueError("n_sections must be positive integer")
        b = rlgc.baseline if isinstance(rlgc, ParameterRef) else rlgc
        if not isinstance(b, RLGC):
            raise TypeError("rlgc must be RLGC or ParameterRef")
        return ComponentInstance._create(
            id=id,
            factory="transmission_line",
            pins=tuple(f"{e}.{c}" for e in ("head", "tail") for c in b.conductors),
            fields={"length": (length, "meter", True), "rlgc": (rlgc, "rlgc", False)},
            kind="complete_line",
            catalog_source=self._src(),
            metadata={"n_sections": n_sections},
            _token=_component_creation_token,
        )

    def interdigitated_capacitor(
        self,
        *,
        id: str,
        terminal_1_to_reference_capacitance: object,
        terminal_2_to_reference_capacitance: object,
        terminal_mutual_capacitance: object,
    ) -> ComponentInstance:
        body = CompositePlan(id=id, library=self)
        body.factory_name = "interdigitated_capacitor"
        c1 = body.add(
            self.capacitor(
                id="terminal_1_to_reference",
                capacitance=terminal_1_to_reference_capacitance,
            )
        )
        c2 = body.add(
            self.capacitor(
                id="terminal_2_to_reference",
                capacitance=terminal_2_to_reference_capacitance,
            )
        )
        cm = body.add(
            self.capacitor(
                id="terminal_mutual", capacitance=terminal_mutual_capacitance
            )
        )
        a = body.bus(id="terminal_1")
        b = body.bus(id="terminal_2")
        body.branch(id="terminal_1_shunt", at=a, elements=(c1,), end=body.ground)
        body.branch(id="terminal_2_shunt", at=b, elements=(c2,), end=body.ground)
        body.series(id="mutual", start=a, elements=(cm,), end=b)
        body.expose_pin(id="terminal_1", at=a)
        body.expose_pin(id="terminal_2", at=b)
        for name, value in (
            (
                "terminal_1_to_reference_capacitance",
                terminal_1_to_reference_capacitance,
            ),
            (
                "terminal_2_to_reference_capacitance",
                terminal_2_to_reference_capacitance,
            ),
            ("terminal_mutual_capacitance", terminal_mutual_capacitance),
        ):
            if isinstance(value, ParameterRef):
                body.expose_parameter(id=name, parameter=value)
        return body.build()

    def symmetric_squid(
        self,
        *,
        id: str,
        josephson_inductance: object,
        junction_capacitance: object = 0 * registry.farad,
        loop_inductance: object | None = None,
    ) -> ComponentInstance:
        if loop_inductance is None:
            raise TypeError("loop_inductance is required")
        body = CompositePlan(id=id, library=self)
        body.factory_name = "symmetric_squid"
        j1 = body.add(
            self.josephson_junction(
                id="junction_1",
                josephson_inductance=josephson_inductance,
                junction_capacitance=junction_capacitance,
            )
        )
        loop = body.add(self.inductor(id="loop", inductance=loop_inductance))
        j2 = body.add(
            self.josephson_junction(
                id="junction_2",
                josephson_inductance=josephson_inductance,
                junction_capacitance=junction_capacitance,
            )
        )
        a = body.bus(id="terminal_1")
        b = body.bus(id="terminal_2")
        body.parallel(id="finite_loop", start=a, branches=((j1,), (loop, j2)), end=b)
        body.expose_pin(id="terminal_1", at=a)
        body.expose_pin(id="terminal_2", at=b)
        body.expose_inductive_branch(id="loop", branch=loop.inductive_branch("self"))
        for name, value in (
            ("josephson_inductance", josephson_inductance),
            ("junction_capacitance", junction_capacitance),
            ("loop_inductance", loop_inductance),
        ):
            if isinstance(value, ParameterRef):
                body.expose_parameter(id=name, parameter=value)
        return body.build()

    def grounded_parallel_linear_lc_resonator(
        self, *, id: str, capacitance: object, inductance: object
    ) -> ComponentInstance:
        return self._resonator(
            id=id,
            grounded=True,
            branch="linear",
            values={"capacitance": capacitance, "inductance": inductance},
        )

    def floating_parallel_linear_lc_resonator(
        self,
        *,
        id: str,
        terminal_1_to_reference_capacitance: object,
        terminal_2_to_reference_capacitance: object,
        terminal_mutual_capacitance: object,
        inductance: object,
    ) -> ComponentInstance:
        return self._resonator(
            id=id,
            grounded=False,
            branch="linear",
            values={
                "terminal_1_to_reference_capacitance": terminal_1_to_reference_capacitance,
                "terminal_2_to_reference_capacitance": terminal_2_to_reference_capacitance,
                "terminal_mutual_capacitance": terminal_mutual_capacitance,
                "inductance": inductance,
            },
        )

    def grounded_parallel_single_junction_resonator(
        self,
        *,
        id: str,
        capacitance: object,
        josephson_inductance: object,
        junction_capacitance: object,
    ) -> ComponentInstance:
        return self._resonator(
            id=id,
            grounded=True,
            branch="junction",
            values={
                "capacitance": capacitance,
                "josephson_inductance": josephson_inductance,
                "junction_capacitance": junction_capacitance,
            },
        )

    def floating_parallel_single_junction_resonator(
        self,
        *,
        id: str,
        terminal_1_to_reference_capacitance: object,
        terminal_2_to_reference_capacitance: object,
        terminal_mutual_capacitance: object,
        josephson_inductance: object,
        junction_capacitance: object,
    ) -> ComponentInstance:
        return self._resonator(
            id=id,
            grounded=False,
            branch="junction",
            values={
                "terminal_1_to_reference_capacitance": terminal_1_to_reference_capacitance,
                "terminal_2_to_reference_capacitance": terminal_2_to_reference_capacitance,
                "terminal_mutual_capacitance": terminal_mutual_capacitance,
                "josephson_inductance": josephson_inductance,
                "junction_capacitance": junction_capacitance,
            },
        )

    def grounded_parallel_symmetric_squid_resonator(
        self,
        *,
        id: str,
        capacitance: object,
        josephson_inductance: object,
        junction_capacitance: object,
        loop_inductance: object,
    ) -> ComponentInstance:
        return self._resonator(
            id=id,
            grounded=True,
            branch="squid",
            values={
                "capacitance": capacitance,
                "josephson_inductance": josephson_inductance,
                "junction_capacitance": junction_capacitance,
                "loop_inductance": loop_inductance,
            },
        )

    def floating_parallel_symmetric_squid_resonator(
        self,
        *,
        id: str,
        terminal_1_to_reference_capacitance: object,
        terminal_2_to_reference_capacitance: object,
        terminal_mutual_capacitance: object,
        josephson_inductance: object,
        junction_capacitance: object,
        loop_inductance: object,
    ) -> ComponentInstance:
        return self._resonator(
            id=id,
            grounded=False,
            branch="squid",
            values={
                "terminal_1_to_reference_capacitance": terminal_1_to_reference_capacitance,
                "terminal_2_to_reference_capacitance": terminal_2_to_reference_capacitance,
                "terminal_mutual_capacitance": terminal_mutual_capacitance,
                "josephson_inductance": josephson_inductance,
                "junction_capacitance": junction_capacitance,
                "loop_inductance": loop_inductance,
            },
        )

    def _resonator(self, *, id, grounded, branch, values):
        body = CompositePlan(id=id, library=self)
        if grounded:
            capacitor = body.add(
                self.capacitor(id="capacitor", capacitance=values["capacitance"])
            )
            first = body.bus(id="terminal_1")
            second = body.ground
        else:
            capacitor = body.add(
                self.interdigitated_capacitor(
                    id="capacitor",
                    terminal_1_to_reference_capacitance=values[
                        "terminal_1_to_reference_capacitance"
                    ],
                    terminal_2_to_reference_capacitance=values[
                        "terminal_2_to_reference_capacitance"
                    ],
                    terminal_mutual_capacitance=values["terminal_mutual_capacitance"],
                )
            )
            first = body.bus(id="terminal_1")
            second = body.bus(id="terminal_2")
        if branch == "linear":
            element = body.add(
                self.inductor(id="inductor", inductance=values["inductance"])
            )
        elif branch == "junction":
            element = body.add(
                self.josephson_junction(
                    id="junction",
                    josephson_inductance=values["josephson_inductance"],
                    junction_capacitance=values["junction_capacitance"],
                )
            )
        elif branch == "squid":
            element = body.add(
                self.symmetric_squid(
                    id="squid",
                    josephson_inductance=values["josephson_inductance"],
                    junction_capacitance=values["junction_capacitance"],
                    loop_inductance=values["loop_inductance"],
                )
            )
        else:
            raise AssertionError("unknown resonator branch")

        def element_use(component):
            if component.kind == "ordinary":
                return component
            return component.between(
                component.pin("terminal_1"), component.pin("terminal_2")
            )

        body.parallel(
            id="parallel_resonator",
            start=first,
            branches=((element_use(capacitor),), (element_use(element),)),
            end=second,
        )
        body.expose_pin(id="terminal_1", at=first)
        # Grounded resonators offer two boundary handles on their one live
        # internal bus; the physical parallel branches still end at ground.
        body.expose_pin(id="terminal_2", at=first if grounded else second)
        if branch == "squid":
            body.expose_inductive_branch(
                id="loop", branch=element.inductive_branch("loop")
            )
        for name, value in values.items():
            if isinstance(value, ParameterRef):
                body.expose_parameter(id=name, parameter=value)
        return body.build()


components = _BuiltinComponents()
