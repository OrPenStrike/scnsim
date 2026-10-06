"""Scope assembly, occurrence overlays, and reusable Composite publication.

One grammar maintains wire equivalence, exposure, consumption, and completeness
invariants across root, inline, and factory-owned scopes."""

from __future__ import annotations

from copy import deepcopy
from types import MappingProxyType

from ..errors import PlanSealedError, SCNSimValidationError
from ..units import Quantity, require_quantity
from .components import ComponentInstance
from .factory_context import _component_creation_token, _factory_context
from .handles import (
    _Net, _physical_inductive_branch, BranchRef, BusRef, CoordinateRef,
    CouplingRef, GroundRef, InductiveBranchRef, LinkRef, ParallelRef, PinRef,
    SeriesRef, TapRef, TwoTerminalUse,
)
from .parameters import ParameterRef
from .physical_values import identifier, quantity_record


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
                if len({id(root) for root in pin_roots}) != len(pin_roots):
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
            if c.kind == "ordinary" and len({id(root) for root in roots}) != len(roots):
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
        from .catalogs import Library

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
        from .provenance import catalog_source_record

        src = catalog_source_record(self.library)
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



# Preserve public class identity through the authoring facade.
SubsystemPlan.__module__ = "scnsim.authoring"
CompositePlan.__module__ = "scnsim.authoring"
