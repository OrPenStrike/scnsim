"""Public circuit-model authoring declarations for the executable dev3 slice.

The Plan remains the sole physical authority.  This module owns primitive
R/L/C authoring only; Composite, RLGC, couplings, and compiled diagrams keep
their declared fail-fast boundary until their respective V1 slices.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal
import unicodedata

from ._scaffold import unavailable
from .errors import PlanSealedError, SCNSimValidationError
from .units import Quantity, registry, require_positive_quantity, require_quantity

if TYPE_CHECKING:
    from .results import CircuitDiagramResult
    from .specs import CircuitDiagramSpec


def _identifier(value: str, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    value = unicodedata.normalize("NFC", value)
    if not value or any(ord(char) < 32 or char in "/\\" for char in value):
        raise ValueError(f"{field} must be a nonempty portable identifier")
    return value


def _quantity_record(value: Quantity, unit: str) -> dict[str, str]:
    """Encode one physical scalar through the single canonical quantity rule."""

    from ._canonical import quantity_envelope

    return quantity_envelope(value, si_unit=unit, registry=registry)


class PinRef:
    """Stable reference to one named electrical terminal of a component."""

    __slots__ = ("_component", "name")

    def __init__(self) -> None:
        unavailable("PinRef construction")

    @classmethod
    def _create(cls, component: ComponentInstance, name: str) -> PinRef:
        instance = object.__new__(cls)
        instance._component = component
        instance.name = name
        return instance

    @property
    def component_id(self) -> str:
        """Owning component ID, useful only for authoring inspection."""

        return self._component.id

    def _endpoint(self) -> dict[str, object]:
        return {"component_path": [self._component.id], "pin_id": self.name}


class ElectricNodeRef:
    """Plan-bound handle to one declared equipotential electric node."""

    __slots__ = ("_plan", "_node")

    def __init__(self) -> None:
        unavailable("ElectricNodeRef construction")

    @classmethod
    def _create(cls, plan: CircuitPlan, node: _PlanNode) -> ElectricNodeRef:
        instance = object.__new__(cls)
        instance._plan = plan
        instance._node = node
        return instance

    @property
    def id(self) -> str:
        """Current public or opaque canonical node identity."""

        return self._node.id

    @property
    def is_public(self) -> bool:
        """Whether this node is selectable as a public Runtime coordinate."""

        return self._node.visibility != "internal"


class CoordinateRef:
    """Public handle to one observable coordinate inside a later composite."""

    __slots__ = ("_component", "name")

    def __init__(self) -> None:
        unavailable("CoordinateRef construction")

    @classmethod
    def _create(cls, component: ComponentInstance, name: str) -> CoordinateRef:
        instance = object.__new__(cls)
        instance._component = component
        instance.name = name
        return instance


class PortRef:
    """Plan-bound handle to one logical Port component."""

    __slots__ = ("_plan", "id", "_node", "role", "reference_impedance")

    def __init__(self) -> None:
        unavailable("PortRef construction")

    @classmethod
    def _create(
        cls,
        plan: CircuitPlan,
        id: str,
        node: ElectricNodeRef,
        role: Literal["terminated", "nonloading_probe"],
        reference_impedance: Quantity,
    ) -> PortRef:
        instance = object.__new__(cls)
        instance._plan = plan
        instance.id = id
        instance._node = node
        instance.role = role
        instance.reference_impedance = reference_impedance
        return instance

    @property
    def node(self) -> ElectricNodeRef:
        """Electric node attached to this logical Port."""

        return self._node


class ParameterRef:
    """Stable public handle to one primitive component parameter."""

    __slots__ = ("_component", "id", "baseline", "unit")

    def __init__(self) -> None:
        unavailable("ParameterRef construction")

    @classmethod
    def _create(
        cls, component: ComponentInstance, id: str, baseline: Quantity, unit: str
    ) -> ParameterRef:
        instance = object.__new__(cls)
        instance._component = component
        instance.id = id
        instance.baseline = baseline
        instance.unit = unit
        return instance

    @property
    def component_id(self) -> str:
        """Owning component ID."""

        return self._component.id

    def _canonical_ref(self) -> dict[str, object]:
        return {"component_path": [self.component_id], "parameter_id": self.id}

    def show(self) -> object:
        """Inspect this primitive's sealed baseline and identity."""

        return MappingProxyType(
            {
                "component_id": self.component_id,
                "parameter_id": self.id,
                "baseline": self.baseline,
                "unit": self.unit,
                "fan_out": (),
                "affine_maps": (),
            }
        )


class InductiveBranchRef:
    """Stable oriented handle to an inductive branch used by a later slice."""

    __slots__ = ("_component", "id")

    def __init__(self) -> None:
        unavailable("InductiveBranchRef construction")

    @classmethod
    def _create(cls, component: ComponentInstance, id: str) -> InductiveBranchRef:
        instance = object.__new__(cls)
        instance._component = component
        instance.id = id
        return instance


class ParameterSpec:
    """Library declaration of one public physical parameter schema."""

    def __init__(self, *, unit: object) -> None:
        unavailable("ParameterSpec construction")


class ParameterSet:
    """Immutable physical values bound to exact ``ParameterRef`` keys."""

    __slots__ = ("_values", "_allow_extrapolation")

    def __init__(
        self,
        values: Mapping[ParameterRef, Quantity],
        *,
        allow_extrapolation: Sequence[ParameterRef] = (),
    ) -> None:
        if not isinstance(values, Mapping):
            raise TypeError("values must map ParameterRef objects to Quantities")
        bound: list[tuple[ParameterRef, Quantity]] = []
        for parameter, value in values.items():
            if not isinstance(parameter, ParameterRef):
                raise TypeError("ParameterSet keys must be ParameterRef objects")
            bound.append(
                (parameter, require_quantity(value, parameter.unit, name=parameter.id))
            )
        bound.sort(key=lambda item: (item[0].component_id, item[0].id))
        if len({parameter for parameter, _ in bound}) != len(bound):
            raise ValueError("ParameterSet contains a duplicate parameter")
        approved = tuple(allow_extrapolation)
        if any(not isinstance(parameter, ParameterRef) for parameter in approved):
            raise TypeError("allow_extrapolation must contain ParameterRef objects")
        approved = tuple(sorted(set(approved), key=lambda item: (item.component_id, item.id)))
        self._values = MappingProxyType(dict(bound))
        self._allow_extrapolation = approved

    @property
    def values(self) -> Mapping[ParameterRef, Quantity]:
        """Ordered read-only mapping used to create an explicitly new request."""

        return self._values

    @property
    def allow_extrapolation(self) -> tuple[ParameterRef, ...]:
        """Exact public parameters authorized for this one request."""

        return self._allow_extrapolation

    def _canonical_record(self) -> dict[str, object]:
        return {
            "type": "parameter_set",
            "bindings": [
                {
                    "parameter": parameter._canonical_ref(),
                    "value": _quantity_record(value, parameter.unit),
                }
                for parameter, value in self._values.items()
            ],
            "allow_extrapolation": [
                parameter._canonical_ref() for parameter in self._allow_extrapolation
            ],
        }


class RLGC:
    """Immutable per-length matrices for a later distributed-line slice."""

    def __init__(
        self,
        *,
        conductors: Sequence[str],
        reference_conductor: str,
        resistance_per_length: object,
        inductance_per_length: object,
        conductance_per_length: object,
        capacitance_per_length: object,
        extraction_frequency: object | None = None,
    ) -> None:
        unavailable("RLGC construction")

    @property
    def conductors(self) -> tuple[str, ...]:
        unavailable("RLGC.conductors")

    @property
    def reference_conductor(self) -> str:
        unavailable("RLGC.reference_conductor")

    @property
    def resistance_per_length(self) -> object:
        unavailable("RLGC.resistance_per_length")

    @property
    def inductance_per_length(self) -> object:
        unavailable("RLGC.inductance_per_length")

    @property
    def conductance_per_length(self) -> object:
        unavailable("RLGC.conductance_per_length")

    @property
    def capacitance_per_length(self) -> object:
        unavailable("RLGC.capacitance_per_length")

    @property
    def extraction_frequency(self) -> object | None:
        unavailable("RLGC.extraction_frequency")


class AffineMap:
    """Declarative calibrated mapping reserved for Composite expansion."""

    def __init__(
        self,
        *,
        input: ParameterRef,
        slope: object,
        intercept: object,
        support: tuple[object, object],
    ) -> None:
        unavailable("AffineMap construction")


class ComponentInstance:
    """One immutable primitive component from an exact ``Library`` factory."""

    __slots__ = ("_factory", "_id", "_pins", "_parameters", "_catalog_id", "_catalog_source")

    def __init__(self) -> None:
        unavailable("ComponentInstance construction")

    @classmethod
    def _create(
        cls,
        *,
        id: str,
        factory: Literal["resistor", "capacitor", "inductor"],
        parameter_id: str,
        baseline: Quantity,
        unit: str,
        catalog_id: str = "scnsim.components",
        catalog_source: Mapping[str, object],
    ) -> ComponentInstance:
        instance = object.__new__(cls)
        instance._id = _identifier(id, field="component id")
        instance._factory = factory
        instance._catalog_id = catalog_id
        instance._catalog_source = MappingProxyType(dict(catalog_source))
        instance._pins = MappingProxyType(
            {name: PinRef._create(instance, name) for name in ("terminal_1", "terminal_2")}
        )
        instance._parameters = MappingProxyType(
            {parameter_id: ParameterRef._create(instance, parameter_id, baseline, unit)}
        )
        return instance

    @property
    def id(self) -> str:
        """Stable component identity inside its parent Plan."""

        return self._id

    @property
    def factory(self) -> str:
        """Built-in primitive factory identity."""

        return self._factory

    @property
    def catalog_id(self) -> str:
        """Reserved stable identity for built-in primitive components."""

        return self._catalog_id

    def pin(self, name: str, *, conductor: str | None = None) -> PinRef:
        """Return one declared primitive terminal."""

        if conductor is not None:
            unavailable("ComponentInstance.pin(conductor=...)")
        try:
            return self._pins[name]
        except KeyError:
            raise KeyError(f"component {self.id!r} has no pin {name!r}") from None

    def parameter(self, name: str) -> ParameterRef:
        """Return one deliberately public primitive parameter."""

        try:
            return self._parameters[name]
        except KeyError:
            raise KeyError(f"component {self.id!r} has no parameter {name!r}") from None

    def inductive_branch(self, name: str) -> InductiveBranchRef:
        """Inductive branch handles arrive with Composite support in dev4."""

        unavailable("ComponentInstance.inductive_branch")

    def coordinate(self, name: str) -> CoordinateRef:
        """Composite coordinates arrive with Composite support in dev4."""

        unavailable("ComponentInstance.coordinate")

    def _canonical_snapshot(self) -> dict[str, object]:
        parameter = next(iter(self._parameters.values()))
        return {
            "component_path": [self.id],
            "catalog_id": self.catalog_id,
            "factory": self.factory,
            "pin_order": ["terminal_1", "terminal_2"],
            "parameter_bindings": [
                {
                    "id": parameter.id,
                    "binding": {
                        "kind": "constant",
                        "value": _quantity_record(parameter.baseline, parameter.unit),
                    },
                }
            ],
            "inductive_branches": [],
            "realization": {
                "kind": self.factory,
                parameter.id: {
                    "kind": "constant",
                    "value": _quantity_record(parameter.baseline, parameter.unit),
                },
            },
        }


class CompositePlan:
    """Declarative private graph reserved for the dev4 Composite slice."""

    def __init__(self, *, id: str, library: Library) -> None:
        unavailable("CompositePlan construction")

    def parameter(self, *, id: str, baseline: object, spec: ParameterSpec) -> ParameterRef:
        unavailable("CompositePlan.parameter")

    def add(self, component: ComponentInstance) -> ComponentInstance:
        unavailable("CompositePlan.add")

    def net(self, *pins: PinRef, id: str | None = None) -> ElectricNodeRef:
        unavailable("CompositePlan.net")

    def ground(self, *pins: PinRef) -> None:
        unavailable("CompositePlan.ground")

    def expose_pin(self, *, id: str, at: ElectricNodeRef) -> PinRef:
        unavailable("CompositePlan.expose_pin")

    def expose_coordinate(self, *, id: str, at: ElectricNodeRef) -> CoordinateRef:
        unavailable("CompositePlan.expose_coordinate")

    def build(self) -> ComponentInstance:
        unavailable("CompositePlan.build")


class _LibraryMeta(type):
    """Keep every catalog subclass free of mutable instance storage."""

    def __new__(
        cls, name: str, bases: tuple[type, ...], namespace: dict[str, object], **kwargs: object
    ) -> _LibraryMeta:
        if bases and any(not isinstance(base, _LibraryMeta) for base in bases):
            raise TypeError("Library subclasses cannot use non-Library bases")
        if namespace.get("__slots__", ()) not in ((), []):
            raise TypeError("Library subclasses cannot declare instance state")
        namespace["__slots__"] = ()
        return super().__new__(cls, name, bases, namespace, **kwargs)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Library catalog types are immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Library catalog types are immutable")


class Library(metaclass=_LibraryMeta):
    """Base class for one immutable, project-owned component catalog."""

    __slots__ = ()

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Library catalogs are immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Library catalogs are immutable")


class _BuiltinComponents(Library):
    """Implementation behind the public ``scnsim.components`` catalog."""

    @staticmethod
    def _primitive(
        *, id: str, factory: Literal["resistor", "capacitor", "inductor"], value: Quantity, unit: str
    ) -> ComponentInstance:
        baseline = require_positive_quantity(value, unit, name=factory)
        parameter_id = {"resistor": "resistance", "capacitor": "capacitance", "inductor": "inductance"}[factory]
        from ._canonical import catalog_source_record

        return ComponentInstance._create(
            id=id,
            factory=factory,
            parameter_id=parameter_id,
            baseline=baseline,
            unit=unit,
            catalog_source=catalog_source_record(components),
        )

    def resistor(self, *, id: str, resistance: Quantity) -> ComponentInstance:
        """Declare a resistor oriented from ``terminal_1`` to ``terminal_2``."""

        return self._primitive(id=id, factory="resistor", value=resistance, unit="ohm")

    def capacitor(self, *, id: str, capacitance: Quantity) -> ComponentInstance:
        """Declare a capacitor oriented from ``terminal_1`` to ``terminal_2``."""

        return self._primitive(id=id, factory="capacitor", value=capacitance, unit="farad")

    def inductor(self, *, id: str, inductance: Quantity) -> ComponentInstance:
        """Declare an inductor oriented from ``terminal_1`` to ``terminal_2``."""

        return self._primitive(id=id, factory="inductor", value=inductance, unit="henry")

    def josephson_junction(
        self,
        *,
        id: str,
        josephson_inductance: object,
        junction_capacitance: object,
    ) -> ComponentInstance:
        unavailable("components.josephson_junction")

    def transmission_line(
        self,
        *,
        id: str,
        length: object,
        rlgc: RLGC,
        n_sections: int,
    ) -> ComponentInstance:
        unavailable("components.transmission_line")

    def interdigitated_capacitor(
        self,
        *,
        id: str,
        terminal_1_to_reference_capacitance: object,
        terminal_2_to_reference_capacitance: object,
        terminal_mutual_capacitance: object,
    ) -> ComponentInstance:
        unavailable("components.interdigitated_capacitor")

    def symmetric_squid(
        self,
        *,
        id: str,
        josephson_inductance: object,
        junction_capacitance: object,
        loop_inductance: object,
    ) -> ComponentInstance:
        unavailable("components.symmetric_squid")

    def grounded_parallel_linear_lc_resonator(
        self,
        *,
        id: str,
        capacitance: object,
        inductance: object,
    ) -> ComponentInstance:
        unavailable("components.grounded_parallel_linear_lc_resonator")

    def floating_parallel_linear_lc_resonator(
        self,
        *,
        id: str,
        terminal_1_to_reference_capacitance: object,
        terminal_2_to_reference_capacitance: object,
        terminal_mutual_capacitance: object,
        inductance: object,
    ) -> ComponentInstance:
        unavailable("components.floating_parallel_linear_lc_resonator")

    def grounded_parallel_single_junction_resonator(
        self,
        *,
        id: str,
        capacitance: object,
        josephson_inductance: object,
        junction_capacitance: object,
    ) -> ComponentInstance:
        unavailable("components.grounded_parallel_single_junction_resonator")

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
        unavailable("components.floating_parallel_single_junction_resonator")

    def grounded_parallel_symmetric_squid_resonator(
        self,
        *,
        id: str,
        capacitance: object,
        josephson_inductance: object,
        junction_capacitance: object,
        loop_inductance: object,
    ) -> ComponentInstance:
        unavailable("components.grounded_parallel_symmetric_squid_resonator")

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
        unavailable("components.floating_parallel_symmetric_squid_resonator")


@dataclass(slots=True)
class _PlanNode:
    id: str
    visibility: Literal["public", "internal", "port_promoted"]
    endpoints: tuple[PinRef, ...]


class CircuitPlan:
    """The single physical authority for one reusable primitive circuit model."""

    __slots__ = (
        "_id", "_sealed", "_components", "_nodes", "_grounded", "_ground_groups", "_ports", "_pin_nodes"
    )

    def __init__(self, *, id: str) -> None:
        self._id = _identifier(id, field="plan id")
        if self._id == "ground":
            raise ValueError("plan id cannot be the reserved ground identity")
        self._sealed = False
        self._components: list[ComponentInstance] = []
        self._nodes: list[_PlanNode] = []
        self._grounded: set[PinRef] = set()
        self._ground_groups: list[tuple[PinRef, ...]] = []
        self._ports: list[PortRef] = []
        self._pin_nodes: dict[PinRef, _PlanNode | Literal["ground"]] = {}

    @property
    def id(self) -> str:
        """User-supplied stable Plan identity."""

        return self._id

    @property
    def sealed(self) -> bool:
        """Whether the Plan has been permanently consumed by a Run."""

        return self._sealed

    @property
    def components(self) -> tuple[ComponentInstance, ...]:
        return tuple(self._components)

    @property
    def nodes(self) -> tuple[ElectricNodeRef, ...]:
        return tuple(ElectricNodeRef._create(self, node) for node in self._nodes)

    @property
    def ports(self) -> tuple[PortRef, ...]:
        return tuple(self._ports)

    def _assert_mutable(self) -> None:
        if self._sealed:
            raise PlanSealedError("CircuitPlan is permanently sealed", stage="plan_mutation")

    def _validate_pin(self, pin: PinRef) -> None:
        if not isinstance(pin, PinRef) or pin._component not in self._components:
            raise SCNSimValidationError("pin does not belong to this Plan", stage="authoring")
        if pin in self._pin_nodes:
            raise SCNSimValidationError("each pin may belong to one node or ground", stage="authoring")

    def add(self, component: ComponentInstance) -> ComponentInstance:
        """Add one built-in primitive component and return its Plan handle."""

        self._assert_mutable()
        if not isinstance(component, ComponentInstance):
            raise TypeError("CircuitPlan.add requires a ComponentInstance")
        if component.catalog_id != "scnsim.components":
            unavailable("CircuitPlan.add custom ComponentInstance")
        if any(existing.id == component.id for existing in self._components):
            raise SCNSimValidationError("component IDs must be unique", stage="authoring")
        self._components.append(component)
        return component

    def net(self, *pins: PinRef, id: str | None = None) -> ElectricNodeRef:
        """Create one complete equipotential node from exactly these pins."""

        self._assert_mutable()
        if not pins:
            raise SCNSimValidationError("net requires at least one pin", stage="authoring")
        if len(set(pins)) != len(pins):
            raise SCNSimValidationError("net cannot repeat a pin", stage="authoring")
        for pin in pins:
            self._validate_pin(pin)
        endpoints = [pin._endpoint() for pin in pins]
        if id is None:
            from ._canonical import internal_node_id

            node_id = internal_node_id(endpoints)
            visibility: Literal["public", "internal", "port_promoted"] = "internal"
        else:
            node_id = _identifier(id, field="node id")
            if node_id == "ground" or node_id.startswith("internal-"):
                raise SCNSimValidationError("node ID is reserved", stage="authoring")
            visibility = "public"
        if any(node.id == node_id for node in self._nodes):
            raise SCNSimValidationError("node IDs must be unique", stage="authoring")
        node = _PlanNode(node_id, visibility, tuple(pins))
        self._nodes.append(node)
        self._pin_nodes.update({pin: node for pin in pins})
        return ElectricNodeRef._create(self, node)

    def ground(self, *pins: PinRef) -> None:
        """Attach one nonempty local pin group to the canonical Plan ground."""

        self._assert_mutable()
        if not pins:
            raise SCNSimValidationError("ground requires at least one pin", stage="authoring")
        if len(set(pins)) != len(pins):
            raise SCNSimValidationError("ground cannot repeat a pin", stage="authoring")
        for pin in pins:
            self._validate_pin(pin)
        self._grounded.update(pins)
        self._ground_groups.append(tuple(pins))
        self._pin_nodes.update({pin: "ground" for pin in pins})

    def add_port(
        self,
        *,
        id: str,
        at: ElectricNodeRef,
        role: Literal["terminated", "nonloading_probe"],
        reference_impedance: Quantity,
    ) -> PortRef:
        """Attach one logical Port to an existing Plan node."""

        self._assert_mutable()
        port_id = _identifier(id, field="port id")
        if not isinstance(at, ElectricNodeRef) or at._plan is not self:
            raise SCNSimValidationError("Port must attach to an ElectricNodeRef from this Plan", stage="authoring")
        if role not in ("terminated", "nonloading_probe"):
            raise ValueError("role must be 'terminated' or 'nonloading_probe'")
        if any(port.id == port_id for port in self._ports):
            raise SCNSimValidationError("port IDs must be unique", stage="authoring")
        if any(port.node._node is at._node for port in self._ports):
            raise SCNSimValidationError("one electric node may own at most one Port", stage="authoring")
        if at._node.visibility == "internal":
            at._node.id = port_id
            at._node.visibility = "port_promoted"
        reference = require_positive_quantity(reference_impedance, "ohm", name="reference_impedance")
        port = PortRef._create(self, port_id, at, role, reference)
        self._ports.append(port)
        return port

    def _validate_complete(self) -> None:
        if not self._components:
            raise SCNSimValidationError("CircuitPlan requires at least one component", stage="plan_seal")
        if any(node.visibility == "internal" and len(node.endpoints) < 2 for node in self._nodes):
            raise SCNSimValidationError(
                "an anonymous node without a Port must join at least two pins",
                stage="plan_seal",
            )
        missing = [
            pin.component_id + "." + pin.name
            for component in self._components
            for pin in component._pins.values()
            if pin not in self._pin_nodes
        ]
        if missing:
            raise SCNSimValidationError("every component pin must be netted or grounded", stage="plan_seal", evidence={"missing_pins": missing})

    def _seal(self) -> CircuitPlan:
        """Validate and permanently seal the Plan for ``CircuitRun``."""

        if not self._sealed:
            self._validate_complete()
            self._sealed = True
        return self

    def _canonical_snapshot(self) -> dict[str, object]:
        """Return the closed primitive payload consumed by canonical encoding."""

        self._validate_complete()
        catalog_sources: dict[str, dict[str, object]] = {}
        for component in self._components:
            source = dict(component._catalog_source)
            existing = catalog_sources.setdefault(component.catalog_id, source)
            if existing != source:
                raise SCNSimValidationError(
                    "components from one catalog must share one captured source identity",
                    stage="plan_seal",
                )
        if len(catalog_sources) != 1:
            raise SCNSimValidationError("dev3 Plan requires one built-in catalog source", stage="plan_seal")
        return {
            "plan_id": self.id,
            "catalog_sources": list(catalog_sources.values()),
            "components": [component._canonical_snapshot() for component in self._components],
            "nodes": [
                {
                    "node_id": node.id,
                    "visibility": node.visibility,
                    "endpoints": [pin._endpoint() for pin in node.endpoints],
                }
                for node in self._nodes
            ],
            "grounded_endpoints": [pin._endpoint() for pin in self._grounded],
            "ports": [
                {
                    "port_id": port.id,
                    "node_id": port.node.id,
                    "role": port.role,
                    "reference_impedance": _quantity_record(port.reference_impedance, "ohm"),
                    "orientation": "node_to_reference",
                }
                for port in self._ports
            ],
            "couplings": [],
        }

    _canonical_record = _canonical_snapshot

    def render_schematic(self, spec: CircuitDiagramSpec | None = None) -> CircuitDiagramResult:
        """Materialize a read-only authoring schematic without sealing the Plan."""

        self._validate_complete()
        from .results import CircuitDiagramResult, _verified_result
        from .specs import CircuitDiagramSpec

        spec = CircuitDiagramSpec() if spec is None else spec
        if spec.representation != "authoring":
            unavailable("CircuitPlan.render_schematic(compiled)")
        try:
            import schemdraw
            import schemdraw.elements as elm
        except ImportError as exc:
            raise RuntimeError("Schemdraw is required for authoring schematics") from exc
        color = "#111827" if spec.theme != "dark" else "#f8fafc"
        drawing = schemdraw.Drawing(show=False, transparent=True)
        drawing.config(unit=2.5, color=color, lw=1.8, fontsize=11)
        node_y = {id(node): 3.0 * (index + 1) for index, node in enumerate(self._nodes)}
        rail_end = 3.0 * (len(self._components) + 1)
        for node in self._nodes:
            y = node_y[id(node)]
            drawing.add(elm.Line(color=color).endpoints((-1.0, y), (rail_end, y)))
            drawing.add(elm.Dot(open=True, color=color).at((-1.0, y)).label(node.id, loc="left", color=color))
        ground_y = 0.0
        drawing.add(elm.Line(color=color).endpoints((-1.0, ground_y), (rail_end, ground_y)))
        for index, component in enumerate(self._components, start=1):
            x = 3.0 * index
            terminal_1 = component.pin("terminal_1")
            terminal_2 = component.pin("terminal_2")
            target_1 = self._pin_nodes[terminal_1]
            target_2 = self._pin_nodes[terminal_2]
            y1 = ground_y if target_1 == "ground" else node_y[id(target_1)]
            y2 = ground_y if target_2 == "ground" else node_y[id(target_2)]
            direction = "down" if y1 >= y2 else "up"
            element = getattr({
                "resistor": elm.Resistor,
                "capacitor": elm.Capacitor,
                "inductor": elm.Inductor,
            }[component.factory](color=color).at((x, y1)), direction)().length(abs(y1 - y2) or 0.2)
            parameter = next(iter(component._parameters.values()))
            label = component.id
            if spec.show_parameter_values:
                label = f"{label}\\n{parameter.baseline:~P}"
            drawing.add(element.label(label, loc="right", color=color))
        for group in self._ground_groups:
            first = group[0]
            component_index = self._components.index(first._component) + 1
            drawing.add(elm.Ground(color=color).at((3.0 * component_index, ground_y)))
        for port in self._ports:
            y = node_y[id(port.node._node)]
            drawing.add(
                elm.Dot(open=True, color=color)
                .at((rail_end, y))
                .label(f"{port.id} ({port.reference_impedance:~P})", loc="right", color=color)
            )
        return _verified_result(
            CircuitDiagramResult, drawing=drawing, representation="authoring"
        )

    def couple_inductive(
        self,
        *,
        id: str,
        inductor_a: InductiveBranchRef,
        inductor_b: InductiveBranchRef,
        coupling_coefficient: object,
    ) -> None:
        unavailable("CircuitPlan.couple_inductive")


components = _BuiltinComponents()
"""The immutable built-in SCNSim primitive catalog."""
