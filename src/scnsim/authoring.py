"""Public circuit-model authoring declarations.

This module is for the person who owns a reusable circuit model.  It declares
components, public or internal electrical nodes, logical Port components,
parameters, and inductive couplings in one :class:`CircuitPlan`.  Solver
requests and results belong to ``scnsim.runtime`` instead.

All call bodies fail intentionally.  The classes and signatures are the
reviewable V1 UX scaffold, not a permissive partial circuit compiler.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal

from ._scaffold import unavailable

if TYPE_CHECKING:
    from .results import CircuitDiagramResult
    from .specs import CircuitDiagramSpec


class PinRef:
    """Stable reference to one named electrical terminal of a component.

    A ``PinRef`` is used only while assigning component terminals to one Plan
    electric node or to the Plan's canonical ground.  It is not a
    port, node voltage, or solver result.
    """

    def __init__(self) -> None:
        unavailable("PinRef construction")


class ElectricNodeRef:
    """Plan-bound handle to one declared equipotential electric node.

    ``CircuitPlan.net()`` returns this authoring handle for both named Public
    nodes and anonymous Internal nodes.  A logical Port may attach only through
    this handle; it cannot create or infer a connection from a component pin.
    After Plan seal, the same handle is the preferred Runtime selector for a
    Public node.  Declared string IDs remain useful at serialization and team
    facade boundaries; anonymous Internal nodes are not selectable.
    """

    def __init__(self) -> None:
        unavailable("ElectricNodeRef construction")


class CoordinateRef:
    """Public handle to one observable coordinate inside a composite.

    A ``CoordinateRef`` may be retained or used by Direct quantity selectors,
    evaluation, and optimization.  It is not an electrical terminal and cannot
    be passed to ``CircuitPlan.net()`` or ``CircuitPlan.add_port()``.  A
    composite author exposes a separate ``PinRef`` when the same internal node
    must also be externally connected.
    """

    def __init__(self) -> None:
        unavailable("CoordinateRef construction")


class PortRef:
    """Plan-bound handle to one logical Port component.

    A Port owns its electric-node attachment, canonical node-to-reference
    voltage/current direction, role, reference impedance, load, and
    backend-lowering identity.  Direct lowers the Port through a node selector
    and reference admittance; HB lowers the same Port to the exact
    JosephsonCircuits ``P`` and ``R_port`` rows.  Case-specific drive current is
    not stored here.
    """

    def __init__(self) -> None:
        unavailable("PortRef construction")


class ParameterRef:
    """Stable public handle to one component parameter.

    Model authors expose these handles to optimization or ``ParameterSet``
    construction.  A composite's internal child parameters are not public
    unless its Library factory deliberately returns a ``ParameterRef`` for
    them.
    """

    def __init__(self) -> None:
        unavailable("ParameterRef construction")

    def show(self) -> object:
        """Inspect baseline, unit, mappings, support, fan-out, and identity."""

        unavailable("ParameterRef.show")


class InductiveBranchRef:
    """Stable oriented handle to an inductive branch used for mutual coupling."""

    def __init__(self) -> None:
        unavailable("InductiveBranchRef construction")


class ParameterSpec:
    """Library declaration of one public physical parameter schema.

    ``unit`` is the preferred authoring/display unit from ``scnsim.units``.
    Runtime dimensionality validation will remain authoritative; this class
    does not introduce a second dimensional type system or own optimization
    bounds.  Every exposed physical ``ParameterRef`` may be rebound through a
    ``ParameterSet``; an ``OptimizationSpec`` separately chooses active search
    variables and finite bounds.
    """

    def __init__(self, *, unit: object) -> None:
        unavailable("ParameterSpec construction")


class ParameterSet:
    """Immutable physical values bound to exact ``ParameterRef`` keys.

    Omit ``parameters=`` on a Run operation to use the sealed Plan baseline.
    Pass a ``ParameterSet`` to evaluate another candidate without mutating the
    Plan or inheriting a reduction lineage from an optimization result.
    """

    def __init__(
        self,
        values: Mapping[ParameterRef, object],
        *,
        allow_extrapolation: Sequence[ParameterRef] = (),
    ) -> None:
        unavailable("ParameterSet construction")

    @property
    def values(self) -> Mapping[ParameterRef, object]:
        """Ordered read-only mapping used to create an explicitly new request."""

        unavailable("ParameterSet.values")


class RLGC:
    """Immutable per-length matrices for one uniform N-conductor line.

    ``conductors`` fixes the public row/column order and excludes the declared
    reference conductor.  R/L/G/C are SCNSim-registry Quantity-valued ``N x N``
    matrices; ``N=1`` is the ordinary one-by-one case.  Each conductor voltage
    is relative to ``reference_conductor`` and positive series current flows
    from the line's ``head`` pin to its ``tail`` pin, matching extractor +z.
    Construct this value directly or use :func:`scnsim.load_q2d_rlgc` for a
    compatible AEDT Q2D raw CSV.  Line length and pi-section count belong to
    the component factory.
    """

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
        """Ordered non-reference conductor labels shared by every matrix."""

        unavailable("RLGC.conductors")

    @property
    def reference_conductor(self) -> str:
        """Declared shunt-reference label excluded from matrix rows/columns."""

        unavailable("RLGC.reference_conductor")

    @property
    def resistance_per_length(self) -> object:
        """Quantity-valued symmetric positive-semidefinite series matrix."""

        unavailable("RLGC.resistance_per_length")

    @property
    def inductance_per_length(self) -> object:
        """Quantity-valued symmetric positive-definite series matrix."""

        unavailable("RLGC.inductance_per_length")

    @property
    def conductance_per_length(self) -> object:
        """Quantity-valued reciprocal Maxwell shunt matrix."""

        unavailable("RLGC.conductance_per_length")

    @property
    def capacitance_per_length(self) -> object:
        """Quantity-valued reciprocal Maxwell shunt matrix."""

        unavailable("RLGC.capacitance_per_length")

    @property
    def extraction_frequency(self) -> object | None:
        """Frozen source extraction frequency, or ``None`` for manual data."""

        unavailable("RLGC.extraction_frequency")


class AffineMap:
    """Declarative calibrated mapping ``output = slope * input + intercept``.

    ``support`` is the closed evidence range of the input ``ParameterRef``.
    Mapping evaluation belongs to ParameterSet binding inside the future Julia
    candidate process; this object is not a Python callback or expression AST.
    """

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
    """One immutable component created by an exact ``Library`` factory.

    The instance carries its Library/version/schema identity.  Consumers use
    only deliberately exposed ``pin()``, ``coordinate()``, ``parameter()``,
    and inductive-branch handles; composite children remain inspection
    evidence.
    """

    def __init__(self) -> None:
        unavailable("ComponentInstance construction")

    def pin(self, name: str, *, conductor: str | None = None) -> PinRef:
        """Return one pin; N-trace line pins also require ``conductor``."""

        unavailable("ComponentInstance.pin")

    def parameter(self, name: str) -> ParameterRef:
        """Return a deliberately public parameter by its schema name."""

        unavailable("ComponentInstance.parameter")

    def inductive_branch(self, name: str) -> InductiveBranchRef:
        """Return a declared oriented inductive branch for mutual coupling."""

        unavailable("ComponentInstance.inductive_branch")

    def coordinate(self, name: str) -> CoordinateRef:
        """Return a deliberately exposed internal analysis coordinate."""

        unavailable("ComponentInstance.coordinate")


class CompositePlan:
    """Declarative private construction graph for one reusable composite.

    A custom Library factory declares public parameters with baselines, adds
    children, composes their pins into internal electric nodes, and explicitly
    exposes only supported pins and coordinates before ``build()``.  Hidden
    children, nodes, and parameters remain implementation evidence.  This
    scaffold declares that authoring boundary but implements none of it.
    """

    def __init__(self, *, id: str, library: Library) -> None:
        unavailable("CompositePlan construction")

    def parameter(
        self,
        *,
        id: str,
        baseline: object,
        spec: ParameterSpec,
    ) -> ParameterRef:
        """Declare one public parameter and its sealed physical baseline."""

        unavailable("CompositePlan.parameter")

    def add(self, component: ComponentInstance) -> ComponentInstance:
        """Add one child component to this composite construction graph."""

        unavailable("CompositePlan.add")

    def net(self, *pins: PinRef, id: str | None = None) -> ElectricNodeRef:
        """Create one complete internal equipotential electric node."""

        unavailable("CompositePlan.net")

    def ground(self, *pins: PinRef) -> None:
        """Attach one nonempty local pin group to the parent Plan ground."""

        unavailable("CompositePlan.ground")

    def expose_pin(self, *, id: str, at: ElectricNodeRef) -> PinRef:
        """Expose an internal node as one public external wiring terminal."""

        unavailable("CompositePlan.expose_pin")

    def expose_coordinate(
        self,
        *,
        id: str,
        at: ElectricNodeRef,
    ) -> CoordinateRef:
        """Expose an internal node for retain/evaluate/optimize, not wiring."""

        unavailable("CompositePlan.expose_coordinate")

    def build(self) -> ComponentInstance:
        """Seal the declaration as one exact-identity composite instance."""

        unavailable("CompositePlan.build")


class Library:
    """Immutable catalog that creates exact-identity component instances.

    SCNSim exports its built-in singleton as ``scnsim.components``.  Each
    snake-case factory returns an immutable :class:`ComponentInstance` that
    retains catalog, source, and factory provenance.  A custom Python package
    may export another named ``Library`` object, but there is no mutable
    Notebook-global registry or string discovery mechanism.
    """

    def __init__(self) -> None:
        unavailable("Library construction")

    def resistor(self, *, id: str, resistance: object) -> ComponentInstance:
        """Declare a resistor oriented from ``terminal_1`` to ``terminal_2``."""

        unavailable("Library.resistor")

    def capacitor(self, *, id: str, capacitance: object) -> ComponentInstance:
        """Declare a capacitor oriented from ``terminal_1`` to ``terminal_2``."""

        unavailable("Library.capacitor")

    def inductor(self, *, id: str, inductance: object) -> ComponentInstance:
        """Declare an inductor from ``terminal_1`` to ``terminal_2``."""

        unavailable("Library.inductor")

    def josephson_junction(
        self,
        *,
        id: str,
        josephson_inductance: object,
        junction_capacitance: object,
    ) -> ComponentInstance:
        """Declare an ideal Josephson element plus optional parallel ``Cj``.

        Both rows are oriented from ``terminal_1`` to ``terminal_2``.
        The executable API will default ``junction_capacitance`` to exact
        ``0 * u.F``.  The scaffold keeps it explicit because the shared unit
        registry needed to create that physical default does not exist yet.
        """

        unavailable("Library.josephson_junction")

    def transmission_line(
        self,
        *,
        id: str,
        length: object,
        rlgc: RLGC,
        n_sections: int,
    ) -> ComponentInstance:
        """Declare a uniform N-conductor RLGC pi ladder from one typed input."""

        unavailable("Library.transmission_line")

    def interdigitated_capacitor(
        self,
        *,
        id: str,
        terminal_1_to_reference_capacitance: object,
        terminal_2_to_reference_capacitance: object,
        terminal_mutual_capacitance: object,
    ) -> ComponentInstance:
        """Declare the complete C1G/C2G/C12 two-terminal lumped model."""

        unavailable("Library.interdigitated_capacitor")

    def symmetric_squid(
        self,
        *,
        id: str,
        josephson_inductance: object,
        junction_capacitance: object,
        loop_inductance: object,
    ) -> ComponentInstance:
        """Declare a finite-loop SQUID from ``terminal_1`` to ``terminal_2``."""

        unavailable("Library.symmetric_squid")

    def grounded_parallel_linear_lc_resonator(
        self,
        *,
        id: str,
        capacitance: object,
        inductance: object,
    ) -> ComponentInstance:
        """Declare a grounded parallel-C/linear-L resonator with pin ``terminal``."""

        unavailable("Library.grounded_parallel_linear_lc_resonator")

    def floating_parallel_linear_lc_resonator(
        self,
        *,
        id: str,
        terminal_1_to_reference_capacitance: object,
        terminal_2_to_reference_capacitance: object,
        terminal_mutual_capacitance: object,
        inductance: object,
    ) -> ComponentInstance:
        """Declare a floating linear resonator from ``terminal_1`` to ``terminal_2``."""

        unavailable("Library.floating_parallel_linear_lc_resonator")

    def grounded_parallel_single_junction_resonator(
        self,
        *,
        id: str,
        capacitance: object,
        josephson_inductance: object,
        junction_capacitance: object,
    ) -> ComponentInstance:
        """Declare a grounded C/JJ resonator with pin ``terminal`` and Cj."""

        unavailable("Library.grounded_parallel_single_junction_resonator")

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
        """Declare a floating C/JJ resonator from ``terminal_1`` to ``terminal_2``."""

        unavailable("Library.floating_parallel_single_junction_resonator")

    def grounded_parallel_symmetric_squid_resonator(
        self,
        *,
        id: str,
        capacitance: object,
        josephson_inductance: object,
        junction_capacitance: object,
        loop_inductance: object,
    ) -> ComponentInstance:
        """Declare a grounded SQUID resonator with pin ``terminal`` and finite loop."""

        unavailable("Library.grounded_parallel_symmetric_squid_resonator")

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
        """Declare a floating SQUID resonator from ``terminal_1`` to ``terminal_2``."""

        unavailable("Library.floating_parallel_symmetric_squid_resonator")


class CircuitPlan:
    """The single physical authority for one reusable circuit model.

    A model developer adds exact Library components, assigns every electrical
    pin to one Public or Internal electric node or the canonical Plan ground,
    declares logical Port components on those nodes, and records mutual
    couplings here.  The ground exists at construction and lowers to backend
    node ``"0"``; callers never declare a second reference.  ``CircuitPlan``
    owns no solver, reduction, optimization, workspace, or result state.  A
    :class:`CircuitRun` later seals the completed Plan for execution.
    """

    def __init__(self, *, id: str) -> None:
        unavailable("CircuitPlan construction")

    def add(self, component: ComponentInstance) -> ComponentInstance:
        """Add one Library-created component and return its Plan-bound handle."""

        unavailable("CircuitPlan.add")

    def net(self, *pins: PinRef, id: str | None = None) -> ElectricNodeRef:
        """Create one complete electric node and return its Plan-bound handle.

        ``id`` creates a Public node coordinate.  Omitting it creates an
        anonymous Internal node with deterministic endpoint-derived identity;
        attaching a Port may later promote that node under the Port ID.  This
        call never incrementally merges a previously declared node, and the
        reserved ID ``"ground"`` is invalid here.
        """

        unavailable("CircuitPlan.net")

    def ground(self, *pins: PinRef) -> None:
        """Attach one nonempty local pin group to the canonical Plan ground.

        Calls may repeat for separate authoring blocks.  Every call targets the
        same physical reference; its endpoint group is retained only so an
        authoring schematic can draw one local ground bus and glyph.
        """

        unavailable("CircuitPlan.ground")

    def add_port(
        self,
        *,
        id: str,
        at: ElectricNodeRef,
        role: Literal["terminated", "nonloading_probe"],
        reference_impedance: object,
    ) -> PortRef:
        """Add one logical Port component to an existing electric node.

        The Port owns the external boundary and load identity.  It does not
        create topology or store a request-specific DC/AC drive.  An anonymous
        node is promoted to a Public coordinate using this Port ID; a named
        node keeps its existing coordinate identity.
        """

        unavailable("CircuitPlan.add_port")

    def render_schematic(
        self,
        spec: CircuitDiagramSpec | None = None,
    ) -> CircuitDiagramResult:
        """Materialize an authoring or compiler-expanded schematic.

        Rendering validates a complete Plan but never seals or mutates it,
        solves an analysis, or writes a workspace.  The returned diagram is
        derived inspection evidence; it cannot become topology authority.
        """

        unavailable("CircuitPlan.render_schematic")

    def couple_inductive(
        self,
        *,
        id: str,
        inductor_a: InductiveBranchRef,
        inductor_b: InductiveBranchRef,
        coupling_coefficient: object,
    ) -> None:
        """Declare signed ``k`` between two oriented inductive branches."""

        unavailable("CircuitPlan.couple_inductive")


components = object.__new__(Library)
"""The immutable built-in SCNSim component catalog."""
