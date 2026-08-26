"""Public circuit-model authoring declarations.

This module is for the person who owns a reusable circuit model.  It declares
components, named electrical topology, ports, parameters, and inductive
couplings in one :class:`CircuitPlan`.  Solver requests and results belong to
``scnsim.runtime`` instead.

All call bodies fail intentionally.  The classes and signatures are the
reviewable V1 UX scaffold, not a permissive partial circuit compiler.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from ._scaffold import unavailable


class PinRef:
    """Stable reference to one named electrical terminal of a component.

    A ``PinRef`` is used only while assigning component terminals to a named
    Plan net or to the Plan's single node-flux reference.  It is not a port,
    node voltage, or solver result.
    """

    def __init__(self) -> None:
        unavailable("PinRef construction")


class ParameterRef:
    """Stable public handle to one component parameter.

    Model authors expose these handles to optimization or ``ParameterSet``
    construction.  A composite's internal child parameters are not public
    unless its Library factory deliberately returns a ``ParameterRef`` for
    them.
    """

    def __init__(self) -> None:
        unavailable("ParameterRef construction")


class InductiveBranchRef:
    """Stable oriented handle to an inductive branch used for mutual coupling."""

    def __init__(self) -> None:
        unavailable("InductiveBranchRef construction")


class ParameterSpec:
    """Library declaration of a physical parameter's unit and mutability.

    ``unit`` is the preferred authoring/display unit from ``scnsim.units``.
    Runtime dimensionality validation will remain authoritative; this class
    does not introduce a second dimensional type system.
    """

    def __init__(self, *, unit: object, variable_capable: bool = False) -> None:
        unavailable("ParameterSpec construction")


class ParameterSet:
    """Immutable physical values bound to exact ``ParameterRef`` keys.

    Omit ``parameters=`` on a Run operation to use the sealed Plan baseline.
    Pass a ``ParameterSet`` to evaluate another candidate without mutating the
    Plan or inheriting a reduction lineage from an optimization result.
    """

    def __init__(self, values: Mapping[ParameterRef, object]) -> None:
        unavailable("ParameterSet construction")


class ComponentInstance:
    """One immutable component created by an exact ``Library`` factory.

    The instance carries its Library/version/schema identity.  Consumers use
    only ``pin()``, ``parameter()``, and declared inductive branch handles;
    composite children remain inspection evidence.
    """

    def __init__(self) -> None:
        unavailable("ComponentInstance construction")

    def pin(self, name: str) -> PinRef:
        """Return a declared electrical pin by its public name."""

        unavailable("ComponentInstance.pin")

    def parameter(self, name: str) -> ParameterRef:
        """Return a deliberately public parameter by its schema name."""

        unavailable("ComponentInstance.parameter")

    def inductive_branch(self, name: str) -> InductiveBranchRef:
        """Return a declared oriented inductive branch for mutual coupling."""

        unavailable("ComponentInstance.inductive_branch")


class Library:
    """Immutable catalog that creates exact-identity component instances.

    SCNSim exports one built-in object as ``scnsim.library``.  A custom Python
    package may export another immutable ``Library`` object, but there is no
    mutable Notebook-global registry or string discovery mechanism.
    """

    def __init__(self) -> None:
        unavailable("Library construction")

    def resistor(self, *, id: str, resistance: object) -> ComponentInstance:
        """Declare a two-terminal linear resistor with pins ``a`` and ``b``."""

        unavailable("Library.resistor")

    def capacitor(self, *, id: str, capacitance: object) -> ComponentInstance:
        """Declare a two-terminal linear capacitor with pins ``a`` and ``b``."""

        unavailable("Library.capacitor")

    def inductor(self, *, id: str, inductance: object) -> ComponentInstance:
        """Declare a linear inductor with pins ``a``/``b`` and branch ``self``."""

        unavailable("Library.inductor")

    def josephson_junction(
        self,
        *,
        id: str,
        josephson_inductance: object,
        junction_capacitance: object,
    ) -> ComponentInstance:
        """Declare an ideal Josephson element plus optional parallel ``Cj``.

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
        resistance_per_length: object,
        inductance_per_length: object,
        conductance_per_length: object,
        capacitance_per_length: object,
        n_sections: int,
    ) -> ComponentInstance:
        """Declare a scalar RLGC pi ladder with fixed positive section count."""

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
        """Declare the explicit two-junction finite-loop SQUID component."""

        unavailable("Library.symmetric_squid")

    def grounded_parallel_linear_lc_resonator(
        self,
        *,
        id: str,
        subsystem_capacitance: object,
        inductance: object,
    ) -> ComponentInstance:
        """Declare a grounded parallel-C/linear-L resonator with pin ``signal``."""

        unavailable("Library.grounded_parallel_linear_lc_resonator")

    def floating_parallel_linear_lc_resonator(
        self,
        *,
        id: str,
        terminal_a_to_reference_capacitance: object,
        terminal_b_to_reference_capacitance: object,
        terminal_mutual_capacitance: object,
        inductance: object,
    ) -> ComponentInstance:
        """Declare a floating three-capacitance/linear-L resonator."""

        unavailable("Library.floating_parallel_linear_lc_resonator")

    def grounded_parallel_single_junction_resonator(
        self,
        *,
        id: str,
        subsystem_capacitance: object,
        josephson_inductance: object,
        junction_capacitance: object,
    ) -> ComponentInstance:
        """Declare a grounded resonator with one nonlinear Josephson branch."""

        unavailable("Library.grounded_parallel_single_junction_resonator")

    def floating_parallel_single_junction_resonator(
        self,
        *,
        id: str,
        terminal_a_to_reference_capacitance: object,
        terminal_b_to_reference_capacitance: object,
        terminal_mutual_capacitance: object,
        josephson_inductance: object,
        junction_capacitance: object,
    ) -> ComponentInstance:
        """Declare a floating resonator with one nonlinear Josephson branch."""

        unavailable("Library.floating_parallel_single_junction_resonator")

    def grounded_parallel_symmetric_squid_resonator(
        self,
        *,
        id: str,
        subsystem_capacitance: object,
        josephson_inductance: object,
        junction_capacitance: object,
        loop_inductance: object,
    ) -> ComponentInstance:
        """Declare a grounded resonator with an explicit finite-loop SQUID."""

        unavailable("Library.grounded_parallel_symmetric_squid_resonator")

    def floating_parallel_symmetric_squid_resonator(
        self,
        *,
        id: str,
        terminal_a_to_reference_capacitance: object,
        terminal_b_to_reference_capacitance: object,
        terminal_mutual_capacitance: object,
        josephson_inductance: object,
        junction_capacitance: object,
        loop_inductance: object,
    ) -> ComponentInstance:
        """Declare a floating resonator with an explicit finite-loop SQUID."""

        unavailable("Library.floating_parallel_symmetric_squid_resonator")


class CircuitPlan:
    """The single physical authority for one reusable circuit model.

    A model developer adds exact Library components, assigns every electrical
    pin to a named net or the one reference, declares external ports, and
    records mutual couplings here.  ``CircuitPlan`` owns no solver, reduction,
    optimization, workspace, or result state.  A :class:`CircuitRun` later
    seals the completed Plan for execution.
    """

    def __init__(self, *, id: str) -> None:
        unavailable("CircuitPlan construction")

    def add(self, component: ComponentInstance) -> ComponentInstance:
        """Add one Library-created component and return its Plan-bound handle."""

        unavailable("CircuitPlan.add")

    def net(self, id: str, *pins: PinRef) -> None:
        """Create one named electrical node and assign all supplied pins once."""

        unavailable("CircuitPlan.net")

    def reference(self, id: str, *pins: PinRef) -> None:
        """Declare the Plan's one node-flux reference and its attached pins."""

        unavailable("CircuitPlan.reference")

    def add_port(
        self,
        *,
        id: str,
        at: str,
        role: Literal["terminated", "nonloading_probe"],
        reference_impedance: object,
    ) -> None:
        """Attach an explicit external boundary to an existing named net."""

        unavailable("CircuitPlan.add_port")

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


library = object.__new__(Library)
"""The immutable built-in SCNSim component catalog."""
