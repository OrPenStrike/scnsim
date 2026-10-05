"""Immutable catalog types, one builtin singleton, and physical factories.

Provenance is resolved lazily at component creation. Factory-owned Composite
assembly is imported only when a factory constructs a body."""

from __future__ import annotations

from functools import wraps
import inspect
import numpy as np

from ..units import registry
from .components import ComponentInstance
from .factory_context import _component_creation_token, _factory_context
from .parameters import ParameterRef
from .physical_values import ElectricalResolution, RLGC


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
    from .provenance import catalog_source_record

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
        self, *, id: str, length: object, rlgc: object,
        n_sections: int | None = None,
        discretization: ElectricalResolution | None = None,
    ) -> ComponentInstance:
        if (n_sections is None) == (discretization is None):
            raise ValueError("exactly one of n_sections and discretization is required")
        if discretization is None:
            if isinstance(n_sections, bool) or not isinstance(n_sections, int) or n_sections < 1:
                raise ValueError("n_sections must be positive integer")
            metadata = {"n_sections": n_sections}
        else:
            if not isinstance(discretization, ElectricalResolution):
                raise TypeError("discretization must be ElectricalResolution")
            metadata = {"discretization": discretization._record()}
        b = rlgc.baseline if isinstance(rlgc, ParameterRef) else rlgc
        if not isinstance(b, RLGC):
            raise TypeError("rlgc must be RLGC or ParameterRef")
        if discretization is not None and (
            np.any(b.resistance_per_length.magnitude != 0)
            or np.any(b.conductance_per_length.magnitude != 0)
        ):
            raise ValueError("ElectricalResolution requires explicitly zero R and G")
        return ComponentInstance._create(
            id=id,
            factory="transmission_line",
            pins=tuple(f"{e}.{c}" for e in ("head", "tail") for c in b.conductors),
            fields={"length": (length, "meter", True), "rlgc": (rlgc, "rlgc", False)},
            kind="complete_line",
            catalog_source=self._src(),
            metadata=metadata,
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
        from .assembly import CompositePlan

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
        from .assembly import CompositePlan

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
        from .assembly import CompositePlan

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



# Public catalog identity remains stable across implementation relocation.
type.__setattr__(Library, "__module__", "scnsim.authoring")
type.__setattr__(_BuiltinComponents, "__module__", "scnsim.authoring")
components = _BuiltinComponents()
