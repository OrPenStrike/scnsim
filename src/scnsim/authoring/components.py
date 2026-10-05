"""Catalog-created bodies and retained physical field bindings.

Factories own construction; assembly owns occurrence overlays. Literal values
and derived-field coefficients retain their existing capture rules."""

from __future__ import annotations

from types import MappingProxyType

from ..errors import SCNSimValidationError
from .factory_context import _component_creation_token, _two_terminal_use_token
from .handles import CoordinateRef, InductiveBranchRef, PinRef, TwoTerminalUse
from .parameters import ParameterRef
from .physical_values import (
    AffineMap, RLGC, RLGCParameterSpec, _checked_field_baseline,
    _retained_literal_field, identifier, quantity_record,
)
from .snapshot import freeze


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



# Preserve public class identity through the authoring facade.
ComponentInstance.__module__ = "scnsim.authoring"
