"""Immutable explicit diagram declarations and their lossless capture binding.

Public refs address captured drawing inventory, never live electrical handles.
Preparation/realization are diagram-owned consumers; this module owns records,
layout serialization, and the Plan facade without any automatic geometry path.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
import json
from types import MappingProxyType
from threading import RLock

from ..canonical import canonical_json_bytes, float64_from_hex, float64_hex, sha256_hex
from ..errors import SCNSimValidationError


def _fail(message: str, **evidence: object) -> SCNSimValidationError:
    return SCNSimValidationError(message, stage="schematic_layout", evidence=evidence)


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _point(value: object) -> tuple[float, float]:
    x, y = value
    return (float(x), float(y))


def _bounds(value: object) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = value
    result = (float(xmin), float(ymin), float(xmax), float(ymax))
    if result[0] > result[2] or result[1] > result[3]:
        raise ValueError("scope frame bounds must be ordered")
    return result


@dataclass(frozen=True, slots=True, init=False)
class DiagramRef:
    """Opaque captured drawing target; inventory is its construction authority."""
    preparation_sha256: str
    key: str
    kind: str
    scope_path: tuple[str, ...]

    def __init__(self) -> None:
        raise TypeError("DiagramRef is supplied by prepare_schematic()")

    @classmethod
    def _create(cls, preparation_sha256: str, key: str, kind: str, scope_path: tuple[str, ...]) -> DiagramRef:
        result = object.__new__(cls)
        for name, value in (("preparation_sha256", preparation_sha256), ("key", key), ("kind", kind), ("scope_path", tuple(scope_path))):
            object.__setattr__(result, name, value)
        return result


@dataclass(frozen=True, slots=True)
class DiagramPose:
    origin: tuple[float, float]
    rotation: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", _point(self.origin))
        if not isinstance(self.rotation, int) or isinstance(self.rotation, bool) or self.rotation not in (0, 90, 180, 270):
            raise ValueError("diagram pose rotation must be 0, 90, 180 or 270")


@dataclass(frozen=True, slots=True)
class DiagramEndpoint:
    target: DiagramRef | str
    contact: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.target, DiagramRef):
            if self.contact is not None:
                raise TypeError("captured contact endpoints have no junction arm")
        elif isinstance(self.target, str):
            from ..specs import DiagramSide
            if not self.target or self.contact not in tuple(side.value for side in DiagramSide):
                raise ValueError("junction endpoints require their local id and cardinal arm")
        else:
            raise TypeError("route endpoint requires a DiagramRef or local junction id")


@dataclass(frozen=True, slots=True)
class DiagramRoute:
    id: str
    start: DiagramEndpoint
    end: DiagramEndpoint
    waypoints: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.start, DiagramEndpoint) or not isinstance(self.end, DiagramEndpoint):
            raise TypeError("route endpoints must be DiagramEndpoint")
        object.__setattr__(self, "waypoints", tuple(_point(p) for p in self.waypoints))


@dataclass(frozen=True, slots=True)
class DiagramJunction:
    id: str
    center: tuple[float, float]
    arms: tuple[object, ...]

    def __post_init__(self) -> None:
        from ..specs import DiagramSide
        object.__setattr__(self, "center", _point(self.center))
        arms = tuple(self.arms)
        if any(not isinstance(side, DiagramSide) for side in arms):
            raise TypeError("junction arms must be DiagramSide")
        object.__setattr__(self, "arms", arms)


@dataclass(frozen=True, slots=True)
class DiagramJump:
    route_id: str
    at: tuple[float, float]
    over_route_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "at", _point(self.at))


@dataclass(frozen=True, slots=True)
class DiagramCaption:
    target: DiagramRef
    at: tuple[float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "at", _point(self.at))


@dataclass(frozen=True, slots=True)
class DiagramLeader:
    target: DiagramRef
    contact: DiagramRef
    knee: tuple[float, float]
    end: tuple[float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "knee", _point(self.knee))
        object.__setattr__(self, "end", _point(self.end))


@dataclass(frozen=True, slots=True)
class DiagramCoupling:
    """Explicit nonconductive guide for one captured mutual-coupling target."""
    target: DiagramRef
    waypoints: tuple[tuple[float, float], ...]
    at: tuple[float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "waypoints", tuple(_point(p) for p in self.waypoints))
        object.__setattr__(self, "at", _point(self.at))


@dataclass(frozen=True, slots=True)
class SchematicScopeLayout:
    scope: DiagramRef
    frame: tuple[float, float, float, float]
    poses: Mapping[DiagramRef, DiagramPose]
    boundaries: Mapping[DiagramRef, tuple[float, float]]
    ports: Mapping[DiagramRef, DiagramPose]
    grounds: Mapping[DiagramRef, DiagramPose]
    junctions: tuple[DiagramJunction, ...]
    routes: tuple[DiagramRoute, ...]
    jumps: tuple[DiagramJump, ...]
    captions: tuple[DiagramCaption, ...]
    leaders: tuple[DiagramLeader, ...]
    couplings: tuple[DiagramCoupling, ...]
    children: Mapping[DiagramRef, SchematicScopeLayout]

    def __post_init__(self) -> None:
        if not isinstance(self.scope, DiagramRef):
            raise TypeError("scope layout requires a captured DiagramRef")
        object.__setattr__(self, "frame", _bounds(self.frame))
        for name in ("poses", "ports", "grounds", "children"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")
            object.__setattr__(self, name, _freeze(value))
        object.__setattr__(self, "boundaries", MappingProxyType({ref: _point(p) for ref, p in self.boundaries.items()}))
        for name in ("junctions", "routes", "jumps", "captions", "leaders", "couplings"):
            object.__setattr__(self, name, tuple(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class ScopeInventory:
    ref: DiagramRef
    parent: DiagramRef | None
    physical_occurrences: tuple[object, ...]
    child_scopes: tuple[DiagramRef, ...]
    contacts: tuple[object, ...]
    ports: tuple[object, ...]
    grounds: tuple[object, ...]
    captions: tuple[object, ...]
    analysis_labels: tuple[object, ...]
    structures: tuple[object, ...]
    required_contact_sets: tuple[object, ...]

    def __post_init__(self) -> None:
        for descriptor in fields(self):
            object.__setattr__(self, descriptor.name, _freeze(getattr(self, descriptor.name)))


@dataclass(frozen=True, slots=True)
class MeasuredFragment:
    """Captured native body and upright text, with no live lowering handles."""
    target: DiagramRef
    occurrence_identity: object
    orientation: int
    body_bounds: object
    occupied_bounds: object
    text_bounds: object
    contacts: Mapping[DiagramRef, object]
    obstacles: tuple[object, ...]
    terminal_access: tuple[object, ...]
    markers: Mapping[str, object]
    scene: object

    def __post_init__(self) -> None:
        for descriptor in fields(self):
            object.__setattr__(self, descriptor.name, _freeze(getattr(self, descriptor.name)))


@dataclass(frozen=True, slots=True)
class GeometryRealization:
    scene: object
    contacts: Mapping[DiagramRef, object]
    obstacles: tuple[object, ...]
    terminal_access: tuple[object, ...]
    bounds: object

    def __post_init__(self) -> None:
        for descriptor in fields(self):
            object.__setattr__(self, descriptor.name, _freeze(getattr(self, descriptor.name)))


@dataclass(frozen=True, slots=True)
class ScopeMeasurement:
    scope: DiagramRef
    preparation_sha256: str
    scope_layout_sha256: str
    bounds: object
    contacts: Mapping[DiagramRef, object]
    obstacles: tuple[object, ...]
    terminal_access: tuple[object, ...]
    markers: Mapping[tuple[DiagramRef, str], object]
    fragment: object

    def __post_init__(self) -> None:
        for descriptor in fields(self):
            object.__setattr__(self, descriptor.name, _freeze(getattr(self, descriptor.name)))


def _refs(value: object):
    if isinstance(value, DiagramRef):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _refs(key)
            yield from _refs(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _refs(item)
    elif is_dataclass(value):
        for descriptor in fields(value):
            if descriptor.name not in {"scene", "fragment"}:
                yield from _refs(getattr(value, descriptor.name))


@dataclass(frozen=True, slots=True, init=False)
class SchematicPreparation:
    identity: Mapping[str, object]
    root: DiagramRef
    scopes: tuple[ScopeInventory, ...]
    native: Mapping[tuple[DiagramRef, int], MeasuredFragment]
    _point: object = field(repr=False, compare=False)
    _spec: object = field(repr=False, compare=False)
    _measurements: dict[tuple[str, str, str], ScopeMeasurement] = field(repr=False, compare=False)
    _measurement_lock: object = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError("SchematicPreparation is supplied by CircuitPlan.prepare_schematic()")

    @classmethod
    def _from_capture(cls, *, identity: Mapping[str, object], root: DiagramRef, scopes: tuple[ScopeInventory, ...], native: Mapping[tuple[DiagramRef, int], MeasuredFragment], point: object, spec: object) -> SchematicPreparation:
        result = object.__new__(cls)
        for name, value in (("identity", _freeze(identity)), ("root", root), ("scopes", tuple(scopes)), ("native", _freeze(native)), ("_point", point), ("_spec", spec)):
            object.__setattr__(result, name, value)
        # Operation-local exact child memoization is not serialized identity.
        object.__setattr__(result, "_measurements", {})
        object.__setattr__(result, "_measurement_lock", RLock())
        return result

    @property
    def refs(self) -> tuple[DiagramRef, ...]:
        return tuple(dict.fromkeys(_refs((self.root, self.scopes, self.native))))

    def measure_scope(self, scope_ref: DiagramRef, scope_layout: SchematicScopeLayout) -> ScopeMeasurement:
        from .diagram.preparation import measure_scope
        return measure_scope(self, scope_ref, scope_layout)

    def to_json(self) -> str:
        return canonical_json_bytes({"identity": dict(self.identity), "root": _encode(self.root), "scopes": _encode(self.scopes), "native": _encode(self.native)}).decode("utf-8")


_LAYOUT_TYPES = {cls.__name__: cls for cls in (DiagramPose, DiagramEndpoint, DiagramRoute, DiagramJunction, DiagramJump, DiagramCaption, DiagramLeader, DiagramCoupling, SchematicScopeLayout)}


def _encode(value: object) -> object:
    """Encode geometry without decimal float loss or serialized live objects."""
    if isinstance(value, DiagramRef):
        return {"type": "DiagramRef", "key": value.key, "kind": value.kind, "scope_path": list(value.scope_path), "preparation_sha256": value.preparation_sha256}
    if isinstance(value, Enum):
        return {"type": type(value).__name__, "value": value.value}
    if isinstance(value, float):
        return {"f64": float64_hex(value)}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Mapping):
        return {"type": "mapping", "items": [[_encode(key), _encode(item)] for key, item in value.items()]}
    if isinstance(value, (tuple, list)):
        return [_encode(item) for item in value]
    if is_dataclass(value):
        return {"type": type(value).__name__, "fields": {descriptor.name: _encode(getattr(value, descriptor.name)) for descriptor in fields(value) if descriptor.name not in {"scene", "fragment"}}}
    raise TypeError(f"unsupported schematic JSON value: {type(value).__name__}")


def _decode(value: object, refs: Mapping[str, DiagramRef]) -> object:
    if isinstance(value, list):
        return tuple(_decode(item, refs) for item in value)
    if not isinstance(value, dict):
        return value
    if set(value) == {"f64"}:
        return float64_from_hex(value["f64"])
    kind = value["type"]
    if kind == "DiagramRef":
        ref = refs[value["key"]]
        if _encode(ref) != value:
            raise _fail("layout ref does not match this preparation", ref=value["key"])
        return ref
    if kind == "DiagramSide":
        from ..specs import DiagramSide
        return DiagramSide(value["value"])
    if kind == "mapping":
        result = {}
        for key, item in value["items"]:
            decoded_key = _decode(key, refs)
            if decoded_key in result:
                raise _fail("layout JSON repeats a mapping target")
            result[decoded_key] = _decode(item, refs)
        return result
    cls = _LAYOUT_TYPES[kind]
    return cls(**{name: _decode(item, refs) for name, item in value["fields"].items()})


def _scope_layout_sha256(scope_layout: SchematicScopeLayout) -> str:
    """Share exact child-layout identity between memoization and realization."""
    return sha256_hex({
        "schema": "scnsim.schematic_scope_layout",
        "schema_version": 1,
        "layout": _encode(scope_layout),
    })


@dataclass(frozen=True, slots=True)
class SchematicLayout:
    preparation: SchematicPreparation
    root: SchematicScopeLayout

    def __post_init__(self) -> None:
        if not isinstance(self.preparation, SchematicPreparation) or not isinstance(self.root, SchematicScopeLayout):
            raise TypeError("SchematicLayout requires preparation and a root scope layout")
        known = set(self.preparation.refs)
        for ref in _refs(self.root):
            if ref not in known:
                raise _fail("layout contains a foreign captured ref", ref=ref.key)
        if self.root.scope != self.preparation.root:
            raise _fail("layout root does not match preparation root")

    def to_json(self) -> str:
        return canonical_json_bytes({"schema": "scnsim.schematic_layout", "schema_version": 1, "identity": dict(self.preparation.identity), "root": _encode(self.root)}).decode("utf-8")

    @classmethod
    def from_json(cls, preparation: SchematicPreparation, text: str) -> SchematicLayout:
        document = json.loads(text)
        if document["schema"] != "scnsim.schematic_layout" or document["schema_version"] != 1:
            raise _fail("unsupported schematic layout JSON schema")
        if document["identity"] != dict(preparation.identity):
            raise _fail("layout JSON binding differs from preparation")
        refs = {ref.key: ref for ref in preparation.refs}
        return cls(preparation, _decode(document["root"], refs))


def _prepare_schematic(plan: object, spec: object, *, parameters: object | None = None) -> SchematicPreparation:
    from .diagram.preparation import capture_schematic
    return capture_schematic(plan, spec, parameters)


def _render_schematic(plan: object, spec: object, *, parameters: object | None = None) -> object:
    from .diagram.pipeline import render_schematic
    if spec.representation == "compiled":
        snapshot = plan._capture_authoring_snapshot()
        point = plan._resolve_parameter_point(parameters, snapshot=snapshot)
        return render_schematic(point, spec)
    layout = spec.layout
    if layout is None:
        raise _fail("authoring rendering requires a complete SchematicLayout; call prepare_schematic() first")
    from .diagram.spec import CircuitDiagramSpec
    capture_spec = CircuitDiagramSpec(theme=spec.theme, show_parameter_values=spec.show_parameter_values, show_provenance=spec.show_provenance)
    preparation = _prepare_schematic(plan, capture_spec, parameters=parameters)
    if dict(preparation.identity) != dict(layout.preparation.identity):
        raise _fail("layout binding differs from the current Plan, point or presentation")
    return render_schematic(preparation._point, spec, layout=layout)


__all__ = ["DiagramRef", "DiagramPose", "DiagramEndpoint", "DiagramRoute", "DiagramJunction", "DiagramJump", "DiagramCaption", "DiagramLeader", "DiagramCoupling", "SchematicScopeLayout", "SchematicLayout", "SchematicPreparation", "ScopeInventory", "MeasuredFragment", "ScopeMeasurement", "GeometryRealization"]
for _name in __all__:
    globals()[_name].__module__ = "scnsim.schematic"
