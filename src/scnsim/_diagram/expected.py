"""Independent expected manifests for certified circuit diagrams.

This module deliberately consumes only the immutable canonical Plan capture,
and, for a compiled projection, the compiler's sealed expanded-graph evidence.
It does not import the authoring semantic IR or any placement data.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, cast

from .. import units
from .._canonical import (
    canonical_json_bytes,
    float64_from_hex,
    quantity_envelope,
    quantity_from_envelope,
    sha256_hex,
)
from ..errors import SCNSimValidationError
from .._authoring_snapshot import ResolvedPlanPoint
from .snapshot import _CapturedPlan

_Representation = Literal["authoring", "compiled"]
_Path = tuple[str, ...]
_Endpoint = tuple[_Path, str]
_GROUND = b"ground"


def _fail(message: str, **evidence: object) -> SCNSimValidationError:
    return SCNSimValidationError(message, stage="schematic_audit", evidence=evidence)


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _fail("canonical diagram input needs an object", field=field)
    return cast(Mapping[str, object], value)


def _sequence(value: object, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise _fail("canonical diagram input needs an array", field=field)
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise _fail("canonical diagram input needs a nonempty string", field=field)
    return value


def _path(value: object, field: str = "component_path") -> _Path:
    result = tuple(_string(item, field) for item in _sequence(value, field))
    if not result:
        raise _fail("component paths cannot be empty", field=field)
    return result


def _endpoint(value: object) -> _Endpoint:
    record = _mapping(value, "endpoint")
    if set(record) != {"component_path", "pin_id"}:
        raise _fail("canonical endpoint fields are malformed")
    return _path(record["component_path"]), _string(record["pin_id"], "pin_id")


def _endpoint_record(endpoint: _Endpoint) -> dict[str, object]:
    return {"component_path": list(endpoint[0]), "pin_id": endpoint[1]}


def _branch_ref(value: object) -> tuple[_Path, str]:
    record = _mapping(value, "inductive_branch_ref")
    if set(record) != {"component_path", "branch_id"}:
        raise _fail("canonical inductive-branch reference fields are malformed")
    return _path(record["component_path"]), _string(record["branch_id"], "branch_id")


def _branch_ref_record(value: tuple[_Path, str]) -> dict[str, object]:
    return {"component_path": list(value[0]), "branch_id": value[1]}


def _token(kind: str, **fields: object) -> str:
    return canonical_json_bytes({"kind": kind, **fields}).decode("utf-8")


def _contact_token(kind: str, *, path: _Path = (), identity: str, terminal: str = "") -> str:
    return _token(
        "diagram_contact",
        contact_kind=kind,
        component_path=list(path),
        identity=identity,
        terminal=terminal,
    )


class _UnionFind:
    def __init__(self) -> None:
        self._parent: dict[bytes, bytes] = {_GROUND: _GROUND}

    def add(self, value: bytes) -> None:
        self._parent.setdefault(value, value)

    def find(self, value: bytes) -> bytes:
        self.add(value)
        parent = self._parent[value]
        if parent != value:
            self._parent[value] = self.find(parent)
        return self._parent[value]

    def union(self, *values: bytes) -> None:
        if not values:
            return
        roots = [self.find(value) for value in values]
        root = min(roots, key=lambda item: (item != _GROUND, item))
        for item in roots:
            self._parent[item] = root


def _endpoint_key(endpoint: _Endpoint) -> bytes:
    return canonical_json_bytes(_endpoint_record(endpoint))


def _net_key(endpoints: Iterable[_Endpoint], *, ground: bool) -> str:
    if ground:
        return "ground"
    records = [_endpoint_record(endpoint) for endpoint in sorted(set(endpoints))]
    if not records:
        raise _fail("a reconstructed expected net has no canonical endpoints")
    return "net-" + sha256_hex(
        {
            "schema": "scnsim.diagram_net_equivalence",
            "schema_version": 1,
            "endpoints": records,
        }
    )


def _quantity_is_zero(value: Mapping[str, object]) -> bool:
    token = value.get("si_value_f64")
    if not isinstance(token, str):
        raise _fail("expected scalar quantity evidence is malformed")
    return float64_from_hex(token) == 0.0


def _quantity_product(
    left: Mapping[str, object], right: Mapping[str, object], intercept: Mapping[str, object]
) -> Mapping[str, object]:
    try:
        left_quantity = quantity_from_envelope(left, registry=units.registry)
        right_quantity = quantity_from_envelope(right, registry=units.registry)
        offset = quantity_from_envelope(intercept, registry=units.registry)
        result = left_quantity * right_quantity + offset
        return cast(
            Mapping[str, object],
            quantity_envelope(result, si_unit=str(intercept["si_unit"]), registry=units.registry),
        )
    except Exception as error:
        raise _fail("unable to resolve one captured affine baseline") from error


@dataclass(frozen=True, slots=True)
class _Component:
    path: _Path
    parent_scope: _Path
    kind: str
    pin_order: tuple[str, ...]
    record: Mapping[str, object]

    @property
    def is_composite(self) -> bool:
        return self.kind == "composite"

    @property
    def is_line(self) -> bool:
        return self.kind == "transmission_line"

    @property
    def is_subsystem(self) -> bool:
        return self.is_composite or self.is_line


@dataclass(frozen=True, slots=True)
class _Branch:
    path: _Path
    owner: _Path
    role: str
    native_kind: str
    pins: tuple[str, str]
    nets: tuple[str, str]
    value: Mapping[str, object]
    reciprocal: bool
    omitted: bool = False

    @property
    def key(self) -> str:
        return _token(
            "authoring_branch",
            component_path=list(self.path),
            branch_role=self.role,
        )

    @property
    def unordered_nets(self) -> tuple[str, str]:
        return cast(tuple[str, str], tuple(sorted(self.nets)))

    def record(self) -> dict[str, object]:
        return {
            "branch_id": self.key,
            "component_path": list(self.path),
            "branch_role": self.role,
            "native_kind": self.native_kind,
            "owner_scope": list(self.owner),
            "terminals": [
                {
                    "pin_id": pin,
                    "net": net,
                    "contact": _contact_token(
                        "native_terminal",
                        path=self.path,
                        identity=self.role,
                        terminal=pin,
                    ),
                }
                for pin, net in zip(self.pins, self.nets, strict=True)
            ],
            "reciprocal_terminal_swap": self.reciprocal,
        }


@dataclass(frozen=True, slots=True)
class _Site:
    path: _Path
    pin_id: str
    net: str
    peer_kind: str
    peer_id: str

    @property
    def key(self) -> str:
        return _token(
            "boundary_site",
            component_path=list(self.path),
            pin_id=self.pin_id,
        )

    @property
    def peer_key(self) -> str:
        return _token("diagram_peer", peer_kind=self.peer_kind, peer_id=self.peer_id)

    def record(self) -> dict[str, object]:
        return {
            "site_id": self.key,
            "component_path": list(self.path),
            "pin_id": self.pin_id,
            "net": self.net,
            "peer": self.peer_key,
        }


@dataclass(frozen=True, slots=True)
class _ExpectedManifest:
    representation: _Representation
    electrical: Mapping[str, object]
    semantic: Mapping[str, object]
    expected_values: Mapping[str, object]
    verified: Mapping[str, object]
    compiled_graph_sha256: str | None
    expanded_graph_sha256: str | None


class _PlanModel:
    """Independent canonical-document expansion used only for expectations."""

    def __init__(self, capture: _CapturedPlan) -> None:
        if not isinstance(capture, _CapturedPlan):
            raise TypeError("expected manifest construction requires _CapturedPlan")
        document = _mapping(capture.document, "captured Plan")
        if document.get("schema") != "scnsim.plan" or document.get("schema_version") != 1:
            raise _fail("captured Plan schema is unsupported")
        if sha256_hex(capture.canonical_bytes) != capture.plan_sha256:
            raise _fail("captured Plan bytes disagree with its SHA-256")
        if canonical_json_bytes(document) != capture.canonical_bytes:
            raise _fail("captured Plan document is not its immutable canonical byte source")
        self.capture = capture
        self.document = document
        self.components: dict[_Path, _Component] = {}
        self._children: dict[_Path, list[_Path]] = defaultdict(list)
        self._bindings: dict[tuple[_Path, str], Mapping[str, object]] = {}
        self._resolved_values: dict[tuple[_Path, str], Mapping[str, object]] = {}
        for item in _sequence(document.get("components"), "components"):
            self._add_component(_mapping(item, "component"), ())
        self._uf = _UnionFind()
        self._bind_topology()
        self._endpoint_roots: dict[bytes, set[_Endpoint]] = defaultdict(set)
        for component in self.components.values():
            for pin in component.pin_order:
                endpoint = (component.path, pin)
                self._endpoint_roots[self._uf.find(_endpoint_key(endpoint))].add(endpoint)
        self._net_by_root = {
            root: _net_key(endpoints, ground=self._uf.find(root) == self._uf.find(_GROUND))
            for root, endpoints in self._endpoint_roots.items()
        }
        self._node_by_id: dict[str, Mapping[str, object]] = {}
        for raw in _sequence(document.get("nodes"), "nodes"):
            node = _mapping(raw, "node")
            node_id = _string(node.get("node_id"), "node_id")
            self._node_by_id[node_id] = node

    def _add_component(self, record: Mapping[str, object], parent_scope: _Path) -> None:
        path = _path(record.get("component_path"))
        if path in self.components:
            raise _fail("captured Plan repeats a component path", component_path=list(path))
        realization = _mapping(record.get("realization"), "realization")
        kind = _string(realization.get("kind"), "realization.kind")
        pins = tuple(_string(pin, "pin_order") for pin in _sequence(record.get("pin_order"), "pin_order"))
        if not pins or len(set(pins)) != len(pins):
            raise _fail("captured component pin order is malformed", component_path=list(path))
        component = _Component(path, parent_scope, kind, pins, record)
        self.components[path] = component
        self._children[parent_scope].append(path)
        for raw in _sequence(record.get("parameter_bindings"), "parameter_bindings"):
            item = _mapping(raw, "parameter_binding")
            parameter_id = _string(item.get("id"), "parameter_id")
            self._bindings[(path, parameter_id)] = _mapping(item.get("binding"), "binding")
        if kind == "composite":
            for child in _sequence(realization.get("children"), "composite children"):
                self._add_component(_mapping(child, "composite child"), path)

    def _bind_topology(self) -> None:
        for component in self.components.values():
            for pin in component.pin_order:
                self._uf.add(_endpoint_key((component.path, pin)))
        for raw in _sequence(self.document.get("nodes"), "nodes"):
            node = _mapping(raw, "node")
            self._union_endpoints(_sequence(node.get("endpoints"), "node endpoints"))
        for raw in _sequence(self.document.get("grounded_endpoints"), "grounded_endpoints"):
            self._uf.union(_GROUND, _endpoint_key(_endpoint(raw)))
        for component in self.components.values():
            if not component.is_composite:
                continue
            realization = _mapping(component.record["realization"], "composite realization")
            nodes: dict[str, tuple[_Endpoint, ...]] = {}
            for raw in _sequence(realization.get("private_nodes"), "private_nodes"):
                node = _mapping(raw, "private_node")
                node_id = _string(node.get("id"), "private_node.id")
                endpoints = tuple(_endpoint(item) for item in _sequence(node.get("endpoints"), "private node endpoints"))
                nodes[node_id] = endpoints
                self._uf.union(*(_endpoint_key(endpoint) for endpoint in endpoints))
            for raw in _sequence(realization.get("public_pin_map"), "public_pin_map"):
                item = _mapping(raw, "public_pin_map item")
                public_id = _string(item.get("public_id"), "public_id")
                private_id = _string(item.get("private_node_id"), "private_node_id")
                if private_id not in nodes:
                    raise _fail("Composite public pin targets no captured private node")
                self._uf.union(
                    _endpoint_key((component.path, public_id)),
                    *(_endpoint_key(endpoint) for endpoint in nodes[private_id]),
                )
            for raw in _sequence(realization.get("grounded_endpoints"), "grounded_endpoints"):
                self._uf.union(_GROUND, _endpoint_key(_endpoint(raw)))

    def _union_endpoints(self, values: Sequence[object]) -> None:
        endpoints = tuple(_endpoint(value) for value in values)
        if not endpoints:
            raise _fail("captured Plan node has no endpoints")
        self._uf.union(*(_endpoint_key(endpoint) for endpoint in endpoints))

    def net(self, endpoint: _Endpoint) -> str:
        root = self._uf.find(_endpoint_key(endpoint))
        try:
            return self._net_by_root[root]
        except KeyError:
            raise _fail(
                "expected electrical endpoint has no reconstructed canonical net",
                endpoint=_endpoint_record(endpoint),
            ) from None

    def parameter(self, path: _Path, parameter_id: str) -> Mapping[str, object]:
        key = (path, parameter_id)
        if key in self._resolved_values:
            return self._resolved_values[key]
        binding = self._bindings.get(key)
        if binding is None:
            raise _fail(
                "captured component has no expected parameter binding",
                component_path=list(path),
                parameter_id=parameter_id,
            )
        kind = binding.get("kind")
        if kind == "constant":
            result = _mapping(binding.get("value"), "constant value")
        elif kind in {"identity", "affine"}:
            source = _mapping(binding.get("input"), "binding input")
            source_path = _path(source.get("component_path"))
            source_id = _string(source.get("parameter_id"), "parameter_id")
            source_value = self.parameter(source_path, source_id)
            if kind == "identity":
                result = source_value
            else:
                result = _quantity_product(
                    source_value,
                    _mapping(binding.get("slope"), "affine slope"),
                    _mapping(binding.get("intercept"), "affine intercept"),
                )
        else:
            raise _fail("captured parameter binding kind is unsupported", kind=kind)
        self._resolved_values[key] = result
        return result

    def component_value(self, component: _Component, parameter_id: str) -> Mapping[str, object]:
        return self.parameter(component.path, parameter_id)

    def immediate_subsystems(self, scope: _Path) -> tuple[_Component, ...]:
        return tuple(
            self.components[path]
            for path in sorted(self._children.get(scope, ()))
            if self.components[path].is_subsystem
        )

    def owner_leaves(self, scope: _Path) -> tuple[_Component, ...]:
        return tuple(
            self.components[path]
            for path in sorted(self._children.get(scope, ()))
            if not self.components[path].is_subsystem
        )

    def node_net(self, node_id: str) -> str:
        node = self._node_by_id.get(node_id)
        if node is None:
            raise _fail("captured Port targets no Plan node", node_id=node_id)
        endpoints = _sequence(node.get("endpoints"), "node endpoints")
        return self.net(_endpoint(endpoints[0]))

    def public_nodes(self) -> tuple[dict[str, object], ...]:
        rows: list[dict[str, object]] = []
        for node_id in sorted(self._node_by_id):
            node = self._node_by_id[node_id]
            visibility = _string(node.get("visibility"), "node.visibility")
            if visibility == "internal":
                continue
            rows.append(
                {
                    "node_id": node_id,
                    "visibility": visibility,
                    "visible_mark": "filled_dot" if visibility == "public" else "port_circle",
                    "net": self.node_net(node_id),
                }
            )
        return tuple(rows)

    def resolve_branch_ref(self, reference: tuple[_Path, str]) -> tuple[_Path, str]:
        seen: set[tuple[_Path, str]] = set()
        current = reference
        while current not in seen:
            seen.add(current)
            component = self.components.get(current[0])
            if component is None:
                raise _fail("coupling references an unknown component path")
            if not component.is_composite:
                declarations = {
                    _string(_mapping(item, "inductive branch").get("id"), "branch_id")
                    for item in _sequence(component.record.get("inductive_branches"), "inductive_branches")
                }
                if current[1] not in declarations:
                    raise _fail("coupling references an unknown physical inductive branch")
                return current
            realization = _mapping(component.record["realization"], "composite realization")
            matches = [
                _mapping(item, "public inductive branch map")
                for item in _sequence(realization.get("public_inductive_branch_map"), "public_inductive_branch_map")
                if _mapping(item, "public inductive branch map").get("public_id") == current[1]
            ]
            if len(matches) != 1:
                raise _fail("coupling branch exposure is missing or ambiguous")
            current = _branch_ref(matches[0].get("target"))
        raise _fail("coupling branch exposure contains a cycle")


def _make_branches(model: _PlanModel) -> tuple[_Branch, ...]:
    branches: list[_Branch] = []
    for component in sorted(model.components.values(), key=lambda item: item.path):
        if component.is_subsystem:
            continue
        if len(component.pin_order) != 2:
            raise _fail(
                "authoring primitive must expose exactly two physical pins",
                component_path=list(component.path),
                kind=component.kind,
            )
        pins = cast(tuple[str, str], component.pin_order)
        nets = (model.net((component.path, pins[0])), model.net((component.path, pins[1])))
        if component.kind == "resistor":
            branches.append(
                _Branch(component.path, component.parent_scope, "resistance", "R", pins, nets, model.component_value(component, "resistance"), True)
            )
        elif component.kind == "capacitor":
            branches.append(
                _Branch(component.path, component.parent_scope, "capacitance", "C", pins, nets, model.component_value(component, "capacitance"), True)
            )
        elif component.kind == "inductor":
            declarations = tuple(_sequence(component.record.get("inductive_branches"), "inductive_branches"))
            if len(declarations) != 1:
                raise _fail("authoring inductor must expose one oriented branch")
            declaration = _mapping(declarations[0], "inductive branch")
            positive = _endpoint(declaration.get("positive_endpoint"))
            negative = _endpoint(declaration.get("negative_endpoint"))
            oriented_pins = (positive[1], negative[1])
            branches.append(
                _Branch(
                    component.path,
                    component.parent_scope,
                    "inductor:" + _string(declaration.get("id"), "branch_id"),
                    "L",
                    oriented_pins,
                    (model.net(positive), model.net(negative)),
                    model.component_value(component, "inductance"),
                    False,
                )
            )
        elif component.kind == "josephson_junction":
            junction = model.component_value(component, "josephson_inductance")
            capacitance = model.component_value(component, "junction_capacitance")
            branches.append(
                _Branch(component.path, component.parent_scope, "josephson_inductance", "JJ", pins, nets, junction, False)
            )
            branches.append(
                _Branch(
                    component.path,
                    component.parent_scope,
                    "junction_capacitance",
                    "C",
                    pins,
                    nets,
                    capacitance,
                    True,
                    omitted=_quantity_is_zero(capacitance),
                )
            )
        else:
            raise _fail(
                "captured authoring primitive is outside the certified native alphabet",
                component_path=list(component.path),
                kind=component.kind,
            )
    return tuple(branches)


def _line_records(model: _PlanModel) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    for component in sorted(model.components.values(), key=lambda item: item.path):
        if not component.is_line:
            continue
        realization = _mapping(component.record["realization"], "line realization")
        rlgc = _mapping(realization.get("rlgc"), "rlgc")
        conductors = tuple(
            _string(item, "conductor")
            for item in _sequence(realization.get("pin_conductors"), "pin_conductors")
        )
        if tuple(component.pin_order) != tuple(
            f"{end}.{conductor}" for end in ("head", "tail") for conductor in conductors
        ):
            raise _fail("line pin order disagrees with its ordered conductor rows")
        rows = [
            {
                "conductor": conductor,
                "row_ordinal": ordinal,
                "head_net": model.net((component.path, f"head.{conductor}")),
                "tail_net": model.net((component.path, f"tail.{conductor}")),
                "head_site": _contact_token(
                    "boundary_site", path=component.path, identity=f"head.{conductor}"
                ),
                "tail_site": _contact_token(
                    "boundary_site", path=component.path, identity=f"tail.{conductor}"
                ),
            }
            for ordinal, conductor in enumerate(conductors)
        ]
        records.append(
            {
                "component_path": list(component.path),
                "conductors": list(conductors),
                "reference_conductor": _string(rlgc.get("reference_conductor"), "reference_conductor"),
                "orientation": _string(rlgc.get("orientation"), "rlgc.orientation"),
                "rows": rows,
            }
        )
    return tuple(records)


def _all_couplings(model: _PlanModel) -> tuple[Mapping[str, object], ...]:
    records: list[Mapping[str, object]] = [
        _mapping(item, "coupling")
        for item in _sequence(model.document.get("couplings"), "couplings")
    ]
    for component in model.components.values():
        if component.is_composite:
            realization = _mapping(component.record["realization"], "composite realization")
            records.extend(
                _mapping(item, "coupling")
                for item in _sequence(realization.get("couplings"), "couplings")
            )
    records.sort(key=lambda item: (_string(item.get("id"), "coupling.id"), canonical_json_bytes(item)))
    return tuple(records)


def _lca(left: _Path, right: _Path) -> _Path:
    result: list[str] = []
    for left_part, right_part in zip(left, right, strict=False):
        if left_part != right_part:
            break
        result.append(left_part)
    return tuple(result)


def _coupling_records(model: _PlanModel) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    for coupling in _all_couplings(model):
        left = model.resolve_branch_ref(_branch_ref(coupling.get("branch_a")))
        right = model.resolve_branch_ref(_branch_ref(coupling.get("branch_b")))
        left_component = model.components[left[0]]
        right_component = model.components[right[0]]
        owner = _lca(left_component.parent_scope, right_component.parent_scope)
        records.append(
            {
                "coupling_id": _string(coupling.get("id"), "coupling.id"),
                "owner_scope": list(owner),
                "branch_a": _branch_ref_record(left),
                "branch_b": _branch_ref_record(right),
                "coupling_coefficient": _plain(coupling.get("coupling_coefficient")),
                "derived_mutual_inductance": _plain(coupling.get("derived_mutual_inductance")),
            }
        )
    return tuple(records)


def _site_for_subsystem(model: _PlanModel, component: _Component, pin_id: str) -> _Site:
    return _Site(
        component.path,
        pin_id,
        model.net((component.path, pin_id)),
        "subsystem",
        canonical_json_bytes({"component_path": list(component.path)}).decode("utf-8"),
    )


def _sites_for_scope(model: _PlanModel, scope: _Path) -> tuple[_Site, ...]:
    sites = [
        _site_for_subsystem(model, component, pin)
        for component in model.immediate_subsystems(scope)
        for pin in component.pin_order
    ]
    if scope:
        component = model.components[scope]
        sites.extend(
            _Site(
                scope,
                pin,
                model.net((scope, pin)),
                "parent_boundary",
                canonical_json_bytes(
                    {"parent_scope": list(component.parent_scope), "component_path": list(scope), "pin_id": pin}
                ).decode("utf-8"),
            )
            for pin in component.pin_order
        )
    else:
        subsystem_nets = {site.net for site in sites}
        for raw in _sequence(model.document.get("ports"), "ports"):
            port = _mapping(raw, "port")
            port_id = _string(port.get("port_id"), "port_id")
            net = model.node_net(_string(port.get("node_id"), "node_id"))
            if net not in subsystem_nets:
                sites.append(_Site((), port_id, net, "port", port_id))
    unique = {site.key: site for site in sites}
    return tuple(unique[key] for key in sorted(unique))


@dataclass(frozen=True, slots=True)
class _MemberUnit:
    id: str
    members: tuple[str, ...]
    nets: frozenset[str]
    grounded: bool


def _relation_and_units(
    branches: Sequence[_Branch], scope: _Path
) -> tuple[list[dict[str, object]], list[_MemberUnit]]:
    owned = tuple(branch for branch in branches if branch.owner == scope and not branch.omitted)
    grouped: dict[tuple[str, str], list[_Branch]] = defaultdict(list)
    for branch in owned:
        grouped[branch.unordered_nets].append(branch)
    relations: list[dict[str, object]] = []
    units: list[_MemberUnit] = []
    consumed: set[str] = set()
    for nets in sorted(grouped):
        members = sorted(grouped[nets], key=lambda item: item.key)
        if len(members) < 2:
            continue
        member_ids = tuple(member.key for member in members)
        relation_id = _token(
            "parallel_relation", owner_scope=list(scope), endpoints=list(nets), members=list(member_ids)
        )
        relations.append(
            {
                "relation_id": relation_id,
                "owner_scope": list(scope),
                "endpoint_nets": list(nets),
                "members": list(member_ids),
            }
        )
        units.append(_MemberUnit(relation_id, member_ids, frozenset(nets), "ground" in nets))
        consumed.update(member_ids)
    for branch in sorted(owned, key=lambda item: item.key):
        if branch.key in consumed:
            continue
        units.append(
            _MemberUnit(branch.key, (branch.key,), frozenset(branch.nets), "ground" in branch.nets)
        )
    return relations, units


def _connected_units(units: Sequence[_MemberUnit], boundary_nets: set[str]) -> list[list[_MemberUnit]]:
    remaining = [unit for unit in units if not unit.grounded]
    neighbors: dict[int, set[int]] = defaultdict(set)
    for left, first in enumerate(remaining):
        for right in range(left + 1, len(remaining)):
            shared = first.nets & remaining[right].nets
            if any(net != "ground" and net not in boundary_nets for net in shared):
                neighbors[left].add(right)
                neighbors[right].add(left)
    groups: list[list[_MemberUnit]] = []
    unseen = set(range(len(remaining)))
    while unseen:
        start = min(unseen)
        pending = [start]
        unseen.remove(start)
        indexes: list[int] = []
        while pending:
            current = pending.pop()
            indexes.append(current)
            for neighbor in sorted(neighbors[current]):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    pending.append(neighbor)
        groups.append([remaining[index] for index in sorted(indexes)])
    return groups


def _interface_records(
    model: _PlanModel, branches: Sequence[_Branch], scope: _Path
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    relations, units = _relation_and_units(branches, scope)
    sites = _sites_for_scope(model, scope)
    sites_by_net: dict[str, list[_Site]] = defaultdict(list)
    for site in sites:
        sites_by_net[site.net].append(site)
    boundary_nets = set(sites_by_net)
    components = _connected_units(units, boundary_nets)
    candidates: list[dict[str, object]] = []
    internal_ids = {unit.id for unit in units if unit.grounded}
    for component in components:
        nets = frozenset(net for unit in component for net in unit.nets)
        incidence = tuple(
            sorted(
                (site for net in nets for site in sites_by_net.get(net, ())),
                key=lambda site: site.key,
            )
        )
        peers = tuple(sorted({site.peer_key for site in incidence}))
        members = tuple(sorted({member for unit in component for member in unit.members}))
        if len(peers) < 2:
            internal_ids.update(unit.id for unit in component)
            continue
        candidates.append(
            {
                "units": tuple(sorted(unit.id for unit in component)),
                "members": members,
                "nets": tuple(sorted(nets)),
                "sites": incidence,
                "peers": peers,
            }
        )

    pair_indexes = [index for index, item in enumerate(candidates) if len(item["peers"]) == 2]
    pair_uf = _UnionFind()
    encoded_indexes = {index: f"candidate-{index}".encode() for index in pair_indexes}
    for value in encoded_indexes.values():
        pair_uf.add(value)
    for offset, left in enumerate(pair_indexes):
        for right in pair_indexes[offset + 1 :]:
            if candidates[left]["peers"] == candidates[right]["peers"] and set(candidates[left]["nets"]) & set(candidates[right]["nets"]):
                pair_uf.union(encoded_indexes[left], encoded_indexes[right])
    merged_groups: dict[bytes, list[int]] = defaultdict(list)
    for index in pair_indexes:
        merged_groups[pair_uf.find(encoded_indexes[index])].append(index)
    selected: list[dict[str, object]] = [
        item for item in candidates if len(item["peers"]) >= 3
    ]
    for indexes in merged_groups.values():
        merged = [candidates[index] for index in indexes]
        selected.append(
            {
                "units": tuple(sorted({value for item in merged for value in item["units"]})),
                "members": tuple(sorted({value for item in merged for value in item["members"]})),
                "nets": tuple(sorted({value for item in merged for value in item["nets"]})),
                "sites": tuple(sorted({site.key: site for item in merged for site in item["sites"]}.values(), key=lambda site: site.key)),
                "peers": merged[0]["peers"],
            }
        )

    interfaces: list[dict[str, object]] = []
    classified: dict[str, str] = {}
    for item in sorted(selected, key=lambda value: (value["peers"], value["members"], value["nets"])):
        sites_value = cast(tuple[_Site, ...], item["sites"])
        peers = cast(tuple[str, ...], item["peers"])
        members = cast(tuple[str, ...], item["members"])
        nets = cast(tuple[str, ...], item["nets"])
        terminal_counts = {
            peer: sum(site.peer_key == peer for site in sites_value)
            for peer in peers
        }
        relation_type = "P" if len(peers) == 2 else "M"
        identity = _token(
            "interface_bundle",
            parent_scope=list(scope),
            peer_signature=list(peers),
            sites=[site.key for site in sites_value],
            members=list(members),
            nets=list(nets),
        )
        interfaces.append(
            {
                "interface_id": identity,
                "parent_scope": list(scope),
                "relation_type": relation_type,
                "peer_signature": list(peers),
                "peer_count": len(peers),
                "sites": [site.record() for site in sites_value],
                "members": list(members),
                "nets": list(nets),
                "terminal_counts": terminal_counts,
                "terminal_class": "multi_terminal" if any(count > 1 for count in terminal_counts.values()) else "single_terminal",
            }
        )
        classified.update({member: identity for member in members})

    stitches: list[dict[str, object]] = []
    for net, net_sites in sorted(sites_by_net.items()):
        unique_sites = tuple(sorted({site.key: site for site in net_sites}.values(), key=lambda site: site.key))
        if len(unique_sites) < 2:
            continue
        peers = tuple(sorted({site.peer_key for site in unique_sites}))
        identity = _token(
            "memberless_stitch",
            parent_scope=list(scope),
            peer_signature=list(peers),
            net=net,
            sites=[site.key for site in unique_sites],
            members=[],
        )
        stitches.append(
            {
                "stitch_id": identity,
                "parent_scope": list(scope),
                "relation_type": "S",
                "net": net,
                "peer_signature": list(peers),
                "peer_count": len(peers),
                "sites": [site.record() for site in unique_sites],
                "members": [],
            }
        )

    classification: list[dict[str, object]] = []
    for unit in sorted(units, key=lambda item: item.id):
        if unit.id in internal_ids:
            role = "parent_internal_or_peripheral"
            relation_id: str | None = None
        else:
            member_targets = {classified[member] for member in unit.members if member in classified}
            if len(member_targets) != 1:
                raise _fail("expected member partition is incomplete or duplicated", unit_id=unit.id)
            role = "interface_member"
            relation_id = next(iter(member_targets))
        classification.append(
            {
                "unit_id": unit.id,
                "members": list(unit.members),
                "classification": role,
                "interface_id": relation_id,
            }
        )
    return relations, interfaces, stitches, classification


def _expected_authoring(model: _PlanModel) -> _ExpectedManifest:
    branches = _make_branches(model)
    active = tuple(branch for branch in branches if not branch.omitted)
    omissions = tuple(branch for branch in branches if branch.omitted)
    lines = _line_records(model)
    ports: list[dict[str, object]] = []
    for raw in _sequence(model.document.get("ports"), "ports"):
        port = _mapping(raw, "port")
        port_id = _string(port.get("port_id"), "port_id")
        node_net = model.node_net(_string(port.get("node_id"), "node_id"))
        ports.append(
            {
                "port_id": port_id,
                "role": _string(port.get("role"), "port.role"),
                "orientation": _string(port.get("orientation"), "port.orientation"),
                "reference_impedance": _plain(port.get("reference_impedance")),
                "node_net": node_net,
                "reference_net": "ground",
                "circuit_contact": _contact_token("port", identity=port_id, terminal="circuit"),
                "boundary_contact": _contact_token("port", identity=port_id, terminal="boundary"),
                "load_signal_contact": _contact_token("port", identity=port_id, terminal="load_signal"),
                "load_reference_contact": _contact_token("port", identity=port_id, terminal="load_reference"),
                "load_kind": "raw_reference_impedance",
            }
        )
    couplings = _coupling_records(model)

    net_contacts: dict[str, set[str]] = defaultdict(set)
    for branch in active:
        for pin, net in zip(branch.pins, branch.nets, strict=True):
            net_contacts[net].add(
                _contact_token(
                    "native_terminal", path=branch.path, identity=branch.role, terminal=pin
                )
            )
    all_sites: dict[str, _Site] = {}
    scopes = [(), *(component.path for component in model.components.values() if component.is_composite)]
    for scope in scopes:
        for site in _sites_for_scope(model, scope):
            all_sites.setdefault(site.key, site)
    for site in all_sites.values():
        net_contacts[site.net].add(
            _contact_token("boundary_site", path=site.path, identity=site.pin_id)
        )
    for port in ports:
        node_net = cast(str, port["node_net"])
        net_contacts[node_net].update(
            cast(str, port[field])
            for field in ("circuit_contact", "boundary_contact", "load_signal_contact")
        )
        net_contacts["ground"].add(cast(str, port["load_reference_contact"]))
    for node in model.public_nodes():
        if node["visibility"] == "public":
            net_contacts[cast(str, node["net"])].add(
                _contact_token("public_node", identity=cast(str, node["node_id"]))
            )

    electrical = {
        "schema": "scnsim.diagram_connectivity_manifest",
        "schema_version": 1,
        "representation": "authoring",
        "nets": [
            {"net": net, "contacts": sorted(contacts)}
            for net, contacts in sorted(net_contacts.items())
        ],
        "public_nodes": list(model.public_nodes()),
        "branches": [branch.record() for branch in sorted(active, key=lambda item: item.key)],
        "transmission_lines": sorted(
            lines, key=lambda item: cast(list[str], item["component_path"])
        ),
        "ports": sorted(ports, key=lambda item: cast(str, item["port_id"])),
        "couplings": [
            {
                key: value
                for key, value in coupling.items()
                if key not in {"owner_scope", "derived_mutual_inductance"}
            }
            for coupling in sorted(
                couplings, key=lambda item: cast(str, item["coupling_id"])
            )
        ],
        "omissions": [
            {
                "branch_id": branch.key,
                "component_path": list(branch.path),
                "branch_role": branch.role,
                "native_kind": branch.native_kind,
                "endpoint_nets": list(branch.nets),
                "reason": "exact_zero",
                "surviving_continuity": list(branch.nets),
            }
            for branch in sorted(omissions, key=lambda item: item.key)
        ],
    }

    regions = [
        {
            "component_path": list(component.path),
            "parent_scope": list(component.parent_scope),
            "region_role": "electrical_box" if component.is_line else "composite_region",
            "visible_id": component.path[-1],
        }
        for component in sorted(model.components.values(), key=lambda item: item.path)
        if component.is_subsystem
    ]
    regions.insert(
        0,
        {
            "component_path": [],
            "parent_scope": None,
            "region_role": "root_envelope",
            "visible_id": None,
        },
    )
    leaf_ownership = [
        {
            "branch_id": branch.key,
            "component_path": list(branch.path),
            "owner_scope": list(branch.owner),
            "visible_local_name": branch.path[-1],
            "branch_role": branch.role,
        }
        for branch in sorted(active, key=lambda item: (item.path, item.role))
    ]
    relations: list[dict[str, object]] = []
    interfaces: list[dict[str, object]] = []
    stitches: list[dict[str, object]] = []
    classification: list[dict[str, object]] = []
    for scope in sorted(scopes):
        scope_relations, scope_interfaces, scope_stitches, scope_classification = _interface_records(
            model, branches, scope
        )
        relations.extend(scope_relations)
        interfaces.extend(scope_interfaces)
        stitches.extend(scope_stitches)
        classification.extend(scope_classification)
    semantic = {
        "schema": "scnsim.diagram_semantic_manifest",
        "schema_version": 1,
        "representation": "authoring",
        "regions": regions,
        "leaf_ownership": leaf_ownership,
        "boundary_sites": [site.record() for site in sorted(all_sites.values(), key=lambda item: item.key)],
        "relations": sorted(relations, key=lambda item: cast(str, item["relation_id"])),
        "interfaces": sorted(interfaces, key=lambda item: cast(str, item["interface_id"])),
        "stitches": sorted(stitches, key=lambda item: cast(str, item["stitch_id"])),
        "member_classification": sorted(classification, key=lambda item: cast(str, item["unit_id"])),
        "port_boundaries": [
            {
                "port_id": port["port_id"],
                "node_net": port["node_net"],
                "colocated_subsystem_peers": sorted(
                    {
                        site.peer_key
                        for site in all_sites.values()
                        if site.net == port["node_net"] and site.peer_kind == "subsystem"
                    }
                ),
            }
            for port in ports
        ],
        "couplings": [
            {
                key: value
                for key, value in coupling.items()
                if key != "derived_mutual_inductance"
            }
            for coupling in couplings
        ],
    }
    expected_values = {
        branch.key: _plain(branch.value)
        for branch in branches
    }
    expected_values.update(
        {
            _token("transmission_line_length", component_path=line["component_path"]): _plain(
                model.component_value(
                    model.components[tuple(cast(Sequence[str], line["component_path"]))],
                    "length",
                )
            )
            for line in lines
        }
    )
    expected_values.update(
        {
            _token("port_impedance", port_id=_string(raw.get("port_id"), "port_id")): _plain(raw.get("reference_impedance"))
            for raw in (
                _mapping(value, "port")
                for value in _sequence(model.document.get("ports"), "ports")
            )
        }
    )
    expected_values.update(
        {
            _token("coupling_coefficient", coupling_id=cast(str, coupling["coupling_id"])): _plain(
                coupling["coupling_coefficient"]
            )
            for coupling in couplings
        }
    )
    expected_values.update(
        {
            _token("derived_mutual_inductance", coupling_id=cast(str, coupling["coupling_id"])): _plain(
                coupling["derived_mutual_inductance"]
            )
            for coupling in couplings
        }
    )
    verified = {
        "schema": "scnsim.diagram_verified_snapshot",
        "schema_version": 1,
        "representation": "authoring",
        "plan_id": model.document["plan_id"],
        "plan_sha256": model.capture.plan_sha256,
        "baseline_parameters": _plain(model.capture.baseline_parameters),
        "source_provenance": _plain(model.capture.provenance),
        "canonical_values": expected_values,
    }
    return _ExpectedManifest(
        "authoring",
        cast(Mapping[str, object], _freeze(electrical)),
        cast(Mapping[str, object], _freeze(semantic)),
        cast(Mapping[str, object], _freeze(expected_values)),
        cast(Mapping[str, object], _freeze(verified)),
        None,
        None,
    )


def _compiled_identity(compiled: Mapping[str, object]) -> tuple[str, str]:
    plan_sha = _string(compiled.get("plan_sha256"), "compiled.plan_sha256")
    expanded = _string(compiled.get("expanded_graph_sha256"), "expanded_graph_sha256")
    expected = sha256_hex(
        {
            "schema": "scnsim.expanded_graph_identity",
            "schema_version": 1,
            "plan_sha256": plan_sha,
            "node_order": _plain(compiled.get("node_order")),
            "resolved_bindings": _plain(compiled.get("resolved_bindings")),
            "expanded_branch_rows": _plain(compiled.get("expanded_branch_rows")),
        }
    )
    if expanded != expected:
        raise _fail(
            "compiled expanded-graph evidence fails its existing identity encoder",
            expected_expanded_graph_sha256=expected,
            actual_expanded_graph_sha256=expanded,
        )
    lineage = _mapping(compiled.get("ref_lineage"), "compiled.ref_lineage")
    original = _mapping(lineage.get("original"), "compiled original lineage")
    compiler = _string(original.get("compiled_graph_sha256"), "compiled_graph_sha256")
    return compiler, expanded


def _compiled_row_projection(row: Mapping[str, object]) -> dict[str, object]:
    """Project one compiler row onto the complete visible matrix grammar."""

    kind = _string(row.get("kind"), "expanded row kind")
    if kind == "transmission_line_audit":
        return {
            "kind": kind,
            "component_path": _plain(row.get("component_path")),
            "conductors": _plain(row.get("conductors")),
            "reference_conductor": row.get("reference_conductor"),
            "n_sections": row.get("n_sections"),
            "length": _plain(row.get("length")),
            "dx": _plain(row.get("dx")),
            "orientation": row.get("orientation"),
            "stations": _plain(row.get("stations")),
            "rlgc_source": _plain(row.get("rlgc_source")),
        }
    if kind == "mutual_inductance":
        return {
            "kind": kind,
            "coupling_id": row.get("coupling_id"),
            "branch_a": _plain(row.get("branch_a")),
            "branch_b": _plain(row.get("branch_b")),
            "coupling_coefficient": _plain(row.get("coupling_coefficient")),
            "derived_mutual_inductance": _plain(row.get("derived_mutual_inductance")),
            "omitted_as_zero": row.get("omitted_as_zero"),
        }
    return {
        "kind": kind,
        "component_path": _plain(row.get("component_path")),
        "section": row.get("section"),
        "station": row.get("station"),
        "end": row.get("end"),
        "row_conductor": row.get("row_conductor"),
        "column_conductor": row.get("column_conductor"),
        "branch_id": row.get("branch_id"),
        "value": _plain(row.get("value")),
        "omitted_as_zero": row.get("omitted_as_zero"),
        "terminal_1_to_terminal_2": row.get("terminal_1_to_terminal_2"),
        "incidence_f64": _plain(row.get("incidence_f64")),
        "physical_positive_incidence_f64": _plain(
            row.get("physical_positive_incidence_f64")
        ),
        "physical_negative_incidence_f64": _plain(
            row.get("physical_negative_incidence_f64")
        ),
    }


def _expected_compiled(model: _PlanModel, compiled: Mapping[str, object]) -> _ExpectedManifest:
    if _string(compiled.get("plan_sha256"), "compiled.plan_sha256") != model.capture.plan_sha256:
        raise _fail("compiled evidence belongs to a different Plan snapshot")
    compiler_sha, expanded_sha = _compiled_identity(compiled)
    node_order = tuple(
        _string(item, "compiled node")
        for item in _sequence(compiled.get("node_order"), "compiled.node_order")
    )
    source_rows = tuple(
        _mapping(item, "expanded branch row")
        for item in _sequence(compiled.get("expanded_branch_rows"), "expanded_branch_rows")
    )
    rows = tuple(_compiled_row_projection(row) for row in source_rows)
    line_audits = tuple(row for row in rows if row.get("kind") == "transmission_line_audit")
    branch_rows = tuple(row for row in rows if row.get("kind") != "transmission_line_audit")
    active = tuple(row for row in branch_rows if row.get("omitted_as_zero") is not True)
    omissions = tuple(row for row in branch_rows if row.get("omitted_as_zero") is True)
    couplings = tuple(row for row in active if row.get("kind") == "mutual_inductance")
    physical = tuple(row for row in active if row.get("kind") != "mutual_inductance")
    ports = []
    for raw in _sequence(model.document.get("ports"), "ports"):
        port = _mapping(raw, "port")
        node_id = _string(port.get("node_id"), "port.node_id")
        if node_id not in node_order:
            raise _fail("compiled Port node is absent from the visible compiler node order")
        ports.append(
            {
                "port_id": _string(port.get("port_id"), "port.port_id"),
                "node_id": node_id,
                "role": _string(port.get("role"), "port.role"),
                "orientation": _string(port.get("orientation"), "port.orientation"),
                "reference_impedance": _plain(port.get("reference_impedance")),
                "reference_node": "ground",
                "load_kind": "raw_reference_impedance",
            }
        )
    ports.sort(key=lambda item: cast(str, item["port_id"]))
    electrical = {
        "schema": "scnsim.diagram_connectivity_manifest",
        "schema_version": 1,
        "representation": "compiled",
        "node_order": list(node_order),
        "expanded_terms": [_plain(row) for row in physical],
        "transmission_lines": [_plain(row) for row in line_audits],
        "couplings": [_plain(row) for row in couplings],
        "omissions": [_plain(row) for row in omissions],
        "ports": ports,
    }
    hierarchy = [
        {
            "component_path": list(_path(row.get("component_path"))),
            "kind": "pi_ladder",
            "n_sections": row.get("n_sections"),
            "conductors": _plain(row.get("conductors")),
        }
        for row in line_audits
    ]
    membership = [
        {
            "component_path": _plain(row.get("component_path")),
            "compiled_kind": row.get("kind"),
            "section": row.get("section"),
            "station": row.get("station"),
            "row_conductor": row.get("row_conductor"),
            "column_conductor": row.get("column_conductor"),
            "omitted_as_zero": row.get("omitted_as_zero"),
        }
        for row in branch_rows
    ]
    semantic = {
        "schema": "scnsim.diagram_semantic_manifest",
        "schema_version": 1,
        "representation": "compiled",
        "compiler_hierarchy": hierarchy,
        "expanded_membership": membership,
        "matrix_evidence": [_plain(row) for row in line_audits],
        "couplings": [_plain(row) for row in couplings],
        "omissions": [_plain(row) for row in omissions],
    }
    expected_values = {
        _token("compiled_term", ordinal=index): _plain(row.get("value"))
        for index, row in enumerate(branch_rows)
        if row.get("value") is not None
    }
    verified = {
        "schema": "scnsim.diagram_verified_snapshot",
        "schema_version": 1,
        "representation": "compiled",
        "plan_id": model.document["plan_id"],
        "plan_sha256": model.capture.plan_sha256,
        "compiled_graph_sha256": compiler_sha,
        "expanded_graph_sha256": expanded_sha,
        "baseline_parameters": _plain(model.capture.baseline_parameters),
        "source_provenance": _plain(model.capture.provenance),
        "resolved_bindings": _plain(compiled.get("resolved_bindings")),
        "canonical_values": expected_values,
    }
    return _ExpectedManifest(
        "compiled",
        cast(Mapping[str, object], _freeze(electrical)),
        cast(Mapping[str, object], _freeze(semantic)),
        cast(Mapping[str, object], _freeze(expected_values)),
        cast(Mapping[str, object], _freeze(verified)),
        compiler_sha,
        expanded_sha,
    )


def build_expected(
    capture: _CapturedPlan,
    *,
    representation: _Representation = "authoring",
    compiled: Mapping[str, object] | None = None,
) -> _ExpectedManifest:
    """Build expected A/B manifests without consulting renderer answer tables."""

    if representation not in ("authoring", "compiled"):
        raise ValueError("representation must be 'authoring' or 'compiled'")
    model = _PlanModel(capture)
    if representation == "authoring":
        if compiled is not None:
            raise TypeError("compiled evidence is invalid for an authoring diagram")
        return _expected_authoring(model)
    if compiled is None:
        raise TypeError("compiled diagrams require exact expanded-graph evidence")
    return _expected_compiled(model, compiled)


# The structured V2 witness deliberately has its own normalized-snapshot
# projection.  The V1 helpers above remain solely for the retained compiled
# row decoder while the renderer is migrated; no V2 certification path uses
# their catalog/tree or inferred-interface model.


@dataclass(frozen=True, slots=True)
class _PointExpectedManifest:
    representation: _Representation
    electrical: Mapping[str, object]
    structural: Mapping[str, object]
    expected_values: Mapping[str, object]
    verified: Mapping[str, object]
    compiled_graph_sha256: str | None = None
    expanded_graph_sha256: str | None = None


def _point_semantic(point: ResolvedPlanPoint) -> Mapping[str, object]:
    if not isinstance(point, ResolvedPlanPoint):
        raise TypeError("structured witness requires ResolvedPlanPoint")
    semantic = _mapping(point.snapshot.semantic_record, "semantic_record")
    if semantic.get("schema") != "scnsim.authoring_snapshot" or semantic.get("schema_version") != 2:
        raise _fail("structured witness requires the normalized V2 authoring snapshot")
    return semantic


def _v2_path(value: object, field: str) -> _Path:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise _fail("structured witness path must be an array", field=field)
    result = tuple(_string(item, field) for item in value)
    return result


def _v2_quantity(point: ResolvedPlanPoint, path: _Path, field: str, unit: str) -> Mapping[str, object]:
    value = point.resolved_fields.get((path, field))
    if value is None:
        raise _fail("resolved point does not exactly cover a displayed physical field", path=list(path), field=field)
    if unit == "rlgc":
        record = getattr(value, "_record", None)
        if not callable(record):
            raise _fail("resolved RLGC field has no structured record", path=list(path), field=field)
        encoded = record()
        if not isinstance(encoded, Mapping):
            raise _fail("resolved RLGC record is malformed", path=list(path), field=field)
        return encoded
    return cast(
        Mapping[str, object],
        quantity_envelope(value, si_unit=unit, registry=units.registry),
    )


def _v2_contact(*, path: _Path | None = None, pin: str | None = None, port: str | None = None) -> str:
    if port is not None:
        return _token("visible_port_contact", port_id=port)
    assert path is not None and pin is not None
    return _token("visible_physical_terminal", component_path=list(path), pin_id=pin)


def _v2_net_map(semantic: Mapping[str, object]) -> tuple[dict[str, str], tuple[dict[str, object], ...]]:
    connectivity = _mapping(semantic.get("connectivity"), "connectivity")
    contacts: dict[str, set[str]] = defaultdict(set)
    for raw in _sequence(connectivity.get("physical_endpoints"), "physical_endpoints"):
        row = _mapping(raw, "physical_endpoint")
        path = _v2_path(row.get("path"), "physical_endpoint.path")
        pin = _string(row.get("pin"), "physical_endpoint.pin")
        net = _string(row.get("net"), "physical_endpoint.net")
        contacts[net].add(_v2_contact(path=path, pin=pin))
    for raw in _sequence(connectivity.get("ports"), "ports"):
        row = _mapping(raw, "port")
        contacts[_string(row.get("net"), "port.net")].add(
            _v2_contact(port=_string(row.get("id"), "port.id"))
        )
    result: dict[str, str] = {"ground": "ground"}
    rows: list[dict[str, object]] = []
    ordered_sources = sorted(
        (source for source in contacts if source != "ground"),
        key=lambda source: canonical_json_bytes(sorted(contacts[source])),
    )
    ordinals = {
        source: f"netv-{ordinal}" for ordinal, source in enumerate(ordered_sources)
    }
    for source, values in sorted(contacts.items()):
        if not values:
            raise _fail("captured electrical net has no visible physical contact", net=source)
        visible = "ground" if source == "ground" else ordinals[source]
        result[source] = visible
        rows.append({"net": visible, "contacts": sorted(values)})
    return result, tuple(sorted(rows, key=canonical_json_bytes))


def _v2_rebased_path(value: object, field: str, *, basis: _Path) -> _Path:
    path = _v2_path(value, field)
    return (*basis, *path) if basis else path


def _v2_endpoint(
    value: object,
    *,
    basis: _Path = (),
    region_paths: frozenset[_Path] = frozenset(),
) -> dict[str, object]:
    endpoint = dict(_mapping(value, "structured endpoint"))
    kind = endpoint.get("kind")
    if kind == "ground":
        if set(endpoint) != {"kind"}:
            raise _fail("ground endpoint has extra fields")
        return endpoint
    if kind not in {"bus", "tap", "pin"}:
        raise _fail("structured endpoint kind is unsupported", kind=kind)
    scope = _v2_rebased_path(endpoint.get("scope"), "endpoint.scope", basis=basis)
    identifier = _string(endpoint.get("id"), "endpoint.id")
    if kind == "tap":
        _string(endpoint.get("bus"), "endpoint.bus")
    if kind == "pin":
        component = _string(endpoint.get("component"), "endpoint.component")
        if not isinstance(endpoint.get("public"), bool):
            raise _fail("pin endpoint public flag is malformed")
        boundary_path = scope if endpoint["public"] is True else (*scope, component)
        if endpoint["public"] is True or boundary_path in region_paths:
            return {"kind": "boundary_pin", "scope": list(boundary_path), "id": identifier}
        return {
            "kind": "physical_pin",
            "path": [*scope, component],
            "pin_id": identifier,
        }
    endpoint["scope"] = list(scope)
    return cast(dict[str, object], _plain(endpoint))


def _v2_element(value: object, *, basis: _Path = ()) -> dict[str, object]:
    row = _mapping(value, "structured element")
    return {
        "path": list(_v2_rebased_path(row.get("path"), "element.path", basis=basis)),
        "pin_1": _string(row.get("pin_1"), "element.pin_1"),
        "pin_2": _string(row.get("pin_2"), "element.pin_2"),
    }


def _v2_scopes(root: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []

    def visit(scope: Mapping[str, object], parent: _Path | None) -> None:
        path = _v2_path(scope.get("path"), "scope.path")
        _string(scope.get("kind"), "scope.kind")
        if path:
            records.append(
                {
                    "path": list(path),
                    "parent": [] if parent is None else list(parent),
                    "local_id": path[-1],
                }
            )
        for child in _sequence(scope.get("children"), "scope.children"):
            visit(_mapping(child, "child scope"), path)
        for body in _sequence(scope.get("component_bodies"), "scope.component_bodies"):
            visit(_mapping(_mapping(body, "component body").get("body"), "component body scope"), path)

    visit(root, None)
    return tuple(sorted(records, key=lambda row: tuple(cast(Sequence[str], row["path"]))))


def _v2_visible_contacts(
    semantic: Mapping[str, object],
) -> tuple[dict[str, tuple[str, ...]], dict[str, _Path]]:
    """Project only physical/Port contacts that can be recovered from ink."""

    connectivity = _mapping(semantic.get("connectivity"), "connectivity")
    by_net: dict[str, list[str]] = defaultdict(list)
    owners: dict[str, _Path] = {}
    for raw in _sequence(connectivity.get("physical_endpoints"), "physical_endpoints"):
        row = _mapping(raw, "physical_endpoint")
        path = _v2_path(row.get("path"), "physical_endpoint.path")
        pin = _string(row.get("pin"), "physical_endpoint.pin")
        contact = _v2_contact(path=path, pin=pin)
        by_net[_string(row.get("net"), "physical_endpoint.net")].append(contact)
        owners[contact] = path[:-1]
    for raw in _sequence(connectivity.get("ports"), "ports"):
        row = _mapping(raw, "port")
        contact = _v2_contact(port=_string(row.get("id"), "port.id"))
        by_net[_string(row.get("net"), "port.net")].append(contact)
        owners[contact] = ()
    return (
        {net: tuple(sorted(set(contacts))) for net, contacts in by_net.items()},
        owners,
    )


def _v2_parent_pin_contacts(point: ResolvedPlanPoint, root: Mapping[str, object]) -> set[tuple[_Path, str]]:
    """Read actual parent pin uses from declarations, not global net equality."""
    contacts: set[tuple[_Path, str]] = set()

    def endpoint(raw: object, owner: _Path) -> None:
        if not isinstance(raw, Mapping) or raw.get("kind") != "pin":
            return
        scope = _v2_path(raw.get("scope"), "pin.scope")
        path = scope if raw.get("public") else (*scope, _string(raw.get("component"), "pin.component"))
        if path[:-1] == owner:
            contacts.add((path, _string(raw.get("id"), "pin.id")))

    def visit(scope: Mapping[str, object]) -> None:
        owner = _v2_path(scope.get("path"), "scope.path")
        for raw in _sequence(scope.get("structures"), "scope.structures"):
            row = _mapping(raw, "structure")
            for key in ("start", "at", "end"):
                endpoint(row.get(key), owner)
            for raw_endpoint in row.get("endpoints", ()):
                endpoint(raw_endpoint, owner)
            for branch in row.get("branches", (row,)):
                for member in branch.get("elements", ()):
                    path = _v2_path(member.get("path"), "member.path")
                    if path[:-1] == owner:
                        for key in ("pin_1", "pin_2"):
                            contacts.add((path, _string(member.get(key), key)))
        exposure = _mapping(scope.get("exposures"), "scope.exposures")
        for raw in _sequence(exposure.get("pins"), "scope.exposures.pins"):
            endpoint(_mapping(raw, "public pin").get("intrinsic_endpoint"), owner)
        for child in scope.get("children", ()):
            visit(child)
        for body in scope.get("component_bodies", ()):
            visit(body["body"])

    visit(root)
    for group in point.snapshot.source_provenance.get("ground_pins_call_groups", ()):
        for pin in group:
            scope = _v2_path(pin.get("scope"), "ground pin.scope")
            endpoint(pin, scope[:-1] if pin.get("public") else scope)
    return contacts


def _v2_structural(
    point: ResolvedPlanPoint,
    semantic: Mapping[str, object],
    net_map: Mapping[str, str],
) -> Mapping[str, object]:
    """Build the source-side expectation for visible ownership evidence only."""

    root = _mapping(semantic.get("scope_hierarchy"), "scope_hierarchy")
    parent_contacts = _v2_parent_pin_contacts(point, root)
    scope_rows = _v2_scopes(root)
    contacts_by_net, contact_owners = _v2_visible_contacts(semantic)
    boundary_rows: list[dict[str, object]] = []

    def visit(scope: Mapping[str, object]) -> None:
        path = _v2_path(scope.get("path"), "scope.path")
        exposure = _mapping(scope.get("exposures"), "scope.exposures")
        pins = tuple(
            _mapping(raw, "scope public pin")
            for raw in _sequence(exposure.get("pins"), "scope.exposures.pins")
        )
        if not path and pins:
            raise _fail("root scope cannot expose a child-boundary pin")
        for pin in pins:
            source_net = _string(pin.get("final_net"), "exposure.final_net")
            try:
                visible_net = net_map[source_net]
            except KeyError as error:
                raise _fail(
                    "public boundary net has no visible physical contact",
                    scope=list(path),
                    net=source_net,
                ) from error
            contacts = contacts_by_net.get(source_net, ())
            inside = sorted(
                contact
                for contact in contacts
                if contact_owners[contact][: len(path)] == path
            )
            outside = sorted(contact for contact in contacts if contact not in inside)
            boundary = (
                {"mode": "only"}
                if len(pins) == 1
                else {
                    "mode": "named",
                    "id": _string(pin.get("id"), "exposure.id"),
                }
            )
            boundary_rows.append(
                {
                    "region": list(path),
                    "boundary": boundary,
                    "parent_incidence": "connected" if (path, pin["id"]) in parent_contacts else "open",
                    "net": visible_net,
                    "inside_contacts": inside,
                    "outside_contacts": outside,
                }
            )
        for child in _sequence(scope.get("children"), "scope.children"):
            visit(_mapping(child, "child scope"))
        for body in _sequence(scope.get("component_bodies"), "scope.component_bodies"):
            visit(_mapping(_mapping(body, "component body").get("body"), "component body scope"))

    visit(root)
    physical = [
        {
            "path": list(_v2_path(row.get("path"), "physical_leaf.path")),
            "owner": list(_v2_path(row.get("path"), "physical_leaf.path")[:-1]),
            "model": _string(row.get("model"), "physical_leaf.model"),
        }
        for row in (_mapping(item, "physical leaf") for item in _sequence(semantic.get("physical_leaves"), "physical_leaves"))
    ]
    ports = [
        {"port_id": _string(row.get("id"), "port.id"), "owner": []}
        for row in (_mapping(item, "port") for item in _sequence(_mapping(semantic.get("connectivity"), "connectivity").get("ports"), "ports"))
    ]
    return cast(
        Mapping[str, object],
        _freeze(
            {
                "schema": "scnsim.diagram_structural_manifest",
                "schema_version": 2,
                "representation": "authoring",
                "regions": list(scope_rows),
                "leaf_ownership": sorted(physical, key=lambda row: cast(list[str], row["path"])),
                "port_ownership": sorted(ports, key=lambda row: cast(str, row["port_id"])),
                "boundary_incidence": sorted(boundary_rows, key=canonical_json_bytes),
            }
        ),
    )


def _v2_verified_source_rows(
    point: ResolvedPlanPoint,
    semantic: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    """Return complete captured authoring facts that are not image claims."""

    rows: list[dict[str, object]] = []

    def add(kind: str, **fields: object) -> None:
        rows.append({"category": "verified_source", "kind": kind, **fields})

    def visit(scope: Mapping[str, object]) -> None:
        path = list(_v2_path(scope.get("path"), "scope.path"))
        for raw in _sequence(scope.get("structures"), "scope.structures"):
            add("authored_operator", scope=path, record=_plain(_mapping(raw, "structure")))
        for raw in _sequence(scope.get("buses"), "scope.buses"):
            add("authored_bus", scope=path, record=_plain(_mapping(raw, "bus")))
        exposures = _mapping(scope.get("exposures"), "scope.exposures")
        for collection, exposure_kind in (
            ("pins", "pin"),
            ("coordinates", "coordinate"),
            ("branches", "branch"),
            ("parameters", "parameter"),
        ):
            for raw in _sequence(exposures.get(collection), f"scope.exposures.{collection}"):
                add(
                    "authored_exposure",
                    scope=path,
                    exposure_kind=exposure_kind,
                    record=_plain(_mapping(raw, "exposure")),
                )
        for child in _sequence(scope.get("children"), "scope.children"):
            visit(_mapping(child, "child scope"))
        for raw in _sequence(scope.get("component_bodies"), "scope.component_bodies"):
            body = _mapping(raw, "component body")
            visit(_mapping(body.get("body"), "component body scope"))

    visit(_mapping(semantic.get("scope_hierarchy"), "scope_hierarchy"))
    connectivity = _mapping(semantic.get("connectivity"), "connectivity")
    for raw in _sequence(semantic.get("physical_leaves"), "physical_leaves"):
        add("oriented_physical_leaf", record=_plain(_mapping(raw, "physical leaf")))
    for raw in _sequence(connectivity.get("node_coordinates"), "node_coordinates"):
        add("authored_node_alias", record=_plain(_mapping(raw, "node coordinate")))
    closure = _mapping(semantic.get("parameter_closure"), "parameter_closure")
    for raw in _sequence(closure.get("definitions"), "parameter_closure.definitions"):
        add("parameter_definition", record=_plain(_mapping(raw, "parameter definition")))
    for raw in _sequence(closure.get("field_bindings"), "parameter_closure.field_bindings"):
        add("parameter_field_binding", record=_plain(_mapping(raw, "parameter binding")))
    add("effective_parameter_point", record=_plain(point.parameter_record))
    provenance = _mapping(point.snapshot.source_provenance, "source_provenance")
    for raw in _sequence(provenance.get("source_units"), "source_provenance.source_units"):
        add("source_unit", record=_plain(_mapping(raw, "source unit")))
    for raw in _sequence(
        provenance.get("ground_pins_call_groups"),
        "source_provenance.ground_pins_call_groups",
    ):
        add("ground_call_group", record=_plain(_sequence(raw, "ground call group")))
    return tuple(
        cast(Mapping[str, object], _freeze(row))
        for row in sorted(rows, key=canonical_json_bytes)
    )


def _v2_authoring(point: ResolvedPlanPoint) -> _PointExpectedManifest:
    from .._canonical import canonical_diagram_digests, canonical_parameters_sha256

    semantic = _point_semantic(point)
    net_map, nets = _v2_net_map(semantic)
    connectivity = _mapping(semantic.get("connectivity"), "connectivity")
    endpoint_net = {
        (_v2_path(row.get("path"), "physical_endpoint.path"), _string(row.get("pin"), "physical_endpoint.pin")):
        net_map[_string(row.get("net"), "physical_endpoint.net")]
        for row in (_mapping(item, "physical endpoint") for item in _sequence(connectivity.get("physical_endpoints"), "physical_endpoints"))
    }
    values: dict[str, object] = {}
    bodies: list[dict[str, object]] = []
    for raw in _sequence(semantic.get("physical_leaves"), "physical_leaves"):
        leaf = _mapping(raw, "physical leaf")
        path = _v2_path(leaf.get("path"), "physical_leaf.path")
        pins = tuple(_string(item, "physical_leaf.pin") for item in _sequence(leaf.get("pin_order"), "physical_leaf.pin_order"))
        fields: list[dict[str, object]] = []
        for raw_field in _sequence(leaf.get("fields"), "physical_leaf.fields"):
            field = _mapping(raw_field, "physical field")
            field_id = _string(field.get("id"), "physical_field.id")
            unit = _string(field.get("unit"), "physical_field.unit")
            value = _v2_quantity(point, path, field_id, unit)
            fields.append({"id": field_id, "unit": unit})
            if unit != "rlgc":
                values[_token("physical_field", component_path=list(path), field=field_id)] = _plain(value)
        record: dict[str, object] = {
            "path": list(path),
            "model": _string(leaf.get("model"), "physical_leaf.model"),
            "pin_order": list(pins),
            "terminals": [{"pin_id": pin, "net": endpoint_net[(path, pin)]} for pin in pins],
            "fields": fields,
            "oriented_branches": _plain(leaf.get("oriented_branches")),
        }
        if record["model"] == "transmission_line":
            rlgc = point.resolved_fields.get((path, "rlgc"))
            conductors = getattr(rlgc, "conductors", None)
            reference = getattr(rlgc, "reference_conductor", None)
            if not isinstance(conductors, tuple) or not conductors or not isinstance(reference, str):
                raise _fail("resolved transmission line lacks ordered RLGC evidence", path=list(path))
            record["conductors"] = list(conductors)
            record["reference_conductor"] = reference
            record["line_kind"] = "CPW" if len(conductors) == 1 else "MTL"
            if len(conductors) == 1:
                record["conductors"] = []
                record["reference_conductor"] = None
                record["n_sections"] = _mapping(leaf.get("model_metadata"), "physical_leaf.model_metadata")["n_sections"]
        bodies.append(record)
    ports = []
    for raw in _sequence(connectivity.get("ports"), "ports"):
        port = _mapping(raw, "port")
        port_id = _string(port.get("id"), "port.id")
        impedance = _mapping(port.get("reference_impedance"), "port.reference_impedance")
        ports.append(
            {
                "port_id": port_id,
                "role": _string(port.get("role"), "port.role"),
                "node_net": net_map[_string(port.get("net"), "port.net")],
                "reference_net": "ground",
                "reference_impedance": _plain(impedance),
                "orientation": "node_to_reference",
                "load_kind": "raw_reference_impedance",
            }
        )
        values[_token("port_impedance", port_id=port_id)] = _plain(impedance)
    couplings = []
    for row in (
        _mapping(item, "coupling")
        for item in _sequence(connectivity.get("couplings"), "couplings")
    ):
        coupling_id = _string(row.get("id"), "coupling.id")
        branches = []
        for field in ("inductor_a", "inductor_b"):
            branch = _mapping(row.get(field), f"coupling.{field}")
            branches.append(
                {
                    "path": list(_v2_path(branch.get("path"), f"coupling.{field}.path")),
                    "branch_id": _string(
                        branch.get("branch_id"), f"coupling.{field}.branch_id"
                    ),
                }
            )
        coefficient = _mapping(
            row.get("coupling_coefficient"), "coupling.coupling_coefficient"
        )
        couplings.append(
            {
                "coupling_id": coupling_id,
                "branches": sorted(branches, key=canonical_json_bytes),
            }
        )
        values[_token("coupling_coefficient", coupling_id=coupling_id)] = _plain(
            coefficient
        )
    electrical = cast(
        Mapping[str, object],
        _freeze(
            {
                "schema": "scnsim.diagram_electrical_manifest",
                "schema_version": 2,
                "representation": "authoring",
                "nets": list(nets),
                "bodies": sorted(bodies, key=lambda row: cast(list[str], row["path"])),
                "ports": sorted(ports, key=lambda row: cast(str, row["port_id"])),
                "couplings": sorted(couplings, key=lambda row: cast(str, row["coupling_id"])),
            }
        ),
    )
    digests = canonical_diagram_digests(point.snapshot, representation="authoring")
    verified = cast(
        Mapping[str, object],
        _freeze(
            {
                "plan_id": semantic["plan_id"],
                **digests,
                "parameters_sha256": canonical_parameters_sha256(point.parameter_record),
                "source_provenance": _plain(point.snapshot.source_provenance),
                "canonical_values": values,
                "source_rows": _v2_verified_source_rows(point, semantic),
            }
        ),
    )
    from .equivalence import normalize_authoring
    from .values import display_projection

    # Keep the frozen verified source quantities above exact. Only the visible
    # authoring correspondence uses the deterministic six-digit projection.
    values = {identity: dict(display_projection(value)) for identity, value in values.items()}
    electrical = _plain(electrical)
    for port in electrical["ports"]:
        port["reference_impedance"] = dict(display_projection(port["reference_impedance"]))
    electrical, structural, values = normalize_authoring(
        electrical, _v2_structural(point, semantic, net_map), values
    )
    return _PointExpectedManifest(
        "authoring",
        cast(Mapping[str, object], _freeze(electrical)),
        cast(Mapping[str, object], _freeze(structural)),
        cast(Mapping[str, object], _freeze(values)),
        verified,
    )


def _v2_compiled(point: ResolvedPlanPoint, compiled: Mapping[str, object]) -> _PointExpectedManifest:
    from .._canonical import (
        canonical_diagram_digests,
        canonical_expanded_graph_sha256,
        canonical_parameters_sha256,
        canonical_plan_snapshot,
        sha256_hex as canonical_sha256_hex,
    )

    semantic = _point_semantic(point)
    plan_sha = canonical_sha256_hex(canonical_plan_snapshot(point.snapshot))
    if compiled.get("plan_sha256") != plan_sha:
        raise _fail("compiled evidence belongs to a different Plan point")
    parameters_sha = canonical_parameters_sha256(point.parameter_record)
    if compiled.get("parameters_sha256") != parameters_sha:
        raise _fail("compiled evidence belongs to a different effective point")
    node_order = tuple(_string(item, "compiled.node_order") for item in _sequence(compiled.get("node_order"), "compiled.node_order"))
    bindings = tuple(_mapping(item, "compiled.resolved_binding") for item in _sequence(compiled.get("resolved_bindings"), "compiled.resolved_bindings"))
    source_rows = tuple(_mapping(item, "compiled.expanded_branch_row") for item in _sequence(compiled.get("expanded_branch_rows"), "compiled.expanded_branch_rows"))
    expanded = canonical_expanded_graph_sha256(
        plan_sha256=plan_sha,
        node_order=node_order,
        resolved_bindings=bindings,
        expanded_branch_rows=source_rows,
    )
    if compiled.get("expanded_graph_sha256") != expanded:
        raise _fail("compiled expanded-graph identity is invalid")
    compiler = _string(compiled.get("compiled_graph_sha256"), "compiled.compiled_graph_sha256")
    rows = tuple(_compiled_row_projection(row) for row in source_rows)
    connectivity = _mapping(semantic.get("connectivity"), "connectivity")
    ports = []
    for raw in _sequence(connectivity.get("ports"), "ports"):
        port = _mapping(raw, "port")
        node = _string(port.get("net"), "port.net")
        if node not in node_order:
            raise _fail("compiled Port node is absent from compiler node order", port=port.get("id"))
        ports.append(
            {
                "port_id": _string(port.get("id"), "port.id"),
                "node_id": node,
                "role": _string(port.get("role"), "port.role"),
                "orientation": "node_to_reference",
                "reference_impedance": _plain(port.get("reference_impedance")),
                "reference_node": "ground",
                "load_kind": "raw_reference_impedance",
            }
        )
    active = [row for row in rows if row.get("kind") != "transmission_line_audit" and row.get("omitted_as_zero") is not True]
    audits = [row for row in rows if row.get("kind") == "transmission_line_audit"]
    omissions = [row for row in rows if row.get("kind") != "transmission_line_audit" and row.get("omitted_as_zero") is True]
    mutual = [row for row in active if row.get("kind") == "mutual_inductance"]
    physical = [row for row in active if row.get("kind") != "mutual_inductance"]
    electrical = cast(Mapping[str, object], _freeze({
        "schema": "scnsim.diagram_connectivity_manifest", "schema_version": 1,
        "representation": "compiled", "node_order": list(node_order),
        "expanded_terms": physical, "transmission_lines": audits,
        "couplings": mutual, "omissions": omissions,
        "ports": sorted(ports, key=lambda row: cast(str, row["port_id"])),
    }))
    structural = cast(Mapping[str, object], _freeze({
        "schema": "scnsim.diagram_semantic_manifest", "schema_version": 1,
        "representation": "compiled",
        "compiler_hierarchy": [{"component_path": row["component_path"], "kind": "pi_ladder", "n_sections": row["n_sections"], "conductors": row["conductors"]} for row in audits],
        "expanded_membership": [{"component_path": row.get("component_path"), "compiled_kind": row["kind"], "section": row.get("section"), "station": row.get("station"), "row_conductor": row.get("row_conductor"), "column_conductor": row.get("column_conductor"), "omitted_as_zero": row.get("omitted_as_zero")} for row in rows if row.get("kind") != "transmission_line_audit"],
        "matrix_evidence": audits, "couplings": mutual, "omissions": omissions,
    }))
    digests = canonical_diagram_digests(point.snapshot, representation="compiled")
    verified = cast(Mapping[str, object], _freeze({
        "plan_id": semantic["plan_id"], **digests,
        "parameters_sha256": parameters_sha,
        "compiled_graph_sha256": compiler,
        "expanded_graph_sha256": expanded,
        "resolved_bindings": _plain(bindings),
        "source_provenance": _plain(point.snapshot.source_provenance),
        "canonical_values": {},
        "source_rows": _v2_verified_source_rows(point, semantic),
    }))
    return _PointExpectedManifest("compiled", electrical, structural, MappingProxyType({}), verified, compiler, expanded)


def build_point_expected(
    point: ResolvedPlanPoint,
    *,
    representation: _Representation = "authoring",
    compiled_evidence: Mapping[str, object] | None = None,
) -> _PointExpectedManifest:
    """Build V2 expectations only from one immutable point/compiler handoff."""

    if representation == "authoring":
        if compiled_evidence is not None:
            raise TypeError("authoring witness does not accept compiled evidence")
        return _v2_authoring(point)
    if representation != "compiled":
        raise ValueError("representation must be 'authoring' or 'compiled'")
    if compiled_evidence is None:
        raise TypeError("compiled witness requires exact same-point compiler evidence")
    return _v2_compiled(point, compiled_evidence)
