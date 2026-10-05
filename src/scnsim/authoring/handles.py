"""Authoring handles and mutable wire equivalence owned by assembly scopes.

Handles retain original Plan/component identity; they own no execution state."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SCNSimValidationError
from ..value_storage import immutable_quantity
from .factory_context import _two_terminal_use_token
from .physical_values import identifier


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



# Preserve public class identity through the authoring facade.
GroundRef.__module__ = "scnsim.authoring"
BusRef.__module__ = "scnsim.authoring"
TapRef.__module__ = "scnsim.authoring"
PinRef.__module__ = "scnsim.authoring"
CoordinateRef.__module__ = "scnsim.authoring"
ElectricNodeRef.__module__ = "scnsim.authoring"
InductiveBranchRef.__module__ = "scnsim.authoring"
PortRef.__module__ = "scnsim.authoring"
TwoTerminalUse.__module__ = "scnsim.authoring"
SeriesRef.__module__ = "scnsim.authoring"
ParallelRef.__module__ = "scnsim.authoring"
BranchRef.__module__ = "scnsim.authoring"
LinkRef.__module__ = "scnsim.authoring"
CouplingRef.__module__ = "scnsim.authoring"
