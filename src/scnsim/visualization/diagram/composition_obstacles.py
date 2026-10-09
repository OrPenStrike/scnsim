"""Measured obstacle ownership and declared terminal access for composition.

Spacing envelopes are not visible obstacles. Producers name every measured
body, label, region and conductive primitive with its actual owner and source;
geometry never infers ownership from a point lying inside a rectangle. A
terminal exemption covers only its declared outward stub, never unrelated ink.
The same immutable records translate once with their measured Block.
"""

from __future__ import annotations

from dataclasses import dataclass
from .scene import Bounds, Point, COORDINATE_TOLERANCE

# Source locators are renderer-local immutable keys, not electrical identities.
Key = tuple[object, ...]

_VECTORS = {"left": (-1, 0), "right": (1, 0), "top": (0, 1), "bottom": (0, -1)}


@dataclass(frozen=True, slots=True)
class OwnedObstacle:
    owner: Key
    scope: tuple[str, ...]
    kind: str
    source: Key
    bounds: Bounds
    hierarchy: tuple[Key, ...] = ()

    def translated(self, dx: float, dy: float) -> OwnedObstacle:
        return OwnedObstacle(self.owner, self.scope, self.kind, self.source,
                             self.bounds.translated(dx, dy), self.hierarchy)

    def transformed(self, *, origin: Point, rotation: int) -> OwnedObstacle:
        """Move measured ink with its complete body; ownership stays intrinsic."""
        bounds = Bounds.around(
            Point(x, y).rotated(rotation).translated(origin.x, origin.y)
            for x in (self.bounds.xmin, self.bounds.xmax)
            for y in (self.bounds.ymin, self.bounds.ymax)
        )
        return OwnedObstacle(self.owner, self.scope, self.kind, self.source,
                             bounds, self.hierarchy)


@dataclass(frozen=True, slots=True)
class TerminalAccess:
    owner: Key
    source: Key
    side: str
    start: Point
    end: Point
    obstacle_sources: tuple[Key, ...]

    def translated(self, dx: float, dy: float) -> TerminalAccess:
        return TerminalAccess(self.owner, self.source, self.side,
                              self.start.translated(dx, dy), self.end.translated(dx, dy),
                              self.obstacle_sources)

    def transformed(self, *, origin: Point, rotation: int) -> TerminalAccess:
        """Rotate the declared access ray, never infer a new exit from bounds."""
        sides = ("right", "top", "left", "bottom")
        side = sides[(sides.index(self.side) + rotation // 90) % 4]
        return TerminalAccess(
            self.owner, self.source, side,
            self.start.rotated(rotation).translated(origin.x, origin.y),
            self.end.rotated(rotation).translated(origin.x, origin.y),
            self.obstacle_sources,
        )

    def permits(self, obstacle: OwnedObstacle, first: Point, second: Point) -> bool:
        """Only the actual obstacle intersection inside this access is legal."""
        if obstacle.owner != self.owner or obstacle.source not in self.obstacle_sources:
            return False
        # Only the terminal-adjacent segment can use this physical access.
        if min(abs(first.x-self.start.x)+abs(first.y-self.start.y),
               abs(second.x-self.start.x)+abs(second.y-self.start.y)) > COORDINATE_TOLERANCE:
            return False
        dx, dy = _VECTORS[self.side]
        if dx:
            if abs(first.y-self.start.y) > COORDINATE_TOLERANCE or abs(second.y-self.start.y) > COORDINATE_TOLERANCE:
                return False
            low = max(min(first.x, second.x), obstacle.bounds.xmin)
            high = min(max(first.x, second.x), obstacle.bounds.xmax)
            return low >= min(self.start.x,self.end.x)-COORDINATE_TOLERANCE and high <= max(self.start.x,self.end.x)+COORDINATE_TOLERANCE
        if abs(first.x-self.start.x) > COORDINATE_TOLERANCE or abs(second.x-self.start.x) > COORDINATE_TOLERANCE:
            return False
        low = max(min(first.y, second.y), obstacle.bounds.ymin)
        high = min(max(first.y, second.y), obstacle.bounds.ymax)
        return low >= min(self.start.y,self.end.y)-COORDINATE_TOLERANCE and high <= max(self.start.y,self.end.y)+COORDINATE_TOLERANCE


def terminal_access(*, owner: Key, source: Key, side: str, point: Point,
                    body_bounds: Bounds, obstacle_sources: tuple[Key, ...],
                    stub: float, clearance: float) -> TerminalAccess:
    """Measure a selected terminal ray using explicitly named physical ink."""
    dx, dy = _VECTORS[side]
    end = point.translated(dx*stub, dy*stub)
    if dx:
        end = Point(max(end.x,body_bounds.xmax+clearance) if dx>0 else min(end.x,body_bounds.xmin-clearance),point.y)
    else:
        end = Point(point.x,max(end.y,body_bounds.ymax+clearance) if dy>0 else min(end.y,body_bounds.ymin-clearance))
    return TerminalAccess(owner, source, side, point, end, obstacle_sources)


__all__ = ["OwnedObstacle", "TerminalAccess", "terminal_access"]
