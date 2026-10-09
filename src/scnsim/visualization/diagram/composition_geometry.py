"""Realize declared schematic geometry from immutable measured fragments.

This owner translates selected native variants and rigidly adopts complete
children. It never packs blocks, chooses routes, creates missing annotations,
or changes electrical topology. Geometry/access facts check visible placement;
the independent ink reconstruction remains the electrical authority.
"""
from __future__ import annotations

from dataclasses import replace
from itertools import pairwise
from collections.abc import Mapping
import json

from ...errors import SCNSimValidationError
from ..schematic import DiagramRef, ScopeMeasurement, _scope_layout_sha256
from .composition_obstacles import OwnedObstacle, TerminalAccess
from .metrics import DEFAULT_METRICS
from .native import (emit_native, jump_arc, junction_mark,
                     measured_obstacles, transform_fragment, transform_text)
from .scene import (Bounds, BoundarySite, ConductivePolyline,
                    COORDINATE_TOLERANCE, GuideMark, NeutralScene, Path, Point,
                    SubsystemRegion)

VECTORS = {"right": (1, 0), "top": (0, 1), "left": (-1, 0), "bottom": (0, -1)}
OPPOSITE = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}


def _fail(message, **evidence):
    return SCNSimValidationError(message, stage="schematic_layout", evidence=evidence)


def _same(first, second):
    return abs(first.x - second.x) + abs(first.y - second.y) <= COORDINATE_TOLERANCE


def _direction(first, second):
    dx, dy = second.x - first.x, second.y - first.y
    if abs(dy) <= COORDINATE_TOLERANCE and abs(dx) > COORDINATE_TOLERANCE:
        return "right" if dx > 0 else "left"
    if abs(dx) <= COORDINATE_TOLERANCE and abs(dy) > COORDINATE_TOLERANCE:
        return "top" if dy > 0 else "bottom"
    raise _fail("declared segment must have positive cardinal length", first=first, second=second)


def _on(point, first, second):
    return (min(first.x, second.x) - COORDINATE_TOLERANCE <= point.x <= max(first.x, second.x) + COORDINATE_TOLERANCE
        and min(first.y, second.y) - COORDINATE_TOLERANCE <= point.y <= max(first.y, second.y) + COORDINATE_TOLERANCE
        and (abs(first.x - second.x) <= COORDINATE_TOLERANCE and abs(point.x - first.x) <= COORDINATE_TOLERANCE
             or abs(first.y - second.y) <= COORDINATE_TOLERANCE and abs(point.y - first.y) <= COORDINATE_TOLERANCE))


def _intersection(a, b, c, d):
    """Exact cardinal intersection, retaining a positive overlap interval."""
    av, cv = abs(a.x-b.x) <= COORDINATE_TOLERANCE, abs(c.x-d.x) <= COORDINATE_TOLERANCE
    if av == cv:
        if abs((a.x-c.x) if av else (a.y-c.y)) > COORDINATE_TOLERANCE:
            return None
        low = max(min(a.y,b.y),min(c.y,d.y)) if av else max(min(a.x,b.x),min(c.x,d.x))
        high = min(max(a.y,b.y),max(c.y,d.y)) if av else min(max(a.x,b.x),max(c.x,d.x))
        if high < low-COORDINATE_TOLERANCE:
            return None
        return (Point(a.x,low),Point(a.x,high)) if av else (Point(low,a.y),Point(high,a.y))
    vertical, horizontal = ((a,b),(c,d)) if av else ((c,d),(a,b))
    at = Point(vertical[0].x,horizontal[0].y)
    return (at,at) if _on(at,*vertical) and _on(at,*horizontal) else None


def _check_path(points, *, route_id, scope):
    for first, second in pairwise(points):
        _direction(first, second)
    segments = tuple(pairwise(points))
    for index, first in enumerate(segments):
        for other_index in range(index + 1, len(segments)):
            relation = _intersection(*first, *segments[other_index])
            if relation is not None and (other_index > index + 1 or not _same(*relation)):
                raise _fail("declared route crosses or retraces itself", scope=scope,
                            route=route_id, intersection=relation)


def _region(bounds, header):
    """The existing rounded ownership contour, in the supplied exact frame."""
    r = DEFAULT_METRICS.label_clearance
    x0,y0,x1,y1 = bounds.xmin,bounds.ymin,bounds.xmax,bounds.ymax
    outline = Path((Point(x0+r,y0),Point(x1-r,y0),Point(x1,y0),Point(x1,y0),
        Point(x1,y0+r),Point(x1,y1-r),Point(x1,y1),Point(x1,y1),Point(x1-r,y1),
        Point(x0+r,y1),Point(x0,y1),Point(x0,y1),Point(x0,y1-r),Point(x0,y0+r),
        Point(x0,y0),Point(x0,y0),Point(x0+r,y0)), "region", True,
        (1,2,4,4,4,2,4,4,4,2,4,4,4,2,4,4,4))
    return SubsystemRegion(bounds, outline, header,
        header.origin if header is not None else Point(x0,y1), "root")


def _exact_keys(actual, expected, *, label, scope):
    actual, expected = tuple(actual), set(expected)
    if len(actual) != len(set(actual)):
        raise _fail(f"explicit {label} target is declared twice",scope=scope)
    actual = set(actual)
    if actual != expected:
        raise _fail(f"explicit {label} declarations do not match captured inventory",
            scope=scope, missing=tuple(set(expected)-set(actual)), extra=tuple(set(actual)-set(expected)))


def _emit_routes(routes, jumps, contacts, junctions, *, obstacles, access, frame, scope):
    """Validate complete authored paths, then split only explicitly named jumps."""
    from .routing import _segment_clear

    paths, endpoints = {}, {}
    for route in routes:
        if route.id in paths:
            raise _fail("duplicate local route id", scope=scope, route=route.id)
        ends = []
        for endpoint in (route.start, route.end):
            if isinstance(endpoint.target, DiagramRef):
                if endpoint.contact is not None:
                    raise _fail("captured contact endpoint does not select a junction arm", route=route.id)
                if endpoint.target not in contacts:
                    raise _fail("route endpoint is not a contact in this realized scope",scope=scope,route=route.id,target=endpoint.target)
                ends.append(contacts[endpoint.target])
            else:
                if (endpoint.target,endpoint.contact) not in junctions:
                    raise _fail("route endpoint does not name a declared junction arm",scope=scope,route=route.id,target=endpoint.target,arm=endpoint.contact)
                ends.append(junctions[endpoint.target, endpoint.contact])
        points = tuple(Point(*value) for value in route.waypoints)
        if len(points) < 2 or not _same(points[0], ends[0]["point"]) or not _same(points[-1], ends[1]["point"]):
            raise _fail("declared route endpoints differ from actual contacts", scope=scope, route=route.id)
        _check_path(points, route_id=route.id, scope=scope)
        if any(not frame.contains(Bounds.around((point,))) for point in points):
            raise _fail("declared route leaves scope frame", scope=scope, route=route.id)
        for row,a,b in ((ends[0],points[0],points[1]),(ends[1],points[-1],points[-2])):
            if _direction(a,b) != row["side"]:
                raise _fail("declared route does not leave its actual terminal side", scope=scope, route=route.id)
        legal = tuple(fact for row in ends for fact in access.get(row["source"], ()))
        for a,b in pairwise(points):
            for obstacle in obstacles:
                # Conductive intersections are checked below and by independent
                # ink reconstruction; they are not rectangular body barriers.
                if obstacle.kind == "conductive":
                    continue
                bounds = obstacle.bounds
                gap = DEFAULT_METRICS.obstacle_clearance
                expanded = replace(obstacle, bounds=Bounds(bounds.xmin-gap,bounds.ymin-gap,bounds.xmax+gap,bounds.ymax+gap))
                if not _segment_clear(a,b,(expanded.bounds,)) and not any(fact.permits(expanded,a,b) for fact in legal):
                    raise _fail("declared route intersects occupied geometry", scope=scope, route=route.id,
                        first=a, second=b, obstacle_source=obstacle.source, obstacle_kind=obstacle.kind)
        paths[route.id], endpoints[route.id] = points, ends
    declared = {}
    for jump in jumps:
        at = Point(*jump.at)
        if jump.route_id == jump.over_route_id:
            raise _fail("jump cannot cross its own route", route=jump.route_id)
        key = frozenset((jump.route_id,jump.over_route_id)),at
        if key in declared:
            raise _fail("crossing has multiple jump declarations", scope=scope, at=at)
        first, second = paths[jump.route_id],paths[jump.over_route_id]
        crossing = [(i,j) for i,(a,b) in enumerate(pairwise(first)) for j,(c,d) in enumerate(pairwise(second))
                    if (hit := _intersection(a,b,c,d)) is not None and _same(*hit) and _same(hit[0],at)
                    and (_direction(a,b) in ("left","right")) != (_direction(c,d) in ("left","right"))]
        # Strict interior crossing prevents an arc from swallowing a terminal
        # or junction arm; the gap remains the existing native measurement.
        if len(crossing) != 1:
            raise _fail("declared jump does not name one transverse crossing", scope=scope, at=at)
        i,j = crossing[0]
        if any(_same(at,end) for end in (*first[i:i+2],*second[j:j+2])):
            raise _fail("declared jump lies on a route vertex", scope=scope, at=at)
        declared[key] = jump
    names = tuple(paths)
    for index,name in enumerate(names):
        for other in names[index+1:]:
            shared = {row["point"] for row in endpoints[name] for peer in endpoints[other]
                      if row["source"] == peer["source"]}
            for a,b in pairwise(paths[name]):
                for c,d in pairwise(paths[other]):
                    hit = _intersection(a,b,c,d)
                    if hit is None:
                        continue
                    if not _same(*hit):
                        raise _fail("declared routes overlap or retrace", scope=scope, routes=(name,other), intersection=hit)
                    at = hit[0]
                    if any(_same(at,point) for point in shared):
                        continue
                    if (frozenset((name,other)),at) not in declared:
                        raise _fail("crossing requires an explicit junction or jump", scope=scope, routes=(name,other), at=at)
    wires, arcs = [], []
    for name,points in paths.items():
        for a,b in pairwise(points):
            side = _direction(a,b)
            dx,dy = VECTORS[side]
            selected = sorted((jump for jump in jumps if jump.route_id == name and _on(Point(*jump.at),a,b)),
                              key=lambda jump: abs(jump.at[0]-a.x)+abs(jump.at[1]-a.y))
            cursor = a
            for jump in selected:
                at = Point(*jump.at)
                half = DEFAULT_METRICS.jump_gap/2
                start,end = at.translated(-dx*half,-dy*half),at.translated(dx*half,dy*half)
                if not _on(start,cursor,b) or not _on(end,cursor,b) or _same(start,cursor) or _same(end,b):
                    raise _fail("native jump span does not fit its declared segment", scope=scope, route=name, at=at)
                wires.append(ConductivePolyline((cursor,start)))
                arc = jump_arc(start,end)
                for obstacle in obstacles:
                    if obstacle.kind != "conductive" and arc.path.bounds.overlaps(obstacle.bounds):
                        raise _fail("declared jump intersects occupied geometry", scope=scope, route=name, source=obstacle.source)
                arcs.append(arc)
                cursor = end
            wires.append(ConductivePolyline((cursor,b)))
    return tuple(wires),tuple(arcs)


def realize_explicit_scope(preparation, scope_layout, *, native_measurements, children: Mapping[DiagramRef, ScopeMeasurement]):
    """Emit exactly one complete declared scope, without automatic geometry."""
    inventory = next(row for row in preparation.scopes if row.ref == scope_layout.scope)
    scope = inventory.ref.scope_path
    frame = Bounds(*scope_layout.frame)
    native_targets = inventory.physical_occurrences
    _exact_keys(scope_layout.poses, (*native_targets,*inventory.child_scopes), label="poses", scope=scope)
    _exact_keys(children, inventory.child_scopes, label="measured children", scope=scope)
    _exact_keys(scope_layout.ports, inventory.ports, label="ports", scope=scope)
    _exact_keys(scope_layout.grounds, inventory.grounds, label="grounds", scope=scope)
    boundary_rows = tuple(row for row in inventory.contacts
                          if row["boundary"] and row["source"]["kind"] == "scope_exposure")
    _exact_keys(scope_layout.boundaries, (row["ref"] for row in boundary_rows), label="boundaries", scope=scope)
    caption_rows = {row["ref"]: row for row in inventory.captions}
    captions = {row.target: Point(*row.at) for row in scope_layout.captions}
    if len(captions) != len(scope_layout.captions):
        raise _fail("caption target is declared twice", scope=scope)
    _exact_keys(captions, caption_rows, label="captions", scope=scope)
    payload = {name: [] for name in ("symbols","boxes","ports","conductive","jumps","guides","text","regions","boundary_sites","node_marks")}
    contacts, obstacles, access, markers = {}, [], {}, {}

    def moved_markers(records, origin, rotation):
        return {sign: {"point": row["point"].rotated(rotation).translated(origin.x,origin.y),
            "path": row["path"].rotated(rotation).translated(origin.x,origin.y)}
                for sign,row in records.items()}

    def adopt(scene):
        for name, values in payload.items():
            values.extend(getattr(scene,name) if name != "regions" else
                          (row for row in scene.regions if row.kind != "root"))

    placements = {**{ref: scope_layout.poses[ref] for ref in native_targets},
                  **dict(scope_layout.ports),**dict(scope_layout.grounds)}
    for target,pose in placements.items():
        measurement = native_measurements[target,pose.rotation]
        moved_captions = {tuple(row["text_path"]): captions[ref] for ref,row in caption_rows.items()
                          if row["target"] == target}
        scene = emit_native(measurement,pose=pose,captions=moved_captions)
        adopt(scene)
        local = []
        for ref,row in measurement.contacts.items():
            moved = {**dict(row), "point": row["point"].translated(*pose.origin)}
            contacts[ref] = moved
            local.append((ref,moved["point"],moved["side"]))
        ink, rays = measured_obstacles(scene,owner=target,scope=scope,contacts=local,metrics=DEFAULT_METRICS)
        obstacles.extend(ink)
        for ray in rays:
            access.setdefault(ray.source[0],[]).append(ray)
        for branch,records in measurement.markers.items():
            markers[target,branch] = moved_markers(records,Point(*pose.origin),0)
    for target,measurement in children.items():
        if measurement.scope != target or measurement.preparation_sha256 != preparation.identity["preparation_sha256"]:
            raise _fail("child measurement has a different capture binding", scope=scope, child=target)
        pose = scope_layout.poses[target]
        origin = Point(*pose.origin)
        scene = transform_fragment(measurement.fragment,origin=origin,rotation=pose.rotation)
        adopt(scene)
        payload["regions"].extend(replace(region,kind="composite") for region in scene.regions if region.kind == "root")
        sides = ("right","top","left","bottom")
        for ref,row in measurement.contacts.items():
            contacts[ref] = {**dict(row),"point": row["point"].rotated(pose.rotation).translated(*pose.origin),
                "side": sides[(sides.index(row["side"])+pose.rotation//90)%4]}
        obstacles.extend(row.transformed(origin=origin,rotation=pose.rotation) for row in measurement.obstacles)
        for row in measurement.terminal_access:
            transformed = row.transformed(origin=origin,rotation=pose.rotation)
            access.setdefault(transformed.source[0],[]).append(transformed)
        for key,records in measurement.markers.items():
            markers[key] = moved_markers(records,origin,pose.rotation)
    for row in boundary_rows:
        ref = row["ref"]
        at = Point(*scope_layout.boundaries[ref])
        side = row["side"]
        # Boundary side is an explicit captured contact fact; only the position
        # is authored. No owner is inferred from containment.
        if side is None:
            edges = tuple(name for name, actual, expected in (
                ("left",at.x,frame.xmin),("right",at.x,frame.xmax),
                ("bottom",at.y,frame.ymin),("top",at.y,frame.ymax))
                if abs(actual-expected)<=COORDINATE_TOLERANCE)
            if len(edges) != 1:
                raise _fail("boundary point does not select one frame side",scope=scope,ref=ref,edges=edges)
            side = edges[0]
        axis = frame.xmin if side == "left" else frame.xmax if side == "right" else frame.ymin if side == "bottom" else frame.ymax
        actual = at.x if side in ("left","right") else at.y
        if abs(actual-axis)>COORDINATE_TOLERANCE or not frame.contains(Bounds.around((at,))):
            raise _fail("scope boundary contact does not lie on its declared side", scope=scope,ref=ref)
        contacts[ref] = {"point":at,"side":OPPOSITE[side],"source":ref.key}
        payload["boundary_sites"].append(BoundarySite(at,None))
    junctions = {}
    for junction in scope_layout.junctions:
        center = Point(*junction.center)
        if junction.id in {key[0] for key in junctions}:
            raise _fail("duplicate local junction id",scope=scope,junction=junction.id)
        sides = tuple(str(getattr(side,"value",side)) for side in junction.arms)
        if len(set(sides)) != len(sides) or len(sides)<2:
            raise _fail("junction requires distinct declared cardinal arms",scope=scope,junction=junction.id)
        for side in sides:
            dx,dy=VECTORS[side]
            tip=center.translated(dx*DEFAULT_METRICS.terminal_stub,dy*DEFAULT_METRICS.terminal_stub)
            wire=ConductivePolyline((center,tip))
            payload["conductive"].append(wire)
            obstacles.append(OwnedObstacle((inventory.ref.key,),scope,"conductive",
                (inventory.ref.key,"junction",junction.id,side),wire.bounds))
            junctions[junction.id,side]={"point":tip,"side":side,"source":(junction.id,side)}
        if len(sides)>=3:
            mark=junction_mark(center)
            payload["node_marks"].append(mark)
            for index,stroke in enumerate(mark.paths):
                obstacles.append(OwnedObstacle((inventory.ref.key,),scope,"conductive",
                    (inventory.ref.key,"junction",junction.id,"mark",index),stroke.bounds))
    header = None
    for ref,row in caption_rows.items():
        if row["target"] in placements:
            continue
        measured = row["run"]
        run = transform_text(measured,origin=Point(captions[ref].x-measured.origin.x,
                                                captions[ref].y-measured.origin.y))
        kind = json.loads(ref.key)["source_kind"]
        # Only child titles identify ownership contours. The root title remains
        # visible text beside an unlabelled root envelope, as audited by ink.
        if kind == "scope_title" and inventory.parent is not None:
            header = run
        elif kind == "boundary":
            at = contacts[row["target"]]["point"]
            index = next(index for index,site in enumerate(payload["boundary_sites"]) if site.point == at)
            payload["boundary_sites"][index] = replace(payload["boundary_sites"][index],visible_label=run)
        else:
            payload["text"].append(run)
        obstacles.append(OwnedObstacle((ref.key,),scope,"text",(ref.key,),run.bounds))
    analysis_rows = {row["ref"]:row for row in inventory.analysis_labels}
    _exact_keys((row.target for row in scope_layout.leaders),analysis_rows,label="analysis leaders",scope=scope)
    for leader in scope_layout.leaders:
        row = analysis_rows[leader.target]
        if leader.contact not in row["contacts"]:
            raise _fail("analysis leader attaches to a different captured contact",scope=scope,target=leader.target)
        start=contacts[leader.contact]["point"]
        knee,end=Point(*leader.knee),Point(*leader.end)
        _check_path((start,knee,end),route_id=leader.target.key,scope=scope)
        measured=row["run"]
        run=transform_text(measured,origin=Point(end.x-measured.origin.x,end.y-measured.origin.y))
        guide=GuideMark("analysis_label",(Path((start,knee),"analysis-label"),Path((knee,end),"analysis-label")),run,(start,))
        payload["guides"].append(guide)
        for index,stroke in enumerate(guide.paths):
            obstacles.append(OwnedObstacle((leader.target.key,),scope,"guide",(leader.target.key,index),stroke.bounds))
        contact=contacts[leader.contact]
        dx,dy=VECTORS[contact["side"]]
        access.setdefault(contact["source"],[]).append(TerminalAccess(
            (leader.target.key,),(leader.contact.key,),contact["side"],start,
            start.translated(dx*DEFAULT_METRICS.terminal_stub,dy*DEFAULT_METRICS.terminal_stub),
            ((leader.target.key,0),)))
        obstacles.append(OwnedObstacle((leader.target.key,),scope,"text",(leader.target.key,"text"),run.bounds))
    coupling_rows = {row["ref"]:row for row in inventory.structures if row["kind"] == "coupling"}
    _exact_keys((row.target for row in scope_layout.couplings),coupling_rows,
                label="coupling guides",scope=scope)
    marker_by_branch = {(tuple(json.loads(target.key)["path"]),branch):records
                        for (target,branch),records in markers.items()}
    for coupling in scope_layout.couplings:
        row = coupling_rows[coupling.target]
        dots = tuple(marker_by_branch[tuple(branch["path"]),branch["branch_id"]]["positive"]
                     for branch in row["branches"])
        points = tuple(Point(*value) for value in coupling.waypoints)
        if len(points)<2 or len(dots)!=2 or not _same(points[0],dots[0]["point"]) or not _same(points[-1],dots[1]["point"]):
            raise _fail("coupling guide endpoints differ from actual positive winding dots",scope=scope,target=coupling.target)
        _check_path(points,route_id=coupling.target.key,scope=scope)
        measured=row["run"]
        at=Point(*coupling.at)
        run=transform_text(measured,origin=Point(at.x-measured.origin.x,at.y-measured.origin.y))
        guide=GuideMark("coupling",(Path(points,"coupling"),*(dot["path"] for dot in dots)),
                        run,tuple(dot["point"] for dot in dots))
        payload["guides"].append(guide)
        for index,stroke in enumerate(guide.paths):
            obstacles.append(OwnedObstacle((coupling.target.key,),scope,"guide",
                (coupling.target.key,index),stroke.bounds))
        obstacles.append(OwnedObstacle((coupling.target.key,),scope,"text",(coupling.target.key,"text"),run.bounds))
        for index,ref in enumerate(row["positive_contacts"]):
            if ref not in contacts:
                # A hidden child's physical pin is never made into a parent
                # electrical endpoint merely to grant a geometry exemption.
                continue
            contact=contacts[ref]
            dx,dy=VECTORS[contact["side"]]
            access.setdefault(contact["source"],[]).append(TerminalAccess(
                (coupling.target.key,),(ref.key,),contact["side"],contact["point"],
                contact["point"].translated(dx*DEFAULT_METRICS.terminal_stub,dy*DEFAULT_METRICS.terminal_stub),
                ((coupling.target.key,index+1),)))
    wires,arcs = _emit_routes(scope_layout.routes,scope_layout.jumps,contacts,junctions,
        obstacles=obstacles,access=access,frame=frame,scope=scope)
    payload["conductive"].extend(wires)
    payload["jumps"].extend(arcs)
    payload["regions"].append(_region(frame,header))
    scene=NeutralScene(**{name:tuple(values) for name,values in payload.items()},bounds=frame)
    # Newly authored conductive/guide ink is preserved for parent ownership;
    # no table here supplies expected electrical connectivity to the auditor.
    for index,wire in enumerate(wires):
        obstacles.append(OwnedObstacle((inventory.ref.key,),scope,"conductive",(inventory.ref.key,"route",index),wire.bounds))
    for index,arc in enumerate(arcs):
        obstacles.append(OwnedObstacle((inventory.ref.key,),scope,"glyph",(inventory.ref.key,"jump",index),arc.path.bounds))
    exposed={row["ref"]:contacts[row["ref"]] for row in boundary_rows}
    scope_access=[]
    for ref,row in exposed.items():
        # An outward boundary ray is a contact fact, not an exemption for any
        # whole child body. Internal ink remains in the complete inventory.
        dx,dy=VECTORS[OPPOSITE[row["side"]]]
        scope_access.append(TerminalAccess((inventory.ref.key,),(ref.key,),OPPOSITE[row["side"]],
            row["point"],row["point"].translated(dx*DEFAULT_METRICS.terminal_stub,dy*DEFAULT_METRICS.terminal_stub),()))
        exposed[ref]={**row,"side":OPPOSITE[row["side"]]}
    digest=_scope_layout_sha256(scope_layout)
    return ScopeMeasurement(inventory.ref,preparation.identity["preparation_sha256"],digest,
        frame,exposed,tuple(obstacles),tuple(scope_access),markers,scene)
