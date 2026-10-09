"""Explicit drawing recipes for the engineer course's named physical Plans.

This is ordinary example code, not a package layout service. Each recipe owns
its frames, individual body poses, wiring bends and junction arms. Shared code
only resolves fresh captured refs and translates measured native contacts;
there is no packing, route search, unknown-Plan completion or renderer fallback.
"""
from __future__ import annotations

import json

from scnsim import (
    DiagramCaption, DiagramEndpoint, DiagramJunction, DiagramLeader,
    DiagramPose, DiagramRoute, DiagramSide, DiagramJump, SchematicLayout,
    SchematicScopeLayout,
)


class _Scope:
    """Mechanical construction of one authored scope declaration."""

    def __init__(self, preparation, path, frame):
        self.preparation = preparation
        self.inventory = next(row for row in preparation.scopes
                              if row.ref.scope_path == tuple(path))
        self.frame = frame
        self.poses = {}
        self.boundaries = {}
        self.ports = {}
        self.grounds = {}
        self.children = {}
        self.points = {}
        self.junctions = []
        self.routes = []
        self.jumps = []
        self.label_positions = {}

    def ref(self, kind, path=(), id=None):
        return next(ref for ref in self.preparation.refs
                    if (record := json.loads(ref.key))["kind"] == kind
                    and tuple(record.get("path", ())) == tuple(path)
                    and (id is None or record.get("id") == id))

    def pin(self, path, id):
        return next(row["ref"] for row in self.inventory.contacts
                    if row["source"]["kind"] == "physical_terminal"
                    and tuple(row["source"]["path"]) == tuple(path)
                    and row["source"]["pin"] == id)

    def boundary(self, path, id):
        return self.ref("contact", path, id)

    def native(self, target, at, rotation=0, *, contact=None):
        fragment = self.preparation.native[target, rotation]
        # The recipe chooses the contact location; measured body extent is
        # never stretched to reach it. This translation creates no new geometry.
        anchor = fragment.contacts[target if contact is None and target.kind in
                                   ("port", "ground") else contact] if contact is not None or target.kind in ("port", "ground") else None
        origin = at if anchor is None else (at[0] - anchor["point"].x,
                                            at[1] - anchor["point"].y)
        pose = DiagramPose(origin, rotation)
        collection = self.ports if target.kind == "port" else self.grounds if target.kind == "ground" else self.poses
        collection[target] = pose
        for ref, row in fragment.contacts.items():
            self.points[ref] = (row["point"].x + origin[0], row["point"].y + origin[1])
        return target

    def child(self, child_scope, layout, origin):
        measured = self.preparation.measure_scope(child_scope, layout)
        self.children[child_scope] = layout
        self.poses[child_scope] = DiagramPose(origin)
        for ref, row in measured.contacts.items():
            self.points[ref] = (row["point"].x + origin[0], row["point"].y + origin[1])

    def expose(self, ref, at):
        self.boundaries[ref] = at
        self.points[ref] = at

    def junction(self, id, at, *arms):
        self.junctions.append(DiagramJunction(id, at, tuple(DiagramSide(arm) for arm in arms)))
        # Current intrinsic junction arm length; this is an explicit course
        # coordinate, not a tunable package clearance or geometry search.
        for arm, delta in {"left": (-.55, 0), "right": (.55, 0),
                           "top": (0, .55), "bottom": (0, -.55)}.items():
            if arm in arms:
                self.points[id, arm] = (at[0] + delta[0], at[1] + delta[1])

    def wire(self, id, start, end, *bends):
        endpoint = lambda ref: DiagramEndpoint(*ref) if isinstance(ref, tuple) else DiagramEndpoint(ref)
        self.routes.append(DiagramRoute(id, endpoint(start), endpoint(end),
                                       (self.points[start], *bends, self.points[end])))

    def labels(self, **positions):
        # Entries are explicit label text -> (actual contact, knee, end).
        self.label_positions.update(positions)

    def finish(self):
        captions = []
        for row in self.inventory.captions:
            target = row["target"]
            if target in self.poses or target in self.ports or target in self.grounds:
                pose = {**self.poses, **self.ports, **self.grounds}[target]
                value = self.preparation.native[target, pose.rotation].scene
                for part in row["text_path"]:
                    value = value[part] if isinstance(part, int) else getattr(value, part)
                at = (value.origin.x + pose.origin[0], value.origin.y + pose.origin[1])
            elif json.loads(row["ref"].key)["source_kind"] == "scope_title":
                at = (self.frame[0] + 2, self.frame[3] - 2)
            else:
                # Every exposed label has its own explicitly declared boundary.
                x, y = self.points[target]
                at = (x - (row["run"].bounds.xmax - row["run"].bounds.xmin) - .8 if x == self.frame[2] else x + .8, y - 1 if x == self.frame[2] else y + .8)
            captions.append(DiagramCaption(row["ref"], at))
        leaders = []
        for row in self.inventory.analysis_labels:
            contact, knee, end = self.label_positions[row["text"]]
            leaders.append(DiagramLeader(row["ref"], contact, knee, end))
        return SchematicScopeLayout(
            scope=self.inventory.ref, frame=self.frame, poses=self.poses,
            boundaries=self.boundaries, ports=self.ports, grounds=self.grounds,
            junctions=tuple(self.junctions), routes=tuple(self.routes), jumps=tuple(self.jumps),
            captions=tuple(captions), leaders=tuple(leaders), couplings=(),
            children=self.children,
        )


def _lc(preparation, path, *, boundary_names=(), bus_label=None):
    """The course's two explicit parallel LC branches, with a left boundary."""
    s = _Scope(preparation, path, (0, 0, 28, 20))
    c = s.ref("occurrence", (*path, "capacitor"))
    l = s.ref("occurrence", (*path, "inductor"))
    c1, c2 = s.pin((*path, "capacitor"), "terminal_1"), s.pin((*path, "capacitor"), "terminal_2")
    l1, l2 = s.pin((*path, "inductor"), "terminal_1"), s.pin((*path, "inductor"), "terminal_2")
    # These named source declarations connect terminal_1 to their signal bus
    # and terminal_2 to ground. Captured refs retain that physical authority.
    cs, cg, ls, lg = c1, c2, l1, l2
    s.native(c, (8, 8), 270, contact=cs)
    s.native(l, (20, 8), 270, contact=ls)
    if boundary_names:
        s.junction("lc_signal", (14, 12), "left", "right", "bottom")
        s.wire("capacitor_signal", cs, ("lc_signal", "bottom"), (8, 10), (14, 10))
        s.wire("inductor_signal", ls, ("lc_signal", "right"), (20, 12))
        first = s.boundary(path, boundary_names[0]); s.expose(first, (0, 12))
        s.wire("public_signal", first, ("lc_signal", "left"))
        if len(boundary_names) == 2:
            # The second published alias is distinct visible boundary incidence.
            second = s.boundary(path, boundary_names[1]); s.expose(second, (28, 12))
            # One additional explicit junction serves this extra actual contact.
            s.junctions[-1] = DiagramJunction("lc_signal", (14, 12),
                                             tuple(DiagramSide(a) for a in ("left", "right", "bottom", "top")))
            s.points["lc_signal", "top"] = (14, 12.55)
            s.wire("alternate_public_signal", second, ("lc_signal", "top"), (25, 12), (25, 16), (14, 16))
    else:
        s.wire("parallel_signal", cs, ls, (8, 12), (20, 12))
    for name, pin, at in (("capacitor", cg, (8, 2)),
                          ("inductor", lg, (20, 2))):
        ground = s.ref("ground", (*path, name), "terminal_2")
        s.native(ground, at, 270, contact=ground)
        s.wire(name + "_ground", pin, ground)
    if bus_label is not None:
        s.labels(**{bus_label: (cs, (8, 9), (4, 9))})
    return s.finish()


def _coupled_lc(preparation, *, coupler_name, alternate=False, bus_label="signal"):
    s = _Scope(preparation, (), (0, 0, 60, 24))
    child = s.ref("scope", ("resonator",))
    child_layout = _lc(preparation, ("resonator",),
                       boundary_names=("terminal", "alternate_terminal") if alternate else ("terminal",),
                       bus_label="terminal_node" if alternate else bus_label)
    s.child(child, child_layout, (26, 0))
    coupling = s.ref("occurrence", (coupler_name,))
    a, b = s.pin((coupler_name,), "terminal_1"), s.pin((coupler_name,), "terminal_2")
    s.native(coupling, (12, 12))
    port = s.ref("port", (), "signal_in"); s.native(port, (7, 12), 180)
    terminal = s.boundary(("resonator",), "terminal")
    s.wire("measurement_to_coupler", port, a)
    s.wire("coupler_to_child", b, terminal)
    s.labels(signal_boundary=(a, (12, 10), (10, 10)),
             resonator_node=(terminal, (26, 10), (24, 10)))
    return SchematicLayout(preparation, s.finish())


def _four_arm(preparation):
    s = _Scope(preparation, (), (0, 0, 60, 60))
    s.junction("central_cross", (30, 30), "left", "right", "top", "bottom")
    # All four actual capacitor occurrences and all four native Port bodies have
    # authored cardinal poses; the native load shape is not independently bent.
    table = {
        "a": ((20, 30), 180, (12, 30), 180, "left"),
        "b": ((40, 30), 0, (48, 30), 0, "right"),
        "c": ((30, 40), 90, (30, 48), 90, "top"),
        "d": ((30, 20), 270, (30, 12), 270, "bottom"),
    }
    starts = {}
    for name, (origin, rotation, port_at, port_rotation, arm) in table.items():
        body = s.ref("occurrence", (f"capacitor_{name}",))
        a = s.pin((f"capacitor_{name}",), "terminal_1")
        b = s.pin((f"capacitor_{name}",), "terminal_2")
        s.native(body, origin, rotation)
        port = s.ref("port", (), f"port_{name}")
        s.native(port, port_at, port_rotation)
        s.wire(f"central_{name}", ("central_cross", arm), a)
        s.wire(f"outer_{name}", b, port)
        starts[name] = a
        x,y = s.points[b]
        s.labels(**{name: (b, (x, y + 2) if arm in ("left", "right") else (x + 2, y),
                             (x + 2, y + 2))})
    s.labels(central=(starts["a"], (20, 28), (22, 28)))
    return SchematicLayout(preparation, s.finish())



def _feedline(preparation, path, *, tap_name="tap", terminated=False):
    s = _Scope(preparation, path, (0, 0, 160, 35))
    left, right = (s.ref("occurrence", (*path, name)) for name in ("left", "right"))
    lh, lt = (s.pin((*path, "left"), name) for name in ("head.signal", "tail.signal"))
    rh, rt = (s.pin((*path, "right"), name) for name in ("head.signal", "tail.signal"))
    s.native(left, (15, 18), contact=lh)
    s.native(right, (90, 18), contact=rh)
    arms = ("left", "right") if terminated else ("left", "right", "bottom")
    s.junction("tap_t", (70, 18), *arms)
    s.wire("left_to_tap", lt, ("tap_t", "left"))
    s.wire("tap_to_right", ("tap_t", "right"), rh)
    if terminated:
        input_port = s.ref("port", path, "input")
        output_port = s.ref("port", path, "output")
        s.native(input_port, (5, 18), 180); s.native(output_port, (155, 18))
        s.wire("input_to_left", input_port, lh)
        s.wire("right_to_output", rt, output_port)
        s.labels(input=(input_port, (5, 15), (8, 15)),
                 middle=(lt, (s.points[lt][0], 15), (s.points[lt][0]+3, 15)),
                 output=(output_port, (155, 21), (152, 21)))
    else:
        input_pin = s.boundary(path, "input")
        output_pin = s.boundary(path, "output")
        tap_pin = s.boundary(path, tap_name)
        s.expose(input_pin, (0, 18)); s.expose(output_pin, (160, 18)); s.expose(tap_pin, (70, 0))
        s.wire("input_to_left", input_pin, lh)
        s.wire("right_to_output", rt, output_pin)
        s.wire("tap_to_boundary", ("tap_t", "bottom"), tap_pin)
    return s.finish()


def _feedline_readout(preparation):
    s = _Scope(preparation, (), (0, 0, 190, 100))
    feedline = s.ref("scope", ("feedline",))
    readout = s.ref("scope", ("readout",))
    s.child(feedline, _feedline(preparation, ("feedline",)), (10, 40))
    s.child(readout, _lc(preparation, ("readout",), boundary_names=("readout_node",)), (90, 0))
    coupling = s.ref("occurrence", ("feedline_readout_coupler",))
    a, b = (s.pin(("feedline_readout_coupler",), name) for name in ("terminal_1", "terminal_2"))
    s.native(coupling, (80, 32), 270)
    tap = s.boundary(("feedline",), "tap")
    node = s.boundary(("readout",), "readout_node")
    s.wire("feedline_to_coupler", tap, a)
    s.wire("coupler_to_readout", b, node, (80, 12))
    left = s.ref("port", (), "feedline_in"); s.native(left, (5, 58), 180)
    right = s.ref("port", (), "feedline_out"); s.native(right, (180, 58))
    s.wire("input_boundary", left, s.boundary(("feedline",), "input"))
    s.wire("output_boundary", s.boundary(("feedline",), "output"), right)
    s.labels(feedline_in=(left, (5, 55), (7, 55)),
             feedline_out=(right, (180, 61), (177, 61)))
    return SchematicLayout(preparation, s.finish())


def _tapped(preparation):
    s = _Scope(preparation, (), (0, 0, 180, 55))
    child = s.ref("scope", ("feedline",))
    s.child(child, _feedline(preparation, ("feedline",)), (10, 5))
    return SchematicLayout(preparation, s.finish())


def _mtl(preparation):
    s = _Scope(preparation, (), (0, 0, 160, 100))
    line = s.ref("occurrence", ("coupled",)); s.native(line, (40, 40))
    # Four explicitly selected signal contacts, not a split into scalar lines.
    for name, pin, rotation, dx, dy in (
        ("readout_head", "head.readout", 180, -15, -4),
        ("filter_head", "head.filter", 180, -15, -4),
        ("readout_tail", "tail.readout", 0, 15, 4),
        ("filter_tail", "tail.filter", 0, 15, 4),
    ):
        contact = s.pin(("coupled",), pin)
        x, y = s.points[contact]
        port = s.ref("port", (), name)
        s.native(port, (x+dx, y), rotation)
        s.wire(name+"_attachment", port, contact)
        s.labels(**{name: (port, (x+dx, y+dy), (x+dx+3, y+dy))})
    return SchematicLayout(preparation, s.finish())


def _floating(preparation):
    s = _Scope(preparation, ("floating",), (0, 0, 80, 50))
    pins = {}
    for name, origin, rotation in (
        ("mutual_cap", (20, 30), 0), ("mutual_ind", (20, 20), 0),
        ("plus_shunt", (10, 10), 270), ("minus_shunt", (50, 10), 270),
    ):
        target = s.ref("occurrence", ("floating", name))
        s.native(target, origin, rotation)
        pins[name] = tuple(s.pin(("floating", name), id) for id in ("terminal_1", "terminal_2"))
    s.junction("plus_upper", (10, 30), "left", "right", "top", "bottom")
    s.junction("plus_lower", (10, 20), "right", "top", "bottom")
    s.junction("minus_upper", (50, 30), "left", "right", "top", "bottom")
    s.junction("minus_lower", (50, 20), "left", "top", "bottom")
    s.wire("plus_capacitor", ("plus_upper", "right"), pins["mutual_cap"][0])
    s.wire("plus_inductor", ("plus_lower", "right"), pins["mutual_ind"][0])
    s.wire("plus_junctions", ("plus_upper", "bottom"), ("plus_lower", "top"))
    s.wire("plus_shunt_signal", ("plus_lower", "bottom"), pins["plus_shunt"][0])
    s.wire("minus_capacitor", pins["mutual_cap"][1], ("minus_upper", "left"))
    s.wire("minus_inductor", pins["mutual_ind"][1], ("minus_lower", "left"))
    s.wire("minus_junctions", ("minus_upper", "bottom"), ("minus_lower", "top"))
    s.wire("minus_shunt_signal", ("minus_lower", "bottom"), pins["minus_shunt"][0])
    boundary = {name: s.boundary(("floating",), name)
                for name in ("plus_1", "plus_2", "minus_1", "minus_2")}
    for name, at in {"plus_1": (0,30), "plus_2": (80,38),
                     "minus_1": (0,40), "minus_2": (80,30)}.items():
        s.expose(boundary[name], at)
    s.wire("plus_1_boundary", boundary["plus_1"], ("plus_upper", "left"))
    s.wire("plus_2_boundary", boundary["plus_2"], ("plus_upper", "top"), (10,38))
    s.wire("minus_1_boundary", boundary["minus_1"], ("minus_upper", "top"), (50,40))
    s.wire("minus_2_boundary", ("minus_upper", "right"), boundary["minus_2"])
    s.jumps.append(DiagramJump("minus_1_boundary", (50,38), "plus_2_boundary"))
    # Each actual source-grounded shunt has its own declared return glyph.
    ground_poses = {("floating", "plus_shunt"): (10,2),
                    ("floating", "minus_shunt"): (50,2)}
    for ground in s.inventory.grounds:
        path = tuple(json.loads(ground.key)["path"])
        s.native(ground, ground_poses[path], 270, contact=ground)
        s.wire("ground_"+path[-1], pins[path[-1]][1], ground)
    return s.finish()


def _capstone(preparation):
    s = _Scope(preparation, (), (0,0,270,110))
    feedline = s.ref("scope", ("feedline",))
    readout = s.ref("scope", ("readout",))
    floating = s.ref("scope", ("floating",))
    s.child(feedline, _feedline(preparation, ("feedline",)), (10,60))
    s.child(readout, _lc(preparation, ("readout",), boundary_names=("readout_node",)), (70,0))
    s.child(floating, _floating(preparation), (125,0))
    pins = {}
    for name, origin, rotation in (
        ("feedline_readout_coupler", (80,55), 270),
        ("readout_to_floating_plus", (110,35), 0),
        ("readout_to_floating_minus", (110,25), 0),
    ):
        target = s.ref("occurrence", (name,)); s.native(target, origin, rotation)
        pins[name] = tuple(s.pin((name,), id) for id in ("terminal_1", "terminal_2"))
    s.junction("readout_attachment", (60,35), "left", "right", "bottom")
    s.junction("floating_split", (90,35), "left", "right", "bottom")
    s.junction("floating_minus_turn", (90,25), "top", "right")
    s.wire("feedline_coupler", s.boundary(("feedline",), "tap"), pins["feedline_readout_coupler"][0])
    s.wire("readout_attachment_left", pins["feedline_readout_coupler"][1],
           ("readout_attachment", "left"), (80,45), (50,45), (50,35))
    node = s.boundary(("readout",), "readout_node")
    s.wire("readout_attachment_bottom", ("readout_attachment", "bottom"), node, (60,12))
    s.wire("readout_attachment_right", ("readout_attachment", "right"), ("floating_split", "left"))
    s.wire("plus_coupler_start", ("floating_split", "right"), pins["readout_to_floating_plus"][0])
    s.wire("minus_turn", ("floating_split", "bottom"), ("floating_minus_turn", "top"))
    s.wire("minus_coupler_start", ("floating_minus_turn", "right"), pins["readout_to_floating_minus"][0])
    s.wire("plus_coupler_child", pins["readout_to_floating_plus"][1],
           s.boundary(("floating",), "plus_1"), (115,35), (115,30))
    s.wire("minus_coupler_child", pins["readout_to_floating_minus"][1],
           s.boundary(("floating",), "minus_1"), (120,25), (120,40))
    s.jumps.append(DiagramJump("minus_coupler_child", (120,30), "plus_coupler_child"))
    port_specs = (("feedline_in", (5,78), 180, ("feedline",), "input"),
                  ("feedline_out", (180,78), 0, ("feedline",), "output"),
                  ("floating_probe_plus", (240,38), 0, ("floating",), "plus_2"),
                  ("floating_probe_minus", (225,30), 0, ("floating",), "minus_2"))
    ports = {}
    for name, at, rotation, child_path, boundary in port_specs:
        port = s.ref("port", (), name); s.native(port, at, rotation); ports[name] = port
        s.wire(name+"_attachment", s.boundary(child_path, boundary), port)
    s.labels(feedline_in=(ports["feedline_in"], (5,75), (7,75)),
             feedline_out=(ports["feedline_out"], (180,81), (177,81)),
             readout_node=(node, (65,12), (65,9)),
             floating_plus=(ports["floating_probe_plus"], (240,41), (244,41)),
             floating_minus=(ports["floating_probe_minus"], (225,33), (229,33)))
    return SchematicLayout(preparation, s.finish())

def engineer_layout(preparation, example):
    """Build only a named course declaration from this preparation's refs."""
    recipes = {
        "primitive_resonator": lambda: _coupled_lc(preparation, coupler_name="coupling_cap", bus_label=None),
        "coupled_lc": lambda: _coupled_lc(preparation, coupler_name="coupling_capacitor"),
        "reusable_lc": lambda: _coupled_lc(preparation, coupler_name="coupling_cap", alternate=True),
        "readout_illustration": lambda: SchematicLayout(preparation, _lc(preparation, (), bus_label="node")),
        "four_arm": lambda: _four_arm(preparation),
        "feedline_readout": lambda: _feedline_readout(preparation),
        "feedline_illustration": lambda: SchematicLayout(preparation, _feedline(preparation, (), terminated=True)),
        "tapped_feedline": lambda: _tapped(preparation),
        "ordered_mtl": lambda: _mtl(preparation),
        "four_port_capstone": lambda: _capstone(preparation),
    }
    return recipes[example]()
