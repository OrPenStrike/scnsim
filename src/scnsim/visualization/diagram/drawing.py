"""Read-only Schemdraw facade which serializes one immutable neutral scene."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from io import BytesIO
import json
from os import fspath
import re
from typing import Any

from .metrics import DEFAULT_METRICS
from .scene import Bounds, NeutralScene, Path, TextRun, scene_digest


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class _SvgCertificate:
    """Identity-only SVG context bound to one already-certified scene."""

    representation: str
    plan_id: str
    plan_sha256: str
    parameters_sha256: str
    connectivity_sha256: str
    semantic_sha256: str
    presentation_sha256: str
    compiled_graph_sha256: str | None
    expanded_graph_sha256: str | None

    def metadata(self) -> str:
        """Return public identity fields, never audit reconstruction payloads."""

        fields: dict[str, str] = {
            "representation": self.representation,
            "plan_id": self.plan_id,
            "plan_sha256": self.plan_sha256,
            "parameters_sha256": self.parameters_sha256,
            "connectivity_sha256": self.connectivity_sha256,
            "semantic_sha256": self.semantic_sha256,
            "presentation_sha256": self.presentation_sha256,
        }
        if self.representation == "compiled":
            assert self.compiled_graph_sha256 is not None
            assert self.expanded_graph_sha256 is not None
            fields["compiled_graph_sha256"] = self.compiled_graph_sha256
            fields["expanded_graph_sha256"] = self.expanded_graph_sha256
        return json.dumps(
            {
                "kind": "scnsim_circuit_diagram_certificate",
                "identity": fields,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def _svg_certificate_from_audit(audit: object) -> _SvgCertificate:
    """Freeze the public identity subset of a final successful audit only."""

    fields = {
        name: getattr(audit, name, None)
        for name in (
            "representation",
            "plan_id",
            "plan_sha256",
            "connectivity_sha256",
            "semantic_sha256",
            "presentation_sha256",
            "compiled_graph_sha256",
            "expanded_graph_sha256",
        )
    }
    representation = fields["representation"]
    plan_id = fields["plan_id"]
    if representation not in {"authoring", "compiled"} or not isinstance(plan_id, str) or not plan_id:
        raise TypeError("diagram audit cannot supply SVG certificate identity")
    required = (
        "plan_sha256",
        "connectivity_sha256",
        "semantic_sha256",
        "presentation_sha256",
    )
    if any(
        not isinstance(fields[name], str) or _SHA256.fullmatch(fields[name]) is None
        for name in required
    ):
        raise TypeError("diagram audit cannot supply complete SVG certificate digests")
    compiled = fields["compiled_graph_sha256"]
    expanded = fields["expanded_graph_sha256"]
    if representation == "compiled":
        if any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in (compiled, expanded)
        ):
            raise TypeError("compiled diagram audit lacks SVG compiler identities")
    elif compiled is not None or expanded is not None:
        raise TypeError("authoring diagram audit unexpectedly carries compiler identities")

    # Parameter identity is intentionally taken only from the witness-verified
    # point row.  Recomputing it here from a mutable Plan or accepting a
    # caller-supplied tag would let a selected-point SVG claim the wrong
    # effective parameter closure.
    data = getattr(audit, "_data", None)
    rows = getattr(data, "verified_rows", None)
    if not isinstance(rows, (tuple, list)):
        raise TypeError("diagram audit lacks verified point identity rows")
    identity_rows = tuple(
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("kind") == "identity"
    )
    if len(identity_rows) != 1:
        raise TypeError("diagram audit lacks one verified point identity row")
    parameters_sha256 = identity_rows[0].get("parameters_sha256")
    if not isinstance(parameters_sha256, str) or _SHA256.fullmatch(parameters_sha256) is None:
        raise TypeError("diagram audit lacks verified effective-parameter identity")
    return _SvgCertificate(
        representation=representation,
        plan_id=plan_id,
        plan_sha256=fields["plan_sha256"],
        parameters_sha256=parameters_sha256,
        connectivity_sha256=fields["connectivity_sha256"],
        semantic_sha256=fields["semantic_sha256"],
        presentation_sha256=fields["presentation_sha256"],
        compiled_graph_sha256=compiled,
        expanded_graph_sha256=expanded,
    )


class _SceneFigure:
    """Minimal Figure adapter accepted by Schemdraw's established save API."""

    def __init__(
        self,
        scene: NeutralScene,
        *,
        color: str,
        background: str,
        certificate: _SvgCertificate | None,
    ) -> None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path as MplPath

        metrics = DEFAULT_METRICS
        self._figure = Figure(facecolor=background)
        self._certificate = certificate
        FigureCanvasAgg(self._figure)
        bounds = scene.bounds
        assert bounds is not None
        width, height = (
            max(2.0, bounds.xmax - bounds.xmin),
            max(2.0, bounds.ymax - bounds.ymin),
        )
        self._figure.set_size_inches(
            width * metrics.figure_inches_per_unit,
            height * metrics.figure_inches_per_unit,
        )
        axes = self._figure.add_axes((0, 0, 1, 1))
        axes.set_xlim(bounds.xmin, bounds.xmax)
        axes.set_ylim(bounds.ymin, bounds.ymax)
        axes.set_aspect("equal")
        axes.axis("off")

        def draw_path(
            path: Path,
            *,
            stroke: str = color,
            linewidth: float | None = None,
            linestyle: str = "solid",
            fill: str = "none",
        ) -> None:
            linewidth = metrics.symbol_linewidth if linewidth is None else linewidth
            vertices = [(point.x, point.y) for point in path.points]
            codes = (
                list(path.codes)
                if path.codes is not None
                else [MplPath.MOVETO] + [MplPath.LINETO] * (len(vertices) - 1)
            )
            if path.closed:
                vertices.append(vertices[0])
                codes.append(MplPath.CLOSEPOLY)
            axes.add_patch(
                PathPatch(
                    MplPath(vertices, codes),
                    fill=fill != "none",
                    facecolor=fill,
                    edgecolor=stroke,
                    linewidth=linewidth,
                    linestyle=linestyle,
                    capstyle="round",
                    joinstyle="round",
                )
            )

        def draw_text(run: TextRun) -> None:
            for glyph in run.glyphs:
                if not glyph.vertices or not glyph.codes:
                    continue
                vertices = [(point.x, point.y) for point in glyph.vertices]
                axes.add_patch(
                    PathPatch(
                        MplPath(vertices, list(glyph.codes)),
                        facecolor=color,
                        edgecolor="none",
                    )
                )

        light_pastels = (
            "#dbeafe",
            "#e0e7ff",
            "#dcfce7",
            "#fef3c7",
            "#fee2e2",
            "#fce7f3",
            "#cffafe",
            "#ede9fe",
            "#ffedd5",
            "#f1f5f9",
        )
        dark_pastels = (
            "#1e3a5f",
            "#312e5e",
            "#164e3a",
            "#5c4300",
            "#5f2020",
            "#5a1f45",
            "#164e63",
            "#3b2b60",
            "#5b2d14",
            "#1e293b",
        )
        paired_pastels = dark_pastels if background == "#111827" else light_pastels

        # A lowerer emits nested scopes bottom-up so their rigid contents can
        # be measured first.  That order is the opposite of paint order: a
        # parent background must never cover a visible child contour or its
        # local-ID header.  Use only the measured region bounds here; scene
        # order remains the deterministic tie-break for disjoint regions.
        visible_regions = tuple(region for region in scene.regions if region.kind != "root")

        def strictly_contains(outer: Bounds, inner: Bounds) -> bool:
            return outer.contains(inner) and outer != inner

        ordered_regions = tuple(
            region
            for _, region in sorted(
                enumerate(visible_regions),
                key=lambda indexed: (
                    sum(
                        strictly_contains(other.bounds, indexed[1].bounds)
                        for other in visible_regions
                        if other is not indexed[1]
                    ),
                    indexed[0],
                ),
            )
        )

        # Paint every background first, from outer scopes to contained scopes.
        # Contours and headers deliberately wait until every fill is complete.
        for region in ordered_regions:
            draw_path(
                region.boundary,
                linewidth=0.0,
                linestyle="solid",
                fill=paired_pastels[0],
            )

        # The ownership grammar is on top of all background fills and below
        # circuit ink.  This is a rendering order, not additional scene data.
        for region in ordered_regions:
            draw_path(
                region.boundary,
                linewidth=metrics.annotation_linewidth,
                linestyle="solid",
            )
            if region.header is not None:
                draw_text(region.header)
        for wire in scene.conductive:
            draw_path(
                Path(wire.points, "wire"),
                linewidth=metrics.conductive_linewidth,
            )
        for jump in scene.jumps:
            draw_path(jump.path, linewidth=metrics.conductive_linewidth)
        for symbol in scene.symbols:
            for path in symbol.paths:
                draw_path(path)
            if symbol.reference_polarity is not None:
                draw_path(symbol.reference_polarity)
            draw_text(symbol.visible_name)
            if symbol.value is not None:
                draw_text(symbol.value)
            if symbol.branch_label is not None:
                draw_text(symbol.branch_label)
        for box in scene.boxes:
            draw_path(box.outline)
            for path in box.paths:
                draw_path(path)
            draw_text(box.title)
            draw_text(box.kind_label)
            if box.length_label is not None:
                draw_text(box.length_label)
            if box.section_label is not None:
                draw_text(box.section_label)
            for row in box.conductor_rows:
                draw_text(row)
            for anchor_label in box.anchor_labels:
                draw_text(anchor_label)
            if box.reference_label is not None:
                draw_text(box.reference_label)
        for port in scene.ports:
            draw_path(port.circle)
            for path in port.paths:
                draw_path(path)
            for label in port.labels:
                draw_text(label)
            for path in port.reference_load.paths:
                draw_path(path)
            draw_text(port.reference_load.visible_name)
            if port.reference_load.value is not None:
                draw_text(port.reference_load.value)
            if port.reference_load.branch_label is not None:
                draw_text(port.reference_load.branch_label)
        for guide in scene.guides:
            for path in guide.paths:
                draw_path(
                    path,
                    linewidth=metrics.annotation_linewidth,
                    linestyle="solid" if path.role == "symbol" or path.closed else "dotted",
                    fill=color if path.role == "coupling" and path.closed else "none",
                )
            if guide.label is not None:
                draw_text(guide.label)
        for site in scene.boundary_sites:
            if site.visible_label is not None:
                draw_text(site.visible_label)
        for mark in scene.node_marks:
            for path in mark.paths:
                draw_path(path, fill=color if mark.filled and path.closed else "none")
            if mark.label is not None:
                draw_text(mark.label)
        for run in scene.text:
            draw_text(run)
        if scene.provenance_band is not None:
            draw_path(
                scene.provenance_band.background,
                linewidth=metrics.annotation_linewidth,
                linestyle="solid",
                fill="#1e293b" if background == "#111827" else "#f1f5f9",
            )
            for line in scene.provenance_band.lines:
                draw_text(line)

    def getimage(self, ext: str = "svg") -> bytes:
        output = BytesIO()
        self._figure.savefig(
            output,
            format=ext,
            bbox_inches="tight",
            pad_inches=0,
            metadata=self._metadata(ext),
        )
        return output.getvalue()

    def save(self, fname: str, transparent: bool = True, dpi: float = 72) -> None:
        self._figure.savefig(
            fname,
            transparent=transparent,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0,
            metadata=self._metadata(
                "svg" if fspath(fname).lower().endswith(".svg") else "png"
            ),
        )

    def _metadata(self, ext: str) -> dict[str, str | None]:
        """Keep SVG identity context out of raster metadata and scene facts."""

        if ext.lower() == "svg" and self._certificate is not None:
            return {"Date": None, "Description": self._certificate.metadata()}
        return {"Date": None}

    def show(self) -> None:
        self._figure.show()


def _frozen_drawing_class() -> type[Any]:
    import schemdraw

    class FrozenDrawing(schemdraw.Drawing):
        """A real Drawing whose rendering is solely the checked frozen scene."""

        def __setattr__(self, name: str, value: object) -> None:
            if getattr(self, "_scnsim_frozen", False):
                raise TypeError("frozen scene drawings do not permit mutation")
            object.__setattr__(self, name, value)

        def __delattr__(self, name: str) -> None:
            if getattr(self, "_scnsim_frozen", False):
                raise TypeError("frozen scene drawings do not permit mutation")
            object.__delattr__(self, name)

        def __init__(
            self,
            scene: NeutralScene,
            *,
            theme: Any,
            certificate: _SvgCertificate | None = None,
        ) -> None:
            from ..presentation import _palette, _require_theme

            if not isinstance(scene, NeutralScene):
                raise TypeError("FrozenDrawing requires a NeutralScene")
            self._scnsim_frozen = False
            self._scnsim_scene = scene
            if certificate is not None and not isinstance(certificate, _SvgCertificate):
                raise TypeError("FrozenDrawing requires an SVG certificate context or None")
            if certificate is not None and certificate.presentation_sha256 != scene_digest(scene):
                raise ValueError("SVG certificate is not bound to this frozen scene")
            self._scnsim_certificate = certificate
            self._scnsim_theme = _require_theme(theme)
            self._scnsim_digest = scene_digest(scene)
            palette = _palette(self._scnsim_theme)
            self._scnsim_color = palette.foreground
            self._scnsim_background = palette.background
            super().__init__(show=False)
            self._scnsim_frozen = True

        def _mutating(self, *_: object, **__: object) -> None:
            if self._scnsim_frozen:
                raise TypeError("frozen scene drawings do not permit mutation")

        add = _mutating
        add_elements = _mutating
        add_svgdef = _mutating
        config = _mutating
        container = _mutating
        move = _mutating
        move_from = _mutating
        pop = _mutating
        push = _mutating
        set_anchor = _mutating
        undo = _mutating

        def draw(self, show: bool = True, canvas: object = None) -> _SceneFigure:
            del canvas
            object.__setattr__(
                self,
                "fig",
                _SceneFigure(
                    self._scnsim_scene,
                    color=self._scnsim_color,
                    background=self._scnsim_background,
                    certificate=self._scnsim_certificate,
                ),
            )
            if show:
                self.fig.show()
            return self.fig

        def get_bbox(self) -> Bounds:
            assert self._scnsim_scene.bounds is not None
            return self._scnsim_scene.bounds

        def get_segments(self) -> tuple[Path, ...]:
            return tuple(
                path for symbol in self._scnsim_scene.symbols for path in symbol.paths
            )

        def save(self, fname: str, transparent: bool = False, dpi: float = 72) -> None:
            """Keep Schemdraw's API while serializing this scene afresh."""

            self.draw(show=False).save(fname, transparent=transparent, dpi=dpi)

        def _repr_svg_(self) -> None:
            # Preserve the existing themed-Drawing rich-display precedence.
            return None

        def _repr_png_(self) -> None:
            return None

        def _repr_mimebundle_(
            self, include: object = None, exclude: object = None
        ) -> dict[str, str]:
            from ..presentation import _adaptive_svg, _notebook_svg_viewer

            svg = self.draw(show=False).getimage("svg").decode("utf-8")
            adaptive = _adaptive_svg(svg, self._scnsim_theme)
            if self._scnsim_theme.value == "auto":
                replacements = {
                    "#dbeafe": "var(--scnsim-region-0)",
                    "#e0e7ff": "var(--scnsim-region-1)",
                    "#dcfce7": "var(--scnsim-region-2)",
                    "#fef3c7": "var(--scnsim-region-3)",
                    "#fee2e2": "var(--scnsim-region-4)",
                    "#fce7f3": "var(--scnsim-region-5)",
                    "#cffafe": "var(--scnsim-region-6)",
                    "#ede9fe": "var(--scnsim-region-7)",
                    "#ffedd5": "var(--scnsim-region-8)",
                    "#f1f5f9": "var(--scnsim-region-9)",
                }
                for source, target in replacements.items():
                    adaptive = adaptive.replace(source, target)
                opening = adaptive.find(">")
                adaptive = (
                    adaptive[: opening + 1]
                    + (
                        "<style>.scnsim-adaptive-svg{--scnsim-region-0:#dbeafe;--scnsim-region-1:#e0e7ff;"
                        "--scnsim-region-2:#dcfce7;--scnsim-region-3:#fef3c7;--scnsim-region-4:#fee2e2;"
                        "--scnsim-region-5:#fce7f3;--scnsim-region-6:#cffafe;--scnsim-region-7:#ede9fe;"
                        "--scnsim-region-8:#ffedd5;--scnsim-region-9:#f1f5f9}"
                        "@media (prefers-color-scheme:dark){.scnsim-adaptive-svg{--scnsim-region-0:#1e3a5f;--scnsim-region-1:#312e5e;"
                        "--scnsim-region-2:#164e3a;--scnsim-region-3:#5c4300;--scnsim-region-4:#5f2020;"
                        "--scnsim-region-5:#5a1f45;--scnsim-region-6:#164e63;--scnsim-region-7:#3b2b60;"
                        "--scnsim-region-8:#5b2d14;--scnsim-region-9:#1e293b}}</style>"
                    )
                    + adaptive[opening + 1 :]
                )
            wanted = None if include is None else set(include)
            rejected = set() if exclude is None else set(exclude)
            bundle: dict[str, str] = {}
            if "text/html" not in rejected and (
                wanted is None or "text/html" in wanted
            ):
                bundle["text/html"] = _notebook_svg_viewer(adaptive, self._scnsim_theme)
            if "image/svg+xml" not in rejected and (
                wanted is None or "image/svg+xml" in wanted
            ):
                bundle["image/svg+xml"] = svg
            return bundle

        def _ipython_display_(self) -> None:
            from IPython.display import HTML, display

            display(HTML(self._repr_mimebundle_()["text/html"]))

    FrozenDrawing.__module__ = __name__
    return FrozenDrawing


def freeze_drawing(
    scene: NeutralScene,
    *,
    theme: Any,
    certificate: _SvgCertificate | None = None,
) -> Any:
    """Return a read-only Schemdraw-compatible facade for one frozen scene."""

    return FrozenDrawing(scene, theme=theme, certificate=certificate)


FrozenDrawing = _frozen_drawing_class()
