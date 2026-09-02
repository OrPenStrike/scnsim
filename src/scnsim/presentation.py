"""Shared, presentation-only light/dark rendering for SCNSim outputs.

Theme selection never participates in Plan, request, Result, or workspace
identity.  Heavy plotting libraries stay lazily imported so the public enum
does not change ordinary package import behavior.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
from typing import Any


class Theme(str, Enum):
    """Presentation theme for diagrams, numerical figures, and reports."""

    AUTO = "auto"
    LIGHT = "light"
    DARK = "dark"


@dataclass(frozen=True, slots=True)
class _Palette:
    background: str
    foreground: str
    secondary: str
    grid: str
    accent: str
    cycle: tuple[str, ...]


_LIGHT = _Palette(
    background="#ffffff",
    foreground="#243044",
    secondary="#5b677a",
    grid="#d8e1ec",
    accent="#2f80d1",
    cycle=(
        "#2f80d1",
        "#c76b00",
        "#2e7d32",
        "#7e57c2",
        "#c44536",
        "#0f766e",
        "#a16207",
        "#be185d",
        "#0e7490",
        "#475569",
    ),
)
_DARK = _Palette(
    background="#111827",
    foreground="#eaf2fb",
    secondary="#a8b7ca",
    grid="#2b3a4f",
    accent="#7bb7f0",
    cycle=(
        "#7bb7f0",
        "#f2a65a",
        "#7dcc8a",
        "#b39ddb",
        "#ff8a80",
        "#2dd4bf",
        "#facc15",
        "#f472b6",
        "#22d3ee",
        "#cbd5e1",
    ),
)


def _require_theme(theme: object) -> Theme:
    if not isinstance(theme, Theme):
        raise TypeError("theme must be a scnsim.Theme value")
    return theme


def _palette(theme: Theme) -> _Palette:
    checked = _require_theme(theme)
    return _DARK if checked is Theme.DARK else _LIGHT


def _theme_variables(palette: _Palette) -> str:
    entries = {
        "bg": palette.background,
        "fg": palette.foreground,
        "secondary": palette.secondary,
        "grid": palette.grid,
        "accent": palette.accent,
        **{f"cycle-{index}": color for index, color in enumerate(palette.cycle)},
    }
    return ";".join(f"--scnsim-{name}:{value}" for name, value in entries.items())


def _theme_css(theme: Theme, *, selector: str) -> str:
    checked = _require_theme(theme)
    base = f"{selector}{{{_theme_variables(_palette(checked))}}}"
    if checked is not Theme.AUTO:
        return base
    return (
        f"{selector}{{{_theme_variables(_LIGHT)}}}"
        "@media (prefers-color-scheme:dark){"
        f"{selector}{{{_theme_variables(_DARK)}}}"
        "}"
    )


def _strip_svg_preamble(svg: str) -> str:
    start = svg.find("<svg")
    if start < 0:
        raise ValueError("renderer did not produce SVG")
    return svg[start:]


def _normalize_svg_ids(svg: str) -> str:
    """Replace renderer-generated SVG IDs without changing their references."""

    identifiers = tuple(dict.fromkeys(re.findall(r'\bid="([^"]+)"', svg)))
    mapping = {
        identifier: f"scnsim-svg-{index}"
        for index, identifier in enumerate(identifiers)
    }
    normalized = re.sub(
        r'\bid="([^"]+)"',
        lambda match: f'id="{mapping[match.group(1)]}"',
        svg,
    )
    normalized = re.sub(
        r'((?:xlink:)?href)="#([^"]+)"',
        lambda match: (
            f'{match.group(1)}="#{mapping.get(match.group(2), match.group(2))}"'
        ),
        normalized,
    )
    return re.sub(
        r'url\(#([^)]+)\)',
        lambda match: f"url(#{mapping.get(match.group(1), match.group(1))})",
        normalized,
    )


def _adaptive_svg(svg: str, theme: Theme) -> str:
    """Return one embeddable SVG with light fallback and optional dark CSS."""

    checked = _require_theme(theme)
    embedded = _strip_svg_preamble(svg)
    if checked is not Theme.AUTO:
        return embedded

    replacements = {
        _LIGHT.background: "var(--scnsim-bg)",
        _LIGHT.foreground: "var(--scnsim-fg)",
        _LIGHT.secondary: "var(--scnsim-secondary)",
        _LIGHT.grid: "var(--scnsim-grid)",
        _LIGHT.accent: "var(--scnsim-accent)",
        **{
            color: f"var(--scnsim-cycle-{index})"
            for index, color in enumerate(_LIGHT.cycle)
        },
    }
    for source, target in replacements.items():
        embedded = embedded.replace(source, target)
    embedded = embedded.replace("<svg ", '<svg class="scnsim-adaptive-svg" ', 1)
    opening_end = embedded.find(">")
    if opening_end < 0:
        raise ValueError("renderer produced malformed SVG")
    css = _theme_css(Theme.AUTO, selector=".scnsim-adaptive-svg")
    style = (
        "<style>"
        f"{css}"
        ".scnsim-adaptive-svg{color-scheme:light dark;background:var(--scnsim-bg)}"
        "</style>"
    )
    return embedded[: opening_end + 1] + style + embedded[opening_end + 1 :]


def _html_fragment(fragment: str, theme: Theme) -> str:
    checked = _require_theme(theme)
    suffix = sha256(f"{checked.value}\0{fragment}".encode("utf-8")).hexdigest()[:16]
    class_name = f"scnsim-presentation-{suffix}"
    selector = f".{class_name}"
    css = _theme_css(checked, selector=selector)
    color_scheme = "light dark" if checked is Theme.AUTO else checked.value
    return (
        f"<style>{css}"
        f"{selector}{{box-sizing:border-box;color:var(--scnsim-fg);"
        f"background:var(--scnsim-bg);color-scheme:{color_scheme};padding:1rem}}"
        f"{selector} table{{border-collapse:collapse}}"
        f"{selector} th,{selector} td{{border:1px solid var(--scnsim-grid);"
        "padding:.35rem .55rem;text-align:left}"
        f"{selector} th{{color:var(--scnsim-secondary)}}"
        f"{selector} svg{{display:block;max-width:100%;height:auto}}"
        "</style>"
        f'<div class="{class_name}">{fragment}</div>'
    )


def _report_html(body: str, theme: Theme) -> str:
    checked = _require_theme(theme)
    css = _theme_css(checked, selector=":root")
    color_scheme = "light dark" if checked is Theme.AUTO else checked.value
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<meta name=\"color-scheme\" content=\"{color_scheme}\">"
        "<title>SCNSim report</title>"
        f"<style>{css}"
        f"html{{color-scheme:{color_scheme};background:var(--scnsim-bg)}}"
        "body{box-sizing:border-box;max-width:76rem;margin:0 auto;padding:2rem;"
        "font-family:system-ui,sans-serif;color:var(--scnsim-fg);background:var(--scnsim-bg)}"
        "table{border-collapse:collapse;margin-block:1rem}"
        "th,td{border:1px solid var(--scnsim-grid);padding:.4rem .6rem;text-align:left}"
        "th{color:var(--scnsim-secondary)}"
        "img,svg{display:block;max-width:100%;height:auto;margin-block:1rem}"
        "a{color:var(--scnsim-accent)}"
        "</style></head>"
        f"<body>{body}</body></html>"
    )


@lru_cache(maxsize=1)
def _themed_figure_class() -> type[Any]:
    from matplotlib.figure import Figure

    class _ThemedFigure(Figure):
        def __init__(self, *args: object, theme: Theme, **kwargs: object) -> None:
            self._scnsim_theme = _require_theme(theme)
            kwargs.setdefault("facecolor", _palette(self._scnsim_theme).background)
            super().__init__(*args, **kwargs)

        def _repr_mimebundle_(
            self,
            include: object = None,
            exclude: object = None,
        ) -> dict[str, str]:
            del include, exclude
            return {"text/html": _figure_svg(self, self._scnsim_theme)}

        def _ipython_display_(self) -> None:
            # Matplotlib's inline backend also registers PNG/SVG formatters for
            # every Figure subclass.  The explicit display hook makes the
            # adaptive HTML bundle the single Notebook representation.
            from IPython.display import HTML, display

            display(HTML(_figure_svg(self, self._scnsim_theme)))

        def savefig(self, *args: object, **kwargs: object) -> object:
            background = _palette(self._scnsim_theme).background
            kwargs["facecolor"] = background
            kwargs["edgecolor"] = background
            kwargs["transparent"] = False
            return super().savefig(*args, **kwargs)

    _ThemedFigure.__module__ = __name__
    return _ThemedFigure


def _themed_subplots(theme: Theme, *args: object, **kwargs: object) -> tuple[Any, Any]:
    checked = _require_theme(theme)
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        *args,
        FigureClass=_themed_figure_class(),
        theme=checked,
        **kwargs,
    )
    palette = _palette(checked)
    flattened = axes.flat if hasattr(axes, "flat") else (axes,)
    for axis in flattened:
        axis.set_prop_cycle(color=palette.cycle)
    return figure, axes


def _finish_figure(figure: Any, theme: Theme) -> Any:
    palette = _palette(_require_theme(theme))
    figure.patch.set_facecolor(palette.background)
    for axis in figure.axes:
        axis.set_facecolor(palette.background)
        axis.tick_params(colors=palette.secondary)
        axis.xaxis.label.set_color(palette.foreground)
        axis.yaxis.label.set_color(palette.foreground)
        axis.title.set_color(palette.foreground)
        for spine in axis.spines.values():
            spine.set_color(palette.grid)
        for gridline in (*axis.get_xgridlines(), *axis.get_ygridlines()):
            gridline.set_color(palette.grid)
        legend = axis.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(palette.background)
            legend.get_frame().set_edgecolor(palette.grid)
            for text in legend.get_texts():
                text.set_color(palette.foreground)
    for text in figure.texts:
        text.set_color(palette.foreground)
    return figure


def _figure_svg(figure: Any, theme: Theme) -> str:
    checked = _require_theme(theme)
    output = BytesIO()
    figure.savefig(
        output,
        format="svg",
        facecolor=_palette(checked).background,
        bbox_inches="tight",
        metadata={"Date": None},
    )
    normalized = _normalize_svg_ids(output.getvalue().decode("utf-8"))
    return _adaptive_svg(normalized, checked)


def _figure_data_uri(figure: Any, theme: Theme) -> str:
    svg = _figure_svg(figure, theme).encode("utf-8")
    return "data:image/svg+xml;base64," + base64.b64encode(svg).decode("ascii")


@lru_cache(maxsize=1)
def _themed_drawing_class() -> type[Any]:
    import schemdraw

    class _ThemedDrawing(schemdraw.Drawing):
        def __init__(self, *args: object, theme: Theme, **kwargs: object) -> None:
            self._scnsim_theme = _require_theme(theme)
            kwargs.setdefault("show", False)
            kwargs.setdefault("transparent", False)
            super().__init__(*args, **kwargs)

        def _repr_mimebundle_(
            self,
            include: object = None,
            exclude: object = None,
        ) -> dict[str, str]:
            del include, exclude
            svg = schemdraw.Drawing._repr_svg_(self)
            return {"text/html": _adaptive_svg(svg, self._scnsim_theme)}

        def _ipython_display_(self) -> None:
            from IPython.display import HTML, display

            svg = schemdraw.Drawing._repr_svg_(self)
            display(HTML(_adaptive_svg(svg, self._scnsim_theme)))

        def _repr_svg_(self) -> None:
            # Avoid a second static MIME candidate shadowing adaptive HTML.
            return None

        def _repr_png_(self) -> None:
            return None

        def save(self, fname: str, transparent: bool = False, dpi: float = 72) -> None:
            del transparent
            super().save(fname, transparent=False, dpi=dpi)

    _ThemedDrawing.__module__ = __name__
    return _ThemedDrawing


def _themed_drawing(theme: Theme, **kwargs: object) -> Any:
    checked = _require_theme(theme)
    drawing = _themed_drawing_class()(theme=checked, **kwargs)
    drawing.config(bgcolor=_palette(checked).background)
    return drawing


__all__ = ["Theme"]
