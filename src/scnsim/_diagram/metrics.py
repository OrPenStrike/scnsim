"""Private, deterministic diagram measurements and text shaping.

The scene layer records glyph outlines rather than leaving text measurement to
an SVG viewer.  This is intentionally independent of Schemdraw's optional
``ziamath`` support, which is unavailable in the supported 0.23 environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .scene import Bounds, GlyphPath, Point, TextRun


@dataclass(frozen=True, slots=True)
class DiagramMetrics:
    """The U2.2 spacing system shared by placement and emitted scenes."""

    unit_length: float = 2.2

    @property
    def native_span(self) -> float:
        return self.unit_length

    @property
    def label_clearance(self) -> float:
        return self.unit_length / 8

    @property
    def terminal_stub(self) -> float:
        return self.unit_length / 4

    @property
    def obstacle_clearance(self) -> float:
        return self.unit_length / 4

    @property
    def routing_lane_pitch(self) -> float:
        return self.unit_length / 2

    @property
    def port_lead(self) -> float:
        return 5 * self.unit_length / 8

    @property
    def port_load_span(self) -> float:
        return self.unit_length

    @property
    def port_circle_radius(self) -> float:
        """Radius of the open Port boundary mark in scene units."""

        return self.unit_length / 16

    @property
    def polarity_mark_span(self) -> float:
        """Small, upright endpoint reference signs, independent of orientation."""
        return self.unit_length / 8

    @property
    def ground_stem(self) -> float:
        """Ground-stem length matching the trusted Schemdraw glyph ratio."""

        return 2 * self.unit_length / 11

    @property
    def ground_bar_step(self) -> float:
        """Separation between bars in the trusted Schemdraw ground glyph."""

        return 3 * self.unit_length / 55

    @property
    def ground_half_width(self) -> float:
        """Half-width shared by ground and capacitor plates at U2.2."""

        return 5 * self.unit_length / 44

    @property
    def grounded_branch_depth(self) -> float:
        return 2 * self.unit_length

    @property
    def panel_gap(self) -> float:
        return 3 * self.unit_length

    @property
    def junction_stagger(self) -> float:
        return self.unit_length / 6

    @property
    def jump_gap(self) -> float:
        return self.unit_length / 12

    @property
    def jump_height(self) -> float:
        return 3 * self.unit_length / 8

    @property
    def primary_text_size(self) -> float:
        return 10.0 / 36.0

    @property
    def port_id_text_size(self) -> float:
        return 9.0 / 36.0

    @property
    def secondary_text_size(self) -> float:
        return 8.0 / 36.0

    @property
    def tertiary_text_size(self) -> float:
        return 7.0 / 36.0

    @property
    def figure_inches_per_unit(self) -> float:
        return 0.65

    @property
    def symbol_linewidth(self) -> float:
        return 1.5

    @property
    def conductive_linewidth(self) -> float:
        return 1.8

    @property
    def annotation_linewidth(self) -> float:
        return 1.0


DEFAULT_METRICS = DiagramMetrics()
_SHAPING_POINT_SIZE = 10.0


@dataclass(frozen=True, slots=True)
class FontFace:
    """One resolved, environment-local font file used in a frozen scene."""

    path: str
    family: str
    sha256: str


@dataclass(frozen=True, slots=True)
class _GlyphTemplate:
    font_sha256: str
    vertices: tuple[Point, ...]
    codes: tuple[int, ...]
    bounds: Bounds
    advance: float
    ascent: float
    descent: float


@lru_cache(maxsize=1)
def _font_faces() -> tuple[FontFace, ...]:
    from matplotlib import font_manager

    candidates = ("DejaVu Sans", "DejaVu Sans Mono", "STIXGeneral")
    resolved: list[FontFace] = []
    for family in candidates:
        path = Path(font_manager.findfont(family, fallback_to_default=True))
        candidate = FontFace(str(path), family, sha256(path.read_bytes()).hexdigest())
        if candidate.path not in {item.path for item in resolved}:
            resolved.append(candidate)
    if not resolved:
        raise RuntimeError("no Matplotlib font is available for diagram text")
    return tuple(resolved)


def _face_for(character: str) -> FontFace:
    """Return the first deterministic local font that contains ``character``."""

    from matplotlib.ft2font import FT2Font

    for face in _font_faces():
        if FT2Font(face.path).get_char_index(ord(character)):
            return face
    raise ValueError(f"diagram font has no glyph for {character!r}")


@lru_cache(maxsize=4096)
def _glyph_template(character: str, size: float) -> _GlyphTemplate:
    """Shape one origin-relative glyph for reuse within this font environment."""

    from matplotlib.font_manager import FontProperties
    from matplotlib.ft2font import FT2Font
    from matplotlib.textpath import TextPath

    from .scene import Bounds, Point

    face = _face_for(character)
    properties = FontProperties(fname=face.path)
    font = FT2Font(face.path)
    # FreeType clamps sub-point requests inconsistently with TextPath. Shape at
    # one stable point size, then scale every geometric fact into scene units.
    font.set_size(_SHAPING_POINT_SIZE, 72)
    glyph = font.load_char(ord(character))
    scale = size / _SHAPING_POINT_SIZE
    advance = float(glyph.linearHoriAdvance) / 65536.0 * scale
    if character.isspace():
        vertices: tuple[Point, ...] = ()
        codes: tuple[int, ...] = ()
        bounds = Bounds(0.0, 0.0, 0.0, 0.0)
    else:
        path = TextPath(
            (0.0, 0.0),
            character,
            size=_SHAPING_POINT_SIZE,
            prop=properties,
        )
        raw_bounds = path.get_extents()
        vertices = tuple(
            Point(float(x) * scale, float(y) * scale) for x, y in path.vertices
        )
        codes = (
            tuple(int(code) for code in path.codes) if path.codes is not None else ()
        )
        bounds = Bounds(
            float(raw_bounds.x0) * scale,
            float(raw_bounds.y0) * scale,
            float(raw_bounds.x1) * scale,
            float(raw_bounds.y1) * scale,
        )
    ascent = (
        float(font.ascender) / float(font.units_per_EM) * _SHAPING_POINT_SIZE * scale
    )
    descent = (
        float(font.descender) / float(font.units_per_EM) * _SHAPING_POINT_SIZE * scale
    )
    return _GlyphTemplate(
        face.sha256,
        vertices,
        codes,
        bounds,
        advance,
        ascent,
        descent,
    )


def shape_text(
    text: str,
    *,
    at: Point,
    size: float = 10.0,
    role: str = "label",
) -> TextRun:
    """Freeze exact Matplotlib glyph outlines, including whitespace advances.

    The caller supplies literal NFC text.  We deliberately do not normalize,
    parse TeX, or substitute a missing glyph: visible IDs must survive exactly
    as authored and unsupported text must fail before a scene exists.
    """

    from .scene import Bounds, GlyphPath, TextRun

    if not isinstance(text, str) or not text:
        raise ValueError("diagram text must be a nonempty literal string")
    if not isinstance(size, (float, int)) or isinstance(size, bool) or size <= 0:
        raise ValueError("diagram text size must be positive")
    size = float(size)
    cursor = at.x
    glyphs: list[GlyphPath] = []
    ascent = descent = 0.0
    for character in text:
        template = _glyph_template(character, size)
        glyphs.append(
            GlyphPath(
                character,
                template.font_sha256,
                tuple(vertex.translated(cursor, at.y) for vertex in template.vertices),
                template.codes,
                template.bounds.translated(cursor, at.y),
                template.advance,
            )
        )
        cursor += template.advance
        ascent = max(ascent, template.ascent)
        descent = min(descent, template.descent)
    occupied = Bounds(at.x, at.y + descent, cursor, at.y + ascent)
    return TextRun(
        text, at, float(size), role, tuple(glyphs), occupied, ascent, descent
    )
