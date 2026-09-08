"""Authoring display projection, separate from exact source numerical evidence.

Visible authoring values round the captured binary scalar to six significant
decimal digits. Observations parse literal ink without rounding it again. The
legacy exact physical parser remains available for compiled scientific evidence.
"""

from __future__ import annotations

from functools import lru_cache
from collections.abc import Mapping
from decimal import (
    Context, Decimal, DecimalException, DivisionByZero, Inexact,
    InvalidOperation, Overflow, Rounded, Underflow, ROUND_HALF_EVEN, localcontext,
)
from typing import cast

from .._canonical import float64_from_hex, quantity_envelope
from ..errors import SCNSimValidationError
from ..units import registry


def _fail(message: str, **evidence: object) -> SCNSimValidationError:
    return SCNSimValidationError(message, stage="schematic_audit", evidence=evidence)


def _envelope(value: Mapping[str, object]) -> tuple[str, str]:
    encoded, unit = value.get("si_value_f64"), value.get("si_unit")
    if not isinstance(encoded, str) or not isinstance(unit, str) or not unit:
        raise _fail("visible scalar requires a canonical scalar quantity")
    try:
        float64_from_hex(encoded)
    except ValueError as exc:
        raise _fail("visible scalar has malformed Float64 evidence") from exc
    return encoded, unit


def _split_visible_scalar(text: str) -> tuple[str, str]:
    if not isinstance(text, str):
        raise _fail("visible scalar text must be a string")
    magnitude, separator, unit = text.strip().partition(" ")
    if not magnitude or not separator or not unit.strip():
        raise _fail("visible scalar must contain an explicit magnitude and unit", label=text)
    try:
        float(magnitude)
    except ValueError as exc:
        raise _fail("visible scalar magnitude is not a finite decimal token", label=text) from exc
    return magnitude, unit.strip()


def parse_visible_scalar(text: str, *, si_unit: str) -> Mapping[str, object]:
    """Decode a literal visible scalar into canonical physical evidence.

    The returned record deliberately retains the actual displayed unit token so
    audits can report it, while correspondence compares the decoded quantity
    and dimensionality rather than a preferred engineering prefix spelling.
    """

    _, displayed_unit = _split_visible_scalar(text)
    try:
        decoded = quantity_envelope(
            registry.Quantity(text), si_unit=si_unit, registry=registry
        )
    except Exception as exc:
        raise _fail("visible scalar is not a complete physical quantity", label=text) from exc
    return {
        "quantity": cast(Mapping[str, object], decoded),
        "displayed_unit": displayed_unit,
        "displayed_text": text,
    }


_PREFIXES = ((-18, "a"), (-15, "f"), (-12, "p"), (-9, "n"),
             (-6, "µ"), (-3, "m"), (0, ""), (3, "k"), (6, "M"), (9, "G"))


def _display_context(precision: int = 28, *, exact: bool = False) -> Context:
    """Own all arithmetic settings, independently of the caller's context."""
    return Context(
        prec=max(28, precision), rounding=ROUND_HALF_EVEN,
        Emin=-999999, Emax=999999, capitals=1, clamp=0, flags=[],
        traps=[InvalidOperation, DivisionByZero, Overflow]
        + ([Inexact, Rounded, Underflow] if exact else []),
    )


def _decimal_text(value: Decimal) -> str:
    if not value:
        return "0"
    try:
        with localcontext(_display_context(len(value.as_tuple().digits), exact=True)):
            return str(value.normalize())
    except DecimalException as exc:
        raise _fail("visible scalar cannot be normalized without loss of precision") from exc


def _project(encoded: str) -> Decimal:
    try:
        exact = Decimal.from_float(float64_from_hex(encoded))
    except ValueError as exc:
        raise _fail("visible scalar has malformed Float64 evidence") from exc
    if not exact.is_finite():
        raise _fail("visible scalar requires a finite value")
    if not exact:
        return Decimal(0)
    with localcontext(_display_context(len(exact.as_tuple().digits) + 8)):
        return exact.quantize(Decimal(1).scaleb(exact.adjusted() - 5),
                              rounding=ROUND_HALF_EVEN)


def display_projection(value: Mapping[str, object]) -> Mapping[str, str]:
    """Expected display evidence from exact captured source bits, not a tolerance."""
    encoded, unit = _envelope(value)
    return {"si_decimal": _decimal_text(_project(encoded)), "si_unit": unit}


@lru_cache(maxsize=128)
def _display_units(unit: str) -> tuple[tuple[int, str], ...]:
    if unit == "dimensionless":
        return ((0, "dimensionless"),)
    return tuple((exponent, format(registry.Unit(prefix + unit), "~"))
                 for exponent, prefix in _PREFIXES)


def parse_display_scalar(text: str, *, si_unit: str) -> Mapping[str, str]:
    """Observe the exact visible decimal and scale; never round observed ink."""
    magnitude, displayed_unit = _split_visible_scalar(text)
    try:
        with localcontext(_display_context()):
            value = Decimal(magnitude)
    except InvalidOperation as exc:
        raise _fail("visible scalar magnitude is not a decimal token", label=text) from exc
    if not value.is_finite():
        raise _fail("visible scalar requires a finite value", label=text)
    scales = {token: exponent for exponent, token in _display_units(si_unit)}
    scales[si_unit] = 0
    if displayed_unit not in scales:
        raise _fail("visible scalar has an unsupported or incorrect unit", label=text,
                    si_unit=si_unit)
    try:
        with localcontext(_display_context(len(value.as_tuple().digits) + 8, exact=True)):
            observed = value.scaleb(scales[displayed_unit]) if value else Decimal(0)
    except DecimalException as exc:
        raise _fail("visible scalar cannot be scaled without loss of precision",
                    label=text) from exc
    return {"si_decimal": _decimal_text(observed), "si_unit": si_unit,
            "displayed_unit": displayed_unit, "displayed_text": text}


@lru_cache(maxsize=512)
def format_visible_scalar(encoded: str, unit: str) -> str:
    """Format the six-significant-digit projection with engineering units."""
    value = _project(encoded)
    units = dict(_display_units(unit))
    exponent = 3 * (value.adjusted() // 3) if value else 0
    if unit == "dimensionless":
        return f"{_decimal_text(value)} dimensionless"
    if exponent not in units:
        # Outside the supported prefix range, scientific base-SI spelling keeps
        # subnormal captured values visible instead of truncating them to zero.
        return f"{_decimal_text(value)} {units[0]}"
    with localcontext(_display_context()):
        scaled = value.scaleb(-exponent)
    number = format(scaled, "f")
    if "." in number:
        number = number.rstrip("0").rstrip(".")
    return f"{number} {units[exponent]}"


def format_envelope(value: Mapping[str, object]) -> str:
    encoded, unit = _envelope(value)
    return format_visible_scalar(encoded, unit)


__all__ = ["display_projection", "format_envelope", "format_visible_scalar",
           "parse_display_scalar", "parse_visible_scalar"]
