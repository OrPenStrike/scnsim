"""Shared scalar presentation over an already decoded Direct quantity.

This owner selects labels, definitions and display units once for HTML and
Figure renderers. It never recalculates quantities or changes their frequency
convention. Strings remain unescaped; the consuming renderer owns escaping.
Identity and original-magnitude data are generated only on explicit request.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from decimal import Decimal, localcontext
import math


@dataclass(frozen=True, slots=True)
class ScalarTable:
    title: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class ScalarPresentation:
    title: str
    tables: tuple[ScalarTable, ...]


_FIELDS = (
    ("root", "Complex angular root"),
    ("frequency", "Frequency"),
    ("linewidth", "Linewidth"),
    ("slope", "Operator slope"),
    ("value", "Complex value"),
    ("magnitude", "Magnitude"),
    ("real", "Real part"),
    ("imag", "Imaginary part"),
    ("zero", "Complex angular zero"),
    ("numerator_slope", "Numerator slope"),
    ("denominator", "Denominator"),
    ("coupling", "Complex coupling"),
    ("branch_a_residue", "Branch A residue"),
    ("branch_b_residue", "Branch B residue"),
    ("evaluation_omega", "Evaluation angular frequency"),
)
_TITLES = {
    "diagonal_root": "Direct diagonal root",
    "operator_element_root": "Direct operator-element root",
    "hybridized_pole": "Direct hybridized pole",
    "transfer_zero": "Direct transfer zero",
    "residue_normalized_coupling": "Direct residue-normalized coupling",
    "response_element": "Direct response element",
}
_PREFIXES = {
    -30: "q", -27: "r", -24: "y", -21: "z", -18: "a", -15: "f",
    -12: "p", -9: "n", -6: "µ", -3: "m", 0: "", 3: "k", 6: "M",
    9: "G", 12: "T", 15: "P", 18: "E", 21: "Z", 24: "Y", 27: "R", 30: "Q",
}
_SYMBOLS = {
    "hertz": "Hz", "radian / second": "rad/s", "ohm": "Ω",
    "siemens": "S", "second": "s", "farad": "F", "henry": "H",
    "volt": "V", "ampere": "A", "1 / second": "s⁻¹",
}


def _scalar(value: object) -> float | complex:
    # Decoders may retain a NumPy scalar or zero-dimensional scalar array.
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, complex):
        return complex(value)
    return float(value)


def _number(value: float | complex, exponent: int = 0) -> str:
    def component(part: float) -> str:
        # Decimal formatting keeps a tiny nonzero component visible even when
        # the shared complex scale would underflow a binary64 division.
        with localcontext() as context:
            context.prec = 20
            scaled = Decimal.from_float(part).scaleb(-exponent)
            return format(scaled, ".12g")

    if isinstance(value, complex):
        sign = "−" if math.copysign(1.0, value.imag) < 0 else "+"
        return f"{component(value.real)} {sign} {component(abs(value.imag))}j"
    return component(value)


def _quantity(value: object) -> tuple[str, str]:
    magnitude = _scalar(value.magnitude)
    unit = str(value.units)
    if unit == "dimensionless":
        return _number(magnitude), "1"
    largest = max(abs(magnitude.real), abs(magnitude.imag)) if isinstance(magnitude, complex) else abs(magnitude)
    exponent = 3 * math.floor(math.log10(largest) / 3) if largest and math.isfinite(largest) else 0
    symbol = _SYMBOLS.get(unit)
    if symbol is not None and exponent in _PREFIXES:
        # Angular and cyclic quantities keep their exact original unit family.
        display_unit = _PREFIXES[exponent] + symbol
    else:
        display_unit = format(value.units, "~")
        if exponent:
            display_unit = f"10^{exponent} {display_unit}"
    return _number(magnitude, exponent), display_unit


def _declared_quantity(value: object) -> object:
    """Use the existing canonical decoder for saved specification envelopes."""
    if hasattr(value, "units"):
        return value
    from .. import units
    from ..canonical import complex_quantity_from_envelope, quantity_from_envelope

    if value["type"] == "complex_quantity_f64":
        return complex_quantity_from_envelope(value, registry=units.registry)
    return quantity_from_envelope(value, registry=units.registry)


def _definitions(spec: Mapping[str, object], lineage: object) -> ScalarTable:
    kind = spec["type"]
    rows: list[tuple[str, ...]] = []

    def text(label: str, meaning: str, value: object) -> None:
        rows.append((label, meaning, str(value), ""))

    def quantity(label: str, meaning: str, value: object) -> None:
        number, unit = _quantity(_declared_quantity(value))
        rows.append((label, meaning, number, unit))

    def branch(label: str, declaration: Mapping[str, object]) -> None:
        if declaration["type"] == "diagonal_root":
            coordinate = declaration["coordinate"]
            text(label, "Diagonal root equation", f"F_View[{coordinate}, {coordinate}](ω) = 0")
            quantity(label + " hint", "Initial Newton basin; not a target", declaration["root_hint"])
        elif declaration["type"] == "hybridized_pole":
            text(label, "Retained block pole: det F_View(ω) = 0", ", ".join(str(item) for item in declaration["coordinates"]))
            quantity(label + " anchor", "Initial complex Newton anchor", declaration["anchor"])
        else:
            raise ValueError("unsupported residue-coupling branch definition")

    if kind in {"diagonal_root", "operator_element_root"}:
        row = spec["coordinate"] if kind == "diagonal_root" else spec["row"]
        column = spec["coordinate"] if kind == "diagonal_root" else spec["column"]
        text("Equation", "Final View operator element", f"F_View[{row}, {column}](ω) = 0")
        quantity("Root hint", "Initial Newton basin; not a target", spec["root_hint"])
        text("Slope", "Derivative with respect to angular frequency", f"dF_View[{row}, {column}] / dω")
    elif kind == "hybridized_pole":
        text("Equation", "Retained coupled-block determinant", "det F_View(ω) = 0")
        text("Coordinates", "Ordered retained block", ", ".join(str(item) for item in spec["coordinates"]))
        quantity("Anchor", "Initial complex Newton anchor", spec["anchor"])
    elif kind in {"transfer_zero", "response_element"}:
        text("Response", "Ordered output ← input", f"{spec['family']}[{spec['output_coordinate']}, {spec['input_coordinate']}]")
        if kind == "transfer_zero":
            text("Equation", "Analytic transfer numerator zero; denominator remains resolved", "N(ω) = 0")
            quantity("Anchor", "Initial complex Newton anchor", spec["anchor"])
        else:
            quantity("Frequency", "Exact declared evaluation frequency; no sweep interpolation", spec["frequency"])
    elif kind == "residue_normalized_coupling":
        branch("Branch A", spec["branch_a"])
        branch("Branch B", spec["branch_b"])
        frequency = spec["frequency"]
        if frequency == "complex_root_midpoint":
            text("Evaluation", "Candidate complex-root midpoint", "(ω_A + ω_B) / 2")
        else:
            quantity("Evaluation frequency", "Fixed declared frequency", frequency)
        text("Coupling", "Full complex residue-normalized quantity; no fitted splitting", "J; Re J, Im J, |J|")
    else:
        raise ValueError(f"unsupported scalar Direct quantity definition: {kind}")
    if kind in {"diagonal_root", "operator_element_root", "hybridized_pole", "transfer_zero"}:
        text("Frequency", "Stored cyclic-frequency projection", "Re(ω) / (2π)")
    if isinstance(lineage, Mapping) and "terminal_coordinates" in lineage:
        text("Final View basis", "Ordered named coordinates", ", ".join(str(item) for item in lineage["terminal_coordinates"]))
    return ScalarTable("Definitions", ("Setting", "Meaning", "Value", "Unit"), tuple(rows))


def _detail_rows(value: object, path: str = "") -> list[tuple[str, str]]:
    if is_dataclass(value):
        return [row for field in fields(value) if not field.name.startswith("_")
                for row in _detail_rows(getattr(value, field.name), f"{path}.{field.name}" if path else field.name)]
    if isinstance(value, Mapping):
        return [row for key, item in value.items()
                for row in _detail_rows(item, f"{path}.{key}" if path else str(key))]
    if isinstance(value, (tuple, list)):
        return [row for index, item in enumerate(value)
                for row in _detail_rows(item, f"{path}[{index}]")]
    return [(path, repr(value))]


def build_scalar_presentation(result: object, *, detailed: bool = False) -> ScalarPresentation:
    """Present every saved scalar field without evaluating any numerical work."""
    from ..results.matrix import DirectQuantityResult

    if not isinstance(result, DirectQuantityResult):
        raise TypeError("scalar presentation requires DirectQuantityResult")
    if not isinstance(detailed, bool):
        raise TypeError("detailed must be bool")
    presentation = result._presentation
    spec = presentation["spec"]
    lineage = presentation.get("ref_lineage")
    title = _TITLES[spec["type"]]
    quantities = tuple((label, value) for name, label in _FIELDS
                       if (value := getattr(result, name)) is not None)
    rows = tuple((label, *_quantity(value)) for label, value in quantities)
    family = result.family or spec.get("family")
    if family is not None:
        rows += (("Response family", str(family), ""),)
    tables = [ScalarTable("Results", ("Quantity", "Value", "Unit"), rows),
              _definitions(spec, lineage)]
    if detailed:
        tables.append(ScalarTable(
            "Original values",
            ("Quantity", "Magnitude (repr; not accuracy)", "Unit", "Retained dtype"),
            tuple((label, repr(_scalar(value.magnitude)), str(value.units),
                   str(value.magnitude.dtype) if hasattr(value.magnitude, "dtype")
                   else "Not retained") for label, value in quantities),
        ))
        detail = _detail_rows(result.identity, "Identity")
        detail += _detail_rows(presentation.get("view"), "View")
        detail += _detail_rows(lineage, "View lineage")
        tables.append(ScalarTable("Identity and View lineage", ("Field", "Value"), tuple(detail)))
    return ScalarPresentation(title, tuple(tables))


__all__ = ["ScalarTable", "ScalarPresentation", "build_scalar_presentation"]
