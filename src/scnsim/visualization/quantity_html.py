"""HTML rendering for the shared scalar Quantity presentation model."""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING

from .presentation import Theme, _require_theme, _theme_css

if TYPE_CHECKING:
    from ..results.base import HtmlPresentation
    from .quantity_presentation import ScalarTable


_ROOT_CLASS = "scnsim-quantity-presentation"


def _column_role(column: str) -> str:
    normalized = column.casefold()
    if normalized == "unit":
        return "unit"
    if normalized in {"value", "numeric value"}:
        return "value"
    if normalized == "meaning":
        return "meaning"
    if normalized in {"quantity", "setting", "field", "result"}:
        return "label"
    return "text"


def _table_html(table: ScalarTable) -> str:
    """Render one plain-text ScalarTable with HTML escaping at the boundary."""

    title = escape(table.title)
    roles = tuple(_column_role(column) for column in table.columns)
    headers = "".join(
        f"<th class=\"scnsim-quantity-{role}\" scope=\"col\">{escape(column)}</th>"
        for column, role in zip(table.columns, roles)
    )
    rows = []
    for row in table.rows:
        cells = []
        for index, value in enumerate(row):
            role = roles[index] if index < len(roles) else "text"
            class_attr = f" class=\"scnsim-quantity-{role}\""
            cells.append(f"<td{class_attr}>{escape(value)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        f"<section class=\"scnsim-quantity-section\"><h3>{title}</h3>"
        "<div class=\"scnsim-quantity-table-wrap\"><table>"
        f"<thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody>"
        "</table></div></section>"
    )


def scalar_show(
    result: object,
    *,
    theme: Theme = Theme.AUTO,
    detailed: bool = False,
) -> HtmlPresentation:
    """Build a notebook-ready HTML view from the shared scalar presentation."""

    from ..results.base import HtmlPresentation
    from .quantity_presentation import build_scalar_presentation

    selected_theme = _require_theme(theme)
    presentation = build_scalar_presentation(result, detailed=detailed)
    sections = "".join(_table_html(table) for table in presentation.tables)
    root_class = f"{_ROOT_CLASS}-{selected_theme.value}"
    selector = f".{root_class}"
    css = _theme_css(selected_theme, selector=selector)
    color_scheme = {
        Theme.AUTO: "light dark",
        Theme.LIGHT: "light",
        Theme.DARK: "dark",
    }[selected_theme]
    html = (
        "<style>"
        f"{css}"
        f"{selector}{{box-sizing:border-box;color:var(--scnsim-fg);"
        "background:var(--scnsim-bg);font:14px/1.45 system-ui,-apple-system,"
        "BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:100%;"
        f"color-scheme:{color_scheme};user-select:text}}"
        f"{selector} h2,{selector} h3{{color:var(--scnsim-fg);"
        "font:inherit;font-weight:650;margin:.8rem 0 .4rem}"
        f"{selector} h2{{font-size:1.15rem}}"
        f"{selector} h3{{font-size:1rem}}"
        f"{selector} .scnsim-quantity-section{{margin:.7rem 0 1rem}}"
        f"{selector} .scnsim-quantity-table-wrap{{max-width:100%;overflow-x:auto;}}"
        f"{selector} table{{border-collapse:collapse;width:100%;"
        "table-layout:auto;font:inherit;color:inherit}"
        f"{selector} th,{selector} td{{border:1px solid var(--scnsim-grid);"
        "padding:.42rem .6rem;text-align:left;vertical-align:top;"
        "overflow-wrap:anywhere}"
        f"{selector} th{{background:var(--scnsim-bg);font-weight:600}}"
        f"{selector} .scnsim-quantity-label{{min-width:8rem}}"
        f"{selector} .scnsim-quantity-meaning{{min-width:12rem}}"
        f"{selector} .scnsim-quantity-value{{text-align:right;"
        "font-variant-numeric:tabular-nums;min-width:6rem}"
        f"{selector} .scnsim-quantity-unit{{white-space:nowrap;width:1%}}"
        "</style>"
        f"<article class=\"{root_class}\" data-detailed=\"{str(detailed).lower()}\">"
        f"<h2>{escape(presentation.title)}</h2>{sections}</article>"
    )
    return HtmlPresentation(html)
