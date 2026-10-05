"""One catalog factory context and reserved body/use construction authority."""

from __future__ import annotations

from contextvars import ContextVar


_factory_context: ContextVar[tuple[object, str] | None] = ContextVar(
    "scnsim_library_factory", default=None
)
_component_creation_token = object()
_two_terminal_use_token = object()
