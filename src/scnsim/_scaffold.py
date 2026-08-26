"""Shared failure behavior for the CONVERGING public API scaffold."""

from __future__ import annotations

from typing import NoReturn


class ScaffoldUnavailableError(NotImplementedError):
    """Raised when a declared SCNSim surface has no candidate implementation yet."""


def unavailable(surface: str) -> NoReturn:
    """Fail rather than making an unimplemented API look successful."""

    raise ScaffoldUnavailableError(
        f"{surface} is declared for SCNSim V1 review but is not implemented in "
        "the current CONVERGING scaffold."
    )
