"""Reusable two-section feedline catalog for the advanced tutorial sequence."""

from __future__ import annotations

from scnsim import (
    ComponentInstance,
    CompositePlan,
    Library,
    ParameterRef,
    RLGC,
    components as builtin_components,
)


class FeedlineLibrary(Library):
    """Project catalog for a feedline with one explicitly exposed tap."""

    def tapped_feedline(
        self,
        *,
        id: str,
        rlgc: RLGC | ParameterRef,
        left_length: object,
        right_length: object,
        n_sections: int = 1,
    ) -> ComponentInstance:
        """Build two independently parameterized N=1 line sections and a tap."""

        rlgc_value = rlgc.baseline if isinstance(rlgc, ParameterRef) else rlgc
        if not isinstance(rlgc_value, RLGC):
            raise TypeError("rlgc must be RLGC or ParameterRef")
        if len(rlgc_value.conductors) != 1:
            raise ValueError(
                "tapped_feedline requires exactly one RLGC conductor; "
                f"got {len(rlgc_value.conductors)}"
            )
        composite = CompositePlan(id=id, library=self)
        left = composite.add(
            builtin_components.transmission_line(
                id="left",
                length=left_length,
                rlgc=rlgc,
                n_sections=n_sections,
            )
        )
        right = composite.add(
            builtin_components.transmission_line(
                id="right",
                length=right_length,
                rlgc=rlgc,
                n_sections=n_sections,
            )
        )
        conductor = rlgc_value.conductors[0]
        input_boundary = composite.bus(id="input")
        tap = composite.bus(id="tap")
        output_boundary = composite.bus(id="output")
        composite.series(
            id="left_section",
            start=input_boundary,
            elements=(
                left.between(
                    left.pin("head", conductor=conductor),
                    left.pin("tail", conductor=conductor),
                ),
            ),
            end=tap,
        )
        composite.series(
            id="right_section",
            start=tap,
            elements=(
                right.between(
                    right.pin("head", conductor=conductor),
                    right.pin("tail", conductor=conductor),
                ),
            ),
            end=output_boundary,
        )
        composite.expose_pin(id="input", at=input_boundary)
        composite.expose_pin(id="tap", at=tap)
        composite.expose_pin(id="output", at=output_boundary)
        for name, value in (
            ("rlgc", rlgc),
            ("left_length", left_length),
            ("right_length", right_length),
        ):
            if isinstance(value, ParameterRef):
                composite.expose_parameter(id=name, parameter=value)
        return composite.build()


components = FeedlineLibrary()
"""Immutable custom component catalog exported by this module."""
