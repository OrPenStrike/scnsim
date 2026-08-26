"""Public synthetic IPF Composite used to review the SCNSim V1 authoring UX.

The declarations intentionally contain no parser, compiler, solver, or hidden
success path.  They show how a custom Library author exposes five physical
parameters, calibrated affine mappings, external pins, and one internal
analysis coordinate while keeping the child topology private.
"""

from __future__ import annotations

from scnsim import (
    AffineMap,
    ComponentInstance,
    CompositePlan,
    Library,
    ParameterSpec,
    RLGC,
    library as sc,
    units as u,
)


class IPFLibrary(Library):
    """Immutable custom catalog for the synthetic intrinsic Purcell filter."""

    def intrinsic_purcell_filter(
        self,
        *,
        id: str,
        readout_rlgc: RLGC,
        filter_rlgc: RLGC,
        coupled_rlgc: RLGC,
    ) -> ComponentInstance:
        """Declare a five-section IPF with five public optimization parameters."""

        component = CompositePlan(id=id, library=self)

        readout_open_length = component.parameter(
            id="readout_open_length",
            baseline=2.4 * u.mm,
            spec=ParameterSpec(unit=u.mm),
        )
        shared_short_length = component.parameter(
            id="shared_short_length",
            baseline=0.9 * u.mm,
            spec=ParameterSpec(unit=u.mm),
        )
        coupled_length = component.parameter(
            id="coupled_length",
            baseline=1.6 * u.mm,
            spec=ParameterSpec(unit=u.mm),
        )
        filter_open_length = component.parameter(
            id="filter_open_length",
            baseline=2.1 * u.mm,
            spec=ParameterSpec(unit=u.mm),
        )
        idc_finger_length = component.parameter(
            id="idc_finger_length",
            baseline=52.0 * u.um,
            spec=ParameterSpec(unit=u.um),
        )

        readout_open = component.add(
            sc.transmission_line(
                id="readout_open",
                length=readout_open_length,
                rlgc=readout_rlgc,
                n_sections=24,
            )
        )
        readout_short = component.add(
            sc.transmission_line(
                id="readout_short",
                length=shared_short_length,
                rlgc=readout_rlgc,
                n_sections=12,
            )
        )
        coupled = component.add(
            sc.transmission_line(
                id="coupled",
                length=coupled_length,
                rlgc=coupled_rlgc,
                n_sections=20,
            )
        )
        filter_short = component.add(
            sc.transmission_line(
                id="filter_short",
                length=shared_short_length,
                rlgc=filter_rlgc,
                n_sections=12,
            )
        )
        filter_open = component.add(
            sc.transmission_line(
                id="filter_open",
                length=filter_open_length,
                rlgc=filter_rlgc,
                n_sections=24,
            )
        )

        calibration_support = (35.0 * u.um, 70.0 * u.um)
        idc = component.add(
            sc.interdigitated_capacitor(
                id="idc",
                terminal_1_to_reference_capacitance=AffineMap(
                    input=idc_finger_length,
                    slope=0.08 * u.fF / u.um,
                    intercept=1.2 * u.fF,
                    support=calibration_support,
                ),
                terminal_2_to_reference_capacitance=AffineMap(
                    input=idc_finger_length,
                    slope=0.09 * u.fF / u.um,
                    intercept=1.0 * u.fF,
                    support=calibration_support,
                ),
                terminal_mutual_capacitance=AffineMap(
                    input=idc_finger_length,
                    slope=0.12 * u.fF / u.um,
                    intercept=0.8 * u.fF,
                    support=calibration_support,
                ),
            )
        )

        feedline_in = component.net(
            readout_open.pin("head", conductor="readout"),
            idc.pin("terminal_1"),
        )
        component.net(
            readout_open.pin("tail", conductor="readout"),
            readout_short.pin("head", conductor="readout"),
        )
        component.net(
            readout_short.pin("tail", conductor="readout"),
            coupled.pin("head", conductor="readout"),
        )
        feedline_out = component.net(
            coupled.pin("tail", conductor="readout"),
        )

        filter_open_tail = component.net(
            filter_open.pin("tail", conductor="filter"),
        )
        component.net(
            filter_open.pin("head", conductor="filter"),
            coupled.pin("head", conductor="filter"),
        )
        component.net(
            coupled.pin("tail", conductor="filter"),
            filter_short.pin("head", conductor="filter"),
        )
        component.net(
            filter_short.pin("tail", conductor="filter"),
            idc.pin("terminal_2"),
        )

        component.expose_pin(id="feedline_in", at=feedline_in)
        component.expose_pin(id="feedline_out", at=feedline_out)
        component.expose_coordinate(
            id="filter_open_tail",
            at=filter_open_tail,
        )
        return component.build()


library = object.__new__(IPFLibrary)
"""Immutable synthetic custom Library object for the public example."""
