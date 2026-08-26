"""Public request declarations for Direct, optimization, HB, and reporting.

A Spec describes *what* a :class:`~scnsim.runtime.CircuitRun` should do.  It
does not execute work, hold a result, or pick a mutable current model.  The
class docstrings below are deliberately user-facing: ``help(SpecName)`` should
answer when that Spec is appropriate before a beginner writes a request.

Constructors fail in this scaffold because validation and identity sealing are
not implemented yet.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from ._scaffold import unavailable
from .authoring import ParameterRef


class DirectSolveSpec:
    """Request a linear frequency-domain response on one selected view.

    Use this when the desired output is the Direct S/Y/Z matrix over a
    frequency grid, optionally with named scalar traces.  It is a *solve*
    request, not a root finder and not the full dynamic operator.  For those
    tasks use ``DiagonalRootSpec``/``HybridizedPoleSpec`` or ``OperatorSpec``.
    """

    def __init__(
        self,
        *,
        frequencies: object,
        traces: Sequence[SParameterTrace] = (),
    ) -> None:
        unavailable("DirectSolveSpec construction")


class DiagonalRootSpec:
    """Select one anchored root of a named Direct-operator diagonal.

    This is the local or "bare-coordinate" resonance associated with one
    selected-view coordinate, not necessarily a hybridized pole of the full
    coupled network.  ``anchor`` selects the intended branch.  The Spec exposes
    quantity selectors such as ``frequency`` and ``linewidth`` for reuse in an
    ``OptimizationSpec``; accessing a selector does not run a solver.
    """

    def __init__(self, *, coordinate: str, anchor: object) -> None:
        unavailable("DiagonalRootSpec construction")

    @property
    def frequency(self) -> object:
        """Selector for ``real(root) / (2 pi)``."""

        unavailable("DiagonalRootSpec.frequency selector")

    @property
    def linewidth(self) -> object:
        """Selector for the positive passive linewidth of the chosen root."""

        unavailable("DiagonalRootSpec.linewidth selector")


class HybridizedPoleSpec:
    """Select an anchored complex pole of a retained coupled block.

    Use this when coupling between several named coordinates is part of the
    physical mode being measured.  SCNSim finds a root of the ordered block
    determinant and records branch/simple-root evidence; it does not relabel a
    diagonal root as a hybridized pole.
    """

    def __init__(self, *, coordinates: Sequence[str], anchor: object) -> None:
        unavailable("HybridizedPoleSpec construction")

    @property
    def frequency(self) -> object:
        """Selector for the chosen pole's physical frequency."""

        unavailable("HybridizedPoleSpec.frequency selector")

    @property
    def linewidth(self) -> object:
        """Selector for the chosen pole's positive passive linewidth."""

        unavailable("HybridizedPoleSpec.linewidth selector")


class TransferZeroSpec:
    """Select an anchored exact zero of one declared transfer function.

    The transfer may be identified by a cofactor or by one ordered S/Y/Z
    projection.  Exactly one selection form must eventually validate.  This is
    an analytic numerator zero with a finite denominator, never the minimum of
    a sampled sweep and never an unresolved pole-zero cancellation.

    The optional arguments expose an unresolved UX edge deliberately: Human
    review still needs to decide whether cofactor selection deserves its own
    typed public object before implementation.
    """

    def __init__(
        self,
        *,
        anchor: object,
        cofactor: object | None = None,
        family: Literal["S", "Y", "Z"] | None = None,
        input_coordinate: str | None = None,
        output_coordinate: str | None = None,
    ) -> None:
        unavailable("TransferZeroSpec construction")

    @property
    def frequency(self) -> object:
        """Selector for the selected zero's physical frequency."""

        unavailable("TransferZeroSpec.frequency selector")


class ResidueNormalizedCouplingSpec:
    """Evaluate local complex coupling between two simple-root branches.

    Use this for coupling derived from the two declared branch residues at one
    explicit evaluation point.  It is not a fitted normal-mode splitting and
    it fails when either required branch is not a unique simple root.
    """

    def __init__(
        self,
        *,
        branch_a: DiagonalRootSpec | HybridizedPoleSpec,
        branch_b: DiagonalRootSpec | HybridizedPoleSpec,
        frequency: object,
    ) -> None:
        unavailable("ResidueNormalizedCouplingSpec construction")

    @property
    def magnitude(self) -> object:
        """Selector for the magnitude of the evaluated complex coupling."""

        unavailable("ResidueNormalizedCouplingSpec.magnitude selector")


class ResponseElementSpec:
    """Evaluate one ordered S/Y/Z matrix element at one exact frequency.

    Use this when an optimization or downstream calculation needs one scalar
    response rather than a sweep.  Input/output names follow the selected
    ``NetworkViewRef`` order; the Spec does not guess S21 or interpolate.
    """

    def __init__(
        self,
        *,
        family: Literal["S", "Y", "Z"],
        input_coordinate: str,
        output_coordinate: str,
        frequency: object,
    ) -> None:
        unavailable("ResponseElementSpec construction")

    @property
    def magnitude(self) -> object:
        """Selector for the scalar response magnitude."""

        unavailable("ResponseElementSpec.magnitude selector")

    @property
    def real(self) -> object:
        """Selector for the scalar response real part."""

        unavailable("ResponseElementSpec.real selector")

    @property
    def imag(self) -> object:
        """Selector for the scalar response imaginary part."""

        unavailable("ResponseElementSpec.imag selector")


class OperatorSpec:
    """Materialize the full Direct dynamic operator on a declared grid.

    Use this for labeled matrix inspection or custom Python calculations that
    are not covered by the small typed quantity catalog.  A full operator is
    not a scalar and therefore cannot be used directly as an optimization
    objective.
    """

    def __init__(self, *, frequencies: object) -> None:
        unavailable("OperatorSpec construction")


class OptimizationVariable:
    """Bind one public component parameter to finite physical search bounds.

    SCNSim canonicalizes the bound Quantities and maps them to dimensionless
    optimizer coordinates.  ``transform='log'`` is explicit; no unit name or
    small SI magnitude silently changes the search coordinates.
    """

    def __init__(
        self,
        *,
        parameter: ParameterRef,
        bounds: tuple[object, object],
        transform: Literal["linear", "log"] = "linear",
    ) -> None:
        unavailable("OptimizationVariable construction")


class QuantitySum:
    """The V1 scalar composition: a sum of same-dimensionality selectors.

    It exists only to express a small auditable objective dependency graph.
    Arbitrary Python callbacks and a general expression language are outside
    V1.
    """

    def __init__(self, *terms: object) -> None:
        unavailable("QuantitySum construction")


class CostObjective:
    """Compare one scalar quantity with one target inside optimization.

    The residual is ``(quantity - target) / scale`` and the total cost adds
    ``weight * abs(residual)**2``.  If ``scale`` is omitted, SCNSim uses the
    nonzero target magnitude, or unity for an exact dimensionless zero target.
    An exact dimensionful zero target requires an explicit physical scale.
    ``weight`` is dimensionless relative importance, not a tolerance or Gate.
    """

    def __init__(
        self,
        *,
        id: str,
        quantity: object,
        target: object,
        weight: object,
        scale: object | None = None,
    ) -> None:
        unavailable("CostObjective construction")


class CMAESSpec:
    """Declare deterministic CMA-ES execution controls, not design success.

    ``max_evaluations`` is a finite work budget.  It does not decide whether a
    circuit meets a Design Target; only a Human-owned accepted Gate may do so.
    The minimal scaffold intentionally omits speculative convergence knobs.
    """

    def __init__(
        self,
        *,
        seed: int,
        max_evaluations: int,
        population_size: int | None = None,
    ) -> None:
        unavailable("CMAESSpec construction")


class OptimizationSpec:
    """Plan one Direct-only, multi-variable, multi-objective search.

    Quantity selectors define how every objective is computed.  Candidate
    binding, Plan compilation, reduction replay, shared quantity evaluation,
    objective aggregation, and CMA-ES remain inside one Julia process; Python
    does not receive a callback for each candidate.
    """

    def __init__(
        self,
        *,
        variables: Sequence[OptimizationVariable],
        objectives: Sequence[CostObjective],
        optimizer: CMAESSpec,
    ) -> None:
        unavailable("OptimizationSpec construction")


class PumpAxis:
    """Name one independent HB fundamental frequency.

    A commensurate harmonic belongs to a higher integer mode of the same axis,
    not a duplicate axis.  Pump amplitude belongs to an ``HBCaseSpec`` drive
    binding, not to this axis.
    """

    def __init__(self, *, id: str, frequency: object) -> None:
        unavailable("PumpAxis construction")


class CurrentDrive:
    """Name one current injection location and Fourier-lattice mode.

    The drive declares *where and at which mode* current can be applied.  Each
    case supplies its own complex Fourier coefficient.  ``mode=()`` is pure DC;
    otherwise tuple rank and ordering follow ``HBSolveSpec.pump_axes``.
    """

    def __init__(self, *, id: str, at: str, mode: tuple[int, ...]) -> None:
        unavailable("CurrentDrive construction")


class HBCaseSpec:
    """One user-named DC/AC operating condition in an HB batch.

    ``currents`` maps request-global ``CurrentDrive`` objects to physical
    Fourier coefficients.  Omitted drives are exact zero.  Case IDs name
    experiments such as ``baseline`` or ``pump_high``; Bias/Pump state is a
    derived result classification, not a lookup key.
    """

    def __init__(self, *, id: str, currents: Mapping[CurrentDrive, object]) -> None:
        unavailable("HBCaseSpec construction")


class HBTruncation:
    """Declare one finite HB operating-point and modulation lattice.

    Per-axis bounds keep the lattice finite.  ``max_intermodulation_order``
    optionally adds the L1 crop ``sum(abs(mode)) <= N``.  The 3WM/4WM flags
    select interactions kept in the same nonlinear balance and linearized
    response; they are not separate solves or a count of pump tones.
    """

    def __init__(
        self,
        *,
        pump_harmonics: tuple[int, ...],
        modulation_harmonics: tuple[int, ...],
        three_wave_mixing: bool,
        four_wave_mixing: bool,
        max_intermodulation_order: int | None = None,
    ) -> None:
        unavailable("HBTruncation construction")


class SParameterTrace:
    """Name one ordered S-matrix projection across a solve frequency grid.

    A trace is a view of the complete matrix result, not another solve.  Direct
    traces use empty mode tuples; HB tuples follow the declared pump-axis
    ordering.
    """

    def __init__(
        self,
        *,
        id: str,
        input_port: str,
        input_mode: tuple[int, ...],
        output_port: str,
        output_mode: tuple[int, ...],
    ) -> None:
        unavailable("SParameterTrace construction")


class HBSolveSpec:
    """Request a shared-basis batch of nonlinear HB operating conditions.

    Axes, drive schema, signal grid, traces, mixing selection, and truncation
    are request-global so every named case is comparable.  ``allow_driven_ptc``
    must be explicit when a PTC view has nonzero DC or AC drive; authorization
    preserves the loaded operating point and compensates only its linearized
    response.
    """

    def __init__(
        self,
        *,
        pump_axes: Sequence[PumpAxis],
        drives: Sequence[CurrentDrive],
        frequencies: object,
        cases: Sequence[HBCaseSpec],
        truncation: HBTruncation,
        traces: Sequence[SParameterTrace] = (),
        allow_driven_ptc: bool = False,
    ) -> None:
        unavailable("HBSolveSpec construction")


class ReportSpec:
    """Choose exact existing Results to assemble into one auditable report.

    Report assembly never searches for "latest" evidence and does not solve,
    interpolate, or create a Design Target.  Each input keeps its exact result
    identity and may be a Direct result, one HB case, or an optimization result.
    """

    def __init__(self, *, inputs: Sequence[object]) -> None:
        unavailable("ReportSpec construction")


class CircuitDiagramSpec:
    """Describe presentation of the original Plan's schematic projection.

    The rendered diagram is inspection evidence only.  It cannot redefine
    topology, display a fabricated reduced equivalent circuit, or invalidate
    numerical receipts when its theme or layout changes.

    Concrete layout/theme controls remain intentionally undecided in this
    scaffold; the empty constructor exposes that open UX without inventing a
    configuration surface.
    """

    def __init__(self) -> None:
        unavailable("CircuitDiagramSpec construction")
