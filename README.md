---
output-file: index.html
---

# SCNSim

**Superconducting Circuit Network Simulation**

SCNSim is being designed as the public Python package for declaring
superconducting-circuit networks, compiling auditable physical operators,
applying explicit reduction pipelines, and solving the selected model with
Direct or harmonic-balance backends.

> **Lifecycle status:** `CONVERGING`. The V1 product, component authoring,
> Runtime, Units, reduction, and failure contracts are being defined with the Human. They are not yet
> accepted, stabilized, implemented, released, or installable.

SCNSim is the network-simulation counterpart to SCGSim:

```text
SCGSim  : geometry -> mesh / EM -> report
SCNSim  : network  -> operator / reduction -> solve -> report
```

## Candidate Notebook UX

The proposed UX first builds one named physical Plan from an exact immutable
Library, then gives that sealed `CircuitPlan` an immutable graph of lazy
`NetworkViewRef` topology and view reductions. `CircuitRun` is the only
execution owner. These are non-executable API sketches; the package does not
exist yet:

```python
from scnsim import CircuitPlan, library as sc, units as u

plan = CircuitPlan(id="example")
input_cap = plan.add(
    sc.capacitor(id="input_cap", capacitance=6.0 * u.fF)
)
resonator = plan.add(
    sc.grounded_parallel_linear_lc_resonator(
        id="readout",
        subsystem_capacitance=110.0 * u.fF,
        inductance=5.8 * u.nH,
    )
)

plan.reference("ground")
plan.net("input", input_cap.pin("a"))
plan.net("readout_node", input_cap.pin("b"), resonator.pin("signal"))
plan.add_port(
    id="signal_in",
    at="input",
    role="terminated",
    reference_impedance=50.0 * u.ohm,
)
```

```python
from scnsim import (
    CMAESSpec,
    CircuitRun,
    CostObjective,
    CurrentDrive,
    DiagonalRootSpec,
    DirectSolveSpec,
    HBCaseSpec,
    HBTruncation,
    HBSolveSpec,
    OptimizationSpec,
    OptimizationVariable,
    PumpAxis,
    ReductionPipeline,
    SParameterTrace,
)

run = CircuitRun(plan=plan, workspace="results/example")

compensated = run.original.reduce(
    ReductionPipeline().ptc(
        "qubit_probe_plus",
        "qubit_probe_minus",
    )
)

qubit_modes = compensated.reduce(
    ReductionPipeline().transform_ports(
        "qubit_probe_plus",
        "qubit_probe_minus",
        id="qubit",
    )
)

feedline = qubit_modes.reduce(
    ReductionPipeline().ports(
        "feedline_in",
        "feedline_out",
    )
)

direct_spec = DirectSolveSpec(...)
direct = run.solve(feedline, direct_spec)

readout_root = DiagonalRootSpec(
    coordinate="readout",
    anchor=6.2 * u.GHz,
)
readout = run.evaluate(feedline, readout_root)

optimization = run.optimize(
    feedline,
    OptimizationSpec(
        variables=(
            OptimizationVariable(
                parameter=resonator.parameter("subsystem_capacitance"),
                bounds=(80.0 * u.fF, 140.0 * u.fF),
            ),
        ),
        objectives=(
            CostObjective(
                id="readout_frequency",
                quantity=readout_root.frequency,
                target=6.2 * u.GHz,
                weight=1.0 * u.dimensionless,
            ),
        ),
        optimizer=CMAESSpec(...),
    ),
)

pump_axis = PumpAxis(id="pump", frequency=7.0 * u.GHz)
dc_bias = CurrentDrive(id="dc_bias", at="flux_port", mode=(0,))
pump_drive = CurrentDrive(id="pump_drive", at="flux_port", mode=(1,))

hb = run.solve(
    feedline,
    HBSolveSpec(
        pump_axes=(pump_axis,),
        drives=(dc_bias, pump_drive),
        frequencies=signal_grid,
        traces=(
            SParameterTrace(
                id="signal_gain",
                input_port="feedline_in",
                input_mode=(0,),
                output_port="feedline_out",
                output_mode=(0,),
            ),
        ),
        cases=(
            HBCaseSpec(id="unbiased", currents={}),
            HBCaseSpec(
                id="dc_only",
                currents={dc_bias: 120.0 * u.uA},
            ),
            HBCaseSpec(
                id="dc_pump",
                currents={
                    dc_bias: 120.0 * u.uA,
                    pump_drive: 0.4 * u.uA,
                },
            ),
        ),
        truncation=HBTruncation(
            pump_harmonics=(3,),
            modulation_harmonics=(1,),
            three_wave_mixing=True,
            four_wave_mixing=True,
        ),
        allow_driven_ptc=True,
    ),
)

hb.show(magnitude="linear")
hb.cases["dc_pump"].s.show(magnitude="db")
```

Port-Termination Compensation (PTC) is one explicit shared topology step.
`transform_ports()` is an independent shared view step that resolves automatic
floating-pair common/differential weights from the bound Plan's complete
external capacitance cut. It neither requires nor inserts PTC. When both steps
are declared, PTC comes first and the same resolved coordinate transform is
used by Direct and every retained HB sideband.

`ports()` retains any ordered N-port view and Schur-eliminates every other port
with zero external current. The same Ref therefore drives Direct and HB
without leaving artificial probe loss in a feedline response. PTC combined
with any nonzero DC or AC operating-point drive is fail-closed unless the HB
request explicitly authorizes the documented loaded-balance interpretation.

An HB solve returns an ordered collection of user-named cases. Case IDs name
the experimental condition; Bias and Pump states are independently derived
from effective DC and nonzero-mode currents. Every case shares the request's
ordered axes, drive schema, mixing model, and truncation; inactive drives carry
exact zero current. `hb.show()` overlays declared
traces across cases and falls back to the requested selected-view matrix
elements when no traces were named. It never guesses S21, mixes incomparable
mode-frequency identities, or silently interpolates.

Direct and HB expose parallel selected-view S/Y/Z matrix families; this common
API does not imply transpose symmetry or reciprocity. Direct
physical quantities use `run.evaluate()`, and the same typed selectors feed
Direct optimization without per-candidate Python callbacks. A nonzero target
automatically normalizes its residual by `abs(target)` before the declared
weight is applied.

All public physical values use the single `scnsim.units` Pint registry. SCNSim
normalizes them to canonical SI for compilation and evidence identity while
returning typed Quantity results for Python use.

## Current contract

Review component creation, explicit finite-loop SQUIDs, resonator factories,
nets, ports, and mutual coupling in
[SCNSim V1 Component Authoring](docs/component-authoring.qmd). Then review the
Run/Ref/Result behavior and unresolved decisions in the
[SCNSim V1 Runtime Contract](docs/v1-runtime-contract.qmd). Together they are
the current product authorities for this `CONVERGING` candidate without
duplicating the reusable physics owned by SCQ_Design.

Reusable scientific meaning remains canonical in SCQ_Design: the
[floating-pair coordinate transform](https://github.com/arfiligol/SCQ_Design/blob/main/docs/knowledge/network-modeling/admittance-coordinate-transforms.qmd#full-external-cut-weighting),
[zero-current Schur boundary](https://github.com/arfiligol/SCQ_Design/blob/main/docs/knowledge/numerical-methods/schur-complement-kron-reduction.qmd#zero-current-schur-versus-matched-wave-submatrices),
[matrix-reference power waves](https://github.com/arfiligol/SCQ_Design/blob/main/docs/knowledge/simulation/port-reference-impedance-semantics.qmd#real-spd-matrix-reference-power-waves),
[Josephson element models](https://github.com/arfiligol/SCQ_Design/blob/main/docs/knowledge/josephson-physics/josephson-current-phase-energy-and-inductance.qmd#josephson-model-contracts),
[finite-loop SQUIDs](https://github.com/arfiligol/SCQ_Design/blob/main/docs/knowledge/josephson-physics/dc-squid-flux-tunability.qmd#finite-loop-mutually-pumped-dc-squid),
and [signed inductive coupling](https://github.com/arfiligol/SCQ_Design/blob/main/docs/knowledge/quantum-circuits/inductive-coupling-coefficient-mutual-self-inductance.qmd).
The Runtime also deep-links canonical
[normalized optimization](https://github.com/arfiligol/SCQ_Design/blob/main/docs/knowledge/numerical-methods/auditable-scientific-optimization.qmd#normalized-weighted-objective)
and
[HB source/mixing semantics](https://github.com/arfiligol/SCQ_Design/blob/main/docs/knowledge/numerical-methods/harmonic-balance-periodic-steady-state.qmd#source-superposition-and-generated-mixing).

## Provenance and boundaries

SCNSim is a separate repository and product, not a Workbench branch. It owns
the reusable circuit-network package/runtime, but not consumer Design Targets,
private notebooks or results, layout/EM simulation, JosephsonCircuits itself,
or the legacy Workbench product.

Reusable implementation may be transplanted only from the exact public
Workbench handoff with recorded Git provenance. Existing Workbench branches,
pull requests, consumers, and bytes remain unchanged. Consumer-specific
D3/IPF/NCUAS source, values, Gates, artifacts, and run evidence must never
enter this public repository.
