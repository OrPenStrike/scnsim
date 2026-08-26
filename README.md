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
> Runtime, Units, reduction, and failure contracts are being defined with the
> Human. The Python package is now an installable API/docstring scaffold, but
> no compiler, solver, workspace, report, or unit-registry behavior is
> implemented, accepted, stabilized, or released.

SCNSim is the network-simulation counterpart to SCGSim:

```text
SCGSim  : geometry -> mesh / EM -> report
SCNSim  : network  -> operator / reduction -> solve -> report
```

## Inspect the API scaffold

The scaffold exists so a model author can use IDE completion or Python's own
help before implementation freezes the UX:

```bash
python -m pip install -e .
```

```python
from scnsim import (
    CircuitPlan,
    DirectSolveSpec,
    ElectricNodeRef,
    OperatorSpec,
    PortRef,
    ReductionPipeline,
    ReportSpec,
    ScalarRLGC,
    load_q2d_scalar_rlgc,
)

help(CircuitPlan)
help(ElectricNodeRef)
help(PortRef)
help(ReductionPipeline)
help(DirectSolveSpec)
help(OperatorSpec)
help(ReportSpec)
help(ScalarRLGC)
help(load_q2d_scalar_rlgc)
```

Every construction or operation raises `ScaffoldUnavailableError`. This is
intentional: the current package is safe for interface review but cannot
produce a fake successful simulation.

## Two Notebook UX layers

SCNSim supports two different users without creating two physical authorities:

| User | Writes once | Reuses later |
|---|---|---|
| Circuit-model developer | Library components, electric nodes, reference, logical Ports, Ref graph, Specs, and project-owned objectives | The exact same model façade and evidence identity |
| Model consumer | Project Design Target and workspace | A few team-owned functions returning typed SCNSim Results |

The package owns the general Plan/Ref/Run/Spec/Result vocabulary. A design team
owns a thin ordinary Python module that hides its repeated topology and
analysis choices. SCNSim does not absorb that team's Design Target or provide a
global model registry.

The persistent examples show both views of the same model:

- [model-author Notebook](examples/simple_resonator/01_model_author.ipynb)
  starts at `CircuitPlan` and explains each Spec in context;
- [model-user Notebook](examples/simple_resonator/02_model_user.ipynb) imports
  the finished [team model façade](examples/simple_resonator/circuit_model.py),
  provides a target, then calls optimization/report functions without
  restating topology.

The tracked `.ipynb` files are the one Notebook source: open them locally in
VS Code to execute cell by cell, read the same files directly on GitHub, or
render them as site pages with Quarto. SCNSim does not maintain a second copied
Notebook authority.

## Choose the operation before the Spec

The method name states the kind of work; the Spec makes that request exact:

| Goal | Terminal call | Spec family | Returned surface |
|---|---|---|---|
| Inspect S/Y/Z over a frequency grid | `run.solve(view, spec)` | `DirectSolveSpec` | Complete selected-view matrices and named traces |
| Inspect HB S/Y/Z for named operating cases | `run.solve(view, spec)` | `HBSolveSpec` | One `HBBatchResult` indexed by case ID |
| Obtain one Direct physical quantity | `run.evaluate(view, spec)` | `DiagonalRootSpec`, `HybridizedPoleSpec`, `TransferZeroSpec`, `ResidueNormalizedCouplingSpec`, or `ResponseElementSpec` | One typed quantity Result without an unrelated response sweep |
| Materialize the Direct operator | `run.evaluate(view, spec)` | `OperatorSpec` | Labeled operator Result |
| Search over typed quantities | `run.optimize(view, spec)` | `OptimizationSpec` | Optimization Result and reusable best `ParameterSet` |

`SolveSpec` therefore means “produce a response surface.” Other evaluation
Specs select a physical quantity or operator. `result.show()` only presents an
existing Result; it never solves again. Passing an evaluation Spec to
`solve()`, or a SolveSpec to `evaluate()`, fails closed.
`SolveSpec` is a prose category, not another public class to construct; the V1
classes in that category are `DirectSolveSpec` and `HBSolveSpec`.

## Candidate Notebook UX

The proposed UX first composes component pins into Public or Internal electric
nodes, then places logical Port components on selected nodes. The sealed
`CircuitPlan` receives one backend-neutral graph of lazy `NetworkViewRef`
reductions expressed in node coordinates. `CircuitRun` is the only execution
owner. These are non-executable API sketches; the package does not yet
implement them:

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
input_node = plan.net(input_cap.pin("a"))
plan.net(
    input_cap.pin("b"),
    resonator.pin("signal"),
    id="readout_node",
)
signal_port = plan.add_port(
    id="signal_in",
    at=input_node,
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

response_view = run.original.reduce(
    ReductionPipeline().retain("signal_in")
)

quantity_view = run.original.reduce(
    ReductionPipeline().retain("readout_node")
)

direct_spec = DirectSolveSpec(...)
direct = run.solve(response_view, direct_spec)

readout_root = DiagonalRootSpec(
    coordinate="readout_node",
    root_hint=6.0 * u.GHz,
)
readout = run.evaluate(quantity_view, readout_root)

optimization = run.optimize(
    quantity_view,
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
        optimizer=CMAESSpec(seed=17, max_evaluations=200),
    ),
)

pump_axis = PumpAxis(id="pump", frequency=7.0 * u.GHz)
dc_bias = CurrentDrive(id="dc_bias", at=signal_port, mode=(0,))
pump_drive = CurrentDrive(id="pump_drive", at=signal_port, mode=(1,))

hb = run.solve(
    response_view,
    HBSolveSpec(
        pump_axes=(pump_axis,),
        drives=(dc_bias, pump_drive),
        frequencies=signal_grid,
        traces=(
            SParameterTrace(
                id="reflection",
                input_port="signal_in",
                input_mode=(0,),
                output_port="signal_in",
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

`root_hint` is a required model-author input that identifies the baseline
simple-root branch. It is not the returned frequency or the optimization
target; a reusable team model declares it once so downstream consumers do not
repeat it.

Port-Termination Compensation (PTC) is one explicit shared topology step that
targets logical `PortRef` objects. Every other reduction verb uses node
coordinates. `transform_pair()` resolves automatic floating-pair
common/differential weights from the bound Plan's complete external
capacitance cut, and `retain()` selects the ordered node-coordinate view while
Schur-eliminating its complement at zero external current.

The final coordinate IDs are also the selected-view wave-channel IDs used by
`SParameterTrace`; Port IDs remain separate except when anonymous-node
promotion deliberately gives both the same string. `CurrentDrive.at` always
uses the returned `PortRef`, never a channel string.

The pipeline never selects a backend. Direct and HB use the same Ref whenever
its final coordinates are realizable through logical Ports. HB performs any
omitted-port elimination on the linearized full mode-port response after the
complete nonlinear balance. PTC combined with any nonzero DC or AC
operating-point drive remains fail-closed unless the HB request explicitly
authorizes the documented loaded-balance interpretation.

An HB solve returns an ordered collection of user-named cases. Case IDs name
the experimental condition; Bias and Pump states are independently derived
from effective DC and nonzero-mode currents. Every case shares the request's
ordered axes, drive schema, mixing model, and truncation; inactive drives carry
exact zero current. `hb.show()` overlays declared
traces across cases and falls back to the requested selected-view matrix
elements when no traces were named. It never guesses S21, mixes incomparable
mode-frequency identities, or silently interpolates.

Direct and HB expose parallel selected-view S/Y/Z matrix families; this common
API does not imply transpose symmetry or reciprocity. The selected matrices
retain every non-compensated logical-Port load and use the induced retained
reference matrix; JosephsonCircuits native S remains reconciliation evidence.
Direct
physical quantities use `run.evaluate()`, and the same typed selectors feed
Direct optimization without per-candidate Python callbacks. A nonzero target
automatically normalizes its residual by `abs(target)` before the declared
weight is applied.

All public physical values use the single `scnsim.units` Pint registry. SCNSim
normalizes them to canonical SI for compilation and evidence identity while
returning typed Quantity results for Python use.

## Current contract

Review component creation, explicit finite-loop SQUIDs, resonator factories,
Public/Internal electric nodes, logical Ports, and mutual coupling in
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
