---
output-file: index.html
---

# SCNSim

**Superconducting Circuit Network Simulation**

SCNSim is being designed as the public Python package for declaring
superconducting-circuit networks, compiling auditable physical operators,
applying explicit reduction pipelines, and solving the selected model with
Direct or harmonic-balance backends.

> **Lifecycle status:** `CONVERGING`. The V1 product, API, Units, reduction,
> and failure contracts are being defined with the Human. They are not yet
> accepted, stabilized, integrated, released, or installable.

SCNSim is the network-simulation counterpart to SCGSim:

```text
SCGSim  : geometry -> mesh / EM -> report
SCNSim  : network  -> operator / reduction -> solve -> report
```

## Candidate Notebook UX

The proposed UX gives one sealed `CircuitPlan` an immutable graph of lazy
`NetworkViewRef` topology and view reductions. `CircuitRun` is the only
execution owner. This is a non-executable API sketch; the package does not
exist yet:

```python
from scnsim import (
    CircuitRun,
    DirectSolveSpec,
    HBSolveSpec,
    NoPumpingSpec,
    ReductionPipeline,
)

run = CircuitRun(plan=plan, workspace="results/example")

compensated = run.original.reduce(
    ReductionPipeline().ptc(
        "qubit_probe_plus",
        "qubit_probe_minus",
    )
)

feedline = compensated.reduce(
    ReductionPipeline().ports(
        "feedline_in",
        "feedline_out",
    )
)

direct = run.solve(feedline, DirectSolveSpec(...))
hb = run.solve(
    feedline,
    HBSolveSpec(
        cases=(NoPumpingSpec(id="pump_off"),),
    ),
)
```

Port-Termination Compensation (PTC) is one explicit shared topology step.
`ports()` retains any ordered N-port view and Schur-eliminates every other port
with zero external current. The same Ref therefore drives Direct and pump-off
HB without leaving artificial probe loss in a feedline response. Pump-on PTC
is fail-closed unless the HB request explicitly authorizes the documented
loaded-balance interpretation.

All public physical values use the single `scnsim.units` Pint registry. SCNSim
normalizes them to canonical SI for compilation and evidence identity while
returning typed Quantity results for Python use.

Automatic floating-node `transform_ports` weighting remains an explicit open
V1 decision and is not implied by this example.

## Current contract

Review the proposed behavior and unresolved decisions in the
[SCNSim V1 Runtime Contract](docs/v1-runtime-contract.qmd), the single current
semantic authority for this `CONVERGING` candidate.

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
