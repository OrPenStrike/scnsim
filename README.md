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
`NetworkModelRef` reductions. `CircuitRun` is the only execution owner. This
is a non-executable API sketch; the package does not exist yet:

```python
from scnsim import CircuitRun, ParameterSet, units as u

run = CircuitRun(plan=plan, workspace="results/example")
readout = run.original.reduce(readout_pipeline)

direct = run.solve(readout, direct_spec)
hb = run.solve(readout, hb_spec)
operator = run.evaluate(readout, operator_spec)

candidate = ParameterSet({
    capacitor.parameter("capacitance"): 95.0 * u.fF,
})
candidate_direct = run.solve(readout, direct_spec, parameters=candidate)
```

All public physical values use the single `scnsim.units` Pint registry. SCNSim
normalizes them to canonical SI for compilation and evidence identity while
returning typed Quantity results for Python use.

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
