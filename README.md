---
output-file: index.html
---

# SCNSim

**Superconducting Circuit Network Simulation**

SCNSim is a notebook-first Python package for declaring superconducting-circuit
networks, compiling auditable physical operators, applying explicit network
views, and running Direct or harmonic-balance analysis and optimization.

SCNSim complements SCGSim: SCGSim owns geometry-to-EM workflows, while SCNSim
owns circuit-network compilation, reduction, solve, optimization, and report
interfaces.

> **Lifecycle status:** `CONVERGING`. Version `1.0.0.dev2` is an installable,
> fail-fast API and docstring scaffold for Human review. It does not yet
> implement a compiler, solver, unit registry, workspace, or report runtime,
> and it cannot produce simulation results.

## Installation

Pin an exact reviewed development revision from another uv project:

```bash
SCNSIM_REVISION=replace-with-reviewed-commit-sha
uv add "scnsim @ git+https://github.com/OrPenStrike/scnsim.git@${SCNSIM_REVISION}"
```

For local documentation and scaffold development:

```bash
git clone https://github.com/OrPenStrike/scnsim.git
cd scnsim
uv sync --locked
```

## Usage

The current honest usage surface is API inspection:

```bash
uv run python -c "import scnsim; print(scnsim.__version__)"
uv run python -c "from scnsim import CircuitPlan; help(CircuitPlan)"
```

Executable authoring and runtime calls deliberately raise
`ScaffoldUnavailableError` until their candidate implementations exist.

## Documentation

| Start here | What it explains |
| --- | --- |
| [Documentation overview](docs/index.qmd) | Reader paths, lifecycle status, and the current review map |
| [Component Authoring](docs/component-authoring.qmd) | Libraries, components, electric nodes, logical Ports, RLGC, and composites |
| [V1 Runtime Contract](docs/v1-runtime-contract.qmd) | Plan/Ref/Run/Spec/Result, reduction, Direct/HB, optimization, and receipts |
| [Simple model author](examples/simple_resonator/01_model_author.ipynb) | Building and packaging one reusable circuit model |
| [Simple model user](examples/simple_resonator/02_model_user.ipynb) | Consuming a team-owned model façade |
| [IPF model author](examples/ipf_optimization/01_model_author.ipynb) | N-trace RLGC, composite authoring, optimization, Direct, and pump-off HB |
| [IPF model user](examples/ipf_optimization/02_model_user.ipynb) | Target-driven reuse without restating circuit topology |

The tracked Notebooks have no saved outputs or execution counts. GitHub renders
them directly, VS Code can open them locally, and Quarto uses the same files as
the documentation source.

## Local documentation preview

After installing [Quarto](https://quarto.org/docs/get-started/):

```bash
uv sync --locked
quarto preview
```

Rendered `_site/` output is local and is not committed or deployed. GitHub Pages
will be considered separately after the relevant V1 semantics are accepted and
a stable `main` line exists.

## License

SCNSim is licensed under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE)
for attribution and the separate MIT-licensed JosephsonCircuits.jl backend
boundary.
