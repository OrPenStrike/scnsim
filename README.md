---
output-file: index.html
---

# SCNSim

**Superconducting Circuit Network Simulation**

SCNSim is a notebook-first Python package for declaring reusable
superconducting-circuit networks, compiling auditable physical operators, and
running Direct, harmonic-balance, and optimization workflows. It owns the
circuit-network layer; geometry and electromagnetic simulation remain SCGSim
responsibilities.

> **Status:** `CONVERGING`. Version `1.0.0.dev2` is an installable, fail-fast
> API scaffold for reviewing the V1 design. It does not yet implement the
> compiler, solvers, unit registry, workspace, or report runtime.

## Install for development

```bash
git clone https://github.com/OrPenStrike/scnsim.git
cd scnsim
uv sync --locked
```

Another repository should pin one reviewed SCNSim commit in its own
`pyproject.toml` and `uv.lock`:

```bash
uv add "scnsim @ git+https://github.com/OrPenStrike/scnsim.git@<reviewed-commit-sha>"
```

Teammates then clone that consuming repository and run `uv sync --locked`;
they do not repeat `uv add`.

## Inspect the current package

```bash
uv run python -c "import scnsim; print(scnsim.__version__)"
uv run python -c "from scnsim import CircuitPlan; help(CircuitPlan)"
```

Executable authoring and runtime calls deliberately raise
`ScaffoldUnavailableError` at this checkpoint.

## Read the documentation

- [Overview](README.md) — product boundary, status, and installation.
- [Tutorial](docs/index.qmd) — a five-part course from primitive circuit to
  reusable optimization and HB workflows.
- [Concept](docs/concepts/physical-authority-and-reusable-composition.qmd) —
  why SCNSim uses Plans, views, typed requests, and explicit ownership.
- [Contract](docs/component-authoring.qmd) — exact public signatures,
  invariants, failure behavior, evidence, and known limits.

Generated Notebooks are available for
[Tutorial 1](https://github.com/OrPenStrike/scnsim/blob/develop/examples/simple_resonator/01_model_author.ipynb),
[Tutorial 2](https://github.com/OrPenStrike/scnsim/blob/develop/examples/reusable_composite/01_composite_plan.ipynb),
[Tutorial 3](https://github.com/OrPenStrike/scnsim/blob/develop/examples/simple_resonator/02_model_user.ipynb),
[Tutorial 4](https://github.com/OrPenStrike/scnsim/blob/develop/examples/ipf_optimization/01_model_author.ipynb),
and [Tutorial 5](https://github.com/OrPenStrike/scnsim/blob/develop/examples/ipf_optimization/02_model_user.ipynb).

## Preview locally

After installing [Quarto](https://quarto.org/docs/get-started/):

```bash
uv sync --locked
quarto preview
```

Rendered `_site/` output is local and is not committed or deployed.

## License

SCNSim is licensed under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE)
for attribution and the separate JosephsonCircuits.jl backend boundary.
