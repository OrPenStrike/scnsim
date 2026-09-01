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

> **Status:** `STABILIZED` for the dev4 reusable-composition scope taught in
> Lessons 6–8. The dev3 execution scope and dev5 Lessons 9–11 candidate remain
> `CONVERGING`; HB execution remains explicit fail-fast work for dev6.

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
uv run python -c "from scnsim import CircuitPlan, components; help(components.capacitor); help(CircuitPlan)"
```

Primitive, Composite, RLGC, PTC/transform, expanded Direct, optimization, and
inventory paths through Lessons 1–11 are executable candidates. HB operations
continue to raise `ScaffoldUnavailableError` until dev6 lands.

## Read the documentation

- [Overview](README.md) — product boundary, status, and installation.
- [Tutorial](docs/index.qmd) — thirteen focused native-API lessons from
  primitive authoring through optimization, reuse, and pump-off HB.
- [Concept](docs/concepts/physical-authority-and-reusable-composition.qmd) —
  why SCNSim uses Plans, views, typed requests, and explicit ownership.
- [Contract](docs/contracts/index.qmd) — public behavior plus the maintainer
  implementation design followed by the current executable Runtime slices.

Each lesson also has a generated Notebook in
[`examples/tutorials/`](https://github.com/OrPenStrike/scnsim/tree/develop/examples/tutorials)
for GitHub preview or VS Code/Jupyter execution.

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
