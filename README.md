---
output-file: index.html
---

# SCNSim

**Superconducting Circuit Network Simulation**

SCNSim is a notebook-first Python package for building an equivalent-circuit
model, checking the circuit that was declared, and asking reproducible network
questions of it. A typical model might contain a parallel LC resonator, a
coupling capacitor, and a terminated feedline Port. The author states those
parts, their values and wiring, which parts belong to a subsystem, and which
subsystem boundaries are public. SCNSim does not infer that engineering intent
from names, numerical values, a picture, or a calculated mode.

The engineer course works toward a complete review package: a diagram
generated from the authored circuit, its independent A/B correspondence audit,
and typed analytical Results that point back to the same Plan while recording
their separately selected View and request. Tutorial authoring figures are
genuine exports tied to their source checkpoints; they are not solver outputs
or evidence that every course cell has executed successfully.

Begin with the [engineer course setup](examples/engineer/setup.qmd), then use
[Chapter 1](examples/engineer/chapter-01/01-build.qmd) to build and inspect one
coupled grounded LC. [Chapter 2](examples/engineer/chapter-02/01-build.qmd)
rebuilds that circuit with independent C and L inputs, completes a three-point
capacitance sweep, and then teaches the established Optimization workflow and
optional C/L spaces. Each complete zero-output Chapter Notebook is its own
clean-kernel execution unit. [Chapter 3](examples/engineer/chapter-03/01-build.qmd)
adds a two-section CPW feedline, grounded readout, and named two-Port S21.
[Chapter 4](examples/engineer/chapter-04/01-physical-plan.qmd) rebuilds the
four-Port capstone, completes its original/PTC core, then offers transform,
quantity, and pump-off HB continuations. The remaining four Chapters are a
published roadmap, not placeholder pages. The former sixteen-Chapter sources remain in a
single Migration legacy group while later batches move their reviewed content.

## Start with the circuit, then choose the question

One complete `CircuitPlan` is the physical and structural source shared by two
separate paths:

```text
explicitly authored CircuitPlan
          |
          +----> Automatic Semantic Circuit Diagram Layout System (ASCDLS)
                 diagram of the authored or compiled Plan
          |
          +----> CircuitRun -> original or selected View -> request -> Result
```

The diagram path translates declared electrical and assembly facts into
automatic geometry, then independently checks that projection against the
Plan. It answers “did the picture preserve the circuit I wrote?” It does not
choose an analysis coordinate, reduction, Port-Termination Compensation (PTC),
parameter override, or solver request.

The analysis path starts from the same Plan. A `CircuitRun` supplies
`run.original`; the user derives a `NetworkViewRef` only when the question
needs different coordinates, compensation, or a retained boundary. A Solve,
Evaluate, or Optimization Spec then states the calculation, and a typed Result
owns the answer and its evidence. The original drawing therefore remains a
drawing of the authored circuit even when an analysis View asks about, for
example, common and differential coordinates of two physical terminals.

| The author must declare | SCNSim may derive |
|---|---|
| components, physical values, wiring, ground, and logical Ports | validated electrical and compiled operators |
| subsystem ownership, series/parallel structure, buses/taps, and public pins | an automatic authoring or compiled diagram |
| independent parameter definitions and their physical-field bindings | selected parameter points and parameterized diagrams |
| the analysis question, selected View, parameter point or space, and request | Direct, harmonic-balance, sweep, optimization, Result, and report evidence |

SCNSim owns this circuit-network layer. Physical geometry and electromagnetic
simulation remain SCGSim responsibilities.

> **Current status:** accepted Direct/HB numerical algorithms and physical
> models are not reopened. Structured authoring, the independent Parameter
> System, and parameterized diagrams remain `CONVERGING` with an integrated
> implementation baseline. See the centralized
> [implementation and evidence status](docs/contracts/index.qmd#implementation-evidence-status)
> for scoped CI, Tutorial, historical-test, and delivery limitations.

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

## Inspect your installed version

```bash
uv run python -c "import scnsim; print(scnsim.__version__)"
uv run python -c "from scnsim import CircuitPlan, components; help(components.capacitor); help(CircuitPlan)"
```

These commands inspect the installed package surface. They do not establish
that the complete TARGET Tutorial is executable. The retained numerical
contracts still cover Direct, HB, optimization, views, exact resolution, and
reports; this documentation rewrite does not change those meanings or add HB
optimization, interpolation, release delivery, or a SCQGate handoff.

## Candidate CI scope

The [central evidence status](docs/contracts/index.qmd#implementation-evidence-status)
distinguishes current runnable checks, numerical regressions blocked by old
fixtures, and superseded historical expectations. Candidate CI success is not
a full-suite pass, whole-course execution, Full V1 restabilization, or Human
acceptance.

## Read the documentation

- [Overview](README.md) — the model-first workflow, product boundary, status,
  and installation.
- [Engineer course](docs/index.qmd) — current Chapters 1–4 and the
  eight-Chapter roadmap, with the remaining four Chapters marked future.
- [Concept](docs/concepts/physical-authority-and-reusable-composition.qmd) —
  why SCNSim separates authored circuit facts, diagram geometry, and analysis
  Views while keeping one Plan authority.
- [Parameters](docs/concepts/units-parameters-and-optimization.qmd) — why a
  named input, its physical binding, a selected point, a sweep, and an
  optimization variable are different objects.
- [Contract](docs/contracts/index.qmd) — public behavior plus the maintainer
  implementation design followed by the current executable Runtime slices.

Each implemented Chapter has one zero-output generated Notebook for GitHub or
VS Code/Jupyter review. QMD fragments are the only editable lesson source; the
aggregate Notebook is a derived transport artifact, not an execution record.
The old sixteen QMD/IPYNB pairs remain under `examples/tutorials/` as migration
legacy with their source bindings intact.
The website displays separately generated authoring SVGs without executing
kernels or solvers. See the [source and figure workflow](examples/README.md)
for the execution boundary. Neither a figure nor a Notebook establishes
whole-course support, Human acceptance, or release status.

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
