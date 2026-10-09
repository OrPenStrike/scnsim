# SCNSim

**Author one superconducting circuit, then ask typed network and quantity
questions against it.**

[![CI](https://github.com/OrPenStrike/scnsim/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/OrPenStrike/scnsim/actions/workflows/ci.yml?query=branch%3Adevelop)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python requirement: >=3.12](https://img.shields.io/badge/python-%3E%3D3.12-blue.svg)](pyproject.toml)
[![Tutorial: Engineer course](https://img.shields.io/badge/tutorial-Engineer%20course-blue.svg)](docs/index.qmd)

![Conceptual SCNSim branding: an orca and penguin beside circuit schematics and network diagrams.](docs/assets/readme-hero-orca-penguin.png)

SCNSim is a notebook-first Python package for reproducible superconducting
circuit-network analysis. A `CircuitPlan` owns physical components, wiring,
ground, Ports, and reusable Subsystems; a `CircuitRun` solves explicit
requests and returns typed Results with request and evidence identities.

`1.0.0a1` is the first public Alpha for trial use. The package's newer
semantics and this documentation remain `CONVERGING`; Alpha is neither a
stable release nor a new scientific-validation claim. Public API and saved
workspace compatibility may change before a stable release.

The current development candidate is `1.0.0.dev18`, requiring Python 3.12. Normal
`CircuitRun.solve`, `evaluate` and `optimize` default to same-process JAX CPU
Float64/Complex128; Float32/Complex64 and explicit Julia are selectable.
[Operation Benchmark](docs/guides/benchmark.qmd) reads recorded operations from
the current Plan leaf without executing another calculation. Generation
checkpoints and boundary diagnostics preserve recovery and numerical evidence.
JAX Optimization defaults to `commit_every_generations=10`; explicit positive
integers select a different group size in the Plan leaf’s SQLite store. Omitted
intervals with explicit Julia keep its existing native value of 1. See the
guide for recovery, durability and recomputation limits.
For same-generation JAX Optimization, `configure_runtime(cpu_threads=N)`
declares the task CPU budget. Persistent candidate threads preserve isolated
candidate preparation and continuation state; capacity is bounded by N and the
population size. Compatible ready JAX assembly requests share a finite batch,
while native sparse work shares the budget. `None` retains serial execution with
the existing environment; 1 is serial. The coordinator alone owns CMA order,
cache admission, winner selection, SQLite commits and durable callbacks.
Population wall time divided by population size is an average, not candidate
latency. There is one scheduler and no public executor selector or fallback.
The dev14 candidate extends JAX evaluation to operator-element roots, hybridized
poles, S/Y/Z transfer zeros, residue-normalized coupling and loaded operators.
Scalar quantity selectors also serve Optimization; operators remain matrix
outputs. These additions remain `CONVERGING` pending source-bound functional
observations; no performance or numerical acceptance is implied. Published
Alpha snapshots and historical outputs retain their original identities.

Scalar root, pole, zero, response and coupling Results offer themed HTML through
`result.show(detailed=False)` and detached Figure tables through
`result.plot(detailed=False)`. Set `detailed=True` to include provenance and
original stored magnitude/unit values. This presentation remains `CONVERGING`
and does not execute or modify a calculation.

## Alpha 1.0.0a2 release notes

This candidate retains `CONVERGING` semantics; it is not a stable release or a
claim of new numerical certainty. Publication and its exact tag remain separate
delivery steps. The immutable `v1.0.0a1` snapshot and historical course outputs
keep their original identities.

Python 3.12 or newer and pinned CPU JAX/CMA dependencies supply the default
same-process JAX path. Julia is optional and explicitly selected; Julia-only
features require the extra and a manually provided compatible executable.
JAX Optimization now defaults to committing every 10 completed generations;
explicit positive intervals remain available, and omitted Julia intervals retain
its existing native value of 1. A larger interval trades fewer commits for more
buffered evidence and possible recomputation after interruption.

Current JAX operation evidence uses a Plan-leaf SQLite store. Same-version exact
checkpoint references resume only committed full CMA/RNG state; checkpoint off
creates no resumable state. Readonly access never repairs a hot journal: recovery
is explicit on the bound Run. Historical workspaces remain readonly without
migration. Public interfaces and saved workspace compatibility may change before
a stable release; pin exact versions and preserve their environments.

## A first Direct S11 result

This Chapter 1 example builds a grounded parallel LC resonator, couples it to
one terminated 50 Ω measurement Port, and requests a named reflection trace.
The physical model and 110 fF / 120 fF values follow the Engineer course.

```python
import numpy as np

from scnsim import (
    CircuitPlan,
    CircuitRun,
    DirectSolveSpec,
    ParameterDefinitions,
    ParameterSet,
    ParameterSpec,
    SParameterTrace,
    components,
    units as u,
)

inputs = ParameterDefinitions(id="readme_lc")
capacitance = inputs.parameter(
    id="capacitance",
    baseline=110.0 * u.fF,
    spec=ParameterSpec(unit=u.fF),
)

plan = CircuitPlan(id="readme_coupled_lc")
resonator = plan.subsystem(id="resonator")
capacitor = resonator.add(
    components.capacitor(id="capacitor", capacitance=capacitance)
)
inductor = resonator.add(
    components.inductor(id="inductor", inductance=5.8 * u.nH)
)
resonator_bus = resonator.bus(id="signal")
resonator.parallel(
    id="parallel_lc",
    start=resonator_bus,
    branches=((capacitor,), (inductor,)),
    end=resonator.ground,
)
resonator_pin = resonator.expose_pin(id="terminal", at=resonator_bus)

signal_bus = plan.bus(id="signal_boundary")
coupling = plan.add(
    components.capacitor(id="coupling_capacitor", capacitance=6.0 * u.fF)
)
plan.series(
    id="coupling", start=signal_bus, elements=(coupling,), end=resonator_pin
)
plan.add_port(
    id="signal_in",
    at=signal_bus,
    role="terminated",
    reference_impedance=50.0 * u.ohm,
)

run = CircuitRun(plan=plan, workspace="workspaces/readme_lc")
reflection = SParameterTrace(
    id="reflection",
    input_port="signal_in",
    input_mode=(),
    output_port="signal_in",
    output_mode=(),
)
direct_spec = DirectSolveSpec(
    frequencies=np.linspace(5.75, 6.25, 401) * u.GHz,
    traces=(reflection,),
)
baseline = run.solve(run.original, direct_spec, parameters=ParameterSet())
selected = run.solve(
    run.original,
    direct_spec,
    parameters=ParameterSet({capacitance: 120.0 * u.fF}),
)
baseline_s11 = baseline.traces["reflection"]
selected_s11 = selected.traces["reflection"]
baseline_s11.show(component="magnitude", magnitude="db")
selected_s11.show(component="magnitude", magnitude="db")
```

The example uses the default same-process JAX backend and does not require
Julia. Explicit `backend="julia"` execution requires the optional Julia support
and an already installed Julia 1.12.6; it may prepare the committed native
project dependencies. The figures below are previously published Chapter 1
Julia results—not generated by this example or the website build—and show the
same model, sample grid, and `signal_in ← signal_in` channel. The figures present
the complete S11 magnitude and phase; the named
`reflection` trace above is a typed projection of that Direct result.

![Certified Chapter 1 authoring projection of the grounded LC, coupler, and terminated Port.](examples/engineer/figures/01-authoring.svg)

*Authoring projection of the physical Plan; it is not a numerical result.*

![Previously published baseline S11 for 110 fF on the 401-point 5.75–6.25 GHz grid.](examples/engineer/figures/01-s11-baseline.svg)

*Stored Chapter 1 baseline Result: 110 fF. The magnitude axis uses a fixed ±0.05 dB display range; the wrapped phase discontinuity is presentation, not solver failure. [Exact complex samples](examples/engineer/figures/01-s11-data.csv).*

![Previously published selected S11 for 120 fF on the same grid.](examples/engineer/figures/01-s11-selected.svg)

*Stored Chapter 1 selected Result: 120 fF, with the same 5.8 nH inductor, 6 fF coupler, 50 Ω Port, and 401-point grid. No dip or resonance is inferred from this plot. [Exact complex samples](examples/engineer/figures/01-s11-data.csv).*

Continue with the [Alpha Tutorial — Engineer course](docs/index.qmd).

## What the model can answer

- **Direct responses:** full finite-grid S, Y, and Z matrices, with named
  channel traces projected from the same Result.
- **Direct quantities:** loaded roots, coupled poles, transfer zeros, and
  residue-normalized couplings without requiring a fitted external model.
- **Views and Optimization:** Views select a derived analysis network without
  changing physical ownership. Optimization binds declared parameters and
  compares typed scalar objectives; a lower cost does not guarantee every
  target, improvement, or a global optimum.
- **Harmonic balance:** explicit pump, drive, operating-point, and response
  requests return their own typed outcomes. Direct and HB traces are comparable
  only when their selected boundary, channel, work point, pump state, and
  linearization agree.

These are research-model results, not universal accuracy or performance
guarantees. Fitting remains useful for measurements and model identification;
SCNSim does not claim automatic global root search or independent model
validation. Physical layout and electromagnetic simulation are SCGSim
responsibilities.

## Install and learn

After the `v1.0.0a2` Alpha tag is published, install the current Alpha with
Python 3.12 or newer and retain the resulting lockfile and environment for
reproducibility:

```bash
uv add "scnsim @ git+https://github.com/OrPenStrike/scnsim.git@v1.0.0a2"
uv run python -c "import scnsim; print(scnsim.__version__)"
```

The immutable first-Alpha [`v1.0.0a1` snapshot](https://github.com/OrPenStrike/scnsim/tree/v1.0.0a1)
remains available for historical trial use; its original outputs retain their
original identities.

For the current development checkout, `uv sync --locked` installs the pinned CPU
JAX/CMA runtime on Python 3.12. The default JAX path does not install JuliaPkg,
discover Julia or launch a numerical child process. Configure its CPU pool before
first execution with `configure_runtime(cpu_threads=...)`; per-call `backend`
and `precision` override the Run defaults. `run.benchmark()` and `resolve()` are
read-only. See the [operation guide](docs/guides/benchmark.qmd).

`run.explain(ref, spec, backend=None, precision=None)` inherits the Run defaults.
Its JAX route uses sparse Python lowering and View realization without Julia,
JIT or a numerical solve. Explicit Julia execution, HB, Julia compiler
preflight and compiled schematics require
`uv sync --locked --extra julia` and manually provided Julia 1.12.6 on PATH or
`PYTHON_JULIAPKG_EXE`. Feature calls never download Julia or change backends.
Historical Engineer-course execution sources explicitly select Julia so their
original outputs retain their original numerical authority.

The Engineer-course generators use repository-local Jupyter tooling and an
optional static-export extra:

```bash
uv sync --locked --group course-generation --extra static-export --extra julia
```

The site uses the vendored, unmodified Askr v0.4.0 `askr-html` format with
the Quiet Quartz light/dark theme. To build it, use Quarto 1.10.18 or later and run
`quarto render --no-execute --no-clean`. Start with the
[Alpha Engineer course](docs/index.qmd); its aggregate Chapter notebooks are the
execution units for Lessons 1–4, while Chapter 7 Lesson 5 has its own standalone
notebook and result artifacts. The [documentation root](https://orpenstrike.github.io/scnsim/)
opens the [1.0.0 development site](https://orpenstrike.github.io/scnsim/1.0.0-dev/).
The version catalog follows one latest Alpha per release line. After the
`v1.0.0a2` tag and its catalog deployment are published, Askr's version menu
will select the separately rendered 1.0.0a2 Alpha site while preserving the
page path when it exists. That deployment is a separate delivery step and is
not claimed live here. Each site has branch-local search and shows its exact
content source; the Alpha content comes from its immutable version tag, with
only the reviewed Askr presentation and Pages profile overlaid at render time.
Quarto's View source controls point to the content
source. A missing page in a current version returns to that version's home;
a retired Alpha URL explains its replacement rather than serving newer content
under the old version. The former `/develop/` path redirects to the matching
development path; `/main/` is not an Alpha alias. Main carries the same
deployment configuration, not a third content version. The site executes no
notebooks or solvers. Both source renders must validate before one Pages
artifact is published. For a consuming
repository preferring a reviewed commit over the Alpha tag, pin that commit
in its `pyproject.toml` and lockfile:

```bash
uv add "scnsim @ git+https://github.com/OrPenStrike/scnsim.git@<reviewed-commit-sha>"
```

## License

Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
