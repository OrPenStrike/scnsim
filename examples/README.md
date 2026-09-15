# SCNSim engineer course sources

`engineer/setup.qmd` is the shared environment entry. `engineer/chapter-01/`
through `engineer/chapter-08/` are the current course Chapters. YAML-free QMD
fragments hold each Chapter's canonical Lesson prose and code. Thin QMD
wrappers make those Lessons readable as separate HTML pages, while each
`chapter.qmd` includes its fragments in order and generates one complete
`chapter.ipynb`.

The web Lessons are not independently runnable: each later Lesson plainly
states the earlier state it requires. The aggregate Chapter is the clean-kernel
execution unit. Open `chapter.ipynb`, restart its Python kernel, and use
**Run All** for Chapters 1–7. Chapter 8 explicitly uses two independent kernels
and tells the reader which half to run in each. Every committed Notebook remains
a zero-output transport artifact rather than an execution record.

A Chapter's relative `workspace="workspaces/..."` path is resolved from the
process's current working directory. Starting Jupyter from the repository root
therefore writes under the root `workspaces/` directory; starting it elsewhere
writes under that directory instead. Directories named `workspaces/` are
ignored at every repository depth to reduce accidental publication, but
ignored does not mean deleted or backed up. Preserve or back up the workspace
evidence when an exact later `resolve()` is required: moving or deleting those
bytes can make that exact Result unavailable.

QMD is the only hand-edited lesson authority. Do not edit the derived Notebook
or duplicate lesson code in its wrappers. The website uses `execute.enabled:
false`; browsing or building documentation starts no kernel, solver, runtime
preparation, or Optimization.

Before generating numerical course evidence, select the repository-local
Jupyter tooling group and the independent static-export extra:

```bash
uv sync --locked --group course-generation --extra static-export
```

Neither selection is a numerical-runtime dependency or a published wheel
extra.

## Generate Chapter 1 evidence explicitly

Chapter 1's diagram, numerical figures, complex data, and quantity table are
generated from the exact fragment cells by one bounded tool:

```bash
uv run --locked --group course-generation --extra static-export \
  python scripts/generate_engineer_chapter1.py \
  --workspace /tmp/scnsim-engineer-chapter1-run
```

The workspace must be new or empty and outside the repository. Generation
executes the Chapter's real Direct and diagonal-root requests; it is never a
Quarto step. Check committed source/artifact identity without importing SCNSim
or running a solver:

```bash
uv run --locked python scripts/generate_engineer_chapter1.py --check
```

`engineer/figures/chapter-01-artifacts.json` binds the aggregate, fragments,
wrappers, package source tree, environment, exact result identities, and these
public artifacts:

- `01-authoring.svg`: certified authoring projection of the baseline Plan;
- `01-s11-baseline.svg` and `01-s11-selected.svg`: public `.s.plot()` magnitude
  and phase presentations, with a fixed ±0.05 dB display-only magnitude range;
- `01-s11-data.csv`: unrounded complex baseline and selected one-port response
  on the exact 401-point grid; and
- `01-quantities.csv`: capacitance, separate unloaded LC calculation, loaded
  root frequency, and loaded linewidth for both points.

The ideal lossless one-port magnitude is approximately 0 dB. The phase remains
readable; its wrapped discontinuity is ordinary angle presentation, not a
numerical failure. Plotly displays the horizontal axis in GHz while the Result
retains its declared frequency units. Numerical plots are typed Result presentations,
not ASCDLS certificates. The loaded root is a separate Result and is not
inferred from a dip or from the unloaded LC formula.

## Generate Chapter 2 evidence explicitly

Chapter 2's point tables, parameter-field figures, and selected-winner diagram
come from its exact fragment cells:

```bash
uv run --locked --group course-generation --extra static-export \
  python scripts/generate_engineer_chapter2.py \
  --workspace /tmp/scnsim-engineer-chapter2-run
```

The workspace must again be new or empty and outside the repository. Verify
the committed bindings without importing SCNSim or running a solver:

```bash
uv run --locked python scripts/generate_engineer_chapter2.py --check
```

`engineer/figures/chapter-02-artifacts.json` binds the aggregate, four
fragments, four wrappers, package source tree, environment, exact Result
identities, winner certificate, and these public artifacts:

- `02-capacitance-sweep.svg`, `.csv`, and `.md`: the three ordered C points at
  fixed 5.8 nH, including typed status and loaded root values;
- `02-optimized-winner.svg`, `.csv`, and `.md`: the genuine returned winner,
  its independent loaded-root readback, and the same Plan rendered at that
  exact `ParameterSet`; and
- `02-optional-grid.svg` plus `02-optional-spaces.csv` and `.md`: the complete
  3 × 2 Cartesian field beside the two separately listed C/L points.

The numerical figures are typed Result presentations, while only the winner
schematic carries an ASCDLS diagram certificate. Neither the optimizer's small
cost nor a generated artifact creates scientific or Human acceptance.

## Generate Chapter 3 evidence explicitly

Chapter 3's typed full-Plan layout outcome, two standalone subcircuit
illustrations, and named two-Port response come from its exact fragment cells:

```bash
uv run --locked --group course-generation --extra static-export \
  python scripts/generate_engineer_chapter3.py \
  --workspace /tmp/scnsim-engineer-chapter3-run
uv run --locked python scripts/generate_engineer_chapter3.py --check
```

`engineer/figures/chapter-03-artifacts.json` binds the Chapter source, package
source, environment, request, Result, and public artifact identities. The
complete Plan's Default diagram truthfully reports the typed
`schematic_layout` failure `fixed connector intersects occupied geometry` and
therefore has no SVG or certificate. `03-feedline-illustration.svg` and
`03-readout-illustration.svg` are certified projections of two separately
declared illustrative Plans, not fragments extracted from or substitutes for
the failed complete Plan. `03-direct-s21.svg`, `.csv`, and `.md` present the
named `transmission` trace at exactly 5.5, 6.0, and 6.5 GHz. Three samples do
not define an interpolated curve or a resolved resonance.

## Generate Chapter 4 evidence explicitly

Chapter 4's explicit diagram and numerical Results are generated independently
of the website:

```bash
uv run --locked --group course-generation --extra static-export \
  python scripts/generate_engineer_chapter4.py \
  --workspace /tmp/scnsim-engineer-chapter4-run
uv run --locked python scripts/generate_engineer_chapter4.py --check
```

`engineer/figures/chapter-04-artifacts.json` binds its six fragments, six
wrappers, aggregate source, package source, environment, diagram certificate,
View lineage, requests, Results, and public artifacts. The required core owns
`04-explicit-capstone.svg`, the raw four-Port `04-raw-direct-s21.*` Result, and
the separately labeled `04-ptc-direct-s21.*` table and figure.
Optional evidence includes the exact 6.2 GHz `04-response-element.*` lookup,
the separate `04-optional-direct-s21.*` Result, and the
`04-pump-off-hb.*` Result. Direct and HB keep their separately declared grids;
the generated summary does not overlay or interpolate them.

## Generate Chapters 5–8 evidence explicitly

Each remaining Chapter has its own bounded generator and source-only checker:

```bash
uv run --locked --group course-generation --extra static-export \
  python scripts/generate_engineer_chapter5.py \
  --workspace /tmp/scnsim-engineer-chapter5-run
uv run --locked --group course-generation --extra static-export \
  python scripts/generate_engineer_chapter6.py \
  --workspace /tmp/scnsim-engineer-chapter6-run
uv run --locked --group course-generation --extra static-export \
  python scripts/generate_engineer_chapter7.py \
  --workspace /tmp/scnsim-engineer-chapter7-run
uv run --locked --group course-generation --extra static-export \
  python scripts/generate_engineer_chapter8.py \
  --workspace /tmp/scnsim-engineer-chapter8-run

uv run --locked python scripts/generate_engineer_chapter5.py --check
uv run --locked python scripts/generate_engineer_chapter6.py --check
uv run --locked python scripts/generate_engineer_chapter7.py --check
uv run --locked python scripts/generate_engineer_chapter8.py --check
```

Chapter 5 generates the certified reusable-Composite authoring diagram.
Its public surface contains both C and L refs, two equivalent ordinary Pins,
and one explicitly published Coordinate; the parent couples directly to one
Pin and leaves the other open. Chapter 6 generates certified Default, T,
Cross, and fixed-reuse diagrams. Chapter 7 records typed `schematic_layout`
failures for both its authoring and finite-pi compiled diagrams; neither
failure has an SVG substitute. Its actual three-point Direct batch still
publishes complete sixteen-channel tables and three figures labeled as
readout-head reflection examples. Chapter 8 launches
two independent Jupyter kernels in one fresh execution directory: the first
persists Results and the standalone Report, while the second reconstructs and
calls only `resolve()` against the unchanged shared workspace. Its manifest
records both kernel identities, the exact matching Result identity, and the
source-only checker rebind separately from the generator that performed the
execution. A source-only rebind starts from one exact previously published
manifest named by the reviewed transition artifact, preserves its historical
execution and redraw identities, and requires every public artifact byte to
equal that prior publication. The transition closes the exact old/new
generator, package-source, teaching-source, and cell hashes, including explicit
additions or removals; measuring new hashes does not authorize them. All eight
candidate manifests are validated before any per-file atomic replacement, so a
validation failure publishes none of them. The publisher does not claim one
filesystem transaction across all eight files. A future redraw needs a separate
reviewed output pair bound to its original verified input Result/data and
producer identities; the current source-only transition authorizes no redraw.

The Chapter 8 generator also has one receipt-bound continuation for a reviewed
run that sealed both Results but stopped during presentation. Maintainers first
run `--verify-resume-only`, then pass the same `--workspace` and exact
`--resume-receipt`. This path reconstructs the same Plan, rejects any changed
catalog/runtime/cell/request/Result identity before binding, and uses public
`resolve()` in two new independent kernels; it never falls back to `solve()` or
`evaluate()`. The published manifest keeps the failed kernel's unavailable
process telemetry distinct from the continuation and restart kernel records.

```bash
uv run --locked --group course-generation --extra static-export \
  python scripts/generate_engineer_chapter8.py \
  --workspace /tmp/preserved-chapter8-run \
  --resume-receipt /tmp/preserved-chapter8-run/pre-tooling-identity.json \
  --resume-receipt-sha256 "$FAILED_RUN_CAPTURE_SHA256" \
  --verify-resume-only
uv run --locked --group course-generation --extra static-export \
  python scripts/generate_engineer_chapter8.py \
  --workspace /tmp/preserved-chapter8-run \
  --resume-receipt /tmp/preserved-chapter8-run/pre-tooling-identity.json \
  --resume-receipt-sha256 "$FAILED_RUN_CAPTURE_SHA256"
```

`FAILED_RUN_CAPTURE_SHA256` is the independently preserved SHA-256 recorded
when the failed-run capture is handed off; the generator never derives or
guesses that trust anchor from the receipt it is validating.

## Former Chapter URL guides

`tutorials/01_*.qmd` through `tutorials/16_*.qmd` are short no-code migration
guides into the current Engineer course. They preserve public URLs without
forming a second syllabus. Their superseded executable bodies, zero-output
Notebooks, checkpoint figures, and diagram generator remain available in Git
history. The
[migration map](../docs/implementation/engineer-course-migration-map.qmd)
records every current destination.

Historical Python fixtures in `tutorials/fixtures/` are **KEEP** implementation
baseline evidence. The current course generators do not import them; CI keeps
the existing dedicated fixture consumer separate from the Engineer Chapters.
Source-bound figures, generated Notebooks, and passing static checks do not by
themselves establish Human acceptance, whole-course execution, release, or
scientific validity.
