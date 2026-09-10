# SCNSim engineer course sources

`engineer/chapter-01/` is the current course entry point. Five YAML-free QMD
fragments hold the canonical Lesson prose and code. Five thin QMD wrappers make
those Lessons readable as separate HTML pages, while `chapter.qmd` includes the
same fragments in order and generates one complete `chapter.ipynb`.

The web Lessons are not independently runnable: Lessons 2–5 plainly state the
earlier state they require. The aggregate Chapter is the clean-kernel execution
unit. Open `chapter.ipynb`, restart its Python kernel, and use **Run All**.
Every committed Notebook remains a zero-output transport artifact rather than
an execution record.

The Chapter's `workspace="workspaces/engineer-chapter-01"` is resolved relative
to the process's current working directory. Starting Jupyter from the
repository root therefore writes under the root `workspaces/` directory;
starting it elsewhere writes under that directory instead. Directories named
`workspaces/` are ignored at every repository depth to reduce accidental
publication, but ignored does not mean deleted or backed up. Preserve or back
up the workspace evidence when an exact later `resolve()` is required: moving
or deleting those bytes can make that exact Result unavailable.

QMD is the only hand-edited lesson authority. Do not edit the derived Notebook
or duplicate lesson code in its wrappers. The website uses `execute.enabled:
false`; browsing or building documentation starts no kernel, solver, runtime
preparation, or Optimization.

## Generate Chapter 1 evidence explicitly

Chapter 1's diagram, numerical figures, complex data, and quantity table are
generated from the exact fragment cells by one bounded tool:

```bash
uv run --locked python scripts/generate_engineer_chapter1.py \
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
- `01-s11-baseline.svg` and `01-s11-selected.svg`: public `.s.show()` magnitude
  and phase presentations, with a fixed ±0.05 dB display-only magnitude range;
- `01-s11-data.csv`: unrounded complex baseline and selected one-port response
  on the exact 401-point grid; and
- `01-quantities.csv`: capacitance, separate unloaded LC calculation, loaded
  root frequency, and loaded linewidth for both points.

The ideal lossless one-port magnitude is approximately 0 dB. The phase remains
readable; its wrapped discontinuity is ordinary angle presentation, not a
numerical failure. The horizontal axis contains frequency in Hz and Matplotlib
shows its `1e9` scale factor. Numerical plots are typed Result presentations,
not ASCDLS certificates. The loaded root is a separate Result and is not
inferred from a dip or from the unloaded LC formula.

## Migration legacy

`tutorials/01_*.qmd` through `tutorials/16_*.qmd`, their same-named zero-output
IPYNBs, and `tutorials/figures/` remain intact migration inputs. They are not a
second current course. The
[migration map](../docs/implementation/engineer-course-migration-map.qmd)
records their future destinations. Existing source/figure bindings remain
valid until a later authorized batch migrates that exact content.

The legacy diagram generator remains separate:

```bash
python scripts/generate_tutorial_diagrams.py
python scripts/generate_tutorial_diagrams.py --check
```

It retains 21 checkpoint outcomes: 18 successful SVG checkpoints, one
truthfully unavailable fixed Default layout, and two successful audit-only
checkpoints. Chapter 12's tapped-feedline Default and full explicit tap-order
figures are current. Chapter 13's complete raw-loaded Default reports the typed
`schematic_layout` failure `fixed connector intersects occupied geometry`; its
successful explicit composition and probe-tee SVGs remain distinct evidence,
not fallbacks for that failure.

Historical Python fixtures in `tutorials/fixtures/` are **KEEP** implementation
baseline evidence. Neither the new Chapter nor its generator imports them.
Source-bound figures, generated Notebooks, and passing static checks do not by
themselves establish Human acceptance, whole-course execution, release, or
scientific validity.
