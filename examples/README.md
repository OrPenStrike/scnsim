# SCNSim micro-course sources

`tutorials/01_*.qmd` through `tutorials/16_*.qmd` are stable Chapter
containers for one model-first course candidate. Each Chapter is a complete
clean-kernel document containing short, sequential Lessons. A new independent
analysis Chapter rebuilds its required circuit before selecting a View or
request, then explains what the resulting diagram or Result means.

Lessons may share the preceding state inside one Chapter. A Chapter never
depends on a hidden builder, fixture, include, or state from another Notebook,
and it does not use a target SCNSim object before explaining why it is needed.

That deliberate Chapter-level repetition makes each Notebook reviewable and
independent of other Chapters' kernel state. It does not create a new
API authority: the Contract owns the meaning, and each Lesson is one teaching
application. Existing Python fixtures are frozen implementation-baseline
evidence and are not the authority for this rewritten course.

The course remains `CONVERGING`, not accepted, stabilized, or released.
Generated authoring figures provide source-bound evidence for their selected
diagram checkpoints, not proof that every cell in all sixteen Chapters has
executed successfully. Compiled-diagram limitations are reported separately;
an authoring image is not substituted for a failed compiled projection.
Direct/HB, View, optimization, Result, and report requests retain their
established numerical meanings. An ASCDLS diagram
projects the authored Plan and does not create or display a selected analysis
View.

Chapter 13 is the optional capstone preview. It declares a complete inline
feedline, readout, and floating circuit before requesting its Plan diagram and
then selecting probe compensation for analysis. Chapters 1–12 explain the
objects and assembly choices that make that complete declaration readable;
Chapters 14–16 rebuild it before asking independent transform, HB, and
Direct/HB questions.

## QMD and generated Notebook authority

Each `.qmd` is the only editable lesson source. Quarto renders it for the site
and generates the same-named `.ipynb` as a committed read-only transport
artifact. Use QMD for documentation editing, IPYNB for GitHub/Jupyter/VS Code,
and regenerate only from QMD. Never hand-edit or reverse-sync generated
Notebooks.

## Generate real authoring figures separately from the website

The diagram-generation entry point is:

```bash
python scripts/generate_tutorial_diagrams.py
```

Run it with the supported development environment; generation is an explicit
execution operation, not a Quarto build step. The generator selects documented
cell IDs from canonical QMD in independent Chapter namespaces. It does not
duplicate Plan builders or guess dependencies. By default it creates a fresh
isolated temporary workspace; `--workspace` can select an execution workspace
explicitly. Keep execution workspaces outside the repository. Chapter 6's figure requires the real declared
optimization and its returned winning parameters; fabricated or substituted
winner values are not permitted.

Check source and artifact currency without importing the circuit package or
running solvers:

```bash
python scripts/generate_tutorial_diagrams.py --check
```

`--only CELL_ID` is for focused diagnosis; it does not publish a complete
manifest.

Only genuine exported `CircuitDiagramResult` authoring SVGs belong in
`tutorials/figures/`. The public manifest,
`tutorials/figures/tutorial-diagrams.json`, binds relative source/cell hashes,
model/parameter-point/diagram identities, SVG hashes, and classified validation
facts; full execution records remain separate. A figure documents its actual
checkpoint, including its A/B and composition checks, not solver correctness
or whole-course execution. Audit-only checkpoints need no duplicate image.

Two Chapter 12 authoring checkpoints—the tapped-feedline Default and explicit
tap-order layouts—currently report `schematic_layout` with
`fixed connector intersects occupied geometry`. Neither has an SVG or diagram
certificate; a different authoring or compiled image must not stand in for
either failure. The 21-checkpoint inventory therefore distinguishes 17 intended
SVG-producing checkpoints, these two unavailable layouts, and two audit-only
checkpoints. The final generated manifest, not that inventory, records which
checkpoints actually succeeded in a particular source-bound batch. Compiled
projection limitations remain separate.

The website includes these SVGs in HTML-only figure blocks with enlargement
and a raw SVG link. Quarto keeps `execute.enabled: false`: browsing or building
the site starts no notebook kernel, solver, or optimization. The generated
IPYNB remains a zero-output transport artifact rather than an executed record.

Validation compares ordered semantic cells, explicit code IDs, source, and
kernel metadata while ignoring generated Markdown cell IDs. Every committed
Notebook must have zero execution counts, outputs, and attachments. Generation
and validation do not imply that the target API is implemented, accepted,
stabilized, or released.
