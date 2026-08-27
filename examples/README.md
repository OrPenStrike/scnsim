# SCNSim UX examples

These examples preserve two distinct user journeys:

- `01_model_author.qmd` shows the circuit-model developer's real authoring,
  inspection, execution, and exact-result workflow.
- `02_model_user.qmd` shows a teammate supplying only the inputs owned by the
  consuming project and calling its reusable façade.

The simple-resonator author builds a one-port reflection Plan from one 6 fF
coupling capacitor and one grounded parallel LC, then inspects its authoring
schematic. The synthetic IPF author instead inspects a mature Composite and its
complete compiler-expanded pi ladders before the model-owned default
optimization, optional immutable override, winner Direct/HB responses, report,
and exact resolve. Reusable model logic remains in the adjacent Python modules.

## Source and generated Notebook authority

Each `.qmd` is the only editable source for its workflow. Quarto renders it for
the documentation site and generates the same-named `.ipynb` as a committed,
read-only transport artifact:

- open the `.qmd` in VS Code with Quarto for source editing or cell execution;
- open the `.ipynb` in VS Code/Jupyter for notebook-native execution; and
- open the `.ipynb` on GitHub for native static rendering.

Never hand-edit or reverse-sync a generated IPYNB. Regeneration is one-way from
a temporary QMD source basename through Quarto's reader-facing IPYNB render
format; the distributed artifact is replaced atomically only after validation.
This avoids Quarto classifying an existing same-name artifact as stale output.

Generated Markdown cell IDs are transport noise. CI instead verifies ordered
semantic cells, explicit code-cell IDs, code source, kernel metadata, and the
required zero-output state. At this `CONVERGING` scaffold checkpoint, manual
execution still fails explicitly before producing simulation evidence.
