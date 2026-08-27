# SCNSim Tutorial sources

The five Tutorials form one linear course. A page may use a public object only
after that object is introduced on the page or listed in its prerequisites:

1. `simple_resonator/01_model_author.qmd` — build a reflective resonator from
   primitive C and L branches, then solve and evaluate it;
2. `reusable_composite/01_composite_plan.qmd` — package those branches as a
   reusable Composite;
3. `simple_resonator/02_model_user.qmd` — publish and consume a team-owned
   model and its default optimization recipe;
4. `ipf_optimization/01_model_author.qmd` — author and inspect an
   optimization-ready N-trace IPF Composite; and
5. `ipf_optimization/02_model_user.qmd` — optimize once and reuse the winner
   for Direct and pump-off HB results.

Adjacent Python modules are reusable model or workflow source, not parallel
Tutorial prose. QMD pages show those files through Quarto native includes and
import the same source from executable cells.

## QMD and generated Notebook authority

Each `.qmd` is the only editable Tutorial source. Quarto renders it for the
documentation site and generates the same-named `.ipynb` as a committed,
read-only transport artifact:

- edit or execute the `.qmd` in VS Code with Quarto;
- open the `.ipynb` in VS Code/Jupyter for notebook-native execution; and
- open the `.ipynb` on GitHub for native static rendering.

Never hand-edit or reverse-sync a generated IPYNB. Regeneration is one-way
from QMD through Quarto's reader-facing IPYNB render. Generated Markdown cell
IDs are transport noise; validation instead compares ordered semantic cells,
explicit code-cell IDs, code source, kernel metadata, and the required
zero-output state.

At this `CONVERGING` checkpoint, manual execution still fails explicitly before
producing simulation evidence.
