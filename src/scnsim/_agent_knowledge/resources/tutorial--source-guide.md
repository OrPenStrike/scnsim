# SCNSim micro-course sources

`tutorials/01_*.qmd` through `tutorials/16_*.qmd` are stable Chapter
containers for one model-first TARGET course. Each Chapter is a complete
clean-kernel document containing short, sequential Lessons. A new independent
analysis Chapter rebuilds its required circuit before selecting a View or
request, then explains what the resulting diagram or Result means.

Lessons may share the preceding state inside one Chapter. A Chapter never
depends on a hidden builder, fixture, include, or state from another Notebook,
and it does not use a target SCNSim object before explaining why it is needed.

That deliberate Chapter-level repetition makes each Notebook reviewable and
runnable on its own once the TARGET runtime exists. It does not create a new
API authority: the Contract owns the meaning, and each Lesson is one teaching
application. Existing Python fixtures are frozen implementation-baseline
evidence and are not the authority for this rewritten course.

All TARGET cells are `CONVERGING` documentation and are not executable on the
current runtime, including Chapters 4–5's independent Parameter System and
parameterized-diagram examples. Their Direct/HB, View, optimization, Result,
and report requests retain established numerical meanings; only the structured
authoring, parameter identity/binding, diagram, and teaching surface is being
rewritten. An Automatic Semantic Circuit Diagram Layout System (ASCDLS) diagram
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

Validation compares ordered semantic cells, explicit code IDs, source, and
kernel metadata while ignoring generated Markdown cell IDs. Every committed
Notebook must have zero execution counts, outputs, and attachments. Generation
and validation do not imply that the target API is implemented, accepted,
stabilized, or released.
