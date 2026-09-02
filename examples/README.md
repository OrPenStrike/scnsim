# SCNSim micro-course sources

`tutorials/01_*.qmd` through `tutorials/13_*.qmd` form one linear native-API
course. Each page answers one focused user need with one operation and
inspection. A page never uses a public SCNSim object before the course
introduces it.

The modules under `tutorials/fixtures/` only reconstruct Plans and Plan-bound
handles needed by adjacent lessons. They do not wrap `CircuitRun`, choose
targets or model policy, or provide a terminal workflow.

## QMD and generated Notebook authority

Each `.qmd` is the only editable lesson source. Quarto renders it for the site
and generates the same-named `.ipynb` as a committed read-only transport
artifact. Use QMD for documentation editing, IPYNB for GitHub/Jupyter/VS Code,
and regenerate only from QMD. Never hand-edit or reverse-sync generated
Notebooks.

Validation compares ordered semantic cells, explicit code IDs, source, and
kernel metadata while ignoring generated Markdown cell IDs. Every committed
Notebook must have zero execution counts, outputs, and attachments. In the
`1.0.0.dev6` implements the Human-`ACCEPTED` Full V1 course taught in Lessons
1–13. Cross-slice stabilization is in progress; Lessons 6–8 retain their
previously stabilized status.
