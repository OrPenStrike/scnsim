# SCNSim UX examples

These notebooks preserve two distinct V1 user journeys around the same simple
grounded readout-resonator model:

- `simple_resonator/01_model_author.ipynb` shows the circuit-model developer
  declaring components, nets, the reference, and a port; it then teaches
  `solve()`/SolveSpec and `evaluate()`/quantity-Spec separately before using
  both exact Results in one report.
- `simple_resonator/02_model_user.ipynb` shows a teammate importing the finished
  model façade, supplying only a Design Target and workspace, then viewing the
  optimization and report.

The model façade owns the required `DiagonalRootSpec.root_hint` once. The
consumer's requested target remains a separate input and is never reused as a
root locator.

`ipf_optimization/` adds a larger but still public synthetic journey:

- `circuit_library.py` declares one five-section Composite using general
  one- and two-trace `RLGC`, shared parameter fan-out, three `AffineMap`
  capacitance calibrations, public pins, and a `CoordinateRef`;
- `circuit_model.py` owns the Plan, PTC/views, five Direct quantities, the
  five-variable `build_default_optimization_spec()` recipe, winner Direct and
  pump-off HB requests, reporting, and exact resolve;
- `01_model_author.ipynb` makes those responsibilities and an optional
  immutable extrapolation override visible; and
- `02_model_user.ipynb` supplies only a synthetic target and workspace.

The notebooks have no outputs and are intentionally non-executable at this
`CONVERGING` scaffold checkpoint.  The `scnsim` package imports for API and
docstring review, but every construction or operation fails explicitly until
the candidate implementation exists.

The `.ipynb` file is the single Notebook authority:

- clone the repository and open it in VS Code for cell-by-cell execution;
- open it on GitHub for a static rendered reading view; and
- run `quarto render` for the SCNSim documentation-site HTML.

GitHub documents native static rendering of committed
[Jupyter notebooks](https://docs.github.com/en/repositories/working-with-files/using-files/working-with-non-code-files#working-with-jupyter-notebook-files-on-github).
Quarto can render the same `.ipynb` source and, by default, does
[not execute its cells during render](https://quarto.org/docs/projects/code-execution.html#notebooks).
SCNSim therefore commits no paired `.qmd` copy and no scaffold outputs.
