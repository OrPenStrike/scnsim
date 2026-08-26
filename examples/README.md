# SCNSim UX examples

These notebooks preserve two distinct V1 user journeys around the same simple
grounded readout-resonator model:

- `simple_resonator/01_model_author.ipynb` shows the circuit-model developer
  declaring components, nets, the reference, a port, a Run, and exact Specs.
- `simple_resonator/02_model_user.ipynb` shows a teammate importing the finished
  model façade, supplying only a Design Target and workspace, then viewing the
  optimization and report.

The notebooks have no outputs and are intentionally non-executable at this
`CONVERGING` scaffold checkpoint.  The `scnsim` package imports for API and
docstring review, but every construction or operation fails explicitly until
the candidate implementation exists.
