---
title: "Use an installed SCNSim package"
---

SCNSim turns an explicitly declared circuit into network calculations and
circuit diagrams. Start with a `CircuitPlan`: its components, physical values,
wiring and subsystem boundaries define the model. A `CircuitRun` then supplies
an Original View; a calculation Spec states the question, and a typed Result
stores the answer. Drawing and calculation use the same model, but drawing
does not choose an analysis View or start a solver.

This guide serves both people and agents using a built distribution. It does
not require an agent service, a source checkout, or access to online docs.

## Know which material applies

The package carries a manifest and UTF-8 resources under
`scnsim/_agent_knowledge/`. They are generated from the canonical public docs
and example sources at build time. The manifest records the distribution
version, a content-derived `bundle_id`, and each resource's source path and
source/content hashes. Use those identities, not a remembered package version
alone, when reporting which instructions were read.

The structured implementation is a **CONVERGING candidate**. The complete
[Direct example](../examples/agent_direct.py) below defines a bounded executable
path: build a grounded LC circuit, solve its original network response, inspect
the result, resolve that exact request, and handle invalid input. It is not a
claim that all Tutorial chapters, ASCDLS layouts, or scientific workflows have
been accepted or validated in every build.

Resources with kind `target-tutorial` retain the course's TARGET warnings and
design proposals. In particular, a historical whole-course “not executable”
callout is not a current per-feature support inventory. Conversely, finding
an example or API description in the bundle does not prove that it has run.
Read the applicable contract and the scoped usage guidance before executing a
workflow. The bundle contains source material, not generated calculation
outputs or an execution receipt.

## Read resources without starting the numerical runtime

After installing a built wheel into the consumer environment, distribution
metadata locates the manifest without importing `scnsim`:

```python
from importlib.metadata import distribution
import json

installed = distribution("scnsim")
manifest_entry = next(
    item for item in installed.files or ()
    if str(item) == "scnsim/_agent_knowledge/manifest.json"
)
manifest_path = installed.locate_file(manifest_entry)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

print(installed.version)
print(manifest["bundle_id"])
print([resource["id"] for resource in manifest["resources"]])
```

This is a metadata inspection example, not a replacement for a validating
resource adapter. An adapter must also check the installed file list, safe
paths, exact version, resource hashes and canonical bundle digest before
returning content. Missing or inconsistent data is an error, not permission
to substitute online docs or a different checkout.

When SCQ MCP is available, use its `search` tool with `package="scnsim"`, a
query such as `Direct response`, and `kind="example"`. Its `read` tool accepts
`package="scnsim"` and the stable ID `example/direct-response`; this guide is
`guide/agent-usage`. Reading returns source text only. The producer does not
install an MCP entrypoint or a shared runtime dependency, and reading a
resource never runs its Python code. Editable installations are outside this
installed-bundle adapter contract; use a wheel in the consumer environment.

## Run one complete Direct calculation

The [standalone Python example](../examples/agent_direct.py) uses this circuit:

- A 110 fF capacitor and 5.8 nH inductor form a grounded parallel resonator.
- A 6 fF capacitor connects that resonator to a 50 ohm terminated logical Port.
- The request evaluates the Original View at 5.5, 6.0, 6.5 and 7.0 GHz.

The file is sequential, with short cell-style sections and no hidden Plan
builder. A subsystem groups the resonator's internal circuit; its public pin
allows the parent to connect it. The root bus used by `add_port()` locates the
measurement boundary. The Port includes its declared load, so this is the
loaded response, not a compensated or reduced model.

Review the text before running it. After explicitly saving the installed
example resource as `agent_direct.py`, execute it with the Python environment
containing the same wheel:

```bash
python agent_direct.py
```

The script creates `workspaces/agent-direct` relative to the working directory
and launches the real Julia backend for `run.solve()`. First use may need Julia
and backend environment preparation; see
[runtime preparation](implementation/julia-runtime-preparation.qmd). No solver
is launched just by inspecting the bundle.

The returned `DirectSolveResult` contains frequencies, channel identities and
complex S data. Four frequencies and one logical Port produce a matrix array
with shape `(4, 1, 1)`. The script prints the real and imaginary values rather
than embedding an expected numerical answer. `S` here is the response in the
selected network's channel basis; the sole raw channel is `signal_in`.

It then calls `run.resolve(original, direct_spec)`. Resolve reads the exact
stored success for the same Plan, View and request; it is not a request to
recompute or choose the latest vaguely similar run. A missing success is an
error. The script compares the returned result identity with the original one.
Finally, it catches the `ValueError` raised by a negative frequency in
`DirectSolveSpec`, demonstrating input validation before a solve is requested.
Backend or execution failures are not disguised as numerical data.

The example preserves physical values and request settings from
[Chapter 2](../examples/tutorials/02_solve_direct.qmd). It deliberately does not
render a diagram, select a transformed View, run HB, sweep parameters, or
optimize a design. Those are separate workflows with their own contracts:
[Views](concepts/compilation-coordinates-and-network-views.qmd),
[parameters](contracts/parameter-system.qmd), and
[calculation/results](v1-runtime-contract.qmd).
