# M0 benchmark and golden observations

Use these tools to measure the exact release-review baseline and compare its
numerical outputs before changing the implementation. The corpus lives in
`circuits.py`; `golden.py` records returned values and actual execution evidence.

## Current benchmark settings

Use frequency grids of 401, 4001 and 10001 points, with the same authored
frequency endpoints, circuits, parameters and optimization budgets. The benchmark
CLI defaults to `--cpu-threads 10`; this declares the SCNSim runtime budget,
not an observed CPU core count. Production defaults remain unchanged.
Before and after comparisons use identical grid and thread settings.

For the R2 compact baseline, use the installed R1 wheel built from
`7f944d10b6835cb392c177d5ab7084efd1b03733`; the comparison uses the final R2
wheel under those same settings. Keep historical 40001-point / two-thread
observations labelled with their original settings.

## Run one operation

Use the allocated Python environment containing the exact baseline wheel.
Use a frozen copy of these benchmark sources outside the package checkout,
with `PYTHONPATH` unset. Set `SCNSIM_SOURCE_COMMIT` to the exact source commit
recorded in the installed wheel's build receipt. For the T10 BEFORE wheel:

```bash
export SCNSIM_SOURCE_COMMIT=68fc9306048d4b660307920775ea2a37acfcb824
```

For AFTER, set it to the final integrated R1 wheel's recorded source commit.
For installed-wheel capture, add `--installed-wheel-only`:

```bash
python benchmarks/run.py --case C1 --operation direct --storage memory \
  --points 401 --chunk-size 64 --cpu-threads 10 --repetitions 5 \
  --source-commit "$SCNSIM_SOURCE_COMMIT" --output /absolute/new/output
```

Check: `summary.json` contains five fresh-process cold samples, five warm
samples, median and interquartile range. `warm_compute` includes only calls
with actual native dispatch; `warm_no_native_dispatch` remains separately
reported, with no compute-time claim. Each worker records machine load,
thread settings, versions, peak process RSS and JAX monitoring events. Raw
compilation logs accompany each process. A lightweight forwarding counter
records actual native dispatches in timed calls. Timing ends when the public operation
returns; serialization and the separate instrumented capture are excluded.

Choose `--storage workspace` for the paired saved operation. Each worker owns
its fresh Workspace. Nonoptimizer operations in both storage modes use the same
declared parameter-point series: the baseline, then successive representable SI
values above it. Each warm call records its actual point and request identity.
These nearby points request computation rather than saved-result lookup.
Optimizers use their original authored baseline and candidate-generation
configuration. Choose operations from the selected case's declared map;
a missing operation raises rather than substituting a different calculation.
`--cma-evaluations` selects the declared 64- or 512-evaluation request.

For tail coverage, use `--chunk-size 64` with `--points` 63, 64, 65 and 127.
Add `--tail-failure` to the assigned failing-tail case. A physical failure stays
recorded as a failure; its observation is useful baseline data. Unexpected
infrastructure or collector errors stop the invocation and retain diagnostics.

## Observe grid-length reuse in one Run

```bash
python benchmarks/run.py --worker --case C1 --operation direct --storage memory \
  --point-counts 63 65 --chunk-size 64 --source-commit "$SCNSIM_SOURCE_COMMIT" \
  --output /absolute/new/grid-reuse
```

Check: both call rows in `worker.json` use the same baseline parameter point,
Run, Plan and View. Each records its actual canonical frequency grid, native
operand shapes, native dispatch count and call-scoped JAX compilation events.
The second length is a grid-reuse observation, separate from the changing-parameter
warm samples. Add `--capture-only` for raw arrays and same-call certificates for
each grid. The ordinary 63, 64, 65 and 127 tail matrix remains separate.

## Optimization measurements

C1's `lbfgsb` operation uses 15000 maximum evaluations, 15000 maximum iterations,
`ftol=2.220446049250313e-09`, `gtol=1e-05`, 20 maximum line-search steps and
publication every accepted iteration. `circuits.py::LBFGSB_CONTROLS` owns these settings.

Both CMA and L-BFGS-B record their own actual public stage counts.
Declared evaluation budgets are workload settings, not observed counts.
The public `optimize(parameters=...)` argument supplies fixed nonactive inputs;
it cannot set active-variable initial values. These fixtures omit that argument
and retain the authored 110 fF initial capacitance. Repeated identical Workspace
requests may return saved results. Such returned calls remain labelled readback, with
unavailable compute means; they are not warmed optimizer execution measurements.
Per-call initial and fixed parameter records retain this boundary.

Compare **mean time per actual evaluation** using `optimization_per_evaluation`
in `summary.json`. Each execution retains public stage evaluations, iterations,
status and termination reason. API wall time covers the public optimization call;
fixture construction and report projection are outside it. Recorded synchronized
native objective phase timings are also reported per evaluation where available;
they include root work inside that objective executable and exclude separate
compilation and host/publication work outside those phases.
These timing bases remain separate. Zero evaluations or missing returned counts
produce an unavailable metric with a reason. A saved readback without native
dispatch retains historical stage counts but contributes no compute comparison.

## Initial-state and root-failure diagnostics

For L-BFGS-B, `local_initial_diagnostic` records the first actual controller cost,
gradient, initial coordinates, bounds and coordinate map. A transparent benchmark
observer forwards the existing `run_local_optimizer/evaluate` call unchanged; it
adds no objective or AD evaluations. Report projection and bound-projected gradient
arithmetic occur after timing. The observer itself adds a forwarding call and
retains one returned observation during the measured call.

The optimizer gradient is already mapped to controller coordinates. Its bound
projection is reported separately from the physical gradient in the returned
Memory Result's first public gradient-bearing trial. Saved history does not expose
that physical gradient; the report says unavailable rather than reconstructing it.
A cached readback has no current controller observation. `optimizer_converged` is
reported as the actual termination reason, without declaring a stationary optimum.
The authored 110 fF point, target and controls remain unchanged; a stage seed is
changed only after the owner interprets the initial-state diagnostic.

Keep C1's original diagonal **Z** TransferZero request as a failure-latency case.
Preserve its anchor and any returned last-iterate/slope evidence without reconstructing missing
values. The matched 8 nH/80 fF parallel LC with 50 Ω loss is a separate S11 success
case declared as `C1-matched-lc` / `transfer_zero` in `circuits.py`; it never
replaces the original request. Its fixed empty ParameterSet is recorded honestly;
identical Workspace repetitions can be readback rather than compute.

## Capture and compare

The timing frontend writes a separate `capture/golden` directory. To collect
one instrumented operation without timing repetitions:

```bash
python benchmarks/golden.py capture -- --worker --capture-only \
  --case C1 --operation direct --storage memory --points 401 \
  --chunk-size 64 --source-commit "$SCNSIM_SOURCE_COMMIT" --output /absolute/new/capture
```

Successful Direct certificates are retained from the same returned native
chunk before the memory sink removes them. Raw checked predicates, codes and
payloads are kept with source-row and lane association. Host materialization
of those chunk payloads happens after the public call; other capture observers
can add overhead within that call. Capture durations are excluded from timing
summaries.

Check: `golden/observation.json` points to SHA-256-named raw files in `blobs/`.
Array records retain dtype, byte order, shape and row-major bytes. Quantity
records retain units. Public Result fields, typed errors, certificates,
primal/derivative equation evidence and ordered optimizer occurrences remain
available alongside the actual identity values.

```bash
python benchmarks/golden.py compare /absolute/left/golden /absolute/right/golden
```

Check: raw self-comparison returns zero differences before using a capture as
the baseline. Raw comparison includes JSON byte order, whitespace and dispatch
chunk boundaries; parsed differences never erase raw-byte differences.

For implementation before/after comparisons, project each retained capture by
source occurrence, then compare the projected directories:

```bash
python benchmarks/golden.py project /absolute/before/golden /absolute/before-comparison
python benchmarks/golden.py project /absolute/after/golden /absolute/after-comparison
python benchmarks/golden.py compare /absolute/before-comparison /absolute/after-comparison
```

Check: each frequency occurrence retains its exact value, units, dtype, physical
axes, outputs, certificates and checked payloads. Status, present/performed flags,
source order, final failure and structured evidence remain semantic. Different
execution of later lanes remains a difference. Original chunk boundaries,
completion order and intermediate reducers stay in the unchanged raw captures;
projection provenance is written beside the comparison directory. Missing or
empty observation inputs raise an error.

The Human supplement of 2026-10-21 separates named source/request identity hashes,
per-execution UUIDs, measured timings and formatted traceback text from the zero-
difference requirement. `project` records those exact values and paths in the adjacent
`*-comparison.provenance.json`, with typed markers preserving their field positions.
`compare` reports their differences under `provenance_differences`; its exit status
uses semantic differences only. Numeric blob hashes, dtype, shape, physical coordinate
IDs, Quantity units and structured failures remain semantic. Encoded JSON evidence
is decoded from its checked raw bytes so the same field rules apply inside it;
its numeric encodings remain intact. The raw archive keeps the original evidence
bytes, whitespace and all identity values.

Private readers, live owners, verification tokens and presentation caches are
excluded from dataclass projection. Unknown public types raise an error.
Evidence is never replaced with object representations or fabricated zeros.

## Measure under comparable conditions

Coordinate the measurement window with other running jobs. Record the actual
load rather than labelling a busy host idle. Cold means a fresh interpreter's
first operation; warm means the same Run and request structure after that first call.
The harness imposes no retired shared-resource quota. An explicit
`--cpu-threads` records the chosen SCNSim runtime configuration.

See `profile.md` for the separately instrumented cProfile, py-spy thread and
JAX traces. Keep those timings separate from ordinary benchmark samples.
Forward-error assertions require the independently estimated observable
condition number specified by the release plan; this raw-byte harness invents
neither condition numbers nor accuracy tolerances.

## Installed-wheel milestone capture and paired timing

Use the exact installed wheel in the parent and every worker. The optional
`--installed-wheel-only` worker mode records the distribution's direct URL,
RECORD hash and actual module origin, and rejects editable or overlay imports.

Golden capture may observe existing internal native, evidence, sink and candidate
outputs read-only when certificates have no public accessor. These hooks run
only for capture; they forward the original call and never add a solve. Error
and SparseEquationCheck serialization uses their public PyTree protocols.
Keep every numeric leaf, certificate, presence/performed flag, ordered failure
and unknown field under the existing comparison rules. Report authorized
numerical changes beside the raw differences instead of removing them.

T10 BEFORE uses the exact `68fc9306048d4b660307920775ea2a37acfcb824` wheel;
AFTER uses the final integrated R1 wheel, each with its own recorded source label.
For paired timing on a shared host, a maintainer controller can supply
`--sample-control-fds EVENTS_FD RELEASE_FD` to a timing worker. Before each
original call the worker writes a ready event and blocks for one `G` byte;
after report projection it writes a complete event. The handshake and load
observations are outside the stopwatch. Before and after workers retain their
own Run while the controller alternates each cold/warm sample. No extra call,
cache clearing or warmup is introduced. T1–T9 keep five cold and five warm
samples; T10 keeps five cold only. Record actual host load and whether the
measurement used an idle window or paired alternation; apply no load threshold.
