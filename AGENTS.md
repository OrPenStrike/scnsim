# Working on SCNSim

The SCNSim Repo Lead owns product code, documentation, tests, CI and packaging.
Bring documentation and Preview work to the same Lead. Use the current task
and the affected source as the starting point; read additional guidance only
when it applies to that work.

## Work within one team

The Lead can make small, closely related changes directly. Delegate substantial
implementation, documentation-site work or long-running work when useful. Use
an existing team member as `docs_writer` for delegated documentation; keep one
writer per path or overlapping hunk. Workers return their work to the Lead
without creating another delegation layer.

Use the shared collaboration and model-routing Skills for worker limits,
model selection and cross-repository ownership. Preserve existing worker IDs,
active assignments and services when responsibilities change.

## Maintain behavior and its documentation together

An authorized feature, fix, refactor or removal includes its directly related
documentation, tests and ordinary local checks. Retiring a feature includes
removing its obsolete dedicated tests. Diagnose failures; do not hide them by
skipping tests, weakening assertions or reducing coverage.

Keep Human-defined research conditions, numerical tolerances, acceptance
criteria and privacy boundaries. Resolve changes to those requirements with
the Human. Follow the task's assigned numerical and verification scope; a
documentation change does not restart simulations or paused work.

Reuse valid results when their relevant inputs and behavior are unchanged.
Check the affected surface after a change. Run a full numerical comparison
only at the milestone assigned by the Human, not after every commit.

## Documentation and Preview

Keep public API explanations, docstrings, tutorials and examples aligned with
the implementation. Introductory tutorials use in-memory storage; advanced
material explains Workspace. Change generated pages and catalogues through
their owning sources and generators.

Preserve an existing Preview and its source snapshot during a handoff. The
Repo Lead owns its operation and shutdown; refreshing a Preview and publishing
the documentation site are separate actions.

## Preserve work and deliver coherent changes

Preserve Human edits, other writers' work, held overlays and unrelated staged
changes. Compose each candidate from its declared base and owned changes.
Historical results remain attached to the source and artifact that produced
them.

Use the existing Git owner for ordinary commit, push and remote readback on
the assigned branch. Keep the current PR and delivery scope. Merge, release,
deployment, dependency pins and external data transfer retain their own
authorization. Use the shared lifecycle Skill for semantic and delivery status;
passing checks does not itself grant Human acceptance.
