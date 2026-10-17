# Goal: organize and archive the rigid stress branch

Completed 2026-10-17: [implemented cleanup and verification](BRANCH_CLEANUP_RESULTS.md).
The requirements below retain the accepted scope.

User direction, 2026-10-17. This is the active next stage on
`Roushelfy/Genesis:rigid-stress-recovery`.
The practical throughput stage is complete; start from its accepted code and
[results](PRACTICAL_THROUGHPUT_RESULTS_20261017.md), not a fresh optimization or
physical-convergence campaign.

## Result

Make this branch understandable and maintainable for a developer or agent who
has not followed the experiments. Keep a small, clear path from the native API
to the Franka/egg example, necessary tests, current model/limitations, recommended
performance settings and exact reproduction commands. Archive material that no
longer earns a place in that active path.

This is actual repository cleanup, not just another inventory or plan.
Choose a sensible structure and complete the moves, reference/import fixes,
deduplication and entrypoint updates. Use engineering judgment rather than
preserving every intermediate script as an active tool.

## What to keep active and what to archive

Keep the Genesis rigid-solver feature, public configuration/output/reset
contracts, useful examples and benchmarks, substantive regression tests,
independent references still needed to validate changes, and a compact current
documentation set. Retain the arbitrary/asymmetric convex/nonconvex contact,
normal/frictional loading and global max/full stress behavior.

Survey the research modules, shell wrappers, patch files, configs, reports,
JSON/scheduler dumps, figures and evidence as well as documentation.
Classify by actual imports, callers and reproduction dependencies, not age or
filename alone. Consolidate duplicates and one-off launchers; archive obsolete
standalone backends, abandoned alternatives, dated progress/audit narratives,
superseded goals and redundant generated material where appropriate.
Do not remove production modules or meaningful tests merely because one demo
does not exercise them.

Preserve the current scientific/performance conclusions, relevant negative
results and known model limits. Old throughput rates from different workloads
must not compete with the current recommendation in the main README.
Historical instructions in archives are historical records, not active goals.

## Archive so it can be recovered

Use a discoverable archive index with the former path, reason, destination or
pinned commit, and an exact retrieval/restoration recipe. A suitable archive
directory, a commit-pinned historical source, or an existing durable evidence
packet can be used; do not duplicate large assets just to change their label.

Moving files inside the checkout does not shrink Git history. Do not rewrite
shared history or force-push to make the repository appear smaller. If checkout
or runtime artifacts are moved outside the active tree, retain a durable,
verifiable location; an unverified local scratch path is not an archive.

Keep canonical evidence bytes, source/configuration identities and hashes
traceable. Use an index/resolver or pinned source for old provenance rather
than silently editing immutable manifests and their original paths. It must
remain possible to reproduce the accepted configuration and inspect the
evidence supporting its claims.

Archived code need not remain part of normal imports or test discovery.
Provide restoration or an isolated entrypoint where execution is still useful.
Avoid leaving many compatibility copies that recreate the clutter.

## Simplify the handoff

Keep GOAL.md, AGENTS.md and README.md short and mutually consistent, with one
active goal and one obvious current usage/reproduction path. Centralize the
stable feature contract, assumptions/limits and accepted results; link to
historical evidence instead of repeating a chronology. Production use must
continue to work without importing the research prototype or archive.

Avoid expanding this cleanup into a new abstraction framework, physics change,
API redesign, giant evidence-publication campaign or long GPU sweep.
Fix concrete code/layout defects needed for the cleanup, and explain any
behavioral change separately rather than hiding it in file moves.

## Verification and completion

Check active imports/entrypoints, relative paths, packaged assets, documentation
links and source-reconstruction recipes after the changes. Run focused tests
appropriate to affected paths and a bounded smoke check of the normal
Franka/egg and benchmark commands where the environment permits.
Unchanged physics does not require rerunning the expensive acceptance matrix;
a semantic numerical change does require its relevant consistency checks.
Label unavailable hardware checks honestly.

Finish with a clean committed branch, a concise current README, usable archive
index and compact summary of what remained active, was consolidated or archived,
why it moved and how it is restored. Report before/after active-tree counts or
size where useful. Completion means the cleanup is implemented and the main
workflow still works, not just that all files have been listed.
