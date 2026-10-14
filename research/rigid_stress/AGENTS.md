# Instructions for the native rigid stress feature

Read `GOAL.md` and `docs/NATIVE_QUADRANTS_PLAN.md` first. They record the user's
2026-10-10 change of direction and supersede older implementation choices and
validation ordering. Follow the root `CLAUDE.md` and `CODING_GUIDELINES.md` for
engine code. Existing research modules and measurements are references; the
production feature belongs to Genesis's existing rigid solver.

Read docs/NONCONVEX_CONTACT_REQUIREMENTS.md for the completed 2026-10-11
extension and its declared contact law. That numerical/performance milestone
is the baseline for the new physical-validation stage below.

## Active physical-validation stage, 2026-10-14

Read `docs/LOAD_MODEL_PHYSICAL_VALIDATION.md` after `GOAL.md`.
The low-mesh performance milestone has closed. The user now authorizes and
prioritizes load-model validation, independent FEM/contact comparisons and
physical mesh/geometry/contact-integration convergence across convex/nonconvex
shapes. This direction supersedes the earlier deferral of those studies.
Keep production arithmetic in Quadrants and use external FEM only as a reference.

- Use Quadrants for the complete production numerical path, including recovery
  solves. External numerical backends in the research prototype are reference
  paths, not acceptable substitutes for the requested native implementation.
- The completed performance phase used inexpensive low-resolution full-shell
  meshes. The new physical-validation stage may refine meshes and integration
  as needed to answer its accuracy questions. Keep long sweeps opt-in.
- Keep same-mesh numerical consistency checks throughout optimization. Check
  complete residuals including the six gauge rows, scalar peak error, contact
  wrench/frame consistency, environment independence and partial resets.
- Preserve arbitrary changing contacts, full geometry, normal plus tangential
  friction, support loads and enabled moments. Smooth histories are a reuse
  opportunity; correctness must survive their invalidation.
- Keep rigid dynamics rigid and make stress recovery optional. Geometry,
  constitutive parameters and mass identify shared immutable operators.
- Maximize measured aggregate environment transitions/s. Count contact/load
  work, solve, residual, global peak, history, selection, resets and policy work
  within their stated benchmark scopes; do not silently change fidelity.

Use the math, CPU FP64 references and archived device evidence as needed, not
as a requirement to rerun the old expensive acceptance sequence. In particular,
the old cuDSS/cuSPARSE implementation, level-6 preset and convergence commands
do not define the new production path or the current iteration workload.

Choose algorithms, placement and tuning based on evidence. Routine tests,
commits and updates to this task branch need no new approval pause. Preserve
unrelated work and upstream branches. Label unavailable hardware tests and
unmeasured throughput honestly.
