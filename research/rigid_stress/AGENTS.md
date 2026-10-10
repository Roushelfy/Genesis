# Instructions for the native rigid stress feature

Read `GOAL.md` and `docs/NATIVE_QUADRANTS_PLAN.md` first. They record the user's
2026-10-10 change of direction and supersede older implementation choices and
validation ordering. Follow the root `CLAUDE.md` and `CODING_GUIDELINES.md` for
engine code. Existing research modules and measurements are references; the
production feature belongs to Genesis's existing rigid solver.

- Use Quadrants for the complete production numerical path, including recovery
  solves. External numerical backends in the research prototype are reference
  paths, not acceptable substitutes for the requested native implementation.
- Develop and tune on inexpensive low-resolution full-shell meshes. Complete
  detailed profiling and the applicable performance optimizations before
  spending GPU time on physical mesh/quadrature convergence studies.
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
