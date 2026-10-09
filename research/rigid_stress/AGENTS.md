# Instructions for implementing this handoff

Read `GOAL.md` and all six docs before implementing the production path.
Follow the root `CLAUDE.md` and `CODING_GUIDELINES.md` when adding engine code.
The imported research scripts are reproducible historical references, not
production design/style templates. Keep the originals recognizable; write
clean, typed integration modules and compare them against these references.

Binding user assumptions:

- Full fixed reference geometry; no quarter/half symmetry reduction.
- Runtime contact locations, directions, counts, footprints and symmetry are unrestricted.
- Normal plus tangential frictional forces; keep pure contact moments when enabled.
- Complete grasp trajectories may be smooth; sudden changes and environment resets must remain correct.
- Global maximum stress only, with no fixed hotspot, fixed contact basis or per-task learned surrogate as the correctness path.
- Maximize total environment transitions/s; allow and tune parallel environments.
- Use a Franka Panda and a hollow egg-shaped **rigid** entity in Genesis.
- Elastic recovery is auxiliary; do not silently substitute deformable FEM dynamics.

The repository currently contains validated CPU math, evidence, an unaccepted
Franka seed and a Torch prototype, not a complete GPU implementation. Do not
report a TODO, host NumPy solve, CPU transfer, dense-debug factor, or synthetic
load benchmark as finished GPU/live-contact work.

Respect measured evidence boundaries and accuracy profiles in `docs/VALIDATION.md`.
Never silently relax physics timestep, contact sampling, residual tolerance,
mesh resolution or output error to create a speedup. Report fidelity tradeoffs
as separate profiles. Keep a CPU FP64/direct correctness oracle and a GPU
direct baseline. If hardware is unavailable, finish CPU integration and
buildable GPU code, record the blocker and reproducible commands, and label
GPU results unmeasured. Do not invent throughput or claim device validation.

Offline preprocessing can use CPU FP64. Runtime shared factors, per-env
histories, contact mapping, residuals and scalar maxima should reside on GPU.
Never reuse history across environment IDs or reset boundaries. Check complete
residuals including the six gauge rows; a reduced residual alone is insufficient.
Physical patch resolution/convergence and equation solve accuracy are distinct.

Do not introduce new approval pauses for routine implementation, tests,
commits or updates on this task branch. Do not overwrite the user's other
branches or push to the upstream organization repository.
