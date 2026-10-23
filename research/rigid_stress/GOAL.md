# Active goal: maximize automatic-radius DPR throughput

**Immediate user goal, 2026-10-23:** maximize throughput of the now-integrated
automatic-radius + Hertz-pressure DPR path. Follow
[the optimization goal](docs/DPR_AUTOMATIC_RADIUS_GOAL_20261023.md#immediate-goal-maximize-native-automatic-dpr-throughput).
Prioritize contained-disk analytical P2 integration, contact-face/geometry and
refined-moment reuse, duplicate/zero-work elimination, fit/scheduling and batch
tuning. Measure max/full complete-scene gains against matched manual DPR and
pure rigid. Preserve the load model and numerical checks; use low-resolution,
focused validation without waiting for broad physical convergence campaigns.

**Latest user correction, 2026-10-23:** follow
[the automatic-radius DPR goal](docs/DPR_AUTOMATIC_RADIUS_GOAL_20261023.md).

Keep the existing DPR load transfer and prefactorized linear FEM. Add the
cheapest useful contact-radius/support estimate, using Hertz or another cheap
declared approximation where appropriate. Feed the finite Hertz/contact pressure
through DPR's existing load transfer into that same FEM. Keep a radius-only
control separately labelled if it retains the original DPR pressure profile.

The runtime path is:

```text
rigid contacts -> cheap radius estimate -> finite contact pressure
               -> original DPR load transfer -> prefactorized FEM
               -> existing maximum/full stress output
```

Stop further development of analytic particular fields, boundary-traction
correction and combined stress/probe fields for this goal. Preserve their
results as historical research. Their sharp-edge singular maxima, standalone
friction-history solvers and two-face pressure references are not prerequisites.

Only rest shape/material are fixed. Preserve arbitrary changing/asymmetric
convex/nonconvex contacts, normal/friction forces, max/full output, batching,
isolation and partial resets inside the existing rigid solver in Quadrants.

Reuse original per-contact radius storage, native Hertz lookup, collision
witnesses and fixed-shape preprocessing. Optimize the actual added work.
Retain established DPR validity/numerical checks; do not introduce unrelated
enrichment-specific requirements or silently modify the source loads.

Prioritize low-resolution implementation, focused numerical consistency and
matched pure-rigid/manual-DPR/automatic-DPR throughput. Use compact independent
physical comparisons, preferably the requested Code_Aster reference, without
blocking throughput work on broad high-resolution convergence sweeps.

Deliver the usable feature on this branch with concise commands, recommended
settings and honest measured errors/costs. The earlier
[hard-contact goal](docs/AUTOMATIC_CONTACT_AND_THROUGHPUT_GOAL_20261020.md)
and sharp-edge/combined-field reports are historical where they conflict with
this correction. Follow the existing storage and reproduction policies.
