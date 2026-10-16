# Application delivery acceptance audit

Acceptance follows [PRACTICAL_VALIDATION_GOAL.md](PRACTICAL_VALIDATION_GOAL.md),
including its explicit separation of engineering delivery, physical qualification
and experimental fracture calibration. The delivered scope is the conditional
native stress/reward feature described in the [application report](PRACTICAL_APPLICATION_RESULTS_20261016.md),
not universal physical accuracy. This audit does not reclassify historical
failures or unresolved reference tails as passes.

| Required result | Direct evidence and disposition |
|---|---|
| Existing Genesis rigid-solver feature; all production arithmetic in Quadrants | Native implementation and public link API remain in the production tree. Examples import no research solver. All899 actual native snapshot files match the recorded source identity; every898 non-synthetic source/asset file also matches its tracked Git blob at cd42aed09. |
| Accept existing independent elastic verification within scope | Existing40 CPU/CUDA bounded-path cases and228 native full fields remain accepted. Source identity is preserved; no unchanged giant verification campaign is repeated. New32 reconstruction fields retain independent full residual, complete raw corner tensor and global-maximum checks. |
| Principal full hollow egg/Franka R/L/M | Four original nonlinear-contact states produce1/2/3/2 actual packets; source/group wrenches reproduce the qualified history exactly. Native replay closes32 state/model/radius variants, using the same material, represented geometry, coordinates and full-inertia convention. M/L, L/R and M/R remain separate; frame-coordinate and balance differences are explicit. |
| Actual omitted-feedback effect | Original rigid/deformable counterparts retain all330 controls/2640 states, original force/actual Panda effort gates, seven tensor snapshots and14 native resolved-load replays. Largest sampled feedback difference23.17% is not an all-time bound. |
| Adequacy and uncertainty of the numerical contact reference | Existing geometry/bulk/contact/time investigations are reused. The480-tet reference is discretely qualified but physically uncertain. Absolute contact/stress accuracy is unverified; simpler prescribed-traction controls support numerical/model conclusions. No unavailable continuum check is reported as passed. |
| Representative nonconvex interior hotspot and convex control | Objective nonlinear U reports resolved-load versus nonlinear recovery and six model fields, inner-bend maxima and source redistribution. Smooth convex controls retain matched integration/distance/model comparisons. Their unresolved spatial/raw-peak tails remain explicit limitations. |
| Arbitrary changing/asymmetric/frictional/multiple contacts, release and reset | Existing native regression and actual8-condition full Franka histories are retained. New controlled traces cover strong/weak grip, release and partial reset; selected peaks reset to zero and unselected peaks remain unchanged. Enabled moments and support identities are preserved. Unsupported pressure moments and actual status36 input remain explicit. |
| Applicable fast/robust models; raw peaks, positions and full maps | All32 rows preserve unaveraged maxima/coordinates and actual raw tensors. Independent cloud checks use each model's unchanged per-contact or aggregate wrench contract. Pressure local-cone violations remain a substantive limitation. Source-prior aggregate conservation and local redistribution are quantified; no continuum-island identity is inferred from source IDs. |
| Physical footprint/profile sensitivity; no hidden fidelity change | Fixed3mm/6mm snapshot assumptions, two reward radius schedules and existing alternative-profile evidence are retained. Radius is not a mesh parameter or inferred contact area. Cold setup, different reference/native timestep and collider scopes are disclosed. No input is adjusted to pass a rejected case. |
| Observable-specific practical reward acceptance | Three restarts per condition/model/radius.1N/4N retains near ties and rank reversals. The predeclared0.2N/4N extension has8/8 tested lift-window rank stability and7/8 lift-window separations beyond the observed band; condition1 stays unresolved. Whole-episode audit has1/8 beyond-band separations and actual phase-dependent preferred-choice reversals; no lift-window-to-episode safety inference is accepted. All460800 observed reward steps are finite. This is numerical proxy discrimination, not real damage calibration. |
| Warmed end-to-end multi-environment cost and memory | Two same-GPU four-scope allocations at32 environments;900 warmup,3x2400 timed steps per scope,96 partial resets/repeat,0 invalid steps. Rigid/live and policy-rigid/policy scopes, native allocations, Torch memory, incremental cost and hardware UUIDs are recorded. Live medians5840.70 pressure and2623.09 source-prior valid env-step/s are conditional engineering rates. |
| Relevant regression after changes | Production arithmetic is unchanged from the qualified source. New recording extension exactly reproduces original source wrenches; all32 independent elastic/cloud checks and eight repeated native application traces close. New source passes Ruff, wrappers and documented Bash commands pass syntax checks. No unrelated test tolerance is changed. |
| Runnable commands and actual reproducibility | Snapshot utility exactly verifies899 files/225384712B. Restoring the published packet verifies53 files/1994054B. Fresh GPU job720758 runs all four robust6mm replays from those restored bytes at unchanged original numerical gates; largest raw-peak relative difference from the original comparison is1.010e-11. |
| Representative figures and editable deliverables | Two actual-data map packets include SVG/PDF,1000dpi PNG and previews; both previews were inspected with view_image and pass repository-report QA. Per-cell tensors remain unsmoothed; Gouraud interpolation is display-only. Existing U/convex maps remain linked. |
| Report, recommended settings and explicit operating-range table | Delivered in the application report, API guidance and runnable command handoff. Recommended source-prior configuration is conditional on declared mesh/material/radius and accepted inputs. Similar choices inside uncertainty, the diagnosed pressure local traction, near-cancelling source-prior inputs and physical fracture thresholds are not claimed reliable. |
| Reproducible evidence without repeated giant publications | Main compact packet practical-application-v2 preserves23 naturally closed jobs, failures, commands, arrays and source identities.791 entries match791 actual runtime originals. Its43.4MB packet reconstructs native package/assets from tracked Git identities instead of duplicating225MB per run. The separate restored-input handoff verifies89 entries against89 actual originals and retains its own natural closure. |

The main packet manifest SHA256 is
`bab8578d1fa989dc818db6a3e49db87dde9614133ae4bc70cf83a268a1a50268`.
Its [manifest](../evidence/20261014-physical/practical-application-v2/manifest.json)
and exact submitted command journals identify all measured scopes.
Actual production, wrapper and scheduler statuses are verified together;
program exit0 alone is not used as proof.

The former universal convergence matrix, extra shapes, full RL training and
experimental fracture calibration remain separately scoped follow-up work.
Cancelled broad refinements retain their confirmed states and partial outputs.
Older background research jobs do not redefine this delivery's acceptance.
Remaining physical uncertainty limits the claims and operating range above.
It does not imply that an unperformed check passed.
