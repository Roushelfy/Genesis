# Moving normal material probes

The experimental normal recovery component can now follow each supplied contact
with 89 material probes: 32 ellipse-edge points, 32 half-radius points, the
center and 24 axial depths from 0.01 to 4 major radii. Generation, location,
tensor evaluation and max/full reductions run entirely in Quadrants. The shared
elastic inverse/factor and existing numerical acceptance remain in use.

Use the exact experimental source
[`a599b9c50`](https://github.com/Roushelfy/Genesis/commit/a599b9c5033da7f6d4e40b79eb9be9a9058d0be0)
and `StressHertzRecovery(..., moving_probes=True)`. The default fixed-query path
is preserved. Public rigid stress options and automatic collision inputs are
not connected to this component yet. This extends the
[normal recovery control](NATIVE_HERTZ_RECOVERY_20261022.md); its flat-patch,
half-space, four-major-radius clearance and normal-only restrictions still apply.

## Material ownership and failure behavior

Initialization constructs tetrahedral neighbors from matching vertex identities.
Each probe starts in the actual source anchor's owning element and follows its
source-to-probe segment through shared faces. A bounded walk never searches
globally or switches to a nearby disconnected surface. Failure invalidates the
environment through the existing error and mechanical-step latch; complete
probe/corner outputs become NaN and the event is counted once per step.
Inactive regions clear their derived locations and do not contribute zero
tensors to the tensile maximum. Probe state is recomputed on every update.

CPU/GPU controls vary centers, principal directions and normal resultants, with
two displaced or opposing regions, independent environments and release/reload.
They check every retained tensor, sampled principal/VM peak, max/full agreement,
one unchanged environment, insufficient-walk failure and next-step recovery.
Separate material segments stay in their own components; a target inside the
other component remains unresolved. Four topology tests pass on each backend,
including nonconvex fixtures, duplicate/interior faces and nonmanifold refusal.
The existing fixed-query complete-field control also passes on both backends.
These checks do not qualify physical stresses on general nonconvex contacts.

## Preserved rotation failure and independent fields

The first moving-probe version passes the unrotated controls but exceeds its
1e-7 tensor comparison target after rotation. Saved failed states show nominal
surface depth up to 1.38e-18 m, about 2.34e-14 major radii. World-coordinate
round trips affect both the native and CPU surface-edge tensors. Their original
failures remain failed. The repaired self-field calculation retains the probe's
defining local coordinates and exact surface depth; arbitrary material queries
still use their actual coordinates. No depth threshold or tensor tolerance is
loosened. The independent CPU reference uses the same parametric query definition.

Complete saved-array inspection covers eight-frame moving/rotated controls and
24 frames from the actual benchmark input generator. It independently reduces
all raw sampled peaks, checks CPU/GPU and rotation, and compares the same-load P2
solution plus a higher-order prescribed-Hertz boundary/particular reference.
Maximum observed changes across the admitted configurations are:

| Quantity | Relative difference |
| --- | ---: |
| Same-load contact probe tensors | 2.33e-9 |
| Same-load complete corner tensors | 1.58e-13 |
| CPU/GPU or rotated probe tensor banks | 6.30e-9 |
| Higher-order prescribed-load probe tensors | 0.000666% |
| Higher-order complete corner tensor banks | 0.1282% |
| Sampled maximum principal stress | 0.00307% |
| Sampled VM maximum | 0.000447% |

The complete corner bank fails the additional 0.1% reference-change target.
Both strict saved-array inspection jobs retain their actual FAILED status;
the completed second inspection reports all metrics and a false strict flag.
The existing goal's separate 5% tensor / 10% peak engineering budgets are met
for these same-mesh, prescribed-load readouts. The numerical 1e-7 target is met.
This distinction does not establish unknown pressure, continuum convergence,
an unsampled global peak or hollow-egg scene accuracy.

## Matched moving-input cost and recommended setting

On the same final allocation, every warmed update includes resident independent
load phases/release, center motion, rotating curvature, Hertz estimation,
boundary integration and acceptance, the regular solve, material-probe location,
combined tensors and sampled maximum reduction. No host readback or reference
audit is inside primary timing. Each median uses three eight-frame blocks.
All variants retain the same 89 probes per region and two region slots.

| Batch | Exact field, order8 max/full (ms) | Far field, order8 | Far field, order6 | Far field, order4 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.45 / 1.46 | 1.41 / 1.51 | 1.28 / 1.33 | 1.20 / 1.24 |
| 64 | 1.95 / 1.98 | 1.75 / 1.77 | 1.60 / 1.62 | 1.50 / 1.52 |
| 256 | 4.26 / 4.32 | 2.95 / 2.99 | 2.54 / 2.58 | 2.27 / 2.30 |
| 1024 | 12.65 / 12.71 | 7.99 / 8.00 | 6.63 / 6.67 | 5.70 / 5.77 |

Order6 passes both moving/rotated controls and the matching input-generator
comparison. It reduces B1024 time another 17.0%/16.5% versus the order8 far rule;
their repeat ranges separate. It is the recommended measured setting for this
qualified normal-box regime. Small-batch exact/far repeats overlap.

Order4 admits the measured trajectory and its independent unrotated comparison,
but refuses a rotated source at 1.048e-4 wrench error against the unchanged 1e-4
gate. Its faster rate remains conditional; it is not a general replacement for
order6. No rejected state is counted as admitted. Separate 24-frame audits save
actual centers, axes, status, loading, elastic validity, wrench error and peaks.
B1024 admits 24576/24576 frames including 17014/17014 loaded frames. Every
saved max/full peak pair and exact/far/quadrature comparison passes its stated
numerical/change gate. These are component rates, not rigid Franka scene rates.

## Reproduction and remaining integration

The [terminal packet](../evidence/20261019-qualified-ranking/hard-native-hertz-moving-probes-v1/README.md)
retains captured workers, all source patches, topology tests, original failures,
strict/engineering flags, actual scheduler states and full-array hashes. Run
captured task lists with the pinned source, existing data-side physical Python,
Slurm `rtx-mid`, image `1_26`, explicit fresh data outputs and private node-local
caches. `check_moving_hertz_probes` accepts backend/control/quadrature and optional
benchmark-input/step arguments. The timer uses `--moving-probes --moving-centers`;
the independent inspector deliberately exits with its preserved strict failure.

Actual live contact association, curved/inner-contact coverage, physical friction
and history/reset integration, and accepted max/full Franka scene measurements
remain required. The complete active goal remains open.
