# Intrinsic source robustness and initial profile

See the [subsequent patch/refinement checkpoint](NATIVE_PATCH_REFINEMENT_PROGRESS_20261011.md)
for the source geometry capture, explicit direct-patch producer, longer
trajectory failures and initial cooperative-fit evidence.

This follows the validated [surface-fan checkpoint](NATIVE_SURFACE_FAN_PROGRESS_20261011.md),
source `209ad577384a2548bb2173498322fbc9fabd9d03`. The active nonconvex goal
is unfinished. The first new-law profile identifies actionable contact costs;
the first B=16 extension also identifies a genuinely incompatible generated
force. Neither result supports declaring an optimization plateau.

## Validated frozen-frame profiling

The normal benchmark now accepts intrinsic contact blocks, the explicit FP64
mesh input and collision-method selection. Its existing frame restoration
reconstructs the pre-integration pose and angular velocity for a frozen solved
contact snapshot. The new preparation check compares the replayed original
positions and forces, final wrench errors, complete residual and stress peak
before timing. At B=4, seed 623001, after 180 live steps, it records:

| Replay check | Measured value |
|---|---:|
| Original position difference | 0 m |
| Largest nodal-force difference | 6.94e-17 N |
| Largest peak-stress difference | 3.94e-9 Pa |
| Largest normalized force error | 8.18e-9 |
| Largest normalized moment error | 4.95e-9 |
| Largest complete residual | 8.07e-12 N |

This retains all final budgets. The initial profile reaches its numerical
checks and kernel measurements but fails while exporting an unused zero-size
buffer through DLPack. The corrected report inspects those buffers without
exporting their null pointers and counts shared aliases once. It exposes both
logical field bytes and distinct native allocation bytes. The corrected B=4
report contains 190 distinct allocations, 76,980,698 native bytes and
76,990,378 logical field bytes. Device-wide free memory is also recorded;
native-array bytes alone are not whole-process GPU memory.

Twenty profile repetitions use the same Q10 surface-fan law, eight edge
subdivisions, FP64, contact block size two, max output, no history, dt=0.01 s,
one substep and current solved contacts `[7, 11, 1, 1]`. The full intrinsic
contact stage costs approximately 244.04 ms on this deliberately small batch,
where each serial contact task leaves much of the GPU idle. Independent
elastic inverse application is approximately 0.182 ms, complete residual
0.064 ms and global peak 0.057 ms. These are intrusive stage-profile timings
on a frozen input, not live throughput measurements.

The Quadrants kernel record assigns 70.58% of measured execution to the
serial vector fit, 14.53% to signed nodal scatter and 12.18% to contact commit,
which regenerates samples and checks their original wrench. Source resolution
is 1.65%. The fit's observed per-call min/mean/max are
13.966/28.182/62.977 ms, across 120 contact-block calls. These are block costs,
not independent per-environment tail measurements. Fine-grained cooperative
fit/reduction, scatter/wrench reuse, block scheduling and large-batch layout
remain applicable candidates. Local candidate profiling and large batches
must determine their end-to-end value; no speedup is claimed yet.

## B=16 failure: distinct from integration and iteration

The unchanged production law with varied B=16, seed 623001, completes 591
steps, then rejects env 13/contact 2. The original force, radius
`0.005564813716696238 m` and actual combined friction
`0.6054935315460935` are preserved. Every earlier checked live frame and four
CPU capture times pass. This partial trajectory does not count as 640-step
acceptance and is not excluded from the evidence.

Independent CPU geometry includes all continuous triangles inside the
Euclidean radius ball, a superset of the intrinsic footprint. Only faces 8
and 11 intersect that ball. An analytic unit polar ray has strict support
approximately `[-2.00e-12, -0.03780946]` on their inward Coulomb cones, while
its dot product with the unit target force is `0.0007000532745`. This proves
that every admissible continuous traction distribution misses the target
force by at least that normalized amount, well outside `1e-8` acceptance.
The native continuous certificate correctly reports raw status 35.

The collision/reference vertices match exactly. Their facet normals differ
by only `1.24e-16`; the produced normal differs from face 8's actual inward
normal by `0.00355719`. The target force is inside the produced-normal cone
to roundoff but outside the actual facet cone, with normalized cone margin
about `-0.000818381`. This input is not repairable by finer stress quadrature,
more fit evaluations or an internal stopping change. Relaxing friction,
altering the original force/radius or accepting the failed environment would
violate the contract.

Two explicit producer ablations also stop at tick 591: selecting GJK and
tightening MPR's fractional portal tolerance to `1e-10`. The latter is a
diagnostic configuration override, not a selected production default. The
rigid solver's perturbed multi-contact path reconstructs contact normals
after returning the discovered features to the original poses; source-geometry
capture and a direct GJK contact-patch control investigate that route. Normal
grasp contacts are not being classified as unsupported. The ordinary generated
input still needs a source-side remedy or a verified physical producer before
it can supply the large-batch baseline.

## Evidence and remaining work

The [complete capsule manifest](../evidence/20261011-nonconvex/online-checkpoint-v4/manifest.json)
contains 229 artifacts (7,651,507 bytes): both initial profiles, their exact
source states and hardware, all three failed B=16 trajectories and original
failure inputs, the independent continuous certificate, commands and logs.
Every original and copied artifact passes SHA-256 verification. Manifest
SHA-256 is
`6ab747deba313e1c3a60d5adfb5012f6d1ae928c8154b4f4d60f59b9a998f53e`.
Hardware remains RTX PRO 6000 Blackwell Server Edition, driver 595.71.05,
Quadrants 1.3.3, image `1_26`. The same low hollow shell is used throughout.

The direct-patch control is a separate collision-input comparison, so any
selected change needs a newly declared matched four-scope baseline. It is
not an optimization of the failed producer with identical input forces.
Primitive/SDF/decomposition/aggregate source coverage, general local refinement,
longer varied lifecycle checks and measured B=1024/2048/larger optimization
remain required. No high-mesh physical convergence study was run.
