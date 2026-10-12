# Native finite-disk refinement and matched scheduling checkpoint

The nonconvex goal remains **incomplete**. This checkpoint extends the
[verified-circle repair](NATIVE_CONTACT_SCHEDULING_20261012.md), retaining all
original component wrenches, radii, actual contact friction and final numerical
budgets. General boundary-crossing refinement, actual policy acceptance,
native cooperative scheduling selection and large-batch throughput remain open.

## Production repair and validation

A fixed Q10 circle can miss a legal compression centroid near its boundary or
a legal normal torque near its Coulomb capacity. More dual iterations cannot
repair that missing discrete representation. Independent finite-area witnesses
demonstrate both defects on the same low-shell face and physical disk.

Six compression-centroid cases require 0.98/0.99 of a 10-micrometre physical
radius, with three directions and actual coefficients 0.5/0.85/1.2. A uniform
vector traction on an interior subdisk of radius 0.25*(1-fraction)*R realizes
the unchanged point wrench. Independent nonnegative-compression LPs reject the
old Q10 cloud. Six additional normal-torque cases require 0.99/0.999 of mu*Fn*R,
including both signs and the same three actual coefficients. Uniform positive
compression with bounded circumferential traction on an interior annulus gives
a finite-area feasible witness. The old disk rule reaches only 0.9740767458*R;
centroid-only radial refinement reaches 0.9934766321*R and still fails the
0.999 torque cases even with 256 evaluations. That failed candidate is retained.

Production now retries unresolved **single-source disks verified to lie wholly
inside one actual material triangle** on nine positive-area integration panels.
The radial partition depends on the larger required compression-centroid and
normal-torque fractions; the angular partition follows the centroid direction
when resolved. The panels cover the entire unchanged disk, use Q10 in each
coordinate and retain the Gaussian width 0.45*R and squared boundary taper.
The required traction law, true material normal and actual coefficient stay
unchanged. This is local integration repair, not a physical convergence study.

Typed SCRATCH flags and two chart coordinates reconstruct all 900 sample
positions, normals and signed P2 bases from shared geometry in Quadrants.
No per-sample geometry cloud is allocated. Already successful contacts retain
their profile, indices, dual, scale and evaluation state exactly. A device
mask selects the retry and preserves the total evaluation count. The canonical
scalar fitting kernel provides all-contact and masked-retry modes through
Python dispatch only; production arithmetic remains Quadrants. Each route
retains 128 evaluations and the 9e-9 internal stopping criterion.

Necessary continuous disk-capacity violations receive analytic six-component
polar certificates, with operand-scaled roundoff and the unchanged original
point error allowance. A centroid beyond R or normal torque beyond mu*Fn*R is
diagnosed only when the strict bound and target separation exceed those
allowances. Other unresolved cases retain their existing diagnostic route.
Shared pair supports and boundary-crossing footprints are not covered by this
particular refinement kernel.

Independent CPU FP64 geometry, profile, vector-fit, signed P2 loads, complete
equilibrium including all six gauge rows, tensor/von Mises and global maximum
comparisons pass. Final relative force/moment budgets remain 1e-8; equilibrium
remains max(1e-11 N,1e-8*||rhs||); stress remains rtol 1e-4/atol 1e-3 Pa.
Regression coverage includes partial reset isolation, release/restoration,
unchanged easy environments, physical centroid/torque rejection controls and
both torque signs. The full native suites pass **54 CPU cases in 454.41 s and
54 CUDA cases in 672.04 s**, after targeted two-case passes. Source hashes close
unchanged on both allocated nodes. Hardware is EPYC 9454 and RTX PRO 6000
Blackwell Server Edition; Genesis 1.4.3, Quadrants 1.3.3, container 1_26.

Run `bash research/rigid_stress/scripts/check_nonconvex_contact.sh cpu` or
`cuda` inside the corresponding allocated node. This suite uses only the low
complete hollow shell: 480 tetrahedra, 810 P2 nodes, 2430 displacement DOFs.

## Completed same-GPU measurement and policy failure

The following comparison uses the **preceding verified-circle law**, without
the new nine-panel retry. Scalar production is compared to experimental QD
sample-parallel fit/commit/scatter with device-masked canonical scalar fallback.
It does not establish native selection or performance of the new refinement.
One RTX PRO 6000 GPU on rtx-209-113, UUID
`GPU-adb3f72e-e837-07af-76f0-f0838bfdf2ed`, driver 595.71.05, executes both
variants. B=32, FP64, maximum output, dt=0.01 s, Q10/s8, pair-union support,
GJK mesh64 patch contacts, seed623101 and the identical varied condition bank
are held fixed. Condition SHA256 is
`156d956bc5fc3827da8df236cf1695a4bf30a9a1e1d939c672f6a57d69d10930`.
Each variant has 600 warmup steps and three measured repetitions.

| Scope | Scalar effective env-step/s, three repeats | Experimental effective env-step/s, three repeats |
|---|---|---|
| Rigid-only | 8086.881, 8112.771, 8114.764 | 8093.641, 8117.649, 8129.826 |
| Recovery-only, frozen real snapshot | 185.449, 185.451, 185.518 | 3339.115, 3338.327, 3340.514 |
| Live rigid plus stress, varying contacts/reset | 197.600, 197.568, 197.137 | 2642.742, 2639.127, 2644.121 |
| Policy plus rigid plus stress | Failed; no valid complete rate | Failed; no valid complete rate |

Divide environment rates by 32 for whole-batch step/s. Live repeats each contain
1280 complete batch transitions, 37 resets, zero invalid/overflow transitions
and all 32 environment failure counters at zero. No environments are excluded.
The repeated mean live gain is about 13.38x. Recovery-only is explicitly a
snapshot replay. Native measured buffers occupy 167,458,738 bytes. The
snapshot intrinsic wall profile is 173.148 ms scalar versus 10.211 ms
experimental; scalar fitting is 148.193 ms, commit 10.412 ms and scatter
10.269 ms. These measurements do not establish an optimization plateau.

Both policy variants reject a load during measurement and produce no result
JSON. The policy is an inference-only Torch FP32 MLP 26->128->128->7, seed99173,
with 1e-4 rad residual scale and observation interoperability included. Its
metadata in another scope does not mean that scope executed policy inference.
No complete policy rollout or RL training throughput is claimed.

A separate untimed diagnostic preserves the first failure before rigid
integration changes the working buffers: call1477, workload tick876, substep0,
environment24, raw contact1, status34. Physical R=0.006465712642 m and actual
mu=0.5196625772 remain unchanged. Both original components use the two actual
sources of geom24/pair22; no component is merged or dropped. External applied
and coupling forces/torques are zero. Independent CPU bounding of the saved
sampled separating ray over both unchanged continuous Euclidean disk/triangle
supersets gives an upper bound +0.0200208790, with positive target margin
0.0015123190. Consequently this ray **does not prove continuum incompatibility**.
Boundary-crossing sampling and continuous feasibility remain to be resolved.
The final scene error loses the current contact detail after rigid integration;
the preserved first-failure record is authoritative for this diagnosis.

## Evidence and remaining work

The [publication manifest](../evidence/20261011-nonconvex/disk-refinement-checkpoint-v1/manifest.json)
records **1133 copied artifacts, 19,571,031 bytes**, and one external 3,728,124-byte
snapshot. Every original and published SHA256 was verified on allocated CPU
compute; manifest SHA256 is
`9e0571743bfe2bf896dad532d16a3b08b07077c238fabf1ee57295908bd9b511`.
It records closed native tests, positive-area CPU/device
proofs, failed prototypes, same-GPU complete/failed scopes and the original
policy failure capture. Large runtime snapshots remain under the project data
directory with explicit hashes. Failed harness/import epochs are preserved and
never counted as numerical passes. New runtime data/caches stay on the data
mount, and no high-resolution physical mesh or quadrature-convergence study ran.
The initial general triangle/disk radial-clipping prototype retains the
physical radius and requires the existing validated intrinsic paths, with the
correct squared radial Jacobian. Its first CPU audit fails a selected-count
comparison and its second native component is unresolved; it is retained as
failed exploratory evidence and is not selected production integration.

Next resolve the legal/general boundary-crossing integration path and actual
policy failure, preserve first-failure diagnostics through the rigid lifecycle,
and settle the separately proved step602 component-model incompatibility
without silently changing component acceptance. Rebuild the baseline on the
final repaired model, qualify native scheduling, evaluate history/layout/tail
packing and further measured opportunities, then measure complete four-scope
1024/2048/larger workloads, seeds, repeated long trajectories and platform
behavior. No completion claim is supported while these remain open.
