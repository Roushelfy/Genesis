# Native current-step anchor-port reuse

The selected native route computes each current anchor-to-seed-port distance
once per contact/environment and reuses it in the min-plus destination-port
pass. All seed norms remain Quadrants FP64 calculations. This removes repeated
norms across destination ports. Shared graph distances, topology, current
anchors, finite radius, Q10 samples, friction cones and final budgets retain
the same definitions.

The typed scratch has the exact built face-port/contact/environment dimensions.
It is overwritten for every valid current contact before consumption. Its
validity follows the current source and footprint status, including contact
birth/death, branch changes, workspace growth and partial reset. The internal
constructor's optional face-port count retains the original uncached route for
standalone callers and the explicit ablation. Production `IntrinsicContacts`
supplies the built count.

## Numerical qualification

CPU and CUDA each pass15 targeted source/prior/lifecycle cases. Their48
independent CPU FP64 full-field comparisons vary three actual friction values
per fixture and include original-point wrench, signed P2 nodal loads, complete
equilibrium with all six gauge rows, tensor/von Mises/peak agreement, dynamic
contact capacity, partial reset and environment isolation.

The worst normalized force/moment error is7.386602756e-10. The largest complete
residual uses0.045856225 of its unchanged budget. The largest independent peak
difference is8.236806025e-6Pa. The paired profile snapshot repeats differ by at
most2.498001805e-16N in nodal force and1.082662493e-8Pa in global peak. Its
maximum normalized wrench errors remain below1e-8.

## Same-GPU measurement

Both ablations include the identical qualified generic EPA bound correction.
They use the same low complete hollow shell (480 affine-P2 tetrahedra,
810 P2 nodes/2430 DOFs), FP64, Q10, s8 surface paths, maximum output,
`body_source_prior`, current native frontier/cooperative fitting/scatter,
mesh64 collision import, GJK patch-on, dt0.01 and one substep. `mesh64` preserves
collision source precision. The collision asset has42 vertices.

Hardware is NVIDIA RTX PRO6000 Blackwell Server Edition,97887MiB, driver595.71.05,
Genesis1.4.3/Quadrants1.3.3 in container1_26. Private base `bc364325` carries the
byte-identical EPA correction selected in `90b2964e` plus this isolated candidate.
All raw source patches and before/after closures are archived.

The profiler pair runs sequentially on GPU4e7dc608. After600 varied warmup
steps,128 frozen actual-snapshot stage passes give:

| Stage | Uncached ms | Reuse ms |
|---|---:|---:|
| Intrinsic contact pipeline wall |40.385310|34.742103|
| Port-distance device kernel |6.574|0.885|
| Linear solve wall |1.571016|1.617451|
| Complete residual wall |0.661387|0.662491|
| Global peak wall |0.459271|0.460399|

The remaining device hotspot is fan-profile sampling (approximately20.1ms).
It remains actionable. Subsequent isolated decisions and measurements are in
[the contact sampling ablations](NATIVE_CONTACT_SAMPLING_ABLATIONS_20261012.md).

The live pair runs sequentially on GPU7ae01237, with seed623102 and independent
varied conditions. Each route includes600 warmup steps followed by three1280
step repeats. Every repeat includes2323 selected-environment resets, controller,
contact parameter updates, integration repair, solver correction/fallback and
native reset-index maintenance. Every environment is counted.

| Live route | Repeat env-step/s | Aggregate effective env-step/s | Batch step/s |
|---|---|---:|---:|
| Uncached |37114.531,37205.404,37225.738|37181.828148|18.155190|
| Current-step reuse |40953.356,41005.768,41040.325|40999.785184|20.019426|

Both routes have zero invalid environment steps and scatter overflows in all
repeats. The aggregate end-to-end speedup is1.102683414x, so reuse is selected.
Native buffers increase from6,108,796,518 to6,111,942,238 bytes. The3,145,720-byte
increment is the actual grown8-contact scratch at B2048 relative to the disabled
scalar allocation. Free device memory is89,579,323,392 bytes in both recorded
rows. Maximum warmup full-residual budget fractions are0.008609276 and0.008723409.

## Reproduction and evidence

The raw archive is
`evidence/20261011-nonconvex/native-port-seed-checkpoint-v1/manifest.json`.
Its454 entries,119 unique blobs and29,588,459 stored bytes preserve both route
sources, original stdout, XML/field properties, full JSONs and allocation logs.
An allocated CPU publisher verifies every original/stored/decompressed hash,
unchanged source closures, all48 oracle fields and all live failure/rate rows.
Manifest SHA256:
`9cffac110841dbd06554f6c2774b8c98d5363bf7d9afa3784cdd9720f7d57d3f`.

Recreate the private candidate and its ablation from the archived source patch,
`probe_port_seed_reuse_v1.py` and `run_port_seed_reuse_v1.sh`. Run modes `off` and
`on` sequentially for scopes `profile` and `live` on one allocated GPU. CPU
regression wrapper v1 closes stress/test/reference files. Its omission of
collider files is disclosed. CUDA wrapper v2 and both measurement pairs also
close all collider sources. The EPA correction is identical across the routes.

The goal remains active. This is one valid B2048 optimization checkpoint.
Current-source four-scope evidence, wider batches/seeds, timed grasp outcomes,
per-environment tail work and the remaining profiling-led candidates/plateau
remain pending. Low-shell numerical consistency is qualified here. Physical
convergence is a later study. Policy inference and RL training are separate
measurement scopes.
