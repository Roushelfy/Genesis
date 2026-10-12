# Native sample-major fan launch

The selected CUDA frontier sampler maps neighboring lanes to Q10 samples of
one contact/environment. Its flattened order changes from
`(sample, contact, environment)` to `(contact, environment, sample)`.
Every task retains its original sample identity, face, anchor, path, area,
profile and local friction cone. Packing remains an atomic sample append with
the same fitting/scatter acceptance. Memory layout and allocation sizes stay
fixed. The CPU route uses its existing dense sampler.

This scheduling change addresses the dominant fan-profile kernel after
[current-step seed reuse](NATIVE_PORT_SEED_REUSE_20261012.md). Neighboring sample
lanes share a contact and its geometric branch decisions. It introduces no
new contact-location, contact-count, friction or symmetry assumption.

## Qualification

CUDA23 source/prior/lifecycle, U/cavity gap/hole and intrinsic footprint cases
pass in413.666s. The unchanged CPU route retains its qualified15-case seed-reuse
matrix (230.632s). The48 recorded independent CPU FP64 source-prior field
comparisons span both backends and three actual friction variants per fixture.
Worst normalized wrench error is7.386598105e-10, full equilibrium/gauge residual
budget fraction0.045856225, and peak difference8.203700418e-6Pa. The CUDA
footprint cases additionally compare full fields against P2Shell references.
Batching, changing sources/capacity, partial reset and isolation checks pass.

Same-snapshot repetitions differ by at most5.551115123e-16N in nodal force
and1.076114131e-8Pa in peak. Final normalized force/moment errors remain below
1e-8. All final budgets are unchanged.

## Paired GPU evidence

Both routes use the qualified EPA correction, selected port-seed reuse,
low full hollow shell (480P2 tetrahedra/810P2 nodes/2430 DOFs), FP64/Q10/s8,
`body_source_prior`, maximum output, native frontier/cooperative fit/scatter,
mesh64/GJK patch-on, dt0.01 and one substep. The raw source epoch is private
basebc364325 plus byte-identical EPA/seed-reuse changes and this isolated
scheduling candidate.

Hardware is NVIDIA RTX PRO6000 Blackwell Server Edition,97887MiB,
driver595.71.05, container1_26, Genesis1.4.3 and Quadrants1.3.3.
The profile pair runs sequentially on GPUa0c812f2 after600 varied warmup steps
and profiles128 actual frozen-snapshot passes.

| Profile stage | Environment-major ms | Sample-major ms |
|---|---:|---:|
| Intrinsic contact pipeline wall |33.604776|18.680484|
| Fan-profile sampling device |20.090|5.580|
| Linear solve wall |1.562340|1.561493|
| Full residual wall |0.658797|0.659080|
| Global peak wall |0.457916|0.458243|

The live pair runs sequentially on GPUe5860f82 with independently varied
seed623102 conditions. Each route warms600 steps and then runs three1280-step
repetitions. Every repetition includes2323 selected-environment resets,
controller, contact-radius/material updates, history/repair/fallback and
observation/access work required by this live scope. All environments count.

| Live route | Repeat effective env-step/s | Aggregate effective env-step/s | Batch step/s |
|---|---|---:|---:|
| Environment-major |40803.143,41016.251,41049.219|40955.913677|19.998005|
| Sample-major |55598.967,55819.145,55872.765|55763.373569|27.228210|

All six repeats have zero invalid environment steps and scatter overflows.
The measured end-to-end speedup is1.361546320x, so sample-major scheduling is
selected. Native memory is6,111,942,238 bytes in both routes. The largest warmup
full-residual budget fractions are0.008033087 and0.008612469. Full raw per-env
failure counters, buffers, device free memory and hardware records are retained.

## Reproduction and remaining work

Recreate `probe_fan_sample_layout_v1.py`, the original Quadrants sampler in
`baseline_fan_samples_v1.py`, both pair launchers and the CUDA regression wrapper
from the archive. Run `off`/`on` sequentially on one GPU for `profile`/`live`.
Both pairs and the new CUDA tests close all collider/stress numerical sources.
The reused CPU seed-reuse closure omits collider sources, as previously disclosed.

Archive:
`evidence/20261011-nonconvex/native-fan-sample-layout-checkpoint-v1/manifest.json`.
Its461 entries/124 unique blobs/29,619,926 stored bytes preserve all raw sources,
patches, logs, XML properties and complete JSONs. An allocated CPU publisher
verifies original/stored/decompressed hashes, source closures, all48 recorded
oracle fields, paired inputs and every live rate/failure row.
Manifest SHA256:
`d3de49c64ece74502db16986e8066480fe711788f1fe4dd45b8695db3b2aa244`.

The goal remains active. Nonnegative path-bound pruning has an isolated gain
and its increment on top of this schedule is being evaluated. Current-source
four-scope/multiple-seed/wider-batch evidence, timed grasp outcomes and tail
costs, dynamic profiling, further applicable candidates and a measured plateau
remain pending. Physical convergence and RL training throughput are separate
studies.
