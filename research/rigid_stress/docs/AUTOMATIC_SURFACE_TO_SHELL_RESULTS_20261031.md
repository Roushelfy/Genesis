# Automatic surface-to-shell preparation: measurements and limits

**Delivered, 2026-10-31.** Implements the [October 31 goal](AUTOMATIC_SURFACE_TO_SHELL_GOAL_20261031.md).
Automatic conversion, focused numerical/mechanical checks, real public entrypoints and measured four-substep defaults are complete.
Use the [running guide](RUNNING_AUTOMATIC_DPR.md). All runtime artifacts remain under
`/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261031-automatic-surface-shell/` (`R` below).
Only code, documentation and a compact measurement manifest belong in Git.

## Geometry and interface

`--backend shell --mesh model.obj --thickness 0.0005` interprets the input as a midsurface and creates a cached
Q4 NPZ. Thickness is metres, independent of `--scale`. Explicit generated/authored NPZs remain accepted. Volume
remains the default; the existing authored egg still uses its established collision/inertia configuration.

Each triangle produces three conforming quads using edge midpoints and a face centroid: 3F quads, V+E+F nodes and
6(V+E+F) elastic DOF. Exact coordinate duplicates are merged; nearby sheets are not tolerance-welded. The importer
applies scene transforms explicitly, including translations of 1e-10 m. External glTF geometry buffers, source,
scale, thickness, preparation recipe and library versions participate in cache identity. There is no remeshing
inside stepping or per environment. CPU work prepares geometry; production elasticity, loading and recovery stay native Quadrants.

Closed and open connected orientable surfaces, concavities and holes are supported by conversion. Duplicate,
degenerate, nonmanifold, intersecting or disconnected inputs are rejected with diagnostics; the single-body gauge
is not applied to multiple free components. Directors follow the native area-weighted convention. Midsurface and
finite-thickness boundary checks include orientation, sampled Q4 Jacobians through thickness, intersections and
positive material volume/inertia. Boundary checks use [PyMeshLab's intersection selection](https://pymeshlab.readthedocs.io/en/latest/filter_list.html#compute-selection-by-self-intersections-per-face).

Collision uses both offset sides and open-edge caps, with convexification, decimation and watertight wrapping disabled.
Mass, COM and inertia use the finite-thickness material volume. The two pressure/display sides remain registered after
joint scaling and centering; caps render gray because no separate three-dimensional cap stress is sampled. The original
sharp-box gripper and explicit planar edge-curvature approximation remain unchanged.

## Numerical and mechanical checks

Fourteen focused tests pass at `28a253b6d`: existing volume behavior, shell units and thickness, analytical plate
mass/COM/inertia, collision registration, winding, cached/reloaded assets, external-buffer invalidation, close gaps,
scene transforms, invalid inputs, both-side coloring, gray caps and an annulus that retains its hole.
Receipts: `R/geometry-tests-v8.xml`, job 768285_0.

Independent CPU and actual RTX PRO 6000 Blackwell controls cover the following generated families:

| Surface | Input triangles | Q4s | Nodes | Elastic DOF |
|---|---:|---:|---:|---:|
| Small egg | 20 | 60 | 62 | 372 |
| Narrow neck | 80 | 240 | 242 | 1452 |
| Open U | 8 | 24 | 35 | 210 |
| Flat strip | 4 | 12 | 19 | 114 |

CPU/native operator and complete tensor/resultant/peak checks include spinning rigid-mode relief, eccentric normal
and friction loading on both sides, slot permutation, release/recontact, an unloaded environment, and original
force/moment/residual checks. GPU same-load full-tensor relative errors are 9.2e-12–5.1e-10. Reloading the generated
240-quad grasp asset or 12-quad plate changes the full tensor by at most 1.4e-12; actual generated/director-aware
native recovery differs from the independent CPU solve by at most 1.5e-10. Repeating the grasp/plate comparison on the exact 17-digit public
fixture gives at most 1.2e-12 auto/reload and 6.1e-11 independent-CPU full-tensor error (`reload-{cpu,gpu}-v2`). See `controls-cpu-v1`, `controls-gpu-v2`,
`reload-cpu-v1` and `reload-gpu-v1`.

Separate nu=0 membrane/bending controls have energy errors below 3.2e-15 / 8.6e-11 and exactly six coarse rigid null
modes. A unit-load clamped strip with 12/48/192 quads has compliance errors 0.8307% / 0.1978% / 0.04786% against the
stated beam plus transverse-shear reference. This bounded mechanical check is distinct from same-mesh solve agreement;
it does not establish curved-shell, three-dimensional contact or sharp-edge peak convergence.

An independent corner-quality audit (`quality-v1`, job 768340_0) measures the generated grasp/notch quads at
42.08–141.79 degrees, minimum corner sine 0.619 and maximum element edge ratio 2.89. The more demanding narrow
neck spans 14.85–166.41 degrees, sine 0.235 and edge ratio 8.39. Subdivision preserves the original surface but does
not improve poor input triangles; valid sampled Jacobians alone are not a mechanical accuracy certificate. These
results and the bounded strip refinement support a simple, deterministic conversion, with low-resolution,
well-shaped source triangles recommended. A separate nonconvex notch control passes CPU/GPU operators, prescribed
both-side loads and reload comparisons: <=1.6e-12 auto/reload tensor error and <=5.5e-11 independent-CPU tensor error
(`notch-native-v1`, job 768338_0). Prescribed-load agreement does not certify automatic contact admission.

## Actual entrypoints and admitted scope

The actual public CLI completes 140 GPU steps on the small egg, an open curved dome and the nonconvex narrow neck.
The final dome and neck use the four-substep default (`entrypoints-final-v1`, `nonconvex-final-v1/group0/neck`). All have nonzero stress;
the small egg is narrower than the fixed finger closure and is a ground-support control. Real 800×600 viewer captures
of automatic and authored assets show occluding facet stress colors (`viewer-v2`, jobs 768287_0/1). Public authored
shell viewer physics and full fields complete 730 steps (`entrypoints-v4`, exit 0). An initial post-return screenshot wrapper failed because it
read a different thread's zero-sized framebuffer; the separate real-viewer capture now succeeds.

A valid conversion does not guarantee Hertz/pressure admission for every resulting rigid contact. At one physical
substep, the imported authored exterior reinterpreted as a midsurface reaches inner-wall contacts at step 121;
the observed negative curvature is rejected. The narrow neck at one substep, broad coarse ellipsoid, flat plate,
U, dimple and notch also retain contact negatives. The final notch still rejects automatic curvature at four substeps,
despite its successful prescribed-load numerical controls; the neck passes at four. A nonnegative pressure-fit
failure is unresolved numerical admission, not proof of physical infeasibility.
No radius, load, friction, residual or force/moment gate is relaxed to admit these cases. Immediate post-substep checks locate the narrow-neck curvature failure at step 3, broad ellipsoid pressure-fit
failure at step 109, and plate pressure-fit failure at step 3; the ten-substep ordinary error polling can otherwise
report later buffers. These are retained negatives. Four physical substeps admit the 730-step converted grasp,
including checks after every physical substep and full-field/partial-reset checks (`first-failure-v2/group0`).
The actual public converted-grasp command also completes 730 steps with its default settings (`entrypoints-final-v1/group0/grasp`).
The public custom-shell default therefore uses four substeps, with explicit `--substeps N`; authored egg and volume
retain one. The 0.01 s outer step is unchanged. Max output spans substeps; the full field is the last substep.

## GPU cost

The current custom-shell default (`01d12d006`) has twelve accepted matched four-substep grasp cells. The automatic
and explicit models use the 80-triangle established exterior as a declared midsurface: 240 quads / 242 nodes /
1452 DOF, with a 0.5 mm material boundary. Its conversion takes 39–43 ms cold / 5.2–5.8 ms warm;
cold scene construction takes 177–207 s including JIT but excluding process imports, `gs.init` and prior conversion. The authored model also has 240 quads / 1452 DOF but different midsurface,
exterior collision, inertia and contact inputs. All three variants in this table use four substeps per 0.01 s outer step.

| B / output | Automatic module / scene ms | Reloaded module / scene ms | Authored module / scene ms |
|---|---:|---:|---:|
| 4 / max | 1.628 / 23.342 | 1.630 / 23.457 | 1.819 / 14.895 |
| 4 / full | 1.635 / 23.185 | 1.631 / 23.018 | 1.819 / 14.784 |
| 256 / max | 2.364 / 26.910 | 2.363 / 26.649 | 2.480 / 17.928 |
| 256 / full | 2.532 / 28.451 | 2.516 / 28.190 | 2.616 / 18.956 |

Module times are for one frozen recovery; scene times include all four physical substeps, rigid dynamics and the
native scripted controller. Complete automatic B256 max/full rates are **9,513 / 8,998 environment-steps/s**; B4
rates are 171 / 173. Auto/reload mechanical inputs match exactly and full tensors agree within 3.2e-13; max/full
peaks match exactly. All twelve cells preserve original acceptance and partial reset. These are three moving
30-step windows after 130-step warmup, not independent repeated trials or universal rates; the final window is faster
as the contact phase changes. See the [compact path/hash/key-number manifest](AUTOMATIC_SURFACE_TO_SHELL_MEASUREMENTS_20261031.json).

For this grasp, native logical payloads are 79,764,509 shared bytes plus 663,420 / 886,140 bytes per environment
(max/full). At B256 this is 249.60 / 306.62 MB. A separate fresh full-mode B256 process samples 4,934 MiB peak GPU
memory; compiler caches were warm, the Python/CUDA process was fresh, and readback/sampling costs are excluded from
primary timing. Actual converted-grasp viewer output is inspected at `grasp-view-memory-v1/group0/grasp-viewer/viewer.png`.
Its separate 800×600 color refresh/readback/PNG export measurement is 1.90 s, including first-use work; this is not a steady
rendering frame rate. Viewer, screenshot and diagnostic readbacks are excluded from primary throughput.

The following retained one-substep ground-support control documents the initial profile and tuning. It is not the
current custom-shell default and is not a successful grasp.

The ground-control native source is `512c1c8ec` (production preparation matches `6bbd768f5`); helper snapshots and hashes live in
each run's `analysis-files.json` and `source.json`. Slurm uses `rtx-mid`, image `1_26`, CUDA FP64, GraphOff, one GPU,
and private node-local compiler caches. The primary workload uses the native scripted Franka controller, planar
k32/q10/cap128, 130 warmup steps, then three 30-step timing windows. These are scene environment-steps per second,
not policy training. Module timing replays a verified frozen contact/geometry context. Readback, acceptance exports,
profiling and viewer work are outside primary timed windows. Original runtime checks stay enabled.

The automatic and explicitly reloaded assets are the same 60-quad / 372-DOF small egg, with the same finite-thickness
collision and material inertia. The authored egg has 240 quads / 1452 DOF, a different shape, original convex exterior
collision, inertia and grasp contacts. Its faster timing is not a conversion-overhead comparison.

| B / output | Automatic module / scene ms | Reloaded module / scene ms | Authored module / scene ms |
|---|---:|---:|---:|
| 4 / max | 99.928 / 101.738 | 99.887 / 101.687 | 1.719 / 3.716 |
| 4 / full | 100.965 / 102.774 | 100.958 / 101.863 | 1.758 / 3.775 |
| 256 / max | 82.840 / 84.649 | 82.756 / 84.615 | 2.278 / 4.516 |
| 256 / full | 85.709 / 87.742 | 85.659 / 87.618 | 2.437 / 4.802 |

Table values are medians of three within-process windows; they do not estimate cross-device confidence intervals.
Automatic B256 max/full complete-scene rates are 3,024 / 2,918 environment-steps/s. B4 is only 39.3 / 38.9. The
12 one-substep control cells pass original acceptance and partial reset. The generated geometry is substantially more expensive
per step in this particular contact workload despite its smaller elastic mesh. Auto/reload contact validity, positions,
forces and radii match exactly; complete tensors agree within 1.3e-12, and max/full peaks within 1.4e-16.

Conversion in the primary automatic runs takes about 30–32 ms cold, with warm mesh-cache reads around 5 ms; small
geometry controls span about 22–35 ms. Cold full scene builds take 167–191 s including JIT. Reloaded processes reuse
compiler caches and build in about 22–23 s; an identical-source/shared-asset pair measures 166.64 s cold / 22.41 s warm full scene builds, with
30.65 ms miss / 3.95 ms hit for its first preparation call (`warm-build-v1`).
Do not equate preparation time with scene startup time.

Automatic native tensor payloads are 16.76 / 16.98 MB at B4 max/full and 155.64 / 169.89 MB at B256. Separate fresh
process sampling reports GPU peaks of 4,750 / 4,752 MiB at B256 (`memory-view-v1`); these include runtime and rigid
solver allocations, and sampling can miss brief peaks. The two batch inventories decompose into 14,555,397 shared bytes plus 551,100 / 606,780 bytes per environment
for max/full; these are native tensor payloads and exclude rigid solver, allocator and runtime memory.

The first accepted B4 profile precedes tuning: pressure fit and its validation account for about 89% / 9% of GPU trace
time. Bounded k8/k16 changes the complete tensor by 2.39e-5 / 7.79e-6 relative to k32 under exactly identical mechanical
inputs, while saving only about 2% of module/scene time in this control. Keep k32: the small speed difference does not
justify changing this loading approximation. There is no universal optimal triangle count or claim that automatic
conversion accelerates an existing authored shell.

## Reproduction and recommendation

Use the existing environment on a compute node. Set data/cache/output paths explicitly as required by
[DATA_LOCATION.md](../../../DATA_LOCATION.md). For example, prepare a small fixture with
`python -m research.rigid_stress.shell_comparison.automatic_shell_geometry --shape dome --output "$RUN/fixture"`,
then run `QD_GRAPH=0 python -m examples.rigid.franka_egg_automatic_stress --backend shell --mesh "$RUN/fixture/dome.obj" --thickness 0.0005 --cache-dir "$RUN/cache" --steps 140`;
add `--viewer` with a usable graphics context. Use `--shape neck` / `neck.obj` for the admitted nonconvex 140-step
example, or `--shape grasp` / `grasp.obj` with `--steps 730` for the shape-compatible full grasp.
The research `submit` helper captures source/helpers and launches single- or multiple-task arrays. On this cluster,
use an array even for one sbatch task: the gs-srun wrapper otherwise enters a different nested-launch path.

Start with a connected, consistently oriented low-resolution midsurface with reasonably shaped triangles. There is
no automatic reduction of a high-poly model: the exact 3F refinement bounds added elements but dense inverse/factor
cost still grows rapidly. Choose thickness from the physical object, check that offsets remain valid, and keep a
prepared authored mesh where its better discretization/contact behavior matters. Use max for headless batches and
full only when tensors/resultants or colors are needed. Existing sharp-edge and continuum limitations remain.

All campaign jobs are terminal, including retained failed cases. Eight unrelated QIPC files remain unchanged;
all nine frozen worktrees created by this campaign were removed after checking jobs and local processes.
Source commits remain in branch history and per-run helper hashes remain on data. Preservation and cleanup
receipts are linked in the compact manifest; prior campaigns and their worktrees were not modified.
