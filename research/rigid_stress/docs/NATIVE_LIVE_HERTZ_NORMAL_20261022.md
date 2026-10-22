# Actual native normal contact recovery

The experimental Hertz route now consumes actual rigid collider contacts inside
the existing rigid solver. Contact association, provenance, curvature/region
construction, predicted pressure/support, elastic recovery, moving material
probes and reductions all run in Quadrants. The scene uses the original native
producer throughout; research callbacks only observe completed substeps.

Use experimental source
[`fc63cf32e`](https://github.com/Roushelfy/Genesis/commit/fc63cf32ea74444168fd328e0bf96ae9b2f0b5f7).
The task branch retains its established engine baselines; select this source
explicitly for the new options. It extends the
[moving-probe component](NATIVE_MOVING_HERTZ_PROBES_20261022.md).

```python
link.configure_stress_recovery(
    gs.options.RigidStressOptions.hertz(
        mesh=reference_mesh,
        young=young,
        poisson=poisson,
        hertz_reduced_modulus=reduced_modulus,
        output_mode="full",
    )
)
```

Configure before scene build. The helper defaults to order8 surface integration.
`hertz_reduced_modulus` supplies the physical combined modulus; omitting it
uses the observed material's `E/(1-nu**2)`, an infinitely stiff opponent model.
The route currently requires a locally planar material source, supported
half-space geometry, four-major-radius depth clearance and a centered normal
wrench. Actual shear, free couples, incompatible source pairs and unresolved
material geometry produce explicit native errors. Curved/inner material
response and physical friction history remain required work.

## Public observations and physical substeps

`get_max_stress()` and `get_max_principal_stress()` aggregate the sampled
von-Mises and signed principal maxima across the physical substeps in the
observed mechanical step. Full mode also exposes `get_stress_field()` and
`get_stress_contact_probes()`: positions, packed tensors, material elements
and an active mask. Full fields describe the latest observed substep. The
89 probes per active region follow the contact's current center, axes and
orientation through actual material-face adjacency.

Manual updates use the existing request API. Selected mutations invalidate
only affected environments, clear source identity and produce NaN derived
fields. Snapshot restore invalidates derived observations; immutable geometry
and elastic operators survive. Borrowed tensor views reflect invalidation,
while copied observations retain their data. Radius overrides are refused for
the predicted-support route.

An actual two-substep audit exposed an inherited clock bug. The checkpoint
ring index can remain zero on every forward-only substep; using it as the
mechanical-step clock repeatedly cleared the peak accumulator. Recovery now
uses the physical global substep modulo the configured substeps. This repairs
both Hertz and established DPR, including the CUDA fused path. Frozen old-source
controls reproduce relative peak misses of 1.88e-5/5.64e-5 for Hertz/DPR on
both CPU/GPU; repaired controls match the actual substep maximum exactly.

The final lifecycle control passes on both backends: six two-substep steps,
separate half-step references, max/full peaks, pure-rigid trajectories, complete
fields, partial reset, selected getters, checkpoint restore, copied/borrowed
views, manual request retention/cancellation and unbatched release/reload.
The original peak and trajectory targets remain 1e-7 relative and 1e-12 m.

## Independent fields and preserved failures

Eight rotated steps on four environments pass for CPU and GPU. Independent
same-load P2 elasticity includes the consistent-mass centrifugal contribution,
then adds the declared Hertz particular field at the actual saved probes and
element corners. Across 64 environment frames, maximum probe/corner relative
differences are 2.67e-9/3.00e-16. Analytical circular support and the Hertz
pressure integral also pass. This establishes numerical recovery and the
analytical control; it does not qualify unknown finite-body pressure or a
continuum global peak.

The first rotated CPU run failed when a probe crossed a tetrahedral vertex.
The old line walk mixed a parameter threshold with barycentric roundoff. The
repair tests current barycentric coordinates and permits a clamped
zero-distance crossing, retaining the original 1e-10 barycentric tolerance,
64-face bound and vertex-identity adjacency. Replaying the exact failed input
reproduces the frozen old-source refusal and locates all 89 probes on repaired
CPU/GPU, with reconstruction error below 2e-16 m. Disconnected material is
never reached by a global nearest search.

A GPU release check initially expected exactly zero stress despite residual
angular velocity. Its 1.06e-30 Pa value agrees with independent centrifugal
elasticity to 2.80e-14 relative. The final test checks this moving release and
then retains the exact-zero assertion after explicitly zeroing angular
velocity. The earlier failure stays FAILED. Constructor-friction, order6 wrench,
probe-location, original clock and replay-validation failures are also retained.

Four established public full-field tests pass across CPU/CUDA. Seven transfer
cases pass on each backend after moving the shared projected surface walk into
the engine; its function body is AST-identical to the captured research helper,
apart from its docstring. These regression sources are independently hash-bound.

## Matched complete scene cost

One allocation measures the same centered sphere/flat-link scene in pure-rigid,
max and full modes, FP64, order8, 48 tetrahedra and 125 P2 nodes. Every scene
step has two physical substeps. Input generation, the public moving-position
setter, dynamics, contact prediction/acceptance, recovery, probes and reductions
are included. Research observers, field readback and visualizer updates are
outside primary timing. Each median uses three eight-step blocks after warmup.

| Batch | Pure rigid (ms) | Max (ms) | Full (ms) | Added max/full (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2.042 | 7.207 | 7.238 | 5.165 / 5.196 |
| 64 | 2.206 | 7.218 | 7.366 | 5.012 / 5.160 |
| 256 | 2.167 | 8.091 | 8.147 | 5.924 / 5.980 |
| 1024 | 2.298 | 9.023 | 9.157 | 6.725 / 6.859 |

B1024 gives approximately 113,500/111,800 environment scene-steps/s for
max/full. Separate 24-step audits admit every environment, including all
24,576 loaded observations per mode at B1024. Max/full peaks meet 1e-7 and
all trajectories match pure rigid within 1e-12 m. Native error latches are
checked after timing, retaining required acceptance throughout each window.
These are actual normal-box scene rates; independently evolving Franka and
friction scenes require their own accepted measurements.

The measured source is
[`b5c28c271`](https://github.com/Roushelfy/Genesis/commit/b5c28c271c888d082ebabd7a58294ba9ff48174e).
The final source changes only the helper's default quadrature from6 to8;
the measured fixtures already explicitly use8. New default/rotated controls
exercise the final source. Order6 remains qualified for the preceding supplied
component workload, but its actual new scene fails the unchanged wrench gate.
Order8 is the measured setting for this native control.

An independent intrusive CPU/CUDA trace observes the original producer and
records Python call costs, GPU tasks/copies and exposed-field memory. Its
nested host durations and potentially overlapping device durations are kept
separate from primary scene wall time. Detailed phase results are retained in
the terminal packet.

| Selected device task/rule | B1 (ms/scene step) | B1024 |
| --- | ---: | ---: |
| Witnessed source resolution, dominant task | 0.836 | 0.918 |
| Hertz residual boundary traction, dominant task | 0.656 | 1.228 |
| Combined corner/probe tensors, dominant task | 0.360 | 1.277 |
| Packed regular elastic application, dominant task | 0.099 | 0.309 |
| Moving-probe kernels, all recorded tasks | 0.049 | 0.163 |
| Normal-law kernels, all recorded tasks | 0.095 | 0.103 |
| Region kernels, all recorded tasks | 0.057 | 0.058 |

These averages divide four traced scene steps; dominant-task rows omit other
tasks in that kernel. The rows are diagnostic and must not be summed into a
scene time. Source resolution is substantial even at B1, whereas the normal
law is a smaller addition. Boundary and combined-field work grow with batch.
The trace retains about 430/452 device launches per scene step and the small
argument/input copies. The exposed unique stress fields occupy 3.32/47.32 MB
at B1/B1024, including shared geometry/operators and probe storage; allocator
and compiler overhead are outside those logical counts. The next optimization
should target the measured source/field work and dispatch, preserving source
identity and full acceptance, rather than attributing the scene addition to
curvature lookup alone.

## Reproduction and remaining goal

The [terminal packet](../evidence/20261019-qualified-ranking/hard-native-hertz-live-normal-v1/README.md)
retains captured workers/task arguments, primary and override source inventories,
patches, actual scheduler states, failures and full-array/trace hashes. Source
replay checks every inventoried file. Run scientific work using the established
Slurm `rtx-mid`, image `1_26`, data-side physical environment, explicit fresh
data outputs and private node scratch. The task lists cover
`check_live_hertz_normal`, `check_live_hertz_lifecycle`,
`check_native_stress_clock`, `check_hertz_probe_walk_roundoff`,
`inspect_live_hertz_normal`, `time_live_hertz_normal` and
`trace_live_hertz_normal`.

The complete [active goal](AUTOMATIC_CONTACT_AND_THROUGHPUT_GOAL_20261020.md)
remains open: integrate and physically verify changing normal/tangential loads
and history, establish useful curved/inner/hollow-egg regimes, optimize measured
input/recovery costs and obtain accepted paired max/full moving Franka scene
throughput. Existing offline shell errors and supplied-component timings retain
their original scope.
