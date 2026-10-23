# Automatic contact size in the original DPR

The implementation follows the [corrected goal](DPR_AUTOMATIC_RADIUS_GOAL_20261023.md):
current contacts feed the existing DPR transfer, prefactorized FEM and max/full
outputs. The native option is available on this branch. CPU/GPU controls,
the matched 130-frame scene window, partial reset and unfused execution pass.
On the measured B64 workload, complete Hertz-profile DPR costs 4.564 ms/step
with full output, versus 2.671 ms for manual DPR and 1.734 ms for pure rigid.
This is a finite-footprint approximation on the original coarse FEM; it does
not establish physical sharp-edge peak accuracy or a complete grasp cycle.

```python
link.configure_stress_recovery(
    gs.options.RigidStressOptions.dpr(
        mesh="elastic.npz",
        radius_model="hertz",
        pressure_profile="hertz",
        box_edge_policy="planar",  # Explicit sharp-box footprint surrogate; see limitations below.
        quadrature=16,
        output_mode="full",
    )
)
```

Use the existing `examples.rigid.franka_egg_stress` entrypoint with
`--radius-model hertz --pressure-profile hertz --box-edge-policy planar --quadrature 16`
for the tested sharp-box scene. `--pressure-profile dpr` keeps
the original pressure shape and changes only the size estimate. The default
`--radius-model manual --pressure-profile dpr` remains the manual baseline.
The controller sets radii only in manual mode. Configure stress before building
the scene, as with the original API.

A compact copy of the unchanged four-condition bank and a one-run task file
are included in [the evidence packet](../evidence/20261023-automatic-dpr-v1/).
From this branch's repository root, submit the admitted 130-frame Hertz-profile
example through the existing allocation/cache wrapper:

```bash
python3 -m research.rigid_stress.shell_comparison.submit \
  --source "$PWD" \
  --tasks research/rigid_stress/evidence/20261023-automatic-dpr-v1/example-tasks.json \
  --output /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/automatic-dpr-example \
  --python /mnt/data/zhaofeng/venvs/rigid-stress-physical/bin/python \
  --minutes 20 --memory-gb 12 --cpus 1 --node-local-cache
```

Choose a fresh output directory for each run. The task calls the existing
`examples.speed_benchmark.rigid_stress` entrypoint with automatic radius,
Hertz pressure, explicit planar box-edge approximation, quadrature 16 and full
output. The submitter records source identity and keeps logs/results on data.
To use the interactive Franka entrypoint inside a configured GPU allocation,
pass `--envs 4 --output-mode full` and the same `--conditions`,
`--radius-model hertz`, `--pressure-profile hertz`,
`--box-edge-policy planar`, `--quadrature 16` and `--steps 130` choices to
`python -m examples.rigid.franka_egg_stress`.

## Declared approximation

For each supplied collision force, use its compressive normal component Fn and
the mean relative principal curvature k at the actual collision witnesses:

    a = (3 Fn / (4 E* k))**(1/3)

Both relative principal curvatures must be positive. This is a circular
approximation for anisotropic curvature, not an elliptic Hertz solution.
The default E* is `young / (1 - poisson**2)`, assuming a rigid counterface.
`contact_modulus` supplies an explicit combined physical modulus in Pa without
changing the link's FEM material. A collision manifold's individual forces get
individual radii; the estimator does not reconstruct a physical contact region.

Primitive curvature is analytic away from edges. Meshes reuse fixed smooth-normal
curvature data; a witnessed vertex supplies a direct lookup, otherwise the query
remains within the actual contacted geom. Signed concave curvature is retained.
The estimate is recomputed from current inputs, so release, slot changes and
partial resets cannot reuse an old radius. There is no contact-history cache.
Strict Hertz mode reports sharp primitive edges and nonpositive relative
curvature as unsupported, separately from DPR force/fit failures. The explicit
`box_edge_policy="planar"` option ignores the singular box curvature only when
the current witness lies on an actual box edge or vertex. The other surface
must still provide positive relative principal curvatures. Accepted contacts
carry `BOX_EDGE_APPROXIMATION` (5); smooth estimates retain `SUCCESS` (1).
This is a declared finite-footprint surrogate, not the Hertz solution for a
sharp edge. Collision geometry, contact loads, DPR transfer and FEM are unchanged.
There is no automatic radius enlargement or pressure-solver fallback.

The Hertz profile is proportional to `sqrt(1 - r**2/a**2)` inside the finite
disk. DPR keeps its original force direction, including friction, and its
nonnegative force/moment fit. A centered planar disk therefore has circular
Hertz normal pressure and proportional shear. Curved/truncated supports can
require a nonconstant correction; this is not an exact Hertz solution there.
No friction-memory or analytic stress field is introduced.

Small supports use quadrature at their physical scale, including when they cross
elastic face edges or vertices. Positive triangle fans are clipped to the
unchanged support ball; complete planar disks use a polar rule. A support smaller
than the anchor face's diameter divided by its Gauss order takes this route
before fitting, since a successful moment fit alone does not prove quadrature
accuracy. This numerical integration change also applies to small manually
specified radii. Signed P2 loads,
original wrench gates, equilibrium residuals, factors and stress outputs remain
in the existing recovery path. Native mesh resolution still limits stress
accuracy; the local radius does not add missing FEM resolution.

## Validation and reproduction

`qipc_comparison.check_automatic_dpr_controls` checks analytic sizes, unload,
tension/edge/flat-curvature rejection, analytic disk/sector P2 loads at face
interiors, mesh edges and vertices, and a SciPy sparse solve of the same coarse
FEM. Translated/rotated primitive controls exercise coordinate roundoff. This is numerical
conformance, not a shell contact-reference qualification.
`qipc_comparison.check_automatic_dpr_scene` exercises the public Franka route
with separate manual, radius-only and Hertz-profile modes; failed frames and
all environments remain in its record.

The final production source is `40775c1d0`; later commits add audits and
documentation. Its CPU/GPU radius and disk/sector controls pass, retaining
the original force/moment and FEM residual checks. Earlier source `df5dc1d97`
also passes the six selected existing DPR regression cases on each backend.
The exact results, source identities and regression logs are in
[the evidence packet](../evidence/20261023-automatic-dpr-v1/README.md).
These checks establish numerical consistency, not physical shell accuracy.

The initial strict automatic mode reaches 108 accepted frames, then frame 108
records a pressure-fit failure and an unsupported curvature classification.
At quadrature 16 the pressure-fit failure disappears in the actual scene,
without changing the load, radius or tolerances. The independent exact-input
CPU replay also passes at orders 16 and 24, while order 10 retains its failure.
The matching pre-integration replay confirms that the unsupported witness is
an actual box edge: local coordinates are (3, -2, -0.0593145) mm on a
6 by 4 by 7 mm box. The strict estimator repeats the same refusal. An earlier
step-end query found a face because the gripper had already moved; that query
cannot classify the earlier contact. The coordinate-roundoff fix passes its
own controls but does not change this genuine sharp-edge refusal.
At sources `44861eef5` and `40775c1d0`, the explicit planar edge surrogate passes all 130 frames
for both pressure profiles in all four environments. The final frame has 10
active contacts, six carrying the edge-approximation status. The CPU/GPU controls
retain strict edge rejection and separately verify the optional edge/vertex
surrogate, unload, tension rejection and flat-relative-curvature rejection.
This 130-frame window covers settling, approach and initial gripping; it is not
a complete 600-frame grasp/release-cycle qualification.

The separate lifecycle runs at `44861eef5` pass both pressure profiles with a
partial reset at frame 110 and Hertz-profile execution with fusion disabled.
Resetting environment 0 clears its peak, while other environments' peaks and
poses remain exactly unchanged. The final optimization only prunes empty
pressure-integration work; it does not change this reset or association code.

The matched-scene audit confirms exactly identical rigid positions across all
130 frames and exactly identical contact forces, normals, positions and friction
at common saved frames 2, 110, 129. The two automatic profiles also have exactly
identical radii. Window peaks on the existing coarse mesh are 78.356, 100.217 and
99.900 kPa for manual, radius-only and Hertz-profile DPR respectively. Their
unweighted tensor-sample differences from manual DPR range from 25% to 40% at
those frames. These are load-model differences, not physical-error estimates.

The extra **bitwise comparison between independent before/after optimization
runs fails**, and its exact assertion remains in the audit. Frames 2 and 110
have identical radii and stress tensors. At frame 129, one radius differs by
6.776e-21 m (relative 1.138e-16). The largest tensor difference is 7.563e-9 Pa
for radius-only DPR (relative 5.841e-14), and 1.168e-9 Pa for Hertz-profile DPR
(relative 2.800e-15). The estimator source is unchanged by the optimization,
but the cause of this bitwise difference has not been isolated. This failed
audit is reported separately from the passing scene-input audit and runtime
numerical checks; no tolerance or assertion was relaxed.

`check_automatic_dpr_box` compares the original DPR/FEM with the retained
independent direct-pressure finite-box reference. It uses only the direct FEM
arrays, never adds an analytic stress field, and reports reference mesh change.
The old direct reference changes 8.03% between its medium/fine grids, so its
differences cannot be promoted to certified continuum errors. An independent
pressure quadrature is checked against analytic sector moments before it serves
as a nodal-load oracle. This control does not qualify hollow-egg contact.

For the prescribed 0.2 N normal load, E=10 GPa, nu=0.3 and relative principal
curvatures 50/80 per metre, the mean-curvature circular radius is 59.4392 micrometres.
Against the elliptical-pressure solution on the **same coarse mesh**, complete
volume tensor differences are 5.146% for the original DPR profile and 1.452%
for the circular Hertz profile. Against the fine direct-pressure reference,
both differ by about 86.2%; even the exact elliptical load on that coarse mesh
differs by 86.18%. This exposes the coarse FEM resolution limit. Native Hertz
nodal loads agree with analytic sector moments to 1.08e-8; the independent
high-order pressure quadrature differs by 2.66e-7. Passing those numerical
checks does not remove the large field discrepancy or reference uncertainty.

The executed CPU/GPU environment probes find neither `run_aster` nor `as_run`
on PATH, and no `code_aster` module in the selected Python. This is an availability
statement about Slurm image 1_26 and that environment, not a claim about every
installation on the cluster.

## Matched throughput

Final measurements use `40775c1d0`, the same level-1 mesh (480 tetrahedra,
2,430 DOFs), fp64, quadrature 16, the selected inverse FEM path and unchanged
four-condition bank. B64 tiles those four conditions. Each batch's scene
variants run on the same RTX PRO 6000 Blackwell Server Edition GPU. The
reported values are medians of three repeats; all 14 scene and six module
configurations have zero invalid environment-steps. Detailed research exports
are disabled, and full fields are computed and retained on the GPU.

| Complete scene, ms per batch step | B4 max | B4 full | B64 max | B64 full |
| --- | ---: | ---: | ---: | ---: |
| Pure rigid | 1.672 | — | 1.734 | — |
| Manual-radius DPR | 2.566 | 2.580 | 2.654 | 2.671 |
| Hertz-sized DPR, original pressure profile | 5.269 | 5.310 | 5.352 | 5.339 |
| Automatic radius, Hertz pressure profile | 4.464 | 4.474 | 4.540 | 4.564 |

Scene timing includes the original rigid solver, controller and stress update
over frames 0–129 after warmup/reset. Exact final rigid positions match across
all variants within each batch. The module measurement instead repeatedly
recovers stress at the matched pre-integration geometry/contact snapshot of
frame 129, which has more loaded contacts than the window average:

| B64 stress module at frame 129, ms | max | full |
| --- | ---: | ---: |
| Manual-radius DPR | 1.011 | 1.018 |
| Hertz-sized DPR, original pressure profile | 7.710 | 7.725 |
| Automatic radius, Hertz pressure profile | 2.534 | 2.547 |

The separate final stage profile repeats each stage 100 times. Snapshot replay
preserves every estimated radius exactly and passes the live FEM load/stress
checks. These are intrusive wall times including launches, not scene latency:

| Stage (ms per batch) | Manual DPR | Hertz-sized DPR | Hertz-profile DPR |
| --- | ---: | ---: | ---: |
| Association including estimator | 0.177 | 0.351 | 0.353 |
| Estimator alone (already inside association) | — | 0.197 | 0.196 |
| Pressure weights and moment fit | 0.530 | 6.820 | 1.697 |
| Original linear FEM solve | 0.210 | 0.206 | 0.195 |

Stage times are not additive. Radius-only DPR needs two or five fit iterations
on 32 of 160 contacts; all Hertz-profile contacts need zero fit iterations in
this snapshot. Resolving the smaller physical supports costs more than the
radius estimate. The final optimization skips exact face misses and two
zero-weight virtual fans for a fully contained disk. Compared with the
preceding implementation, radius-only full module time falls from 10.268 to
7.725 ms; Hertz full module time is roughly unchanged (2.496 to 2.547 ms).
These separate runs do not establish a small speed difference beyond run/GPU
variation. Both complete timing sets, raw hashes, GPU identities and repeat
values are retained in [timing.json](../evidence/20261023-automatic-dpr-v1/timing.json).

An earlier ground-only frame-107 profile is retained on data; it cannot stand
in for the loaded gripper snapshot. Final buffer accounting includes the
shared immutable curvature arrays.

The existing `examples.speed_benchmark.rigid_stress` accepts both new options.
Use matched `rigid`, `live` and `recovery` scopes and max/full modes; keep
`--trajectory-metrics` disabled for production timing. Detailed audits and
profiling run separately. Its association stage includes the estimator when
enabled; obsolete Gaussian-only diagnostic subdivisions are disabled then.

Submit scientific work using the existing `shell_comparison.submit` workflow,
Slurm `rtx-mid`, image `1_26`, the data-side environment and explicit new output
paths under the established project data directory. Frozen sources, task files,
compiler failures and results are retained in `automatic-dpr-*` campaigns.

Use the explicit planar edge policy and quadrature 16 for this measured
sharp-box workload. The strict default remains `--box-edge-policy unsupported`.
Full-cycle admission, converged hollow-egg stress accuracy and physical
sharp-edge surface maxima remain outside the demonstrated scope.
