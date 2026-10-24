# Running native thin-shell DPR

The optional shell backend observes rigid motion through the existing stress
recovery API. It does not deform the collision geometry or feed elastic
displacement back into rigid dynamics. Volume FEM remains the default.

## Configure before building

```python
link.configure_stress_recovery(
    gs.options.RigidStressOptions.dpr(
        mesh="shell.npz",
        backend="shell",
        shell_thickness=0.0005,
        young=1e10,
        poisson=0.3,
        density=2000.0,
        radius_model="hertz",
        pressure_profile="hertz",
        quadrature=16,
        output_mode="full",
    )
)
```

Use an NPZ containing midsurface `vertices[n,3]` in metres and integer
`quads[m,4]` with consistent orientation. Optional `directors[n,3]` specify
the thickness directions; otherwise native preparation computes adjacent-face
averages. All coordinates use the authored link frame. Thickness and material
are fixed before building. Unsupported or inverted geometry raises an error.
The included coarse and medium hollow-egg examples have matching shell assets
and explicit geometry metadata under `examples/rigid/assets/hollow_egg/`.

Only the mature finite-pressure patch loading is supported by this backend.
The physical surfaces are the bilinear outer and inner offsets of the
midsurface. Tangential forces retain their eccentric rotational loads.
The native support-ball pressure law is retained; the CPU study's projected
Hertz law is a separate model.

The sharp gripper example explicitly selects `box_edge_policy="planar"`.
That keeps the sharp collision geometry and uses the documented planar
curvature surrogate for radius estimation at box edges. It is not a rounded
edge model or a proof of local edge-pressure accuracy. The public default
edge policy continues to reject unsupported edge geometry.

## Observe stress

```python
sampled_max_pa = link.get_max_stress()
stress_pa, von_mises_pa = link.get_stress_field()
membrane_N_per_m, bending_N, shear_N_per_m = link.get_shell_resultants()
```

With `output_mode="full"`, stress has shape `[B,n_quads,2,4,6]`: outer/inner
sides, four Gauss points, then `xx,yy,zz,xy,xz,yz` in the authored link frame.
Von Mises omits the component axis. The point order is counterclockwise from
`(-1/sqrt(3),-1/sqrt(3))`. Membrane and bending resultants have shape
`[B,n_quads,4,6]`; transverse shear has shape `[B,n_quads,4,3]`. Unbatched scenes
omit `B`. `output_mode="max"` avoids complete-field/resultant storage.

The maximum covers sampled **in-plane membrane-plus-bending stress** over the
observed scene step's physical substeps. Full fields describe the last
observed physical substep. They do not include three-dimensional near-contact
normal stress. Reset, checkpoint restore and invalid recovery invalidate
derived fields with NaN. Partial resets preserve other environments.
`update_mode="manual"` uses the existing `link.request_stress_update()` API.

## Run the included grasp

On a compute node with the project environment and explicit data/cache paths:

```bash
python -m examples.rigid.franka_egg_stress \
  --envs 64 --steps 600 --level 1 --stress-backend shell \
  --radius-model hertz --pressure-profile hertz \
  --box-edge-policy planar --quadrature 16 --output-mode full
```

The existing benchmark exposes the same backend choice. Use `--scope live`
for the complete robot workflow, `--scope recovery` for a frozen actual
loaded contact snapshot, and `--scope profile` for intrusive stage timings.
For complete-cycle comparisons use `--warmup 600 --steps 600 --repetitions 3
--minimum-seconds 0`. Recovery snapshots use `--warmup 130`. Keep the same
saved condition bank, precision, geometry, controller and cadence across
comparisons; `--scope rigid` supplies the pure-rigid baseline. Keep all failed
environment steps and report both attempted and admitted rates.

In this workspace, submit through
`research.rigid_stress.shell_comparison.submit`, with explicit `--source`,
`--output`, `--tasks` and `--python`, plus `--node-local-cache`. It captures
engine and analysis sources, runs scientific work on Slurm `rtx-mid` with
image `1_26`, and keeps durable outputs/logs on data. See `DATA_LOCATION.md`.
Do not run scientific imports or experiments on the login node.

## Numerical and performance choices

Start with FP64 and the default automatic factor/response selection,
translation-only gauge, minimum-degree ordering, unscaled factor and fused
pipeline. Automatic-radius Hertz pressure enables verified current-step
moment reuse. It does not reuse stale contact history. Overflow retains the
complete original integration fallback.

Use full output when fields/resultants are required; max mode saves memory.
Compare `--full-inverse`, `--dense-surface-loads`, `--inverse-precision 32`,
`--method direct`, `--factor-scaling`, `--factor-ordering natural`, and
`--unfused-pipeline` only with complete residual/field checks and measured
costs. FP32 inverse storage retains FP64 arithmetic and residual correction;
it is distinct from full FP32 arithmetic, which failed the strict native
plate acceptance control. Factor scaling never changes the residual operator.

The coarse hard-contact egg has an approximately 45-micrometre footprint in
a 0.5-mm wall. Its local contact peak is unresolved. Global membrane/bending
checks and native/CPU agreement do not certify a physical global 3D maximum.
