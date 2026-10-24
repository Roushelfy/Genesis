# CPU shell feature-mesh atlas

This study evaluates the user's preprocessed locally refined mesh bank. It
does not modify Genesis production code and does not run deformable dynamics.
All numerical work here is CPU NumPy/SciPy and the existing independent shell
research core. It is not a Quadrants/GPU throughput result.

## Exact-geometry plate control

`plate_atlas.py` builds one coarse shell and three preselected locally refined
shells, each covering a different contact region. All are the same 40 mm square
plate, thickness 0.5 mm, E=10 GPa, nu=0.3, with its whole perimeter clamped.
Tensor-product refinement introduces fine strips as well as a local fine
patch; this is explicitly not a minimal octree/quadtree mesh.

Every current pressure disk remains at its actual continuously varying point.
Normal and tangential resultant forces are integrated with the same finite
Hertz profile. Analytical transverse moments plus adaptive longitudinal
quadrature produce Q4 nodal loads, independently of the main polar transfer.
Tangential loads include the top-surface eccentric rotation couple. The broad
1.2/1.8/2.4/3.6 mm disks are prescribed physical load controls, not a claimed
automatic-radius correction or an egg pressure reference.

All mesh stiffnesses and factorizations are built before querying. Each contact
selects one feature mesh; contact responses are evaluated and summed at the
same 3,481 physical material probes. The full in-plane top/bottom tensors are
combined before VM evaluation. Soft blending combines the same force response
between adjacent meshes with weights summing to one.

Twenty cases cover thirteen moving contacts, three additional radii, two
separated/nearby contacts, three asymmetric contacts, and complete release.
`results/results.json` retains complete numerical gates, diagnostics, setup
costs, references, switching tests and costs. `audit.json` passes nine focused
implementation checks, including stress-map equivalence (2.80e-16), independent
pressure-transfer comparison (<1e-8), wrench/residual acceptance, exact
coarse-plus-delta algebra, normal source balance and unload.

| Model | DOF | Description |
|---|---:|---|
| Coarse | 486 | 5 mm global spacing |
| Features | 2,622 / 2,964 / 2,964 | 1 mm patch spacing, coarse exterior |
| Joint | 9,570 | Wider combined fine region, ~0.75 mm |
| Reference check | 14,910 | Wider region, ~0.5 mm |
| Finest reference | 28,482 | Wider region, ~0.375 mm |

Against the finest sampled shell reference over the nineteen loaded cases:

| Method | Full sampled tensor difference | Sampled VM peak difference |
|---|---:|---:|
| Coarse | 21.82--32.02% | 21.69--48.94% |
| Selected feature atlas | 10.68--17.21% | 0.062--4.483% |
| Smooth blend | 10.58--17.21% | 0.161--5.349% |
| Joint | 3.24--4.37% | 0.122--1.789% |
| Reference check | 2.71--3.60% | 0.008--1.699% |

The tensor metric is an **unweighted Frobenius norm on common probes**, not a
continuous area L2 norm. Peaks are only over these probes and both surfaces.
The remaining finest-grid variation is explicitly material; no continuum or
3D contact ground truth is claimed. Smaller peak error can coexist with a much
larger complete-field error, and the peak output alone must not certify fields.

At a feature-selection boundary, moving the contact by only 20 nm causes a
hard-selection tensor jump of 8.975/8.998%. The actual finer-reference motion
changes the field by only 2.49e-6 relatively; smooth blending changes it by
2.45/2.48e-6. Thus hard switching introduces a real artifact, and smooth
operator blending removes the observed finite jump in this bounded control.
Blending is still an approximation, not a conforming Galerkin reconstruction.

`coarse_exterior_ablation.py` tests one matched two-contact case with a 2.5 mm
coarse exterior and the same 1 mm feature patch. Coarse error improves from
27.93% to 18.19%, but atlas field error changes only 14.55% to 14.23%; its peak
difference changes 3.18% to 3.76%. The residual complete-field error is therefore
not removed by improving only the exterior at this fixed patch resolution.

The first online solve/stress timings are diagnostic and can overlap other CPU
jobs. They exclude load integration and all setup; they must not be reported
as complete workflow cost or as equal-accuracy speedups. A dedicated performance
confirmation is separate from this accuracy script. Feature construction and
shared factor storage are explicitly retained rather than hidden in online cost.

## Closed egg supplement

`egg_supplement.py` samples the same authored 22/22/30 mm egg with taper -0.18
and 0.5 mm normal-offset wall as the previous CPU study. A nonuniform cubed
surface grid has shared nodes on all six charts. Each feature refines two
opposing cap regions and associated strips. Corresponding Q4 geometries differ
across meshes, so comparisons include geometry-discretization error.

Each source group is a complete opposed normal pair at the egg's widest
latitude. Its force and moment are balanced before solution; inertia relief is
disabled. Two such source groups test superposition without mesh-dependent mass
or artificial clamp ambiguity. There are five moving angles, two simultaneous
balanced pairs, and the physical automatic-radius control (~45.47 micrometres).
All seven final cases are numerically admitted. Their feature-atlas field
differences are 35--65% for manual mm-scale disks and 45.25% for the automatic
tiny disk; the joint-to-finer reference still changes 23--30%. These are
**negative/unresolved accuracy results**, despite accurate loads and solves.
The automatic tiny-disk sampled peak difference is 41.09%.

Initial explicit `mu=0` fixtures hit the inherited transfer oracle's lack of an
absolute roundoff allowance in its cone check: exact normal forces at arbitrary
angles acquire ~1e-16 tangential roundoff after normalization. Preserve
`egg_results/initial_zero_friction.json` and its source snapshot. Final normal
fixtures request no friction-cone law (`None`), retain identical forces and
all integration/wrench gates, and record tangential roundoff <=2.80e-16. This
does not add friction to the egg fixture or patch the frozen contact oracle.

Neither this shell atlas nor shell refinement restores missing 3D sigma_nn.
The positive flat-plate result cannot be transferred to the egg or used to
assert a qualified physical global maximum.

## Reproduction

From this directory, with the surrounding Genesis research package available:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python plate_atlas.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python egg_supplement.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python coarse_exterior_ablation.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python audit.py
python plot_egg.py
```

Source lookup finds a surrounding Genesis checkout, or the task's initial
`genesis-atlas` clone. Large field NPZs are reproducible outputs; JSON reports,
PNG summaries and initial failure evidence are retained.
