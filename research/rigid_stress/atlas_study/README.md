# Feature-mesh atlas: theory and CPU validation

Research requested on 2026-10-24: precompute meshes with local refinement at
representative surface regions, then select/combine their responses online.
This package contains theory, executable CPU experiments and retained failures.
It does not implement a native Genesis/Quadrants feature or change production
defaults. It is supplementary to the existing thin-shell integration goal.

## Read first

- [REPORT.md](REPORT.md): measured results, qualifications and recommendations.
- [THEORY_ZH.md](THEORY_ZH.md): self-contained main derivation in Chinese.
- [THEORY.md](THEORY.md): extended assumptions, proofs and error decomposition.
- [EXPERIMENT_MANIFEST.json](EXPERIMENT_MANIFEST.json): source hashes and cases.

The safe first algorithm is to preserve each current physical DPR traction,
choose a mesh fine near that traction, evaluate its whole-body response and sum
the stress tensors at common physical points. Mesh centres select a solver;
they do not replace actual contact positions. Equivalent coarse-plus-full-delta
notation avoids repeating the coarse response. All basis/factor data may be
prepared once; no mesh or factor is rebuilt for the online queries.

This mixed-space sum is generally **not** the Galerkin solution of the joint
refinement. Exact linearity of each solve is not a proof of atlas accuracy.
Static condensation of a common FE model is a different, exact algebraic
operation; its coupling and assumptions are derived and tested separately.

## Experiments and their scope

| Folder/file | What it verifies |
|---|---|
| `algebra.py` | Energy/residual identities, non-Galerkin counterexample, coarse-residual blind spot, force/radius and peak-blending counterexamples |
| `affine_control.py` | 3D P2 constant strain under independently integrated analytic tractions |
| `freebody_control.py` | Actual free-body P2 FEM, consistent mass and common inertia relief; single-mesh superposition |
| `volume/` | 3D P2 box, conforming local tetrahedral refinement, actual moving finite circular pressure and friction, multiple contacts, common physical stress probes and switching |
| `shell/` | Linear shell plate atlas, eccentric friction, moving/radius/multiple loads, smooth switching; bounded closed-egg negative controls |
| `nonconvex/` | 2D P2 nonconvex narrow-neck unit-response model; contact-only versus fixed-neck refinement; remote sampled hot regions |
| `performance/` | Matched CPU factor/response paths including joint-mesh response controls; exact full-interface static-condensation check |

All field differences are explicitly defined common-point tensor norms, not
continuous integrated norms. Finer-mesh changes are refinement sensitivity,
**not rigorous error bars**. Reported maxima are over stated samples; sharp
corners and unresolved tiny Hertz contacts do not acquire a bounded/accurate
continuum maximum merely because a numerical check passes.

The shell plate and nonconvex tensor grids include fine strips as well as a
fine region. The 3D volume suite uses conforming local tetrahedral bisection.
All exact planar/polyhedral domains share physical geometry; curved-egg mesh
geometry differences are another error contribution. Cases whose entire
pressure footprint is not in the nominal finest core are identified, not
removed or treated as fully resolved.

The nonconvex model is a normalized linear operator experiment. Unit loads can
produce large formal displacements; linear force rescaling preserves its error
ratios and puts a physical realization in the small-strain regime. It is not
a large-deformation simulation. Its remote peak is a sampled neck/corner-region
hot spot with explicit exclusions, not a certified continuum maximum.

## Important retained failures

- Initial unbounded element-finder fallback exhausted memory or stalled; bounded
  point-query batches replace that execution path. This does not change meshes
  or FEM equations. Interrupted/resource-failed attempts remain documented.
- An existing shell oracle rejects explicit zero-friction normal loads because
  of roundoff in its zero-allowance check. The initial failures are retained;
  final purely normal fixtures remove only that input cone check and record
  the actual tiny shear roundoff, retaining force/moment and load integration.
- The nonconvex strict raw load-relative residual target `1e-9` remains failed
  at about `1.94e-9`. It is not relabeled passed. Extended-precision residual
  correction changes the sampled field by only about `9.21e-11`; numerical
  roundoff does not explain the percent-level discretization differences.
- Hard atlas selection produces observable jumps even for smooth contact
  motion. Source-parameter blending removes the measured artificial jump;
  it does not by itself prove accurate stress.

## Reproduce

Dependencies: NumPy, SciPy, Matplotlib, scikit-fem and threadpoolctl. Rendering
the optional self-contained HTML report additionally uses markdown2 and
latex2mathml. Gmsh is not required. The shell suite imports the existing
`shell_cpu` research implementation from the enclosing Genesis checkout.

From the repository root:

```bash
cd research/rigid_stress
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python -m atlas_study.algebra
python -m atlas_study.affine_control
python -m atlas_study.freebody_control
python -m atlas_study.volume.study --help
python -m atlas_study.shell.plate_atlas --help
python -m atlas_study.performance.benchmark --help
python -m atlas_study.performance.condensation_check
```

Consult each suite's README before execution: some research runners have no
argument parser and execute/overwrite default evidence when invoked, including
with `--help`. Preserve committed evidence or use a fresh checkout and new output
directories. Do not launch large references and timing jobs concurrently.
In particular, `python -m atlas_study.nonconvex.study` runs the entire nonconvex
suite and has no `--help` parser.

The full performance bank reuses the 3D accuracy suite's physical loads and
common output probes. A tiled B64 bank is four copies of 16 distinct cases,
not 64 independent conditions. CPU module timings exclude rigid/Franka dynamics,
policy inference, rendering and GPU/native launch/copy behavior. Initial setup,
precomputed-operator memory and repeated online work are separate.

No universal same-error speedup, whole-egg physical contact qualification,
native GPU throughput or online deformable-contact solve is claimed.
