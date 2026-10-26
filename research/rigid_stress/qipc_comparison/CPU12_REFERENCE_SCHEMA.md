# CPU twelve-feature reference export, schema v1

Run the independent CPU exporter only in a Slurm `rtx-mid` allocation using
image `1_26`, with the output under the project data directory:

```sh
python -m research.rigid_stress.qipc_comparison.export_cpu12_reference \
  --output /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261026-cpu12/oracle-v1
```

Options `--ncoarse 3 --subdiv 4 --quadrature 8` are the published defaults.
Nondefault exports are labelled adaptations and do not claim to reproduce the
published sanity numbers. Existing nonempty output directories are rejected.
The manifest pins the primary commit and checks that every CPU Python source
matches it before and after export. No scientific import precedes the Slurm guard.

All NPZ archives load with `allow_pickle=False`. Integer IDs are zero-based.
Physical shell DOFs are interleaved `(ux,uy,uz,theta_x,theta_y,theta_z)`.
Sparse archives use SciPy `save_npz` CSR format. The manifest records the shape,
dtype, SHA256 of each array's C-order bytes, and complete archive hash. The job
receipt should separately hash `manifest.json`.

## Inputs permitted in native runs

- `inputs/model.npz`: `vertices`, `quads`, `directors`, `directions`,
  `coarse_vertices`, `coarse_directions`, `coarse_nodes`, `parent_corners`,
  `parent_ij`, `ncoarse`, `subdiv`, `macro_quads`, `macro_coefficients`,
  `macro_centers`, `macro_radii`, `face_tables`, `node_macro_indptr`,
  `node_macro_indices`, `macro_neighbor_indptr`, `macro_neighbor_indices`,
  `fixed`, `coarse_fixed`, `young`, `poisson`, `thickness`, `density`,
  `drilling_factor`, `egg_axes`, `egg_beta`.
- `parent_corners` is `(n_physical_nodes,4)` of **physical mother node IDs**.
  Its corner order is `(--),(+-),(++),(-+)`; `parent_ij` is `(n,2)` of integer
  `(si,sj)` in `0..subdiv`. Both follow the original first-macro registration.
  Compute bilinear weights in Quadrants from `si/subdiv,sj/subdiv`; keep zero
  weights harmlessly or remove them with their corresponding entries. Coarse
  nodes are explicit, so a cardinal P row need not be inferred from parents.
- `macro_quads` uses **coarse local node IDs**, unlike `parent_corners`.
  The two incidence relations use ordinary CSR integer lists.
- `fixed` is the original CPU's six mother scalar translation pins;
  `coarse_fixed` gives the corresponding six scalar coarse coordinates.
- `inputs/selection.npz`: `centers`, `fill_angle`, `support_angle`, `core_angle`,
  `narrow_fraction`, `maximum_radius`.
- `cases/{published43,boundary18,static16,static64}.npz`: source-axis arrays
  `direction (S,3)`, `radius (S,)`, `force (S,3)`, `has_force (S,)`,
  `normal_force`, `friction`, `tangent`, `has_tangent`, `cap_tangent`,
  `has_cap_tangent`, `side`, `source_environment`, `source_names`; case-axis
  `case_names`, `case_families`, and CSR `case_offsets (C+1,)`. Optional vector
  values are zero where their flag is false. Sources retain original order.
  The release environment has an empty interval, not a dummy source.
- Static banks additionally give scalar `reference_bank="published43"` and
  `reference_case_indices`. Static64 is four copies of the static16 tile.
  Boundary cases add `boundary_index`, `epsilon`, `crossing_side` and
  `pair_case_indices (9,2)`; `cases/boundary_locations.json` identifies the
  three pair/junction controls and fixed forces/tangents.

Frozen geometry and constants describe inputs. They do not authorize feeding
CPU-computed floating interpolation weights, operators, loads or solutions into
native preparation or timing. Native floating computations remain in Quadrants.

## Independent validation oracles

- `oracles/operators/{P_nodes,P}.npz` contain the exact CPU bilinear maps.
- `oracles/operators/mother/{K,K_free,S}.npz` contain physical, unscaled
  operators. `numerics.npz` contains `rigid_modes`, `mass_modes`, `mass_gram`,
  `relief_coefficients`, `nodal_mass`, `rotary_mass`, `mass_diagonal`,
  `center_of_mass`, `axes`, `gauss_weights`, `gauss_points`, `B_membrane`,
  `B_bending`, `B_shear`, `element_dofs`, `free`, `coarse_free`, `field_weights`.
- `oracles/operators/feature-XX/{K,T,S}.npz` and `space.npz` contain the exact
  feature operators, `active_macros`, `local_nodes`, `local_dofs`, and
  `relief_projection`. `T` has **3516 free columns**, whereas native raw maps
  may keep 3522 columns with six masks. Demonstrate the column-removal mapping.
- Each operator directory includes independent `factor_L.npz`, `factor_U.npz`,
  and `factor.npz` (`scale`, `perm_r`, `perm_c`), for the CPU's diagonally
  equilibrated SuperLU factors. `inventory.json` reports rank proofs, dimensions,
  nonzeros, factor fill/storage and CPU preparation time.
- `oracles/<bank>/loads.npz`: source-column `raw`, `balanced`,
  `relief_acceleration`, `source_force`; shared `radial`, `angular`,
  `cap_weights`; `(S,4*q*q,...)` arrays `cap_directions`,
  `location_{nodes,weights,positions,normals,macro_ids,element_ids,local}`,
  `physical_positions`; source resultants `expected_moment`, `recovered_force`,
  `recovered_moment`; environment-column `raw_environment`,
  `balanced_environment`.
- `mother/source_response.npz` stores source-column `q`, `stress_local`.
  `mother/output.npz` stores environment-column `q` (sum of source solutions,
  as in the published selection study), separately solved `q_grouped`,
  `unscaled_residual`, `grouped_vs_source_superposition`, and complete outputs.
- `source_features/feature-XX.npz`: all source-column `projected_rhs`,
  `projected_raw_rhs`, `z`, `unscaled_residual`,
  `projected_relief_relative_difference`.
- For `continuous`, `nearest`, `narrow`, `<selector>/output.npz` contains
  `q`, `stress_local`, `stress_full9`, `peak`, `N`, `M`, `Q`.
  `<selector>/selection.npz` gives `weights (S,12)`, group CSR
  `group_features`, `group_environments`, `group_source_indptr`,
  `group_sources`, `group_weights`, case-axis `groups_per_case`,
  `contributions_per_case`, `relative_field_error`, `relative_peak_error`,
  `grouped_vs_source_superposition`.
- `<selector>/groups/feature-XX.npz` uses final axis indexed by its
  `environments`, storing `grouped_force`, `projected_rhs`, `z`,
  `unscaled_residual`. Zero-group features have a zero-length final axis.
  Group order is feature-major, then ascending environment ID.
- Boundary18 additionally exports `top2` as a negative junction control and
  `boundary_jumps.json`. `summary.json` reports all computed selectors.

Static16/static64 keep the full small source input manifests. Their large
oracle payloads are references in `oracles/<bank>/alias.npz`:
`reference_bank`, `case_indices`, `source_indices`, `case_offsets`.
Gather the final source or environment axis from published43. Grouped outputs
map each referenced feature/environment to its new environment. This freezes
the mathematical column values; it does not claim a separately timed CPU run
or independent trajectories for four repeated tiles.

`norms.json` states the exact component, sample and residual conventions.
Local stresses are side-major, then element, Gauss, component; full9 tensors
are `(side*element*Gauss, row-major tensor9, environment)`. Native element-major
two-side six-component tensors need explicit conversion. `published.json`
compares the three selector summaries with the existing published JSON and
keeps its negative-control summaries. All oracles and these numerical audits
stay outside production execution and GPU timing.
