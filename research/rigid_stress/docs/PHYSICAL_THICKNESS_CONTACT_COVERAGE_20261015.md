# Egg thickness fields and actual contact-refinement coverage

The [complete physical-validation objective](LOAD_MODEL_PHYSICAL_VALIDATION.md) remains active. Three fixed-surface thickness levels retain unresolved tensor changes despite stable energy and raw peaks. An independent contact-position audit rejects the apparent small-cap adaptive plateau: every active sample in the first two levels lies outside the refined caps. Four uniform fixed-polyhedron levels also retain substantial field and peak changes. Sparse reference ordering preserves all saved arrays and full stress fields at two resolutions; these numerical checks do not establish physical convergence.

## Fixed-surface thickness axis

The two/three/four-layer meshes contain 200892/301338/401784 affine P2 tetrahedra. An independent reconstruction of every boundary face and oriented surface-flux integral verifies identical actual outer/inner boundaries, volume 3.742463856329281e-6m³ and mass 0.00748492771265856kg at unchanged density 2000kg/m³. COM is unchanged. Physical traction positions, normals, weights and vectors remain identical in all five cases; this isolates through-thickness discretization on the declared fixed polyhedron. This mass belongs to the fine static FEM geometry, not the original coarse Franka egg.

All 15 complete FEM fields are retained. The audit independently reconstructs raw corner maxima, peak coordinates and energy/work, checks saved fields are finite, and verifies exact source/mesh/traction identities. Complete FEM equilibrium budgets in this audit are recorded from the independent FEM driver; they are not freshly reassembled by the postprocessing audit. Producer jobs 716343/716344 and audit 716353 are terminal COMPLETED with closed source identities.

| Case | Layers | Raw global peak (Pa) | Energy (J) | Recorded FEM budget fraction |
|---|---:|---:|---:|---:|
| opposing | 2 | 143156.924 | 9.46512302e-09 | 0.0046527 |
| opposing | 3 | 143295.535 | 9.4687488e-09 | 0.0098312 |
| opposing | 4 | 143313.528 | 9.46994451e-09 | 0.017784 |
| bending | 2 | 141819.343 | 8.73557765e-09 | 0.0046774 |
| bending | 3 | 142068.739 | 8.73891617e-09 | 0.0096956 |
| bending | 4 | 142209.984 | 8.7400178e-09 | 0.017582 |
| asymmetric | 2 | 232636.352 | 1.68664806e-08 | 0.0049127 |
| asymmetric | 3 | 232910.316 | 1.68729645e-08 | 0.0098931 |
| asymmetric | 4 | 233043.413 | 1.68751025e-08 | 0.019007 |
| near_slip | 2 | 150769.911 | 1.00160615e-08 | 0.0035831 |
| near_slip | 3 | 151031.898 | 1.00198549e-08 | 0.008123 |
| near_slip | 4 | 151143.467 | 1.00211053e-08 | 0.013995 |
| multiple | 2 | 143948.489 | 1.11260112e-08 | 0.0049526 |
| multiple | 3 | 144082.398 | 1.11299102e-08 | 0.010907 |
| multiple | 4 | 144098.257 | 1.1131194e-08 | 0.019145 |

The declared initial diagnostic requires all five observable changes below 3%, applied only to a credible trend/tail. None of the ten adjacent case/level comparisons meets it. The 3→4-layer results below retain the original 36 common physical points and the original norm definitions; percentages describe adjacent changes, not exact continuum errors.

| Case | Tensor (%) | von Mises (%) | Displacement (%) | Raw peak (%) | Energy (%) |
|---|---:|---:|---:|---:|---:|
| opposing | 3.82345 | 2.34851 | 0.01626 | 0.01255 | 0.01263 |
| bending | 4.14833 | 2.19680 | 0.01631 | 0.09932 | 0.01260 |
| asymmetric | 4.92495 | 5.13225 | 0.01499 | 0.05711 | 0.01267 |
| near_slip | 4.84777 | 2.96377 | 0.01633 | 0.07382 | 0.01248 |
| multiple | 3.61952 | 2.28065 | 0.01561 | 0.01101 | 0.01153 |

## Stress traces and additional interior probes

The original 36 points are30 seeded directions plus the six axes at the original analytic midwall coordinates. On the two-layer mesh, original indices 30/32/35 have barycentric margin below 1e-8; the minimum is 4.44e-16. A second audit samples 12 signed directions around every point at 10/1/0.1nm without averaging or choosing a preferred trace. A neighborhood tensor diameter near 2.7–2.8% persists as the perturbation shrinks in the two-layer cases, so exact-interface traces deserve explicit reporting.

To test whether interfaces explain the full discrepancy, the audit additionally declares 64 directions (seed2026101505) at four Gauss–Legendre interior depths, yielding 256 fixed common physical points. Every added point has barycentric margin at least 0.000139750177 across the three meshes. Their3→4 tensor changes remain:

| Case | Original36-point tensor change (%) | Additional256-point tensor change (%) |
|---|---:|---:|
| opposing | 3.82345 | 4.01308 |
| bending | 4.14833 | 4.02326 |
| asymmetric | 4.92495 | 3.77455 |
| near_slip | 4.84777 | 4.07591 |
| multiple | 3.61952 | 4.16443 |

Interface ambiguity therefore does not explain away the unresolved thickness response. The added points diagnose the discrepancy; they neither replace the original acceptance observable nor define a volume-weighted global norm. Original global raw maxima remain unchanged. The v1 trace audit is preserved; v2 corrects its metadata key for the 1e-8 barycentric threshold without changing fields or metrics. Both jobs 716828/716834 are terminal COMPLETED. Five/six layers (502230/602676 tetrahedra) and a complete 2/3/4/5/6-layer audit are scheduled as716835/716836/716837 and count as no accepted evidence yet.

## An apparent contact plateau fails coverage

The first two original local CAD references use50µm numerical marking caps with 100/50µm target edges and 17340/22152 tetrahedra. Their final raw peaks6.421227/6.421114MPa are close. Independent five-state audits pass unchanged 1e-9N body and mapped force gates, but a separate coverage audit 716833 finds all eight active samples outside the caps in every saved state of both runs. The unsigned normal sample-load fraction outside is 100%; sample angle/cap-angle ratios range12.4949–34.8402. Incident active trace edges remain275.90/752.47/1002.45µm (minimum/median/maximum). Thus the near-equal peaks do not demonstrate that active contact was refined.

The coverage diagnostic uses the same author-frame angular parameter as mesh marking; it is not a geodesic distance, resolved contact boundary or traction-area integral. Inside-cap samples would still not rule out missed contact on coarser geometry elsewhere. The 32340-tet third small-cap case is also retained with all five independently balanced states and a6.421114MPa final raw peak; its contact coverage is not inferred from the first two cases. The original 61812-tet reference remains outside this checkpoint while running.

Corrected geometry adds every independently audited sample position and support directions from its actual normal-force vector as numerical marking anchors. Geometry/material/density, original six held initializer points, contact law/friction, controls, timestep and force budgets remain fixed. These anchors guide mesh construction only; the solver can contact any surface location.

| Numerical marking radius | Target CAD surface edges (µm) | Paired tetrahedron counts |
|---|---|---|
|500µm (v4)|100 /50 /25 /12.5|90240 /282168 /994368 /3728064|
|150µm (v5)|100 /50 /25 /12.5|31668 /64716 /171228 /545988|

Both completed builders retain full paired CAD/polyhedron assets, matching connectivity and exact old vertices/initializer anchors. The paired polyhedron retains its original boundaries/volume/mass; CAD geometry approximation and represented mass change explicitly at fixed density. The background is still global level 3, with about 0.845% volume/mass error, and two thickness layers remain. The 150µm caps reduce numerical refinement extent; they do not change a physical footprint radius. Actual first/second-level ordered references716883/716886, independent audits716884/716887 and coverage audits716885/716888 are scheduled only. A credible contact/geometry tail still requires more than two useful levels and measured coverage.

## Four fixed-polyhedron levels

The new30720-tet producer716500 and independent all-five-state audit 716501 complete the uniform fixed-polyhedron sequence. Job716881 reconstructs the common6.25ms state on 2400 identical physical probes, independent force balances, full raw corner fields and objective displacement/energy. Body mass/COM, true boundary geometry, finger meshes/axes/masses and held physical points are checked. All levels still have only two active body contact samples.

| Tetrahedra | Active lumped contact support (m²) | Raw global lower peak (MPa) |
|---:|---:|---:|
| 480 | 9.65135906e-05 | 0.156452585 |
| 1920 | 2.41280205e-05 | 0.28422139 |
| 7680 | 6.03186193e-06 | 0.569475319 |
| 30720 | 1.50788797e-06 | 1.20354372 |

The 7680→30720 changes are tensor 27.18622%, von Mises12.90384%, objective displacement8.40235%, nonlinear energy 12.33112% and raw peak 52.68345%. Relative spatial changes normalize by the finer field or value. The raw peak rises from0.569475 to1.203544MPa, an increase of111.34% relative to the coarser level. Support quarters again while the peak grows. This is unresolved spatial/contact concentration, not proof of either a converged finite footprint or a mathematical singularity. Volume and contact resolution still vary together here.

## Complete reference-ordering qualification

An isolated patch changes the two sparse LU calls to `permc_spec="MMD_AT_PLUS_A"`; production code remains unchanged. At7680 tetrahedra, job 716810 re-solves all five original states from the byte-identical initializer and job 716826 independently audits their original force gates. At30720, baseline716805 and candidate716824 both finish naturally, with independent all-state audits716806/716825. Job716882 compares every saved array and reconstructs all corner/quadrature tensors from those complete histories without another solve. Configurations differ only by output path; geometry/contact/map/initializer bytes agree.

| Tetrahedra | States | Maximum full-corner tensor change | Maximum full-quadrature tensor change | Maximum raw-peak change |
|---:|---:|---:|---:|---:|
| 7680 |5| 5.70499e-15 | 1.46464e-12 | 1.66533e-15 |
| 30720 |5| 3.37553e-15 | 5.2182e-13 | 8.88178e-16 |

All original array budgets (relative 1e-5, absolute 1e-12), complete field relative 1e-6 and physical 1e-9N gates remain unchanged. The 30720 baseline/candidate maximum body/mapped independently reconstructed residuals are1.30e-13/2.41e-13N and 1.32e-13/2.44e-13N. Their final raw peak is 1.756437MPa with five active nodes and three stencils; it is a short CAD diagnostic, distinct from the fixed polyhedron above.

The 7680 baseline/candidate execute 15 linear solves and one fresh factor each, with total recorded linear times 191.8673/61.5339s. They run on different nodes (`cpu-206-205`/`cpu-207-129`), though the CPU model matches. These are single-run descriptive observations, not a controlled3.118× speedup, warmed native throughput or verified physical accuracy/cost.

## Reproduction and immutable evidence

Runtime root: `$RIGID_STRESS_DATA_ROOT/runs/20261014-physical-validation`. Reusable entry points retain exact arguments, before/after source hashes, CPU/package versions, scheduler states and failures. The main commands are:

```bash
bash research/rigid_stress/scripts/audit_thickness_probe_traces.sh FRESH_RUN --axis-audit CLOSED_AXIS_AUDIT --run BASE_FEM LAYERS3_FEM LAYERS4_FEM --mesh BASE_MESH LAYERS3_MESH LAYERS4_MESH --original-probes BASE_PROBES_NPZ
bash research/rigid_stress/scripts/audit_adaptive_contact_coverage.sh FRESH_RUN --run CLOSED_CONTACT_RUNS --audit MATCHED_CLOSED_AUDITS
bash research/rigid_stress/scripts/compare_saved_paired_references.sh FRESH_RUN --baseline CLOSED_BASELINE --candidate CLOSED_CANDIDATE --baseline-audit BASELINE_AUDIT --candidate-audit CANDIDATE_AUDIT
```

The complete exact commands and job provenance are inside each packet. Baseline states referenced by the new four-level comparison remain in the preceding immutable generic-stencil/paired-contact packets. Complete new producers and independent audits are included here. Large NPZ fields are preserved as exact ordered32MiB gzip byte chunks with full-file, part and stored-byte hashes. Every array/record is reconstructible. A separate64MiB synthetic binary fixture qualifies this publication path; it is not physical evidence. The new runtime verifier also checks actual producer paths, distinguishing those originals from derived staging chunks and indices.

| Packet | Manifest SHA256 | Archived entries | Verified actual producer files | Reconstructed large originals |
|---|---|---:|---:|---:|
|[egg-thickness-fields-v1](../evidence/20261014-physical/egg-thickness-fields-v1/manifest.json)|`915665a4ada17beb8ef5d4d21b678e8de4b2e5f5086a44440cc0d68228b3de52`|450|347|15|
|[reference-ordering-qualification-v2](../evidence/20261014-physical/reference-ordering-qualification-v2/manifest.json)|`18cd50329997e6cb34a0af66c5a646f0a02c1d7dd4a55cb47dfd20b50c828e4a`|308|308|0|
|[egg-contact-coverage-v1](../evidence/20261014-physical/egg-contact-coverage-v1/manifest.json)|`8420e532ae5449c7ab3981f86f665945f8b1c0da6f391608acb7b28f69cf695c`|1434|1434|0|

The ordering packet predates byte staging for binary fields and its308 verified originals are direct producer paths. The contact packet also retains the full thickness and ordering verification records. No staged part is counted as an actual producer file.

Full convex/nonconvex convergence, load-model sensitivity, egg/U resolved contact, original approach/lift/support histories, replay versus feedback and physically verified warmed costs remain required. The new layer/contact jobs are scheduled work, not completion evidence.
