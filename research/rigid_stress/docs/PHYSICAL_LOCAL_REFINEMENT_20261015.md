# Local contact refinement and finer reference numerical equivalence

The complete [physical-validation goal](LOAD_MODEL_PHYSICAL_VALIDATION.md)
remains active. This checkpoint qualifies an affordable local-mesh study around
the unresolved actual-finger contact loads and extends reference numerical
equivalence to3840 tetrahedra. Mesh qualification and scheduled solves do not
establish a contact/stress tail, full-arm matching or physical accuracy/cost.

## Finer cached reference closes its original field comparison

Run714827 completes all64 original held PD targets and80 inertial steps with
3840 body tetrahedra. Original mapped/body force budgets remain1e-9N; maximum
complete residuals are3.8225e-10/3.8113e-10N. The unaveraged nonlinear global
lower/upper peaks are226120.525349/226120.525677Pa.

Independent audit714839 compares all80 states with the original solved3840-tet
reference: full unaveraged corner Cauchy tensors differ by at most4.5215e-11
relative, body displacement1.7872e-11 and contact force8.3398e-11. Every original
complete force gate passes in both runs. The cached method performs343 recorded
linear solves with14 fresh LU factorizations and1560 GMRES iterations. This is
numerical reference qualification, not a controlled timing or physical accuracy
claim. The30720-tet run714840 starts only after this audit succeeds.

## Fixed geometry and contact-centered local mesh

`localized_franka_mesh.py` anchors two refinement centres at active body load
nodes from the closed original480-tet run, at the fixed1e-8N activity threshold.
Each level marks cells whose bounding-box distance to either fixed centre is
at most1mm, then uses unmodified scikit-fem12 marked-tetrahedron bisection.
The original seed polyhedron and seed arrays remain fixed. The factory retains
all actual compound finger surfaces and masses, material, original six held
physical points, actuation, friction and gravity. Refined consistent mass and
centre of mass are guarded by the original numerical checks in the factory.

Qualification714873 checks positive cell volumes, original vertices and seed
array identity, volume, centre of mass and boundary area at all four levels.
Total volume is3.264924105200664e-6m3 and boundary area0.013893944984008m2;
both are preserved. The installed mesh implementation and its hash are saved.

| Marked-refinement rounds | Body tetrahedra | Body P2 DOFs | Largest edge incident on a contact vertex (mm) |
|---|---:|---:|---:|
|0|480|2430|17.23234|
|3|1099|5217|6.01186|
|6|3993|17718|1.50297|
|9|20696|87987|0.53851|

These incident-edge diagnostics concern this seed's contact vertices; global
maximum edge remains18.68655mm away from the refined region. Smallest global
edge alone does not establish useful contact resolution. The fixture still
uses fixed faceted geometry; CAD/geometry error remains a separate axis.

Four failed mesh diagnostics are retained.714858 fails JSON serialization of
an integer-valued mesh property.714862 exceeds the declared60000-cell study
cap at round11 with94403 cells.714865 rejects an incorrect test expectation
about the installed tie rule.714866 rejects a candidate custom exact-edge
tie rule because boundary area increases7.2795%. That candidate is removed;
its topology is never used for accepted physical solves. Installed scikit-fem
adds seeded perturbations only to its edge-ordering copy, preserving physical
positions; this checkpoint makes no package-defect claim or shared-library edit.
No geometry or residual tolerance is relaxed to accept any failed candidate.

## Physical studies and remaining acceptance

Array714874 schedules four original64-target/80-step histories on each of two
axes. Coupled runs refine volume and body collision triangles together;
frozen-contact runs retain the original contact triangles at their original
physical nodes while refining the same volume neighbourhood. Both retain
the original1e-9N mapped/body force gates. The array depends on successful mesh
qualification714873. `compare_spatial_franka.py --axis local_levels` reports
explicit pairwise axis changes at common material probes. The traction audit
reconstructs either contact representation, retains unsmoothed raw forces and
reports active area, force/area concentration and separate island wrenches.
Extended audit714868 reproduces every original traction/island observable
exactly on the three previously solved collision-only cases.

Both local level0 controls714874_0/4 complete. Every npy payload in all83
original matched geometry/state archives is byte-identical to the qualified
cached coarse reference, on both coupled and frozen-contact paths. Their80
states meet the original complete force gates; finer local results remain live.
First local level3 runs714874_1/5 also complete all80 states with1099 tetrahedra.
Coupled/frozen raw lower peaks are562064.441114/562104.725277Pa, with retained
whole-volume upper bounds only0.000328Pa above each lower value. Largest complete
mapped residuals are2.599e-10/2.621e-10N; minimum whole-volume determinant bounds
are0.99995984/0.99995983. Peak growth from the coarse156447Pa survives fixed
contact triangles. It is evidence of unresolved local stress sensitivity, not
a demonstrated singularity or a physical tail. First common-field/traction
audits714898/714899 complete with the original manufactured-field and force
pullback gates. At2400 fixed material probes and10 common times, full tensors
change43.0296%, von Mises21.3203%, objective displacement19.7577% and energy
33.1750%, while contact force resultants change0.0115784%. The global lower
peak moves to the opposite contact neighbourhood, at reference position
`[-2.26835e-5,-0.021961834,0.002086607]`m; it is not a fixed peak-location tail.
Two active body nodes/islands persist. Their combined lumped current area
falls from96.51359 to6.45833mm2 and maximum unsmoothed normal force/area rises
from8.28321 to160.64966kPa. This is a discrete concentration diagnostic,
not independently resolved physical patch area. Reconstructed force pullback
matches saved source loads exactly at both levels.
Dependent common-field/traction audits714891/714892 and714893/714894 run after
the first three coupled/frozen levels complete successfully. They count as
scheduled checks until their scheduler state and complete original gates pass.

The original third U reference remains live near its8h allocation limit.
A scheduler request to extend it is denied; no running source or process is
changed. Independent cached U controls/fine studies714883 use the same original
physical configurations and complete force gates. Their convergence and
same-case field equivalence are not yet established. The original job continues.

Next acceptance requires solved local fields and credible spatial/contact tails,
then matched full egg/Franka and interior-hotspot references, applicable
identical-input load-model comparisons, resolved-load replay versus feedback,
and warmed native cost at verified physical accuracy. All original A/B/C,
geometry, history, extrema and reproducible-deliverable requirements remain.

## Reproduction and immutable evidence

The [checkpoint manifest](../evidence/20261014-physical/local-refinement-checkpoint-v1/manifest.json)
retains10 terminal runs, including all four failed diagnostics, and888 exact
original files.446 input/executable source identities are independently checked.
Its SHA256 is `8f7f9c63b0dd269eaccd7588886f6695a06c69cbc03698008a6d40b9af39b59b`.
The adjacent verification compares every archived file with its original;
Slurm outcomes, arguments, exact submission commands, meshes, fields, installed
mesh implementation, package/hardware records and both zero-control array
comparisons are included. Live studies are excluded.

The [first local-field checkpoint](../evidence/20261014-physical/first-local-fields-checkpoint-v1/manifest.json)
adds both solved1099-tet cases and their common-field/traction audits,406 exact
files and148 independently checked source identities. Its SHA256 is
`9a232aa92c3cfeb9907093caef9b5767b7414bb58f90721ee75a335b1f04f7a2`.
Its adjacent verification checks all complete original artifacts.

Use `scripts/run_localized_franka.sh RUN qualify --closure-run CLOSURE
--contact-run ORIGINAL --radius 0.001 --levels 0 3 6 9` for geometry checks.
Its `solve` mode uses `--closure-run CLOSURE --local-contact-run ORIGINAL
--local-levels LEVEL --local-radius 0.001 --body-refinement 0 --cold-closure
--control-steps 64 --dt 0.00125 --steps 80`; add `--freeze-contact` for the
separated volume axis. Invoke through the pinned project cluster wrapper and
choose a fresh runtime directory below the established data root. Qualification
and solve source snapshots have separate input/executable before/after hashes.
