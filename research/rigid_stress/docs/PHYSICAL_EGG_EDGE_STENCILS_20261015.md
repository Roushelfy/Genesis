# Egg edge contact and controlled local refinement

The complete [physical-validation objective](LOAD_MODEL_PHYSICAL_VALIDATION.md)
remains active. The7680-tet CAD reference now completes all five dynamic states
with independent force/global-field audits. Its newly exposed edge–edge contacts
are handled by an isolated reference-source patch, preserving the toolkit's
original friction magnitude and every original physical input/force budget.
Production Quadrants kernels and rigid dynamics do not change.

## Reference contact coverage

The failed uniform CAD jobs716502/716504 complete their64-target original held
initializers, then encounter a FaceVertex-only assertion in the reference's
centered force reconstruction. Direct observation of the7680-tet closure finds
three `EdgeEdgeNormalCollision` stencils. No dynamic state precedes that failure.
Job716502 ends FAILED11:0 with actual wrapper139;716504 ends FAILED1:0 with
wrapper1. Both retain their actual source/EXIT identities and initializer arrays.
Their pending audits716503/716505 are cancelled after corresponding recovery
work is arranged; pending cancellation produces no accepted audit evidence.

`generic_stencil_types.patch` applies only to isolated copies of the reference
force evaluator and independent force assembler. It supports the toolkit's
point–point, edge–point, face–point and edge–edge stencils. Centered local
coordinates retain the unchanged normal-potential gradient, including its
edge–edge mollifier and gradient. The friction magnitude comes from the original
unmollified scalar barrier law. This follows the pinned IPC1.6.0
[friction initialization](https://github.com/ipc-sim/ipc-toolkit/blob/v1.6.0/src/ipc/collisions/tangential/tangential_collision.cpp)
and [barrier-force definition](https://github.com/ipc-sim/ipc-toolkit/blob/v1.6.0/src/ipc/potentials/barrier_potential.cpp);
the [normal gradient](https://github.com/ipc-sim/ipc-toolkit/blob/v1.6.0/src/ipc/potentials/normal_potential.cpp)
is retained separately. Dividing a mollified normal resultant by its quadrature
weight would change the friction convention whenever the mollifier is below1.

Job716616 qualifies all four stencil types against an independently written
clamped-log scalar-force expression and24 directional derivative checks
(three directions × two step sizes × four stencils), all within1e-6 relative
derivative error. An explicit nearly parallel edge–edge fixture has normal
resultant/original friction-magnitude ratio0.64; the toolkit's tangential
constructor agrees with the unmollified expression. This fixture verifies the
reference-force convention, not physical contact convergence.

Jobs716727/716728 independently reconstruct all ten unchanged1920-tet
polyhedron/CAD dynamic states under that generalized arithmetic. Maximum
saved contact-force differences are2.09e-17/4.17e-17N, with unchanged1e-9N
body/mapped force gates. The actual recovered7680-tet edge contacts have
unit mollifier ratios in the observed states; the synthetic0.64 case protects
the non-unit branch rather than attributing a nonexistent change to this run.

## Recovered CAD state and fixed-polyhedron trend

Job716729 copies the7680-tet original initializer, reference/contact geometry
and node map byte-identically, starts from the same zero velocity and resolves
all five requested states. Independent job716730 checks every previous-state
integration, contact/friction term, consistent mass/armature, capped jaw PD and
raw global stress bound. Maximum body/mapped residuals are2.07e-13/4.11e-13N.
The final raw peak interval is846802.5831336–846802.5831338Pa, with five active
body contact nodes, three edge–edge stencils and whole-volume determinant lower
bound0.9999206170. The global peak is at
[2.82761087327e-5,−0.0223474216882,−0.00589914498614]m in reference coordinates.
These are short fixed-hand/two-jaw diagnostics, not the complete original motion.
The30720-tet failed initializer has a separately submitted generalized recovery
and independent audit; its live results are excluded from this checkpoint.

On the fixed original polyhedron, jobs716498/716499 complete and independently
audit the7680-tet five-state reference. Job716735 independently compares the
same6.25ms state on480/1920/7680 tetrahedra, with2400 identical physical probes,
identical actual finger inputs/initializer constraints and verified represented
mass/COM. Every refined boundary vertex and facet centroid lies on a supported
seed facet to at most1e-12m. These samples supplement the paired generator's
fixed-boundary/integral qualification; they are not a Hausdorff bound.

|Tetrahedra|Raw global peak lower bound, MPa|Active body contact nodes|Lumped support area, m²|
|---:|---:|---:|---:|
|480|0.156453|2|9.65136e-5|
|1920|0.284221|2|2.41280e-5|
|7680|0.569475|2|6.03186e-6|

The last pair still changes objective common-point tensor47.2215%, von Mises
23.8608%, objective displacement22.4789%, energy21.5366% and raw peak50.0907%.
Unexplained concentration remains visible. Three levels do not establish a tail
and these values do not prove a nonlinear singularity. Surface/bulk/contact
resolution still vary together on this particular branch.

The first comparison attempt716732 incorrectly required1e-12 per-component
agreement with a coarse historical force array evaluated using world-coordinate
FP64 arithmetic. That diagnostic fails at1.55e-11N. The corrected comparison
follows the existing matched-state protocol: independently reconstruct original
body/mapped1e-9N gates and retain the historical force-array difference separately.
No numerical-equivalence pass is claimed for that stricter historical comparison.
Both attempts and their exact source snapshots are preserved.

## Local CAD resolution by physical edge length

Fixed-round adaptive build716746 is cancelled after its second saved level
reaches108336 tetrahedra and a live40GB resident-memory observation. Actual
wrapper0 and source-after records are retained, while its final report is absent;
it is explicitly incomplete. Builds716784/716792 instead refine until declared
local CAD surface edge bounds are met. The latter includes analytic support
directions of the actual finger axes and both observed polyhedral/CAD closure
contact anchors. Anchors guide numerical mesh marking only: all collision
locations remain unrestricted and no physical contact radius/law changes.

|Paired level|Target local CAD edge, µm|Measured maximum, µm|Tetrahedra per family|
|---:|---:|---:|---:|
|1|100|70.9553|17340|
|2|50|35.4811|22152|
|3|25|17.7414|32340|
|4|12.5|8.87092|61812|

Both mesh families retain all six original held points and exactly matched
connectivity at each level. All earlier vertices survive exactly within each
family. The polyhedron branch preserves original boundaries, volume/mass, COM
and inertia; the CAD branch stays on the same declared normal-offset shell at
fixed material/density. Its global mass deficit remains approximately0.84508%,
because regions outside the fixed50µm marking caps retain the level3 background
approximation. Two thickness layers remain explicit. The cap size is a mesh
choice, not a measured contact radius or a contact-convergence claim.

Four actual CAD closure/dynamic references and separate independent audits are
submitted on these meshes. Their results remain live and excluded. Further work
must inspect contact coverage of the marked regions, volume/thickness and
contact-integration/regularization axes, global geometry error and complete
histories before treating a stress value as physically resolved.

## Reproduction and evidence

Use the existing pinned scientific environment through the matching shell
wrappers. Exact Slurm commands are in `meta/generic-stencil-progress-commands-v1.json`
and `meta/adaptive-cad-contact-commands-v1.json` under the established runtime
root. The opt-in modules are `qualify_generic_stencils`,
`resume_paired_egg_contact`, `compare_paired_polyhedron` and
`adaptive_egg_geometry`; original canonical reference files remain intact.

The immutable packet is
`evidence/20261014-physical/generic-stencil-checkpoint-v1/manifest.json`.
It preserves successful/failed/cancelled jobs, source before/after identities,
complete saved arrays and field/probe data, package versions and actual runtime
inputs. The separate verifier checks archived/decompressed bytes against their
runtime originals. This checkpoint does not complete the original application,
U/egg physical tails, remaining load-model/feedback comparisons or warmed cost
at independently verified physical accuracy.

The packet verifies1358 runtime originals and984 source identity records;
manifest SHA256 is
`8a0d5b6517cd24a84bd1e8f7e995629632aece89492cad8412229ed4f0d33768`.
An additional isolated sparse-ordering qualification is running against the
complete7680-tet five-state baseline, including every saved array, corner and
quadrature tensor and original force gate. Its numerical results and any cost
benefit remain unverified and are excluded from this packet.
