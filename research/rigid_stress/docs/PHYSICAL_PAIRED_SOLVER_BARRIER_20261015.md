# Fine paired Newton recovery and the fourth barrier-distance control

The [full physical-validation objective](LOAD_MODEL_PHYSICAL_VALIDATION.md)
remains active. Exact block elimination now solves the previously failing first
dynamic equation of the171228-tet egg reference, after complete original coarse
state/force/field qualification. This linear correction alone accepts no fine
nonlinear state. Separately, the fourth U barrier distance completes all744
original stages and their independent audits. Neither result establishes spatial
or contact-pressure convergence, application/model accuracy or warmed native cost.

## Preserve the original fine physical problem

The earlier fine reference717975 retains all64 accepted held initializer
increments and their saved closure, followed by an actual first-dynamic
`MemoryError`. Its physical timestep remains0.00125s, with the original geometry,
mass, material, friction, gravity, jaw actuation and zero initial velocity.
Capture719038 reconstructs this unchanged first Newton equation from the saved
closure and writes the full matrix, original coordinate map, full gradient,
right-hand side and state arrays before a metadata serialization exception.
Both fine719038 and coarse719039 retain real wrapper1; independent postprocessing
719056/719057 audits these already-written arrays without inventing successful
capture exits or accepting a physical state.

The fine mapped residual is exactly the retained original0.11034389143016814N.
Its reduced matrix has856172 coordinates and75040966 stored entries. The last
eight coordinates, explicitly identified by the existing coordinate map, are
six rigid modes and two jaws. Their graph degrees are848255,846303,847402,
856171,856171,856171,201 and201, versus at most260 for an ordinary elastic
coordinate. These measurements motivate separating the declared border; they
do not prove a particular internal cause of the full-factorization failure.

Small rerun719060 checks the metadata integer-cast repair. Comparator719070
finds byte-identical coarse matrix/map files and identical values, shapes and
dtypes for every saved state array. The state archive bytes differ: for its
three long-double displacement arrays, differences lie exclusively in the
six nonvalue padding bytes of each16-byte x87 storage slot. Raw state-file
identity is explicitly false, and both actual producer codes remain retained.

## Qualify factorization changes before the large equation

Both candidates retain the complete matrix and every original coefficient.
The first permutes the elastic coordinates with seeded METIS, places the eight
declared border coordinates last and factors the full matrix. The second uses
the same elastic ordering with exact Schur block elimination:

\[
H=\begin{bmatrix}A&B\\C&D\end{bmatrix},\qquad
Z=A^{-1}B,\qquad S=D-CZ.
\]

The solver retains distinct nonsymmetric coupling blocksB andC. It solves the
eight-coordinate Schur equation and reconstructs the elastic correction using
the elastic sparse factor. It introduces no coefficient pruning, new penalty,
physical regularization, timestep or initializer change. Limited iterative
refinement, when used, evaluates the same full original matrix and existing
linear budget. Internal preconditioner vectors are not independently accepted
physical solutions.

|Candidate|Original1920-tet states|Maximum full-corner tensor relative change|Maximum full-quadrature tensor relative change|
|---|---:|---:|---:|
|Declared border last, full sparse LU|5|6.895933e-15|2.535164e-12|
|Exact eight-coordinate Schur elimination|5|7.020492e-15|2.760210e-12|

Candidates719045/719084, independent force audits719046/719085 and complete
field comparisons719047/719086 all close successfully. Every saved state
array is checked; both initializer archives are byte-identical to the original
qualified baseline. Both sets of five original force audits are retained for
each comparison. The maximum complete body/mapped force residual across the
Schur comparison is1.474750e-11N, within the unchanged1e-9N physical gates.
These full-field checks establish numerical equivalence on these original
states rather than physical convergence.

The qualified border-last full LU still fails on the captured fine equation
in719064 with actual `MemoryError`. Exact Schur test719087 instead completes:

|Fine linear diagnostic|Measured value|
|---|---:|
|Complete matrix dimension / stored entries|856172 /75040966|
|Elastic block dimension / stored entries|856164 /64979452|
|Elastic LU stored entries|810468709|
|Schur dimension /2-norm condition number|8 /400.341431|
|Original complete linear residual|3.992453e-13N|
|Original cached-Newton linear budget|1.103439e-12N|
|Observed ordering/factor phase wall time|87.566317s|

The matrix arrays remain unchanged. The measured factor phase is a single
reference observation, not a controlled speedup or warmed production cost.
The historical standalone summary's phrase “only the permutation” is too
narrow for its Schur case: exact block elimination also changes the numerical
factorization. The immutable report remains unchanged; this scope correction
is recorded explicitly in the publication summary.

Fine continuation719100 restores the same saved closure and uses the qualified
Schur solver for the original five dynamic steps. Its dependent independent
state audit719101, actual contact-coverage audit719102 and three-level
common-field comparison719103 remain execution work until their actual results
close. The comparator retains all158340 original common probes and raw global
fields for31668/64716/171228-tet covered CAD levels. A converged linear equation
does not bypass these original nonlinear, force, field or coverage gates.

## Fourth barrier distance: complete stages, still a separate error axis

Producer718912, independent auditor718913 and four-level comparator718914
complete with actual wrapper0 and closed copied/input sources. The0.3125µm
barrier retains the original845-tet body, pad5, geometry, material, mass,
initialization coordinates,0.05N per shoe, gravity, friction0.8, per-stage
1e-8m tangential mollifier and all12/604/64/64 original loading stages.
All744 current-friction/full-field audits pass the original1e-9N gates;
the maximum constrained residual is9.869805e-10N. Final body/mapped residuals
are8.317031e-11/1.156224e-10N. These loading stages are not physical time samples.

The four-level comparator checks the three adjacent pairs and two additional
baseline-to-finer pairs at every stage. It retains all absolute norms, original
denominators, finite material frames, source forces/moments, energy and raw
whole-volume bounds. It does not claim all six possible pair combinations.

|Barrier comparison|Final finer-normalized tensor L2 change|Final absolute tensor L2 (Pa·m^1.5)|Maximum absolute over744 stages /ordinal|
|---|---:|---:|---:|
|2.5→1.25µm|0.0699867%|0.0208763|5.151377 /599|
|1.25→0.625µm|0.1138120%|0.0339477|2.612841 /601|
|0.625→0.3125µm|0.0687428%|0.0205038|1.320112 /602|
|2.5→0.625µm|0.1828867%|0.0545512|7.757819 /601|
|2.5→0.3125µm|0.2404321%|0.0717134|9.062091 /602|

The new adjacent final displacement mass-RMS change is2.339814e-11m. The
new level's history-wide raw peak interval is[1689580.219156,1689774.279950]Pa
at ordinal616, versus final[112410.109835,112410.965020]Pa. As before, the
larger peak occurs during held initialization; neither final scalar agreement
nor the smaller last adjacent tensor change summarizes the complete path.
Near-zero relative denominators retain the established explicit-null rule,
with every absolute difference and raw extremum preserved.

This controls numerical barrier distance at fixed coarse spatial/contact
resolution. It does not close the hard-contact continuum limit, pressure
concentration, spatial discretization, loading-history independence or full
egg/Franka matching. The earlier editable figure remains the explicitly
three-level subset; the fourth level is reported here and in its full archive.

## Reproducibility and remaining acceptance

The bounded [solver publication](../evidence/20261014-physical/paired-dynamic-solvers-v1/manifest.json)
retains all14 closed attempts, real failure logs, exact coarse histories and
the complete fine captured equation. Its manifest SHA256 is
`199de963e2cecfcdbe03cb2e8467dc4ccdad54f99dd5e06d13e13525e9315391`.
The original coarse-baseline and fine-failure parent archives are fully
reverified; referenced actual runtime input hashes are checked without
duplicating parent-covered arrays. The549940568-byte matrix is preserved as
ordered32MiB chunks with exact full-file reconstruction. Runtime verification
checks995 archive entries,978 actual originals and one reconstructed large file.
The pending nonlinear continuation is explicitly excluded as successful evidence.

The [fourth-distance archive](../evidence/20261014-physical/fixed-history-barrier-four-v2/manifest.json)
has manifest SHA256
`2444ab8ab3775159c6f766a81784ec5ba0a1e9e034285fdc3f1fc585eeda2805`.
It verifies3152 archive entries and3152 actual runtime originals, retaining its own full producer/audit/comparison,
references the reverified three-level parent and preserves the first publication
check's actual failure: that check expected six pairs while the scientific
comparator declares five. The repaired check verifies the exact declared pair
identities; no experiment, scientific array or original gate was changed.
Exact job submissions remain in the archived command journals and scheduler
SubmitLine records. Reusable entry points are the capture/audit/factor wrappers,
`run_schur_metis_paired_reference.sh`,
`publish_paired_dynamic_solver_checkpoint.py`, and
`publish_barrier_thickness_checkpoint.py --kind barrier-four`.

All original A/B/C requirements remain. Next acceptance requires the closed
fine nonlinear reference and independent coverage/common-field results,
separated bulk/contact/geometry convergence, physically matched egg and
nonconvex contact, remaining load-model/local-island sensitivity, resolved
native replay versus feedback and warmed costs at demonstrated accuracy.
Production Quadrants and rigid dynamics are unchanged.
