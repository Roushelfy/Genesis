# Native continuous-wrench diagnosis and original-point termination

Source base is `87d04e451f18879836667b47632c654012d53ef9`. This follows the
[feasible-fit and continuous incompatibility diagnosis](NATIVE_WRENCH_DIAGNOSIS_20261011.md).
The active nonconvex goal remains incomplete: the aggregate-wrench modeling
decision, source/refinement work and matched four-scope performance remain open.

The [publication manifest](../evidence/20261011-nonconvex/native-continuous-wrench-checkpoint-v1/manifest.json)
records 376 copied raw artifacts (5,020,288 bytes) and four data-side runtime
artifacts (85,152,235 bytes). Original and published bytes pass SHA256 checks;
large snapshots and the full trajectory summary remain in the project data
directory under the storage policy. Manifest SHA256 is
`5bdad10fa5bd18f18409c6ffeda59b489c61631f36df79258874927b08288ef4`.
The first publisher rejected a large snapshot before copying any files; its
source and failed log are retained. The corrected publisher records and verifies
large artifacts in place.

## Original-point numerical budget

The dual constraints remain conditioned about the reference anchor with the
unchanged physical radius. Internal termination now checks the moment error
about the unchanged original force point:

`g_M_original = g_M_anchor - cross((p-anchor)/radius, g_F)`.

Both force and original-point moment errors are normalized by the declared
force budget scale. Internal tolerance remains 9e-9, with 128 evaluations and
the previously validated adaptive Newton/trust step. Final original-point
force/moment acceptance remains 1e-8, complete equilibrium including six gauge
rows remains `max(1e-11 N, 1e-8 ||rhs||)`, and stress remains
`rtol=1e-4, atol=1e-3 Pa`. The elastic operator and output lifecycle are unchanged.

This removes an anchor/original-point mismatch in termination. It does not
relax the independently recomputed final sample/nodal wrench or local cones.

## Production six-component diagnostics

`stress/compatibility.py` adds a bounded native six-dimensional polar search
only for already validated unresolved traction fits. It uses uniform sample
constraint penalties and the existing native Coulomb projection/Jacobian and
small-matrix direction solver. A strict polar correction changes the candidate
certificate rather than the load. Sample separation is checked independently
of convergence of the penalty subproblem.

A separate native kernel proves separation on a conservative continuous support
superset. It includes every actual material-boundary triangle intersecting any
unchanged Euclidean source ball, including branches excluded by intrinsic paths.
The local polar is convex on each affine face; subtriangle vertex maxima bound
its entire interior. AABB distance conservatively excludes subtriangles outside
the ball. Floating allowances include coordinate-subtraction operands.

Depth-first quarter subdivision has depth budget 8 and total visited-node budget
16,384. Its exact required pending queue has `1+3*8=25` entries per built
contact/environment slot, rather than an arbitrary padded queue. The diagnostic
state is typed SCRATCH state and participates in the normal stress data lifecycle.
All numerical search, geometry bounds and status updates are Quadrants kernels.

Depth, workspace or node exhaustion leaves continuous compatibility unresolved;
it never establishes physical incompatibility. The proof also recomputes the
current target margin and final original-point budget. An old stored margin
cannot certify a changed target inside that budget. The transformed polar error
bound is proportional to

`||r_F + cross(r_M,(p-anchor)/radius)|| + ||r_M||`.

Successful continuous separation uses `CONTINUOUS_WRENCH_INFEASIBLE=6` and raw
status36, distinct from continuous force-only status35 and sampled status34.
The normal rigid solver error reports the link, environment, original contact,
radius and actual friction. No original force, moment, radius, friction, source
or rigid dynamics is modified. A failed recovery raises the existing error;
it does not return an accepted incomplete stress field.

The search is sufficient rather than a universal decision procedure. A strict
continuous polar ray may not exist for a conservative superset, or bounded
search may not find one. Such cases remain unresolved. Source ambiguity,
unresolved intrinsic integration and numerical fitting retain their distinct
existing semantics. This work does not select aggregate partitioning or a
different traction law.

## Same-mesh regression and independent proof

The complete combined suite passes 51 CPU cases in 411.91 s and 51 CUDA cases
in 740.39 s, with unchanged source hashes before/after execution. It includes
all previous source/footprint/vector-load/signed-scatter/full-equilibrium
fixtures and the original-point termination change.

The new 2324-byte step602 fixture preserves both exact original sources. Its
regression exercises the saved incompatible component, another feasible
original component, a separately declared feasible force/free-moment input,
an inactive environment, proof depth/node exhaustion, current-budget rechecking
despite a stale stored margin, and release without environment leakage.
The manufactured feasible input is a test control; it does not change live
per-original-component acceptance or substitute a pair total in production.

Independent CPU FP64 triangle/disk clipping and analytic circle-arc bounds audit
the newly generated production ray on the same actual material geometry and mu.

| Production ray backend | Search evaluations | Visited subtriangles | Target margin | Independent continuous polar upper bound |
|---|---:|---:|---:|---:|
| CPU | 79 | 52 | 2.006010599e-3 | -2.095701089e-4 |
| CUDA | 42 | 52 | 2.045878812e-3 | -1.203061162e-4 |

Both rays strictly separate the exact original component. The other original
component remains accepted. The offline audit has before/after source records,
unlike the earlier post-run-only continuous-bound records. This is a geometric
certificate audit on the unchanged low mesh/Q10 law, not physical mesh or
quadrature convergence.

The first diagnostic regression fails one new case on both backends while
twelve old cases pass: a `_` loop variable reused a closest-point return
placeholder, causing a Quadrants syntax error. Renaming it fixes compilation.
The failed frozen epoch is retained. Fixture extraction initially attempted
bitwise witness/anchor equality; their existing roundoff difference fails that
check. The corrected extraction checks operand-scaled FP64 geometric roundoff
and preserves both original arrays unchanged. No wrench/stress budget changes.

## Actual B1024 trajectory

The ordinary native scalar pipeline, mesh64, Q10 squared surface fans, s8,
declared pair support, GJK patch contacts, seed623201 and varied real friction
complete 602 batch steps, or 616,448 environment transitions. At the next step,
the rigid solver reports exactly env66/original contact1/status36:

`original wrench is incompatible with the continuous reference-surface friction cones inside the physical radius`.

Radius remains 4.69197214 mm and actual combined mu remains 0.447395472. The
new failed input's position, force, normal, radius, friction, witness, anchor,
anchor face, geom, pair and method are bitwise identical to the prior exact
step602 capture. This checks the intended current input rather than a later
overwritten scene frame.

The trajectory retains all 1024 environments and four independent CPU full-field
captures at 0/160/320/480: 4096 complete fields, with largest recorded full
equilibrium residual 1.580504619e-11 N, within each original budget. It passes
the former step517 fitting failure. The abort is fully disclosed. This is
every-step correctness/diagnostic evidence with oracle/JIT overhead, not
full-horizon B1024 acceptance, a throughput measurement or a plateau.

Hardware remains RTX PRO 6000 Blackwell Server Edition, 97,887 MiB, driver 595.71.05,
Genesis 1.4.3, Quadrants 1.3.3, container 1_26. CPU checks use AMD EPYC 9454.
Reproduce the combined suite through the existing
`bash research/rigid_stress/scripts/check_nonconvex_contact.sh cpu` and `cuda`
commands in allocated nodes; its source/fixture patterns include the new test.
The archived launchers additionally include lifecycle checks, the native
certificate export/audit, and the exact1024-environment live workload.

Production cooperative scheduling, failure compaction and end-to-end profiling
remain required. The new diagnostic dispatch/storage and fallback cost must be
included in the next matched baseline. Legal-contact robustness and the full
large-batch performance goal are not claimed complete.
