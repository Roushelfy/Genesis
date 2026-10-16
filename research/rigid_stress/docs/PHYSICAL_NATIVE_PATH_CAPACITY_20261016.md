# Native intrinsic path capacity repair

The [eight-level finite-load follow-up](PHYSICAL_EIGHT_LEVEL_REFERENCES_20261016.md)
exposed a production capacity defect before any accepted fine native field.
`StressSurfacePaths` allocated all-pairs float distances and visited flags.
The111286-tetrahedron ellipsoid has125710 ports and5074032 directed graph edges.
Its float64 distance table alone requires126424032800 bytes, with15803004100
entries exceeding Quadrants' supported signed32-bit indexing.

## Native change and preserved problem

The shared graph retains every original port, edge, length, triangle identity
and8-way subdivision. Graphs with at most4096 ports keep their precomputed
distance table and original contact arithmetic. This bounds the float64 table
to128MiB. Larger graphs use native multi-source Dijkstra with a binary heap,
seeded by every port of the witnessed anchor triangle. Each contact/environment
has its own heap, inverse positions and distance vector.

Search stops extending paths at the current physical radius. Every edge and
final target segment has nonnegative length, so discarding graph paths of
length at least that radius preserves every compact-profile contribution.
The verified planar walk and exact adjacent-triangle unfolding remain. The
graph still supplies a witnessed upper bound on continuous intrinsic distance,
with the same port-discretization error. Physical radii, profile laws, source
constraints, normals, friction and acceptance budgets remain fixed.

For this125710-port qualification, fifteen contact/environment slots need
15085200 bytes for distance vectors and15085260 bytes for heap indices/counts.
These are path-workspace figures, excluding shared graph, elasticity and other
buffers. The standalone all-pairs query helper requires precomputed paths;
production footprints dispatch through the per-contact search as needed.

## Verified numerical scope

CPU719886 and CUDA719887 each naturally complete12 original nonconvex
profile/vector-recovery cases with matching before/after source hashes. The
expanded existing test runs both distance algorithms on U and cavity fixtures,
three integration/storage modes and three independent environments. Original
assertions cover every profile sample, excluded nearby sheets, constrained
tractions, signed nodal scatter, force/moment balance, full tensor, raw peak,
gauge residual, streaming, moved anchors/radii, release and exact restoration.
No assertion or numerical tolerance is weakened.

CPU719910 also completes the actual125710-port graph check. An independent
SciPy solve builds edge lengths freshly from port coordinates and introduces
explicit directed anchor nodes. It compares all5656950 entries over three
stages using all five original finite-load snapshots. All within-radius
distances match exactly in this CPU run, support membership matches at every
port, and restoring original contacts restores the full original distance
array exactly. Changed-radius/anchor/release stages are numerical lifecycle
controls, with no physical convergence claim.

An isolated intentional mutation scales graph edges by1.1. Job719924 naturally
fails the unchanged independent profile assertion:112 of2880 samples differ
at the original2e-12 relative/2e-20 absolute tolerance. The mutation touches
only a separate data-side source copy. It confirms the existing physical
profile check detects a wrong graph search.

## Preserved failures and source identity

Original native jobs719781–719784 retain their complete OOM/illegal-address
logs and source closures. The first CPU development attempt719871 retains its
actual static-expression compilation failure. Its correction branches on tensor
rank, which Quadrants can resolve statically.

The first GPU test719872 started during an attempted pending-job cancellation.
The cancellation precondition failed, but a subsequent edit incorrectly proceeded.
Although its numerical program returned0 with12 passing tests, source closure
failed and Slurm returned1. This run is excluded from qualification evidence.
The second CPU/CUDA runs independently pass with matching sources. The race
journal and mismatching before/after records are retained explicitly.

Long follow-ups now execute a fixed, fully hashed data-side snapshot of the
complete Genesis package, package assets, research drivers and wrappers. Each
run verifies the snapshot before and after execution and uses its own cache.
The wrapper records numerical-program and complete-wrapper exits separately.
It includes source/snapshot verification failures in the final wrapper exit.

The [immutable capacity archive](../evidence/20261014-physical/native-intrinsic-capacity-v1/manifest.json)
contains6396 entries, all checked against actual runtime originals, including
the full source snapshot, successful tests, excluded/intentional failures and
launch/cancellation journals. Its manifest SHA256 is:

```text
9fc23ff51bfd0b6c91c24806c6dd599277140ed31f59f254aecdf855c61fb407
```

The first metadata-publisher attempt also failed because a verifier argument
was missing. Its source, traceback and partial scheduler metadata remain in
the archive. The corrected attempt completes full runtime-byte verification.

## Pending evidence and original objective

Large CUDA qualification719911 now naturally completes with matching source
and full snapshot verification. All5656950 entries have matching support
membership and maximum within-radius distance difference8.6736173799e-19m.
Exact original restoration also passes. The
[separate CUDA supplement](../evidence/20261014-physical/native-intrinsic-large-cuda-v1/manifest.json)
reverifies the original capacity archive and all actual new runtime bytes. Its
manifest SHA256 is `794c919d364fac0a0df84adec52e766614d78b07bfe863c91a4a5093fe10b0fc`.
The original archive retains its then-current queued status.

This releases four new five-load/four-model native jobs719912/914/916/918
and independent auditors719913/915/917/919. Both actual added ellipsoid/capsule
levels retain original8-port resolution,10-point native quadrature, all models,
physical inputs and numerical budgets. Pending jobs establish zero accepted
physical cases. Additional pair-union and self-balanced source-prior regressions
719945/719946 cover both algorithms. The CPU run completes all8 cases; CUDA
remains queued. Their separately scoped evidence will be retained after closure.

This checkpoint establishes bounded native path storage with the numerical
scope above. Actual fine traction/scatter/elastic fields,
physical geodesic and contact convergence, model/island bias, resolved
egg/Franka and interior-hotspot references, feedback and physically qualified
warmed cost remain required by the complete original objective.
