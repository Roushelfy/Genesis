# Native hollow-egg admission and geometry gap

The actual native Franka/hollow-egg rollout rejects its first loaded step.
Separate four-contact controls on the unchanged level1 hollow mesh recover
complete fields with order16 boundary integration. These results distinguish
the contact geometry gap from finite-shell elasticity. The full
[hard-contact goal](AUTOMATIC_CONTACT_AND_THROUGHPUT_GOAL_20261020.md) remains active.

## Actual Franka rollout

Slurm739120 uses uncompressed source795805290, four independently initialized
environments, level1 geometry, float64 and the existing moving controller.
The controller's manual pressure radius input is disabled for the Hertz route.
Steps0–9 are unloaded. The first loaded frame, step10, fails in all four
environments. Input and friction processing pass, but field status2 rejects
the source geometry before elastic recovery. No loaded update is admitted.

The retained frame and read-only inspection locate the cause. Each resultant
acts at a coarse mesh vertex. Its smooth estimated inward normal differs from
the witnessed triangle's inward normal by22.63569 degrees. The frame is
orthonormal to9.23e-16, the center lies on the material boundary to6.82e-19m,
and the entire material remains in its inward half-space. The existing
`1e-10` normal-alignment gate therefore rejects a real geometry mismatch.
Changing equilibrium or friction solver tolerances cannot fix this mismatch.
The original rejection, source identities and complete frame arrays remain.

## Finite-shell component control

The component control places four separate normal loads at actual outer-facet
centers on the same level1 volume, with forces0.20/0.12/0.18/0.10N and declared
relative principal curvatures50/80 per metre. Its finite-radius law, all element
corners and89 moving probes run in native Quadrants. This isolates recovery
from the Franka source association and smooth-contact approximation.

| Boundary rule | Maximum wrench error | Native accepted environments | Independent complete field check |
| --- | ---: | ---: | --- |
| Order8 | 0.0284832 | 0/4 | Refused before a usable field |
| Order16 | 0.0000405193 | 4/4 | Pass on CPU and CUDA |

The unchanged wrench gate is`1e-4`. With order16, independent P2 derivatives
and consistent-mass inertial relief, combined with the same prescribed Hertz
particulars, differ by at most`3.59e-11` at element corners and`2.22e-9` over
the complete probe bank. Both remain below the original`1e-7` field gate.
This is a same-load numerical control. It supplies no independent pressure,
curved-contact, frictional shell error or matched scene-throughput claim.

The original order16 reader round-tripped surface probes through world
coordinates. At contact edges, coordinate changes below`3.30e-18m` changed
its reference field by up to`1.36e-7` relatively. The corrected reader uses
the probes' declared exact local coordinates, matching the existing production
definition, and separately records that round-trip sensitivity. The original
field assertion failures remain preserved.

## Repair and reproduction

Source[`d26b5b3d4`](https://github.com/Roushelfy/Genesis/commit/d26b5b3d46efe614ed715fef293f8cc49bcdf0c7)
repairs a standalone normal-component initialization bug introduced by the
optional shear container: a zero-length scalar Quadrants array was invalid.
Unused scalar-shaped placeholders follow the existing inactive-buffer
convention. Actual friction routes already supply their own history buffers.
Existing rotated normal-component/max/full controls pass on CPU and CUDA.
The Hertz option documentation now describes the implemented vector history.

The first shell worker also retained an array-axis error before reaching the
engine. Campaigns`hard-native-hertz-shell-admission-v1` through`v4` preserve
that failure, the initialization defect, the order8 rejection and both versions
of the independent reader. Slurm739126/739131/739135/739143 record their actual
terminal states. Inspection of the Franka frame completes within739131 even
though that group's later shell tasks fail.

Use the captured tasks in the
[evidence packet](../evidence/20261019-qualified-ranking/hard-native-hertz-shell-admission-v1/README.md)
with `shell_comparison/submit.py`, the stated immutable source, explicit fresh
data outputs and node-local caches. Scientific execution uses Slurm`rtx-mid`,
image`1_26` and the existing data-side physical environment. Full arrays stay
on data. `check_live_hertz_franka`, `check_native_hertz_shell` and
`inspect_hertz_franka_admission` are the three new research entrypoints.

Remaining work is a useful curved-source/finite-boundary approximation with
matched physical error, cheaper accurate near-boundary integration, and accepted
moving Franka max/full throughput. Relaxing the face-normal gate would conceal
the present geometry mismatch, and order16 alone does not address it.
