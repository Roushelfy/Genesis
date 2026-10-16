# Eight-level finite-load reference results

This follows the [original finest-field qualification and mesh setup](PHYSICAL_EXTENDED_MATRIX_20261016.md).
All twelve additional five-case producers and six separate three-level auditors
have now completed. The eight-level summary719836 also completes naturally.
This closes those scheduled reference calculations while retaining the complete
[physical-validation objective](LOAD_MODEL_PHYSICAL_VALIDATION.md).

## Independent numerical evidence

Each separate auditor freshly assembles stiffness, consistent mass, rigid modes,
continuous boundary loads and the complete right-hand side. It reconstructs
every original corner tensor and every quadrature component, original common
probe, strain energy and whole-domain raw maximum on the original finest and
two added meshes. All90 fields pass the unchanged original budgets.

- Full nodal reintegration difference: exactly0N across all90 fields.
- Complete force residual: at most0.0795939695 of its original budget.
- All1553366925 quadrature tensor components are checked. Maximum difference
  from reconstructed corner fields is1.7366224192e-7Pa.
- All18 producer/auditor source closures match their copied source files.
  All336 distinct recorded input files were rehashed successfully.

The eight-level summary retains all240 original/new geometry-load-level rows,
all original probe coordinates and outside indices, raw peaks and positions,
energies, and every adjacent comparison. Fresh complete field auditing covers
the latest three levels. Earlier five levels retain their documented historical
audit scopes. Source closures, numerical agreement and small interior-probe
changes do not establish a continuum error bound.

## Observed final changes

Each range spans the same five declared loads: opposing, bending, asymmetric,
near-slip and multiple patches. Physical radius remains3mm, with the original
profiles, strengths, friction, material and original probe sets. Geometry and
volume approximation still change together in this CAD family.

| Geometry | Finest tetrahedra | Final common-probe tensor change | Final raw global-peak change | Cases with all five observables below3% on both latest pairs |
|---|---:|---:|---:|---:|
| Ellipsoid |315446|0.382–0.416%|2.618–16.933%|1/5|
| Capsule |326676|0.649–0.769%|2.890–14.097%|1/5|
| Thin neck |205437|0.529–1.967%|0.330–10.683%|1/5|
| U |294458|0.603–0.624%|0.000176–0.797%|5/5|
| Narrow gap |446367|0.537–0.548%|6.803–8.804%|0/5|
| Cavity |483087|1.930–2.145%|3.617–12.615%|0/5|

The five observables are the original probe tensor, probe von Mises stress,
gauge-aligned probe displacement, raw whole-domain maximum and strain energy.
The two-pair3% column is the previously declared diagnostic, with no fitted
limit or asserted continuum accuracy. All30 summary tails explicitly retain
`physical_convergence_accepted=false`.

For ellipsoid opposing loads, the final raw maximum falls from6089.0677Pa to
5215.6721Pa while the common-probe tensor changes0.404482%. Capsule opposing
loads retain14.097157% raw-peak change. These are unresolved physical/numerical
tails even though the independently reconstructed discrete solutions pass.
The next physical diagnosis must separate fixed-boundary volume approximation
from CAD/contact-trace approximation and inspect the raw hotspot locations.
Neither more agreement on the same assembled matrix nor discarding the raw
maximum answers that question.

## Actual native follow-up failure

Original native jobs719781–719784 attempted all five loads and four declared
models on the two added ellipsoid/capsule levels, at unchanged8 edge subdivisions
and10-point native quadrature. All fail before any accepted physical cases.
The first fails allocating the all-pairs port table. The others report CUDA
illegal-address failures, after warnings that index ranges exceed32-bit support.
Their scheduler exits are1/6/6/6, while saved program exits are1/134/134/134.
Every before/after source closure matches. All failures remain retained.

The production path constructs a dense `n_ports` by `n_ports` distance table,
plus visited storage. A boundary with more than100000 ports requires more than
80GB for the float64 table alone and exceeds the supported index range. A native
per-contact search on the unchanged graph is being qualified. Port resolution,
physical radius, model constraints and original acceptance budgets remain fixed.
Four impossible, unstarted dependent auditors719823–719826 were cancelled with
a preserved scheduler-state journal. Successful reference jobs remain intact.

## Reproduction and remaining work

Under the existing project runtime root, the relevant run names are:

```text
extended-finite-SHAPE-level1-v1
extended-finite-SHAPE-level2-v1
extended-finite-SHAPE-three-level-audit-v1
extended-six-shape-eight-level-summary-v1
meta/extended-finite-complete-audit-closure-v1.json
meta/native-finite-failed-dependency-cancellation-v1.json
```

`SHAPE` takes the six names in the table. The original setup report records
all producer/auditor job IDs and commands. Recreate the complete comparison
with `scripts/summarize_extended_finite_references.sh NEW_RUN --runtime-root P`.
The retained `meta/extended-six-shape-eight-level-summary-commands-v2.json`
records the actual summary submission after checking completed dependencies.

The complete60 new field arrays, six separate15-field audits and combined
240-row summary now have bounded immutable publications. Each archive was
verified against its actual runtime originals, including exact reconstruction
of every original file at least50MB from ordered32MiB byte ranges. Earlier
mesh/qualification/reference arrays remain in reverified parent archives.

| Complete field archive | Manifest SHA256 |
|---|---|
| [ellipsoid](../evidence/20261014-physical/extended-finite-ellipsoid-fields-v1/manifest.json) | `aeb012371abac7bf30b28ff11b0e782563fd25c5e7219bcd3ae0242fe652a506` |
| [capsule](../evidence/20261014-physical/extended-finite-capsule-fields-v1/manifest.json) | `8c915c55c14f422ad4a3db588cb82a194429f140034ab648fd8c0e5e638234b5` |
| [thin_neck](../evidence/20261014-physical/extended-finite-thin_neck-fields-v1/manifest.json) | `f6875dfe8658855940f0a007b3fe169fb6df5810296f7c90a5374958f908f67b` |
| [u](../evidence/20261014-physical/extended-finite-u-fields-v1/manifest.json) | `48525fa194eb2ce0dced45eef26047428a95bef5713d9c718f2007cdbb9e4a80` |
| [narrow_gap](../evidence/20261014-physical/extended-finite-narrow_gap-fields-v1/manifest.json) | `5349e4a7596702d37449232412c181140a4e4ce1bde3e6f1a33a842c8518a8c4` |
| [cavity](../evidence/20261014-physical/extended-finite-cavity-fields-v1/manifest.json) | `85cbe666065489aa9e1ea66e02564c14c894564e4036a5f144773480b9ef2f66` |
| [Eight-level summary](../evidence/20261014-physical/extended-eight-level-summary-v1/manifest.json) | `14f8af8b163409aae2ab07a4f32d3872732ffed48e044b2a8ad99cbf4358aaa8` |

Updated eight-level figures remain required. The
[native path-capacity checkpoint](PHYSICAL_NATIVE_PATH_CAPACITY_20261016.md)
preserves original failures and qualifies the bounded search at its stated
numerical scope. Native model/scatter/elastic replays still require fresh
independent audits. Separated geometry/volume/contact/distance axes, patch and
island diagnostics, resolved egg/Franka and interior-hotspot comparisons,
replay versus feedback, and physically qualified warmed costs remain required.
