# Parallel-plate diagnostic figures

Claim anchor: `plates-admission-before-ranking`.

Main claim: equilibrium, interface admission and spatial convergence are
different requirements; the current plate data expose distinct failures and
do not establish a continuum stress-accuracy ranking.

Required materials are present: continuous force/position journals, endpoint
field arrays, common-column metric, solver receipts and the native rejection
dump. No missing evidence is filled by an inferred successful field. The
audience is the repository's numerical-methods reviewer; English labels,
183mm width, editable Matplotlib source and PNG/grayscale review first.
SVG/PDF are exported only after rendered review. No generated illustration.

| Figure/panel | Claim and evidence | Form | Role |
| --- | --- | --- | --- |
| mechanics/a | Same command does not imply same actual force at0.3s; QIPC/native/Shell journals | Time/actual-force curves, command dashed | Reveals incomplete Shell settling |
| mechanics/b | Equal-force plate travel differs; actual endpoint force and plate coordinate | Endpoint dots, log displacement axis | Native and QIPC0.1/0.3/1N; Shell hollow unadmitted dot |
| maps/a-d | Complete material-space outer tangential fields differ; missing/rejected and unequal-load fields are explicit | Unwrapped material azimuth/height scatter with one common logarithmic tensor-norm scale | QIPC/DPR/SCR/Shell; SCR remains blank with status40 |
| contact-loads/a-f | The reference load remains on polar nodes; a native reconstruction radius is not a measured reference footprint | Upper/lower rest-material x/y point maps with common normal-force scale | Actual QIPC nodes, native rigid source points and original Shell impulse/dt nodes; N, not pressure Pa |

Material passports: the figure receipt records hashes for every source journal,
start/result receipt and common-field NPZ. No smoothing, temporal averaging,
outlier exclusion or clipped color extrema. Displayed surface samples do not
replace raw full-mesh extrema in the tables. There are no statistical error
bars: these are deterministic individual runs, and numerical/spatial controls
remain separately tabulated. Shell is a diagnostic field at a different actual
load, explicitly labeled in its panel and caption.

The primary choice is curves for continuous mechanical history and common-scale
maps for spatial structure. A numerical table is the alternative for exact
errors; a mean-only bar chart or a three-method accuracy league table would
hide failure and reference uncertainty. Use blue/orange/green plus differing
line styles/markers for methods; a monotonic sequential color map for norms.

Body-text handoff: force closure is best for settled QIPC/DPR; original Shell
still carries only about0.0275N at0.3s. QIPC contact is concentrated on one
polar node per plate in both tested surface grids. SCR has no admitted static
field, so the corresponding map is absent rather than zero. The stress peak
and global tangent field remain strongly mesh-dependent; useful mechanics and
same-traction elasticity diagnostics do not qualify continuum stress errors.

QA: inspect color and grayscale at intended size, check labels/colorbar and
missing-field annotation, preserve source data trace and editable vector text.
Open issue: these are diagnostic report figures, not a qualified accuracy
comparison or submission-ready paper result.

Final source selection uses geometry-aligned native-equilibria-v2,
reference-loads-v1 and the original Shell tight-operator diagnostic. QIPC means
the separately pinned16-epsilon line-search roundoff candidate, as explicitly
defined in the results and caption; it is not unmodified-main qualification.
