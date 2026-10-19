# Continuous pad friction and field sensitivity

The selected receipts in [manifest.json](manifest.json) retain exact source
paths, sizes and hashes. Complete fields, journals and full logs remain on
the data mount. The source pins and task arguments identify every run.

The1ms and0.5ms full-egg histories both pass normal and sustained tangential
mechanical windows. They retain fixed smoothing speed1e-4m/s, friction0.3,
free egg motion and1m/s2 horizontal body acceleration after the normal hold.
Independent audits check actual pressure, work-transfer, lagged friction,
current Coulomb cone, force/moment, inertia and solver exits. Complete-field
timestep differences stay below3.45e-7 and raw-peak differences below4.06e-8.

The retained motion decomposition shows small sustained translation and
rotation, with negligible nonrigid velocity. These are slowly creeping
mechanical states, not pinned or zero-velocity static equilibria. An independent
friction reference still needs compatible history and kinematics.

The separate500um0-versus12-patch preconditioner comparison changes complete
fields by at most1.02e-7. The newly admitted250um Shell field still differs
from the provisional250um solid by25.97% in the full tangential tensor and
41.87% in bending, despite0.00952% difference in the raw Mises maximum.
The solid reference remains spatially unqualified. No final error ranking,
friction-reference validation or matched cost claim is made here.

Use the campaign `tasks.json` with the established Slurm submitter and a fresh
data-side output directory; keep old results immutable. The engine friction
patch is linked from the adjacent
[regression checkpoint](../pad-refinement-friction-v1/README.md).
