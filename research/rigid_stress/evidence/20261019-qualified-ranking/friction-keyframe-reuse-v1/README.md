# Friction keyframe cost and response-reuse checkpoint

See the [report](../../../docs/FRICTION_KEYFRAME_COST_REUSE_20261019.md).
The manifest retains329 size/SHA256-verified payloads totaling6,028,562 bytes.
All selected execution groups are terminal; the expected failing exit codes
are checked by the retained publication source. Original data-side files are
unchanged. Files above1MB, including the final rejected SCR field, remain on
data with paths and hashes in `large_files_retained_on_data`.

- 728543: DPR/SCR/repaired250um Shell complete1000-step histories and own-run
  primary-cost audits with full observation every200 steps, two warmup levels
  and three measured warm levels. This does not admit native every-step mode.
- 728468 group1: original every-step SCR failure after37 written rows, retained
  separately from the earlier packet's still-running publication cutoff.
- 728529: unchanged actual observer-call sequence, immutable before-call input,
  and failed observation38 with complete rejected field. This is intrusive,
  not primary timing; all46 sources return status34.
- 728542/728575/728586: read-only input/final-packet inspection and conservative
  surface-cone certificates. The pure-tangential SCR force is incompatible with
  those unilateral cones. Tiny residues are not a meaningful sustained load.
- 728491: optional full-compliance response reuse,24 contact controls, StVK
  controls, complete400-step history, momentum audit and field/common/raw-peak
  equivalence. Predeclared targets and a reproducible metadata gate are retained.

Use captured task lists through the established Slurm workflow with fresh
outputs on data. Exact task inputs and engine overrides remain in the receipts;
captured analysis inventories identify all source modules. Reusing a result's
path in a task means analyzing that original run, not a new rerun automatically.
`publication-source.py` verifies and copies terminal evidence; it does not run
scientific calculations. `meta/audit-response-reuse-v1.py` checks completed
Slurm-produced results against the original quantitative targets.

Fine nonlinear reference trajectories are deliberately excluded while running.
This checkpoint provides no qualified friction continuum reference, native
every-step friction error/cost rank, or whole-goal completion claim.
