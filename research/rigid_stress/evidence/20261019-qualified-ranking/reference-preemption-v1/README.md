# Reference preemption and submission controls

See the [current report](../../../docs/FRICTION_KEYFRAME_COST_REUSE_20261019.md).
The manifest preserves59 byte/SHA256-verified payloads totaling473,402 bytes.

The original728537 allocations were preempted, then automatically requeued;
the replay failed the existing-output guard. Duplicate scheduler records retain
both events. Neither reference completed or passed its final momentum audit.
Legacy zero shell returncodes must not be treated as scientific success.
Original data-side partial histories and sources are unchanged.

The corrected helper disables requeue for its fresh-output-only protocol and
records explicit TERM/INT/HUP failures. Seven final stdlib/shell controls pass.
The first control harness had a parent/child readiness race; its partial outputs
and diagnosis are retained, alongside the corrected child-ready control.

Only immutable submission/source receipts from retry728589 are selected;
its active results are excluded. Both fresh allocations have `Requeue=0`,
with unchanged scientific task arguments,128GB allocations and480-minute caps.
This packet claims neither completed reference qualification nor goal completion.
