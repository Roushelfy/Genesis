# Native production-throughput pilot, 2026-10-19

The [priority protocol](ACCURACY_AND_THROUGHPUT_PROTOCOL.md) separates physical
accuracy, stress-module replay and actual scene throughput. This campaign
starts the production path immediately, while fine-reference qualification
continues independently. It uses the existing native low-resolution Franka
benchmark and preserves the older, separately scoped throughput results.

## Fixed paired experiment

Job728966 runs15 fresh processes: B1/B8/B32, each with pure rigid, DPR maximum,
DPR full, SCR maximum and SCR full. Each batch's five processes run sequentially
on the same allocated GPU. Cross-batch hardware must be checked before treating
the curve as same-GPU scaling. Each process uses900 warmup steps followed by
three1200-step repeats at dt0.01s/substeps1, covering12s per repeat. The minimum
wall-time extension is disabled so paired scenes cover identical physical time.
These are warm repeats within one process, not independent-process repeats.

The fixed eight-condition bank/seed623016, level1 full hollow egg, mesh64
collider, MPR/patch-off, history0 and existing native controller are shared.
SCR uses intrinsic body_source_prior; DPR uses pressure. The measured path
includes native collision/dynamics, controller dispatch, required acceptance,
recovery and GPU max/full buffers at every physical step. It excludes research
journals, independent FEM audits, per-step host field dumps, trajectory-metric
kernels and intrusive profiling. Cold build/JIT and warmup are recorded apart.
There is no policy inference here: report env scene-steps/s, not RL transitions/s.

Default collider sources and the existing nominal6mm radius schedule differ
from the primary physical-pad full-source2mm configuration. Physical-error
cells for this production experiment remain pending. The approximately8%
normal-pad field errors cannot be attached to its rates.

## Source identity and cancelled first submission

The isolated source is pinned at92692aa612666b015ac4b506a39b9e7d3dfdf998.
All858 captured files were rehashed before the replacement submission,
including examples/speed_benchmark. The first job728958 was cancelled because
it was submitted before worktree checkout/index completion and captured an
empty source inventory. All three array elements are terminal CANCELLED;
none contributes an admitted timing. Its invalid patch remains on data with
an identity, not as a source of the replacement.

The submitter now rejects an empty tracked inventory or missing tracked
regular source files, retaining gitlink identities separately. Both rejection
controls and a complete858-file capture pass. The replacement uses a fresh
output directory, unchanged numerical code and identical planned workloads.

## Admission and next work

No pilot rate is published until terminal scheduler/task exits, actual source
hashes, identical step counts, runtime invalid-environment counters and output
configuration are checked. Post-run poses and warmup contact/force observations
can provide initial conformance evidence; complete path/max/full/environment
controls remain separate obligations. A successful prefix of a failure is not
a valid row, and physical-pad every-step friction failures remain unchanged.

Next measure representative changing-contact GPU-resident module replay,
separate production conformance, detailed bottleneck phases and sampled process
memory. The existing benchmark's recovery scope repeats one frozen actual
contact snapshot; it cannot stand in for the required history replay. Expand
batching after small-batch correctness and bottlenecks are established. B32
alone is not a practical large-batch capacity result. Sparse observation and
host readback are separate configurations still to be measured.

## Reproduction and storage

The [input packet](../evidence/20261019-qualified-ranking/production-throughput-pilot-inputs-v1/README.md)
contains the exact15 tasks, source manifests, job/workload scripts, guard
controls and cancellation evidence. Use its tasks with the existing submitter:

```bash
python -m research.rigid_stress.shell_comparison.submit \
  --source "$PINNED_NATIVE_SOURCE" --output "$NEW_DATA_RUN" \
  --tasks "$PILOT_TASKS" --python "$GENESIS_PYTHON" \
  --minutes 60 --concurrency 2 --cpus 4 --memory-gb 64 \
  --node-local-cache --keep-going
```

Use the recorded condition bank and source revision. The runner executes on
rtx-mid through GENESIS_IMAGE_VER=1_26 gs-srun. Outputs/logs are under the existing
project data directory; temporary compilation caches are isolated on the
compute node. No active environment, runtime data or compatibility link moves.
