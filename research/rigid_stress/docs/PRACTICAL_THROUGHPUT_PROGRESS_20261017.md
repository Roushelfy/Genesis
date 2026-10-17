# Practical throughput progress, 2026-10-17

The [large-batch goal](PRACTICAL_THROUGHPUT_GOAL.md) is **open**. This is an
initial measured baseline and optimization checkpoint, not a capacity plateau
or a final recommendation. The previous application accuracy limitations remain.
The [machine-readable checkpoint](PRACTICAL_THROUGHPUT_CHECKPOINT_20261017.json),
[frozen contract](PRACTICAL_THROUGHPUT_BASELINE_20261017.json) and
[scheduler export](PRACTICAL_THROUGHPUT_SCHEDULER_20261017.txt) retain the initial
evidence; later submitted jobs below are explicitly still pending verification.
The [subsequent experiment checkpoint](PRACTICAL_THROUGHPUT_EXPERIMENTS_20261017.md)
closes B16384, supplies measured baseline scaling/memory curves, expands moving
profiling to a complete controller period and records the small or negative
results of the initial optimization candidates. Bounded candidate full-field
and complete-grasp trace comparisons are also closed there.
The merged-capacity candidate described below is now superseded.

## Frozen workload and provenance

Runtime root, relative to the data root in `DATA_LOCATION.md`:
`runs/20261017-practical-throughput/`.

`meta/baseline-contract-v1.json` freezes the exact previous arguments,
stress options, conditions and source identities before production edits.
Its SHA256 is `ca1cd9a6a2d321a671b511524e2856209b76760826f4afaa2e76e6844a39b084`.
The baseline reuses the immutable `practical-native-application-v1` snapshot
from the physical-validation run root. All899 files matched the current
production/application source before editing; the snapshot manifest SHA256 is
`2da41428d67e99c88fb85561ba7112a7a2e7ac5ae44892ed65229a8ca6970b5a`.
The frozen eight-condition bank SHA256 is
`1d40197808823971cb0f5d76e51da266aaaae6cb28453467392a3e591e4bacfa`.

The primary law is intrinsic `body_source_prior`, with the480-tet P2 full
shell/2430 DOF, nominal6mm radius schedule, quadrature10, ports8, history4,
max output, mesh64 collider, MPR and patch-off. The original timestep0.01s,
one substep, material, mass, scripted controller, condition diversity and
policy are retained. Each headline scope warms900 steps then measures three
repetitions of2400 steps, including release/reset and its stated control/policy
costs. The policy scope includes inference, not full RL training.

Every benchmark child checks its immutable snapshot before and after execution,
records actual program/wrapper exits and verifies that no snapshot files were
added or removed. Parent runs also close the condition-bank hashes. Headline
eligibility additionally requires terminal successful Slurm accounting. A
completed child in a running parent remains provisional.

## Closed baseline measurements

Single NVIDIA RTX PRO6000 Blackwell Server Edition GPU per allocation,
101975851008 device bytes, driver595.71.05. Values below are medians of three
valid env-step/s measurements, with zero invalid environment steps and zero
scatter overflows in every repetition. The four scopes at each width execute
on the same allocated GPU, in separate processes.

| Environments | Rigid | Rigid + stress | Policy + rigid | Policy + rigid + stress |
| ---: | ---: | ---: | ---: | ---: |
|32|11293.87|2615.39|9821.80|2524.08|
|256|84845.51|16803.28|75216.98|16271.65|
|2048|434466.83|45118.75|398893.63|44814.51|

Jobs724933/724934/724935 completed0:0. Baseline B8192 job724936 remains
incomplete, although its live child measures52658.16 env-step/s median and
12067143254 native-buffer bytes with no invalid steps. This is provisional
until the enclosing four-scope job closes.

Baseline B32768 job724962 **failed** with an actual CUDA allocation OOM during
profile growth in warmup, before live timing. Its successful rigid scope does
not establish robust-workload capacity. The pending larger B65536 job724972
was cancelled after verifying its pending state; this is not an OOM observation
or a physical pass. B49152 memory diagnostic724971 remains separate. Baseline
B16384/B24576 jobs724974/724975 and the workspace-size2 B32768 trial724976
refine the measured capacity endpoint and test whether the memory reduction
improves usable batch width.

At B2048, native stress buffers account for3071620694 bytes. An existing
`contact_block_size=2` ablation grows to the complete actual union of9 contacts,
accounts for2371923542 bytes and measures45506.49 live env-step/s median
(job724940, completed0:0). This is a22.78% native-buffer reduction and only
about0.86% rate difference on different comparable allocations; it does not
yet prove a throughput gain or improved capacity. These counts exclude other
allocators and are **not peak process VRAM**.

## What profiling establishes

The separate moving B256 Torch/CUPTI trace (job724937, completed0:0) covers50
steps after the timed rollout. It records18306 GPU kernels,2924 HtoD and206
DtoH copies. Its50-step GPU duration sums include102.946ms in primary
cooperative traction fitting,96.887ms in the inverse solve,51.741ms in cached
traction scatter,36.677ms in source resolution and19.492ms in frontier fan
sampling. The trace window is not a complete-episode average. Nested CPU,
runtime and GPU intervals overlap; their sums cannot be added.

The separate B2048 frozen-snapshot profile (job724938, completed0:0) measures
20.355ms for the intrinsic contact pipeline,1.635ms for the linear solve,
0.669ms for complete residual checks and0.438ms for the global peak.
These synchronized repeated stages are diagnostics, not moving headline costs.
The frozen contact-count histogram is `{1:512,4:256,5:256,6:512,7:256,9:256}`.

Every10th warmup step at B2048 records contact counts0–9 (mean3.282), fit
evaluation counts0–10 (mean2.027), zero elastic correction/fallback and zero
integration retry counts. The observed history-hit fraction is37.05%.
These warmup samples do not establish a full-trajectory distribution.

## Optimization experiments

The first candidate skipped the packed-count read only when the workspace
could hold the entire raw contact capacity. The actual application has750 raw
slots and a16-contact workspace, so this condition never activates. Its paired
50-step trace (job724961, completed0:0) has the same206 DtoH copies and18306
kernels. It provides **no application speedup evidence**. The immutable
`complete-union-count-v1` snapshot and its submitted comparisons are retained;
the candidate is superseded.

The current candidate merges the packed-count and initial profile-capacity
reads. It prepares provisional profiles, reads both required capacities once,
and rebuilds the complete union when necessary **before** traction fitting or
load assembly. It retains exact growth, CPU/noncompact paths and the original
traction/residual tolerances. No contact is omitted from an accepted solve.

Its formatted immutable source is `source-snapshots/merged-capacity-readback-v2`,
manifest SHA256
`d09856ecf82468db5eef9fe8a60fee003b21d7ecf237c9a74f372d0c1e1ad0cd`.
Tests explicitly extend complete release/recontact coverage alongside existing
dynamic workspace growth, invalid-member diagnostics, full fields, max/full
observations, checkpoint restoration and partial resets. Jobs724966/724967
(CPU/GPU),724968/724969 (B32/B2048 four scopes) and724970 (moving trace)
were submitted. CPU job724966 completed0:0 with all three selected lifecycle
tests passing, including full release/recontact. GPU validation and headline
speedup remain **unproven** at this checkpoint.

Moving trace724970 completed0:0 and source closure passes. Its50-step window
has156 DtoH copies versus206 in the baseline, and174 calls each to runtime memcpy
and stream synchronization versus224 each. It adds50 collection kernels and50 HtoD argument transfers.
Its separate one-repeat rate is16690.68 versus16643.53 baseline env-step/s;
this small difference is insufficient evidence of a material speedup. Final
selection must account for the added launch as well as the avoided fence.

## Reproducing and continuing the measurement

The exact Slurm arguments and runnable orchestration scripts are in
`meta/*-command.json` and `meta/*.sh` under the runtime root. All science uses
Slurm, `GENESIS_IMAGE_VER=1_26 gs-srun` and the established physical environment;
all outputs/caches are on the data mount. The immutable snapshot's
`research/rigid_stress/scripts/run_practical_native_benchmark.sh` provides
source closure. Existing archived application inputs need not be regenerated.

The metadata-only summarizer accepts a saved scheduler export:

```bash
sacct -j JOB_IDS --format=JobID,State,ExitCode,Elapsed,NodeList -P > "$task_root/meta/sacct-checkpoint.txt"
/usr/bin/python3 research/rigid_stress/scripts/summarize_practical_throughput.py \
  --root "$task_root" --sacct "$task_root/meta/sacct-checkpoint.txt" \
  --output "$task_root/analysis/checkpoint.json"
```

Use a fresh output name. It preserves incomplete observations and labels
intrusive diagnostics, checks wrapper/source/scheduler closure, and admits
only matched successful unprofiled scopes as primary headline points.

Remaining delivery work: finish and refine large-batch/capacity measurements;
measure peak process memory separately from native allocation counts; validate
and select material optimizations with ablations; complete the CPU/GPU and
optimized large-batch checks; retain a bounded complete-grasp check and the
whole-episode versus named-phase distinctions; deliver scaling/memory curves,
matched final rates, runnable final commands and a measured plateau/bottleneck
report. Existing7/8 lift-window discrimination remains distinct from1/8
whole-episode discrimination; force choices and reporting windows stay fixed.
