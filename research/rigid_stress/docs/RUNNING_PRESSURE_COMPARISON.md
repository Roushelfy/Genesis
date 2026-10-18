# Reproduce the pressure/intrinsic comparison

Use the [comparison report](PRESSURE_COMPARISON_RESULTS_20261018.md) for decisions
and the [evidence packet](../evidence/20261017-pressure-comparison/comparison-v1/README.md)
for compact inputs, raw timing repeats, source recipes and inspected figures.
All scientific execution on this cluster uses Slurm and
`GENESIS_IMAGE_VER=1_26 gs-srun`. Host-side Git, source reconstruction and stdlib
metadata validation do not require a numerical allocation.

## Reconstruct the measured native source

From the task checkout, select fresh paths on the data mount:

```bash
task_repo="$PWD"
task_packet="$task_repo/research/rigid_stress/evidence/20261017-pressure-comparison/comparison-v1"
task_root="/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/pressure-reproduce-$(date -u +%Y%m%dT%H%M%SZ)"
task_source="$task_root/native-source"
task_bank="$task_repo/research/rigid_stress/evidence/20261017-throughput/source-recipes-v1/inputs/conditions.npz"
task_python=/mnt/data/zhaofeng/venvs/rigid-stress-physical/bin/python
mkdir -p "$task_root/cache/cuda"
python3 research/rigid_stress/evidence/20261017-throughput/source-recipes-v1/rebuild_snapshot.py \
    --repository "$task_repo" --recipe "$task_packet/native-source-recipe" \
    --output "$task_source" --report "$task_root/source-rebuild.json"
```

The recipe reconstructs899 source/asset files from commit
`36ba93ce732e21a8bd973fb3085671d39555c5f3`, plus its empty provenance patch.
Manifest SHA256 is
`8aaeed17e16822cdf084d155c59469d8f68e40a1a14329096c3072d19d42878d`.
The bank SHA256 is
`1d40197808823971cb0f5d76e51da266aaaae6cb28453467392a3e591e4bacfa`.
Do not relabel a newly edited snapshot as this measurement.

## Paired warmed rates on one GPU

The runner starts separate processes for rigid, pressure live, intrinsic live,
policy+rigid, pressure policy and intrinsic policy on the same GPU allocation.
Choose `task_width` from32,2048,8192,16384,32768. The largest cohort can take
roughly two hours; retain the original900-step warmup and three2400-step
repeats. Policy scope includes inference and rollout, rather than RL training.

```bash
task_width=16384
cat > "$task_root/job.sh" <<EOF
#!/usr/bin/env bash
set -eo pipefail
source /mnt/data/shared/config/env.sh
GENESIS_IMAGE_VER=1_26 gs-srun --ntasks=1 /usr/bin/env \\
    CUDA_CACHE_PATH="$task_root/cache/cuda" \\
    "$task_python" "$task_repo/research/rigid_stress/scripts/run_pressure_comparison.py" \\
    --source "$task_source" --conditions "$task_bank" \\
    --output "$task_root/core-b$task_width-v1" --envs "$task_width"
EOF
sbatch --parsable --array=0-0 --partition=rtx-mid --gres=gpu:1 \
    --cpus-per-task=8 --mem=96G --time=03:00:00 \
    --output="$task_root/job-%A_%a.log" "$task_root/job.sh"
```

The outer script uses `set -eo pipefail` because the shared helper reads optional
unset variables. Submit even a single job as an array (`--array=0-0`); the
cluster's `gs-srun` helper uses the array allocation to retain the intended GPU.
The runner refuses existing output paths. Every child verifies source hashes
before/after, import origins, complete inventory, program and wrapper exits.

For a new source revision, create a new immutable source using
`scripts/snapshot_practical_native.py --output FRESH_SOURCE` and record its
manifest. Source capture uses stdlib; execute the benchmark through Slurm.

## Select settings explicitly

The literal public-default run uses `--public-default-stress`; the example
constructs `RigidStressOptions(mesh=...)` and rejects conflicting overrides.
For independent use inside the allocation/container:

```bash
bash "$task_source/research/rigid_stress/scripts/run_practical_native_benchmark.sh" \
    "$task_root/default-live" --scope live --envs 2048 \
    --level 1 --warmup 900 --steps 2400 --repetitions 3 --minimum-seconds 0 \
    --varied --seed 623016 --conditions "$task_bank" \
    --egg-geometry mesh64 --collision-method mpr --contact-patch off \
    --public-default-stress --compact-log
```

To select intrinsic, replace `--public-default-stress` with
`--contact-model intrinsic --intrinsic-load-partition body_source_prior --history 0`.
For the pressure tuning candidate, replace it with
`--contact-model pressure --scatter-tasks-per-env 128 --contact-wrench-reuse on
--contact-moment-reuse on --packed-block-size 512 --coalesced-residual on`.
These preserve physical integration and the original numerical budgets.

The runner exposes these as profiles `pressure-default`, `intrinsic` and
`pressure-tuned`. For a matched tuning experiment, use
`--profiles pressure-default pressure-tuned` with all four scopes. The
`pressure-scatter128`, `pressure-scatter256`, `pressure-fast` and
`pressure-history4` profiles reproduce the bounded B2048 tuning screen.
They are settings of the same pressure law.

Use `--output-mode full --scopes live` for matched full-field cost. Compare it
with a fresh max run on the same allocated GPU. Full mode changes the requested
public output default, so the runner records its explicit configuration rather
than marking it as the literal all-default object.

## Keep diagnostics separate

Use fresh runner outputs with `--memory --repetitions 1` for100ms NVML process
samples, or `--trace --scopes live --repetitions 1` for the600-step changing
contact trace. They are intrusive diagnostics, excluded from normal rates.
`pressure-unfused` additionally disables pipeline and rigid-post fusion to
expose meaningful pressure kernel names; it does not decompose the fused
headline wall time. Do not add overlapping CPU/API and GPU durations.

The default intrinsic initial contact workspace is16. Capacity with a separately
tuned initial workspace2 in the prior report is a different setting. A failed
capacity/load run has no successful complete-rollout rate. Preserve its log,
contacts, requested duration and failure classification.

## Accuracy and reward replay

Restore the existing53-file independent contact packet using the command in
[the application guide](RUNNING_PRACTICAL_APPLICATION.md#reconstruct-the-published-rlm-inputs).
Then run `scripts/reconstruct_franka_packets.sh FRESH_RUN NATIVE_SOURCE CONTACT_PACKETS
--radii 0.003 0.006 --models pressure body_source_prior` inside a GPU Slurm
allocation/container. Keep the four original states and all original free
moments. Their reference dt differs from the throughput workload; this is a
conditional load-reconstruction comparison.

The packet's `analysis-source-recipes/` reconstruct the executed reference,
oracle and reward helpers with the same stdlib rebuild command as above. Put
each restored analysis tree ahead of the native tree on `PYTHONPATH`.
The small `inputs/ellipsoid.npz` and `inputs/narrow_gap.npz` are the exact
existing control meshes. From the restored reference tree, inside a CPU
allocation/container, run for each geometry and radius0.003/0.0024:

```bash
"$task_python" -m research.rigid_stress.load_model_validation \
    --mesh "$task_packet/inputs/ellipsoid.npz" --geometry ellipsoid \
    --cases opposing asymmetric near_slip multiple \
    --models pressure body_source_prior --backend cpu --method direct \
    --radius 0.003 --output "$task_root/control-ellipsoid-r0.003-v1/results"
```

Use `--geometry narrow_gap` with its corresponding mesh for the close-sheet
control. Four pressure near-slip inputs include free contact moments and are
unsupported; retain those rows. Do not erase moments to turn them into successes.

For full-field moving CPU-FP64 oracle checks, execute the restored
`research.rigid_stress.native_trajectory_oracle` in a GPU allocation/container:

```bash
"$task_python" -m research.rigid_stress.native_trajectory_oracle \
    --envs 16 --steps 1500 --seed 623016 --output-mode full \
    --conditions "$task_bank" --egg-geometry mesh64 \
    --collision-method mpr --contact-patch off --scatter-tasks-per-env 128 \
    --contact-wrench-reuse on --contact-moment-reuse on \
    --packed-block-size 512 --coalesced-residual on \
    --output "$task_root/pressure-oracle-fast128-v1/oracle.json"
```

Run from the native source root with the oracle overlay on `PYTHONPATH`. The
executed launchers include original32/128/256-capacity and forced-fast variants.
They compare complete tensors, raw maximum, original wrench and full gauge at
120 sampled fields over1500 steps, including32 partial resets.

For complete-grasp reward traces, run the existing
`scripts/run_practical_reward_proxy.sh FRESH_RUN NATIVE_SOURCE --conditions BANK
--model MODEL --low-limit 0.2 --radius-scale SCALE --history 0` in a GPU
allocation/container. Use MODEL=`pressure`/`body_source_prior` and SCALE=`1`/`0.5`.
The exact four launched commands are retained in `executed-launchers/`.
Each records three complete1200-step episodes after900 warmup, including the
separate0.2N/4N groups. Its sampled time is not throughput evidence.

## Summaries, figures and closure

The executed launchers preserve original absolute paths as provenance; substitute
your fresh root and input locations when replaying. The analysis tools use the
published cohort names. Match those names or adapt a copied analysis script and
record the new source hash. The compact summary itself is portable.

Extract `executed-tools.tar.xz` and `executed-launchers.tar.xz` into a fresh
data-side provenance directory. `provenance-source-inventory.json` records each
member's original path, bytes and SHA256, including all executed plotting versions.
The readable canonical tools under `source/` remain the current entrypoints.

`scripts/summarize_pressure_comparison.py --root RUN_ROOT --output FRESH_OUTPUT`
validates raw benchmark metadata using stdlib. It requires the frozen
`meta/contract-v1.json` and `source-v1` manifest; the packet retains both contracts
and the exact executed inputs. It distinguishes normal, memory, trace and
capacity records. Scheduler terminal state must also be checked; a partial
summary is not completion evidence.

Run numerical `analyse_pressure_models.py`, `scripts/analyse_pressure_rewards.py`,
the packet's `source/summarize_pressure_accuracy.py` and the Matplotlib renderer
through a CPU Slurm allocation/container, with explicit data-side output/tmp/cache paths.
`source/plot_comparison.py --metadata COMPARISON_JSON --models MODEL_JSON
--rewards REWARD_JSON --output FRESH_FIGURES` exports all three figures.
The packet includes the exact plotting source, style helper, captions, input
hashes and QA. Reinspect rendered previews if data, fonts or source changes.
`source/qa_exports.py` checks exact183mm width, SVG text, embedded TrueType PDF
fonts and450dpi PNG, then rasterizes the PDF for visual inspection. Its isolated
PyMuPDF1.26.5 dependency is recorded in the packet; the
[official font API](https://pymupdf.readthedocs.io/en/latest/document.html#Document.get_page_fonts)
and [rasterization API](https://pymupdf.readthedocs.io/en/latest/recipes-images.html#how-to-increase-image-resolution)
describe those export checks. This QA does not alter the scientific source data.
