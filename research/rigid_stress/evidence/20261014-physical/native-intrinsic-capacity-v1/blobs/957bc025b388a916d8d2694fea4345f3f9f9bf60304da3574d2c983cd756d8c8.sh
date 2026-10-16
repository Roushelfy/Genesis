#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs"
task_epoch="$task_root/20261014-closeout"
task_tools="$task_epoch/publication-tools"
test ! -e "$task_tools"
mkdir -p "$task_tools/source"
task_sources=(research/rigid_stress/scripts/{summarize_nonconvex_closeout,verify_nonconvex_publication,publish_nonconvex_checkpoint,pin_nonconvex_test_sources}.py
    research/rigid_stress/scripts/{check_nonconvex_final,measure_nonconvex_output_modes}.sh
    tests/rigid/test_stress_surface_fan.py)
sha256sum "${task_sources[@]}" > "$task_tools/source.sha256"
cp --parents "${task_sources[@]}" "$task_tools/source"
cp "${BASH_SOURCE[0]}" "$task_tools/launcher.sh"
git rev-parse HEAD > "$task_tools/base-revision.txt"
git diff HEAD > "$task_tools/source.patch"
task_runs=()
for task_batch in 1024 8192; do
    for task_scope in live policy; do
        for task_route in max-before full max-after; do
            task_runs+=("$task_epoch/output-modes-b${task_batch}/$task_scope/$task_route")
        done
    done
done
task_runs+=("$task_epoch/regression-v2-cpu" "$task_epoch/regression-v2-cuda"
    "$task_epoch/surface-fan-equilibrium-cpu" "$task_epoch/surface-fan-equilibrium-cuda"
    "$task_root/20261011-nonconvex/native-final-copied-edge-v1/b36864-623102/live"
    "$task_root/20261011-nonconvex/native-final-copied-edge-v1/b36864-623102/policy"
    "$task_root/20261011-nonconvex/native-final-copied-edge-v1/b40960-623102/live"
    "$task_root/20261011-nonconvex/native-final-copied-edge-v1/b38912-623102/live"
    "$task_epoch/final-audit-v2")
task_extras=("$task_root"/20261011-nonconvex/closeout-*.log "$task_root"/20261011-nonconvex/closeout-*.err
    "$task_epoch"/output-modes-b*-position-control.json "$task_epoch/output-modes-b1024-summary.json"
    "$task_epoch/restored-artifacts.json" "$task_epoch/publication-verification-v3/result.json"
    "$task_epoch/publication-verification/verifier.py" "$task_epoch/publication-verification-v2/verifier.py"
    "$task_epoch/publication-verification-v3/verifier.py"
    "$task_epoch/closeout-publication-711586_0.log" "$task_epoch/closeout-publication-711586_0.err")
sha256sum "${task_sources[@]}" > "$task_tools/source-after.sha256"
cmp "$task_tools/source.sha256" "$task_tools/source-after.sha256"
task_runs+=("$task_tools")
"$RIGID_STRESS_ENV/bin/python" -m research.rigid_stress.scripts.publish_nonconvex_checkpoint \
    --runtime-root "$task_root" \
    --destination research/rigid_stress/evidence/20261011-nonconvex/native-nonconvex-closeout-20261014-v1 \
    --staging-root "$task_epoch/publication-staging" \
    --run "${task_runs[@]}" --extra "${task_extras[@]}" \
    --summary "$task_epoch/final-summary-v2.json" \
    --limitations 'Numerical engine76537c2f is unchanged. Complete source/bank/physical-budget/outcome checks;135 current CUDA cases and114 CPU passes/21 CUDA-only skips are qualified by the full original suite plus10 current cases in the sole modified test module. The original two cross-load residual assertion failures remain retained. Full/max same-state accuracy keeps original budgets; policy modes are matched by bank/configuration/network and outcomes, not by bit-identical evolving loads/positions. Every cross-state position/peak difference and same-mode control is retained. Output traces are separate intrusive50-step passes. Capacity failures have no throughput result. No global-optimum, physical mesh convergence or full RL-training claim. Publication sources reconstruct incomplete pretest file lists from their recorded clean Git revision/import/empty tracked diff; those files are explicitly identified as post-start supplements.'
mkdir -p "$task_epoch/closeout-publication-verification"
"$RIGID_STRESS_ENV/bin/python" -m research.rigid_stress.scripts.verify_nonconvex_publication \
    --originals \
    --manifest research/rigid_stress/evidence/20261011-nonconvex/native-nonconvex-closeout-20261014-v1/manifest.json \
    --output "$task_epoch/closeout-publication-verification/result.json"
