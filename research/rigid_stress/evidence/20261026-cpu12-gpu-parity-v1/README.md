# CPU12 compact evidence packet

Generated 2026-10-27T02:40:55.306720+00:00. Run root: `/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261026-cpu12-gpu-parity`. Maintained source: `09bc9fa4a0e442c145d4a147da93058e5180d016`.

This packet records terminal execution receipts and declared numerical/spatial
semantics separately. It does not mark the user goal complete. Review
`requirements-ledger.json` before making any completion claim.

`index.json` maps campaign/group/task, actual worker hash, source pin, exact CLI
and available result summaries. `external-artifacts.jsonl.gz` records SHA256 and
bytes for every collected task artifact, including arrays, complete fields and
logs left on data. Retained payloads are content-addressed and deduplicated.
Gzip JSON preserves exact raw paired rows/source receipts where labelled raw;
derived summaries are labelled separately. Large600-row result histories remain
external. The packet is capped at10MiB; optional omissions are explicit.

Only task-specific `.returncode.txt`/`.returncode` receipts permit task output
collection. Missing receipts remain pending, including stopped jobs with a
separate cancellation journal. Their potentially live output trees are not
read. A task RC0 is execution success, not numerical/spatial qualification.
Caches and complete analysis-source trees are excluded. Actual small worker
files may be retained individually; full engine code is reproduced by Git pin.

## Verification and reproduction

```sh
sha256sum -c SHA256SUMS
gzip -dc external-artifacts.jsonl.gz
: "${CPU12_REPRO_SOURCE:?choose a fresh source checkout}"
: "${CPU12_PACKET_OUTPUT:?choose a fresh packet directory}"
git worktree add --detach "$CPU12_REPRO_SOURCE" 09bc9fa4a0e442c145d4a147da93058e5180d016
python -m research.rigid_stress.qipc_comparison.package_cpu12_evidence \
  --run-root /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261026-cpu12-gpu-parity --output "$CPU12_PACKET_OUTPUT" \
  --maintained-source 09bc9fa4a0e442c145d4a147da93058e5180d016 --git-source "$CPU12_REPRO_SOURCE"
```

Each terminal task has its original module/arguments and reproduction command in
the index. Use that campaign's recorded engine pin and captured analysis-source
inventory, an `rtx-mid` Slurm allocation, image `1_26`, private caches, and a new
data-side output. Exact captured workers remain at their external source paths.
Do not execute scientific modules on a login node or overwrite any original run.
Packet creation itself uses only the standard library and changes no run files.

`preservation.json` reports current read-only observations of the original12
files and23 frozen sources from the preserved start receipt. Unknown/missing
paths stay explicit. Device-memory point/sampled measurements, intrusive profile
results, approximate-space residuals and exact-mother residuals retain their
source declarations; packaging does not reinterpret them as new validation.

To add a completed target ledger, edit a separate copy and rerun this packager
with `--requirements-ledger <COPY>` into a fresh directory. This preserves all
payload hashes and never promotes a pending task or historical result to a pass.
