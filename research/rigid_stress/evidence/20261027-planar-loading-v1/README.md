# Planar loading DPR compact evidence packet

Generated 2026-10-27T23:20:52.501604+00:00. Run root: `/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261027-planar-loading-dpr`. Maintained source: `6dedc79579ac0d113e4baac26eb538bcf51ca57d`.

This packet records terminal execution receipts and declared numerical/spatial
semantics separately. It does not mark the user goal complete. Review
`requirements-ledger.json` before making any completion claim.

The decoded full index maps campaign/group/task, actual worker hash, source pin, exact CLI
and available result summaries. `external-artifacts.jsonl.gz` records SHA256 and
bytes for every collected task artifact, including arrays, complete fields and
logs left on data. Retained payloads are content-addressed and deduplicated.
Gzip JSON preserves exact raw paired rows/source receipts where labelled raw;
derived summaries are labelled separately. Large600-row result histories remain
external. The packet is capped at10MiB; optional omissions are explicit.

For planar primary-paired-* terminal benchmark tasks, mandatory gzip payloads preserve every original repeat field and vector plus unabridged identity/configuration metadata. Other large diagnostics remain externally hashed. These are original measurements with no new qualification assessment.

Existing top-level primary/memory/profile summary gzip files and the final offline verifier are mandatory exact payloads, without the optional metadata 4MiB input cutoff. The 10MiB packet limit still applies; missing summaries are not created or certified by packaging.

Planar index.json is a small gzip reference: index.json.gz contains the complete original indented index with every task, field and artifact. The reference binds compressed and decompressed bytes/SHA256; both files are covered by SHA256SUMS. Optional payload selection uses bounded deterministic planning and never removes mandatory summaries, raw rows or reproduction/ledger inputs.

Only task-specific `.returncode.txt`/`.returncode` receipts permit task output
collection. Missing receipts remain pending unless a separate cancellation journal records the stopped campaign. Recorded cancellations have their own index status and task list. Their potentially live output trees are not
read. A task RC0 is execution success, not numerical/spatial qualification.
Caches and complete analysis-source trees are excluded. Actual small worker
files may be retained individually; full engine code is reproduced by Git pin.

## Verification and reproduction

```sh
sha256sum -c SHA256SUMS
gzip -dc index.json.gz
gzip -dc external-artifacts.jsonl.gz
: "${PLANAR_REPRO_SOURCE:?choose a fresh source checkout}"
: "${PLANAR_PACKET_OUTPUT:?choose a fresh packet directory}"
git worktree add --detach "$PLANAR_REPRO_SOURCE" 6dedc79579ac0d113e4baac26eb538bcf51ca57d
python -m research.rigid_stress.qipc_comparison.package_cpu12_evidence \
  --run-root /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261027-planar-loading-dpr --output "$PLANAR_PACKET_OUTPUT" \
  --maintained-source 6dedc79579ac0d113e4baac26eb538bcf51ca57d --git-source "$PLANAR_REPRO_SOURCE" \
  --study planar --requirements-ledger "${PLANAR_REQUIREMENTS_LEDGER:?choose the reviewed ledger}"
```

Each terminal task has its original module/arguments and reproduction command in
the index. Use that campaign's recorded engine pin and captured analysis-source
inventory, an `rtx-mid` Slurm allocation, image `1_26`, private caches, and a new
data-side output. Exact captured workers remain at their external source paths.
Do not execute scientific modules on a login node or overwrite any original run.
Packet creation itself uses only the standard library and changes no run files.

`preservation.json` reports current read-only observations of the original12
files and33 frozen sources from the preserved start receipt. Unknown/missing
paths stay explicit. Device-memory point/sampled measurements, intrusive profile
results, approximate-space residuals and exact-mother residuals retain their
source declarations; packaging does not reinterpret them as new validation.

To add a completed target ledger, edit a separate copy and rerun this packager
with `--requirements-ledger <COPY>` into a fresh directory. This preserves all
payload hashes and never promotes a pending task or historical result to a pass.
