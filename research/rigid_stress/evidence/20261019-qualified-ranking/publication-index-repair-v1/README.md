# Publication payload closure repair

The Git ignore rule for logs omitted169 manifest-listed payloads totaling24,922
bytes. They are now explicitly indexed with their original bytes. Existing
manifests and SHA-256 values are unchanged. The inventory is in
[restored-payloads.json](restored-payloads.json). Two failed replay logs preserve
their original trailing whitespace; source/documentation whitespace checks
exclude immutable evidence.

The final publication check reads manifests and payloads from a Git tree,
verifies encoded sizes/hashes and, where recorded, decoded sizes/hashes. It
requires every manifest-listed copied payload to exist in that tree, rather
than relying on ignored local files. The tree receipt is retained in the
data-side meta directory because recording the containing commit here would
be self-referential. Run the reader from the repository root:

```bash
python3 research/rigid_stress/evidence/20261019-qualified-ranking/publication-index-repair-v1/verify_git_tree.py --tree HEAD --output /path/to/data/tree-verification.json
```

The four new result packets retain their original manifests; this repair also
restores missing tiny logs in earlier same-goal packets. Scientific fields,
raw histories, solver tolerances and admission criteria are unchanged.
