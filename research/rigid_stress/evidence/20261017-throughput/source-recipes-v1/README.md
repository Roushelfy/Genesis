# Exact benchmark source recipes

These small recipes reconstruct both executed899-file source trees from commit
`a2bac22db5c972805a9150fffa64f32331d15f72` and exact archived overrides.
Both reconstructions have passed every file size/SHA256 and inventory check.
The committed production engine matches the selected tree; its override restores
the historical measurement file. `source.patch` is retained as provenance.
Restoration uses the committed files and overrides described in `recipe.json`.

| Recipe | Actual source manifest SHA256 | Use |
| --- | --- | --- |
| `baseline` | `2da41428d67e99c88fb85561ba7112a7a2e7ac5ae44892ed65229a8ca6970b5a` | Original matched history4 workload |
| `selected` | `b0f2f9cdaac8b90cfffe6a691c1f2ef1cdb0c6eeb4d356e2632e72f7a072ec10` | Geometry cache and masked surface recovery, with explicit history4 or history0 |

The host-side rebuild uses Python stdlib and Git objects. Set `task_checkout`
to a checkout containing the commit above, `task_packet` to this directory,
and both output variables to fresh paths under the project data directory:

```bash
/usr/bin/python3 "$task_packet/rebuild_snapshot.py" \
  --repository "$task_checkout" --recipe "$task_packet/selected" \
  --output "$task_rebuilt_source" --report "$task_rebuild_report"
```

Use `--recipe "$task_packet/baseline"` for the original source. The source tree
includes its exact native benchmark wrapper, assets, manifest and import checks.
`inputs/conditions.npz` is the exact eight-condition bank; `baseline-contract.json`
retains the original frozen physical/numerical/control configuration.
`executed-environment.json` records the actual container Python/package versions.

Run scientific benchmarks through Slurm and `GENESIS_IMAGE_VER=1_26 gs-srun`
with the established environment, explicit data-side cache/tmp/output paths,
and the commands in `docs/RUNNING_PRACTICAL_THROUGHPUT.md`. Keep history4 code
comparisons separate from the explicitly qualified history0 tuning. The
reconstruction QA proves source bytes; numerical/application qualification
and repeated runtime measurements remain separate evidence.

Full raw traces, arrays, caches and reconstructed trees stay under the data root.
This packet is a compact source/input supplement, not a physical damage calibration.
