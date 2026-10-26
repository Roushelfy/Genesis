# Native local-refinement atlas evidence

The [report](../../docs/LOCAL_REFINEMENT_RESULTS_20261026.md) records the
completed shell/volume evaluation of real `T_i=[P,B_i]` and smaller `K_i`,
per-feature aggregation and merged whole-domain recovery. The
[six-cell audit](metadata/requirements-final.json) distinguishes numerical
acceptance, spatial accuracy, component gains and production retention.

Smaller spaces pass native structural controls but fail selected combined-field
budgets after bounded tuning. Accurate low-model libraries approach mother
dimensions and remain slower in paired complete workflows. Aggregation and
one recovery have useful internal gains; their atlas dependency is unretained.
Production `genesis`/`examples` match `4bc4ada25`, with twenty cleanup tests
passing. These conclusions have the models, loads, sampling and hardware scope
given in the report.

## Compact records

- [Summary index](metadata/index.json): campaign/source/analysis references and
  SHA-bound JSON/CSV records, including all 552 benchmark repeats.
- [Accuracy](metadata/accuracy.json), [geometry](metadata/geometry.json),
  [joint/coarse/local controls](metadata/joint.json),
  [component increments](metadata/components.json),
  [selector comparison](metadata/selector.json),
  [profiling/memory](metadata/profiles.json) and
  [benchmarks](metadata/benchmarks.json).
- [Production regressions](metadata/production-regressions.json) and
  [source preservation](metadata/preservation-final.json).
- [Exact small input banks](inputs/manifest.json): four conditions and the
  heterogeneous 64-condition bank. Large arrays, traces, logs and caches stay
  under the data-side run root in `metadata/index.json`.

Historical failed attempts are retained. A return code of one can identify a
completed spatial rejection; inspect its numerical and spatial fields.
The signed-load cancellation experiment is an algebra control and has no
physical-contact qualification claim. Joint unions that reach mother size
are reported at that size.

## Runnable sources

The complete experimental research tree is
`822ad285497f79d3a24f3ce2f0dd7a58ec6cd699`; its engine is unchanged from
`a971d61d127148401cff2de0d0552daa60e77105`. Use an isolated worktree:

```bash
git worktree add --detach /path/to/new-atlas-checkout 822ad2854
```

The maintained tree contains research commands whose feature imports require
that experimental engine. Read each campaign's `source.json`,
`analysis-files.json`, `tasks.json` and `workload-*.sh` together. Some campaigns
combine an earlier frozen engine with a later captured research worker.
The [ledger](../../docs/LOCAL_REFINEMENT_PROGRESS_20261026.md) records those
interfaces and the retained failed attempts.

Run numerical preparation, tests and independent scientific oracles through
Slurm `rtx-mid`, image `1_26`, using the data-side environment. Supply a new
output directory and private cache paths. Existing examples use
`research.rigid_stress.shell_comparison.submit`; replace source, interpreter,
input and output paths for the local installation. Do not overwrite captured
campaigns or edit frozen engines.

The stdlib summary can be regenerated from the maintained tree:

```bash
python3 -m research.rigid_stress.qipc_comparison.summarize_local_atlas \
  --run /path/to/retained-data-run \
  --output /path/to/new-summary
```

The index covers generated summary files. The separately authored requirement,
regression and preservation receipts bind that index to the delivery decision.
