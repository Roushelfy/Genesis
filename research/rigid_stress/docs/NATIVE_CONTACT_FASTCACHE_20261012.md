# Contact fastcache alone: not selected

Fastcache rewraps the same22 native Quadrants contact kernels with
`fastcache=True`; no graph, numerical body or acceptance budget changes.
Its snapshot diagnostic reduces contact wall16.908571 to16.506932ms, but the
complete changing-contact workload does not show a stable speed benefit.

Original, fastcache and original-after routes run sequentially on the same
GPU, each with B2048/seed623102,600 warmup steps and three1280-step repeats.
Current numerical source is6a861556. Low full hollow shell FP64/Q10/s8,
`body_source_prior`, max output, mesh64/GJK patch-on, dt0.01/substep1;
RTX PRO6000 Blackwell Server Edition,97887MiB, driver595.71.05,
container1_26, Genesis1.4.3/Quadrants1.3.3. Timed metrics, observations,
contact/radius updates, all environments,2323 resets per repeat and required
history/fallback work are included.

| Route | Effective env-step/s | Whole batch step/s |
|---|---:|---:|
| Original before |57593.109187|28.121635|
| Fastcache |57794.949165|28.220190|
| Original after |58152.571315|28.394810|

Pooling both original runs gives57871.488158 env-step/s. Candidate ratio is
0.998677432. The original-before/after spread is larger than the apparent
gain against the first original run; fastcache is not retained.

All nine repeats have zero invalid environment steps/overflows. Each completes
3142 successful/zero failed lift cycles. Final rigid positions and every
per-environment grasp/contact/descent counter agree exactly. Warmup complete
residuals retain their unchanged budgets. Raw stress work/count differences
and every environment remain preserved rather than being selected for speed.

Reproduce with the archived dispatcher probe and original/fastcache/original
launcher. Archive `native-contact-fastcache-checkpoint-v1` closes all source
hashes and all measured rows; its allocated CPU publisher verifies originals,
stored and decompressed artifacts, including environment-level equality.
It contains214 entries/77 unique blobs/22,903,624 stored bytes.
Manifest SHA256:
`d080488ad36896cb7132469582b7cd100cc549c2c9c8a486e129fffc20247407`.
Independent contact/field oracles and graph diagnostics are published in
[the graph audit](NATIVE_CONTACT_GRAPHS_20261012.md); this live checkpoint is
not an additional independent contact oracle.

The separate dynamic audit identifies fitting and serial tails as actual
remaining costs. Wider matched scopes/seeds, tail optimization and a plateau
remain pending. No physical convergence or RL training throughput claim.
