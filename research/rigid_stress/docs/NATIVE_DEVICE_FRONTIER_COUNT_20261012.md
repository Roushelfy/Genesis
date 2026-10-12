# Device frontier maximum: standalone candidate not selected

The fan sampler can read its current maximum face-frontier count directly on
device, removing the per-step host read before its sample-major launch. The
isolated implementation preserves the exact original sample identities,
radius/profile, friction and all acceptance budgets. It uses the selected EPA,
port-seed reuse, sample-major launch, port-distance bound and exact contact
capacity; final traction/scatter fusion is absent from this isolated pair.

B2048/623102 runs the low full hollow shell FP64/Q10/s8,
`body_source_prior`, mesh64/GJK patch-on, dt0.01/substep1 on RTX PRO6000
Blackwell Server Edition,97887MiB, driver595.71.05, container1_26,
Genesis1.4.3/Quadrants1.3.3. Both routes contain600 warmup steps, followed
by128 frozen profile passes or three1280-step live repeats. Paired routes
run sequentially on the same GPU.

Contact snapshot wall improves17.000367 to16.861936ms. Matched live improves
56953.609719 to57053.977519 effective env-step/s, ratio1.001762273. That
0.176% difference is within repeat scatter and does not support a selected
speed benefit. Both routes have zero failures/overflows and2323 resets per
repeat, with5,497,390,686 native bytes. The standalone change is not merged.
The separately evaluated complete graph also consumes the device frontier
count, and its dynamic comparison likewise gives no selected speed benefit.

CUDA23 cases pass in428.580s. The24 recorded independent CPU FP64 full fields
retain maximum normalized wrench error7.386597204e-10, complete residual
budget fraction0.045380680 and peak difference8.064977010e-6Pa. Snapshot
nodal/wrench/peak checks retain their original budgets. CPU route is unchanged.

Archive `native-device-frontier-count-checkpoint-v1` preserves the private
source patch, paired launchers and regression/oracle reports. An allocated
CPU publisher verifies384 original/stored/decompressed entries,117 unique
blobs/29,570,730 stored bytes, source closures and complete measured rows.
Manifest SHA256:
`3f288e4b5574385a29482e305f8918fe138b5b1ef3ec2b34f016fe5ece565533`.

This decision concerns a small host-read optimization at the measured batch,
not proof of maximum aggregate throughput. Wider current-source scopes/seeds,
dynamic/tail profiling and a measured plateau remain part of the active goal.
