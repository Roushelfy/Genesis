# Current-source matched four-scope checkpoint

Source `6a861556` includes qualified native nonconvex mapping, EPA correction,
port-seed reuse, sample-major sampling, port-distance pruning, exact contact
capacity and fused final traction acceptance/scatter. Experimental graph,
fastcache and standalone device-frontier-count changes are absent.

B1024/seed623101 uses one saved condition bank for four sequential scopes on
GPUdd177447, RTX PRO6000 Blackwell Server Edition,97887MiB,
driver595.71.05, container1_26, Genesis1.4.3/Quadrants1.3.3. The low full hollow
shell has480 affine P2 tetrahedra/810 P2 nodes/2430 DOFs. All use FP64/Q10/s8,
max output, `body_source_prior`, mesh64/GJK patch-on and dt0.01/substep1.
The condition bank fixes initial poses, friction/radius parameters, episode
delays and controller targets. Its SHA256 is
`3c1f34dc432508902dd761bee04f799d4b46a9c4731a52c63a165dd711582375`.
Rigid-only creates the bank; the other scopes load the identical bytes.

Each scope has600 warmup steps and three1280-step repeats. Dynamic rigid,
live and policy repeats each contain1163 selected-environment resets and
include timed trajectory metrics/position observation, control, parameter
updates and required history/fallback work. Recovery-only repeats one actual
frozen contact snapshot with its correct pre-integration frame; preparation,
JIT and diagnostic exports are outside timing. It is not a dynamic rollout.

| Scope | Effective env-step/s | Whole batch step/s | Native stress bytes |
|---|---:|---:|---:|
| Rigid-only |126761.284293|123.790317|0|
| Recovery-only |88441.290445|86.368448|2,785,603,166|
| Live rigid plus stress |38804.596541|37.895114|2,785,603,166|
| Policy plus rigid plus stress |40039.169392|39.100751|2,785,603,166|

Zero invalid environment steps or scatter overflows occur in every stress
repeat. Rigid and live final positions match exactly, and every per-environment
grasp/contact/descent counter is identical. Each dynamic repeat completes1571
successful/zero failed lift cycles and77400 passing/zero failing lift-window
observations. Weak-grip and release descents remain in the reported workload.

Policy is FP32 inference of a26→128→128→7 tanh network with1e-4rad residual
action scale; warmup FP64 reference action error is1.414570586e-11rad, below
the1e-9rad budget. Its actions change the actual trajectory. Its slightly
greater rate than live therefore does not measure a policy-induced speedup.
No RL optimizer/backpropagation/training throughput is claimed.

Live fit-evaluation p50/p90/p99/max per environment is11/16/22.77/270;
maximum samples is1144. One environment-frame uses integration retry per
repeat; elastic corrections/fallbacks and numerical failures are zero. These
are parallel work proxies, not per-environment GPU wall-time attribution.
All environment arrays and repeat-specific counts are preserved. Maximum
recorded warmup complete residual budget fractions are0.008796313 recovery,
0.009023787 live and0.008416894 policy. Snapshot wrench/nodal/peak checks keep
the final1e-8 wrench and1e-3Pa absolute peak budgets.

Archive `native-matched-scopes-b1024-checkpoint-v1` contains the saved bank,
all four source closures/results/logs, reusable launcher and allocated CPU
publication checks. All297 original/stored/decompressed entries and81 unique
blobs/14,354,776 stored bytes pass verification. Manifest SHA256:
`bea57c1a074533b72b05f411d11053ede2556528a3aa861519bb8128e3e9d75c`.
It is a single-seed checkpoint, not a final maximum or
plateau. B4096 four scopes and dynamic profiling are being extended; further
batches/seeds and final optimization decisions remain under the active goal.
