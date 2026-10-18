# Native stress sampling requests and lifecycle checks

Native links can select `RigidStressOptions(update_mode="manual", ...)`
before building, then call `link.request_stress_update()` after applying inputs
and before the desired `scene.step()`. Requested steps observe every physical
substep and retain the sampled-step maximum. Unrequested steps use ordinary
rigid post integration and skip auxiliary stress work for those links. The
default `every_step` behavior remains available.

Getters retain the latest sampled observation between requests. Exporters must
record its sample time. Existing geometry/dynamics notices invalidate affected
environments and cancel pending link requests. Full fields remain NaN after
invalidation until the next observed step. The public
[API](../../../docs/NATIVE_API.md) documents this contract.

Job726729 passes4 CPU and5 CUDA cases from the extended existing native
lifecycle/shared-operator capabilities. They check physical contact/load and
independent stress/displacement oracles, max/full fields,1/4 substeps,
fused/nonfused post, history4, three environments, partial reset/restore,
ordinary rigid trajectories, unbatched and mixed manual/automatic links.
All original numerical tolerances/assertions remain. Engine and test source
hashes are verified unchanged during each worker.

Job726732 copies the engine to private compute-node temporary storage and
deliberately removes the fused-path sampling condition. The same
manual/full/4-substep test then fails its retained-maximum assertion. This
confirms the test detects unwanted recovery on an unrequested step. The true
pytest exit1, assertion traceback, exact alternate import path and injected
patch are retained. The negative-control helper's expected-failure admission
is separate from a passing numerical test.

`positive/` and `negative/` retain full JSON/JUnit results and lossless logs.
`test-source/` contains all executed Python test sources and configuration.
`receipts/` records exact engine revision, complete source inventory/patch,
tasks, submission and true worker return codes. Reconstruct analysis workers
from [the first window packet's source](../short-window-v1/source/) plus this
packet's campaign-specific `source-variants/` overlay. Shared operators and
independent research controls remain in that base source. `capture.py` only
copies/hashes existing bytes and losslessly encodes logs.

This establishes the sampling/lifecycle capability. Actual application
profiling, sparse full-grasp DPR/SCR/Shell timing and selected-observable
physical-reference qualification remain required for the active comparison.
