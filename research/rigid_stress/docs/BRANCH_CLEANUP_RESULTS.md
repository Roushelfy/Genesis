# Rigid stress branch cleanup

The branch now exposes the native feature, current example/benchmark,
regression references and accepted results through a compact research tree.
[RUNNING.md](RUNNING.md) is the current usage/test entrypoint;
[RUNNING_PRACTICAL_THROUGHPUT.md](RUNNING_PRACTICAL_THROUGHPUT.md) provides exact
baseline/selected reproduction. The [archive index](../archive/README.md)
records former paths, reasons, fixed revisions and retrieval commands.

## What remains active

- The complete native Quadrants implementation, public options/output/reset
  contracts, Franka/egg example/controller, benchmark, packaged assets and all
  existing regression tests/fixtures. Production physics and existing test assertions are unchanged.
- Independent CPU/FEM oracles imported by tests; native trajectory/field checks;
  reward and R/L/M replay dependencies used by the application handoff.
- Current API, model assumptions, max/full contract, final application and
  throughput results, their acceptance reports and three usage guides.
- Seven immutable evidence/source packets: five final throughput/source packets
  and two compact application/reproduction packets. Their original bytes,
  manifests and SHA256 identities remain unchanged.
- The original pending job716318 audit wrapper and nine-file source closure,
  retained to avoid changing a pending task's inputs. Its failed dependency
  remains visible; this does not reopen that research campaign.

## What was consolidated or archived

Standalone Torch/cuSPARSE/cuDSS implementations, imported early experiments,
unused campaign modules/patches, one-off launch/publish scripts, intermediate
JSON/scheduler dumps, dated progress/audit narratives, superseded goals and
older generated evidence are available at the pinned pre-cleanup commit
`1606ee3a9e9628a935ad5e446984f920af3adc66`.
The [catalogue](../archive/catalogue.tsv) groups every archived path.
Historical plans and open convergence questions retain their original context.
Current reports link historical supporting records directly to that revision.

Nine application figure/QA copies were byte-identical to retained packet blobs;
the application report now uses those canonical blobs. Current plot reproduction
uses the exact source/style shipped with the final figure packets rather than
separate launcher/plotter copies. The duplicate snapshot-rebuild script is also
archived; the source-recipe packet contains the exact executable rebuild helper.

The root README no longer describes production GPU delivery as unfinished.
The current running guide replaces a mixture of historical backend commands.
New snapshots include the stable API contract in place of a superseded goal;
historical899-file recipes keep their original identities. The focused-test
wrapper now locates its checkout independently of its caller's directory and
records only captured source changes, avoiding an unrelated evidence diff.
The example CLI now exposes its existing `seed` and `conditions` constructor
arguments, so the normal example can replay the same accepted input bank as
the benchmark. Defaults and recovery tolerances remain unchanged. A random
varied-bank smoke attempt was rejected by the original contact-load validity
check; that failure remains recorded and does not qualify random inputs.

## Size and validation

Before cleanup, `research/rigid_stress/` contained 158,637 tracked files totaling
73,415,051,427 bytes. The archive removes 157,999 files / 73,362,522,867 bytes
from the active checkout. The retained tree has 641 files, about 52.6 MB,
including the new catalogue, result report and optional reference requirements.
This measures tracked file contents, not filesystem allocation or Git history.
Shared history, runtime data, compatibility symlinks and unrelated worktrees
were preserved. No migration of live data was performed.

Verification records and stdout/stderr are under the established data root in
`runs/20261017-repository-organization/`. The checks cover the resulting layout:

| Check | Result |
| --- | --- |
| Canonical evidence | All 577 retained files match original Git blobs; all 1,037 manifest entries pass stored/decoded SHA256 and size checks. |
| Exact historical sources | Both baseline and selected recipes rebuild all 899 files with their original manifest identities. |
| Imports, assets and links | Active research imports resolve, production imports no research backend, and active local/pinned documentation targets resolve. All production modules and test fixtures remain in place. |
| Archive recovery | The older baseline packet restores with all 18 payload hashes; an isolated sparse historical source worktree is clean and excludes the large evidence collection. |
| Application handoff | All 53 contact-packet files / 1,994,054 bytes restore exactly; the retained reward-log audit preserves phase-dependent limitations. |
| CPU regression | Job725373 completes0:0: 11 passed; 1 CUDA-only moment-reuse case skipped. Independent FEM, max/full output, partial reset and checkpoint cases pass with captured source hashes unchanged. |
| GPU workflow | Job725378 completes0:0 on RTX PRO6000 Blackwell: the frozen-bank 600-step example and 900-warmup/16-step live benchmark pass, with zero invalid environment steps/overflows; the CUDA-only moment/isolation regression passes. All 899 snapshot files, the condition bank and captured test sources remain unchanged. |
| Static checks | Changed Python passes Ruff; retained shell scripts pass Bash syntax; added/modified content passes Git whitespace checks. |

The example/benchmark smoke uses the exact accepted eight-condition bank.
Random generated inputs are a separate domain: the original varied example
attempt triggered the preserved finite-footprint load-admissibility rejection.
It is not counted as a passing smoke check. Initial launch failures and the
pre-test cancellation for an unbounded evidence diff remain in the runtime
records. No failed check was converted into a numerical pass, and no old
convergence or optimization campaign was repeated.
