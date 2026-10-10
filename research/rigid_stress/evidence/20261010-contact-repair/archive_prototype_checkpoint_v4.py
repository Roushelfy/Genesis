"""Publish terminal research jobs, preserving generated bodies and provenance."""
import json
import os
import sys
from pathlib import Path

from research.rigid_stress.scripts import archive_completed_evidence as archive
from research.rigid_stress.scripts import summarize_audits as audits
from research.rigid_stress.scripts import verify_contact_evidence as verify

run = Path(os.environ["RIGID_STRESS_DATA_ROOT"]) / "runs/20261010-contact-repair"
publication = Path("research/rigid_stress/evidence/20261010-contact-repair")
sys.argv = ["audits", "--output", str(run / "wrench-reuse-audits-v1-summary.json"),
            str(run / "wrench-reuse-audit-overflow-b1024-v1.json"), str(run / "wrench-reuse-audit-normal-b32768-v1.json")]
audits.main()
old = run / "build-options-guard-tests-rejected-v1.py"
(run / "build-options-guard-tests-validated-v2.py").write_text(old.read_text().replace("        scene.get_state()\n", "        tuple(scene.rigid_solver.stress_recovery.data)\n"))
prefixes = (
    "wrench-reuse-micro-", "wrench-reuse-full-oracle-", "wrench-reuse-overflow-oracle-",
    "wrench-reuse-base-live-b1024-", "wrench-reuse-reused-live-b1024-",
    "wrench-reuse-base-live-b32768-", "wrench-reuse-reused-live-b32768-",
    "wrench-reuse-audit-", "wrench-reuse-audits-", "wrench-reuse-small-pair-", "wrench-reuse-large-pair-",
    "wrench-reuse-v1-footprint-", "wrench-reuse-probe-frozen-", "audit_wrench_reuse_v1",
    "contact-init-micro-", "contact-init-probe-frozen-",
    "residual-layout-micro-", "residual-layout-full-oracle-", "residual-layout-probe-frozen-",
    "native-balance-odd-style-v3", "native-balance-tests-styled-v3",
    "build-options-guard-", "archive_prototype_checkpoint_v4", "probe_balanced_launch_v3",
)
existing = {item["runtime_path"] for item in json.loads((publication / "manifest.json").read_text())["artifacts"]}
names = [p.name for p in sorted(run.iterdir()) if p.is_file() and p.name.startswith(prefixes)
         and str(p) not in existing and p.suffix in (".json", ".log", ".py", ".patch")]
assert len(names) > 40, len(names)
sys.argv = ["archive", "--run", str(run), "--output", str(publication), "--revision", "59b16811-native-balance-t8-research-and-guard-checkpoints", *names]
archive.main()
sys.argv = ["verify", "--manifest", str(publication / "manifest.json"), "--originals", "--output", str(run / "prototype-checkpoint-v4-evidence-verification.json")]
verify.main()
