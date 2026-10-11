"""Single writer for explicitly terminal native moment validations and diagnostics."""
import json
import os
import sys
from pathlib import Path

from research.rigid_stress.scripts import archive_completed_evidence as archive
from research.rigid_stress.scripts import verify_contact_evidence as verify

run = Path(os.environ["RIGID_STRESS_DATA_ROOT"]) / "runs/20261010-contact-repair"
publication = Path("research/rigid_stress/evidence/20261010-contact-repair")
existing = {item["runtime_path"] for item in json.loads((publication / "manifest.json").read_text())["artifacts"]}
names = [
    "archive_moment_checkpoint_v7.py",
    "validated-scheduling-checkpoint-v6-evidence-verification.json",
    "validated-scheduling-checkpoint-v6-analysis-archive.log",
    "native-contact-moments-v4.patch",
    "native-contact-moments-v5.patch",
    "native-contact-moments-v6.patch",
    "native-contact-moments-gpu-v4.log",
    "native-contact-moments-oracle-seed510000-v4.log",
    "native-contact-moments-gpu-v5.log",
    "native-contact-moments-unbatched-gpu-v5.log",
    "native-contact-moments-gpu-v6.log",
    "native-contact-moments-cpu-v6.log",
    "native-episode-probe-frozen-v2.py",
    "native-rollout-profile-probe-frozen-v1.py",
    "contact-moment-derived-gram-probe-frozen-v1.py",
    "run_native_contact_moments_tests_v4.sh",
    "run_native_contact_moments_tests_v5.sh",
    "run_native_contact_moments_tests_v6.sh",
    "run_native_contact_moments_oracle_v4.sh",
    "run_native_contact_moments_oracle_v5.sh",
]
bases = [
    "native-contact-moments-oracle-seed510000-v5",
    "native-contact-moments-oracle-seed623001-v5",
    "native-contact-moments-overflow-substeps4-v5",
    "contact-moments-base-live-b32768-v1",
    "contact-moments-reused-live-b32768-v1",
    "contact-moments-live-pair-b32768-v1-summary",
    "contact-moment-derived-gram-micro-b1024-v1",
    "contact-moment-derived-gram-micro-b32768-v1",
    "native-episode-extended-policy-b32-seed623001-v2",
    "native-rollout-profile-policy-b32-v1",
]
bases += [
    f"native-episode-extended-{scope}-b{batch}-seed{seed}-v2"
    for scope in ("live", "policy") for batch in (1024, 2048) for seed in (510000, 623001)
]
for base in bases:
    names += [f"{base}{suffix}" for suffix in (".json", ".log", ".variant.json", ".gram-variant.json", ".trace.json") if (run / f"{base}{suffix}").is_file()]
names = [name for name in names if str(run / name) not in existing]
assert len(names) > 50, len(names)
sys.argv = ["archive", "--run", str(run), "--output", str(publication), "--revision", "eba92370-terminal-moment-trials-and-59b16811-native-v6-validation", *names]
archive.main()
sys.argv = ["verify", "--manifest", str(publication / "manifest.json"), "--originals", "--output", str(run / "moment-checkpoint-v7-evidence-verification.json")]
verify.main()
