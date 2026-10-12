"""Archive closed first-failure lifecycle tests and independent actual policy continuum diagnosis."""

import hashlib
import json
import os
import shutil
from pathlib import Path

root = Path(__file__).parent
destination = Path("research/rigid_stress/evidence/20261011-nonconvex/policy-diagnosis-checkpoint-v1")
assert not destination.exists()
folders = ["first-failure-regression-v1","policy-triangle-clip-v2","policy-triangle-clip-v3",
           "policy-clip-socp-v1","policy-clip-certificate-v1","policy-clip-certificate-v2"]
skip = {"cache","tmp","__pycache__",".pytest_cache"}
files = []
for folder in folders:
    for current,dirs,names in os.walk(root/folder):
        dirs[:] = [name for name in dirs if name not in skip]
        files.extend(Path(current)/name for name in names)
    for before in (root/folder).rglob("source.sha256"):
        if not any(part in skip for part in before.relative_to(root/folder).parts):
            assert before.read_bytes() == before.with_name("source-after.sha256").read_bytes(),before
for backend in ("cpu","cuda"):
    assert "9 passed" in (root/f"first-failure-regression-v1/{backend}/tests.log").read_text()
continuous = json.loads((root/"policy-clip-socp-v1/continuous/summary.json").read_text())
socp = json.loads((root/"policy-clip-socp-v1/results/summary.json").read_text())
assert continuous["classification"] == "continuous_incompatible"
assert continuous["conservative_polar_upper_bound"] < -4e-4
assert continuous["certificate_margin"] > 1e-3
assert socp[1]["status"] == "PrimalInfeasible" and socp[0]["status"] == "Solved"
for name in (
    "run_first_failure_regression_v1.sh","triangle_disk_clip_candidate_v2.py",
    "probe_policy_triangle_clip_v2.py","probe_policy_triangle_clip_v3.py",
    "run_policy_triangle_clip_v2.sh","run_policy_triangle_clip_v3.sh",
    "probe_policy_clip_socp_v1.py","run_policy_clip_socp_v1.sh","probe_policy_continuous_v2.py",
    "probe_policy_clip_certificate_v1.py","probe_policy_clip_certificate_v2.py",
    "run_policy_clip_certificate_v1.sh","run_policy_clip_certificate_v2.sh",
    "archive_policy_diagnosis_v1.py","run_archive_policy_diagnosis_v1.sh",
):
    files.append(root/name)
destination.mkdir(parents=True)
entries,external = [],[]
for original in sorted(set(files)):
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    if original.stat().st_size >= 20_000_000 or (original.suffix == ".npz" and original.stat().st_size >= 1_000_000):
        external.append({"original":str(original),"bytes":original.stat().st_size,"sha256":digest})
    else:
        artifact = destination/"raw"/original.relative_to(root)
        artifact.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(original,artifact)
        entries.append({"original":str(original),"artifact":str(artifact.relative_to(destination)),"bytes":original.stat().st_size,"sha256":digest})
manifest = {"scope":__doc__,"source_base_revision":"32bba7a9a6d014b952e619747e534ff2611963cc",
    "first_failure_regression":{"cpu_pass":9,"cuda_pass":9,"cpu_seconds":103.23,"cuda_seconds":160.08},
    "independent_policy_diagnosis":{"socp":socp,"continuous_bound":continuous},
    "prototype_native_certificate":json.loads((root/"policy-clip-certificate-v2/cuda/results/summary.json").read_text()),
    "note":"Both original components, both finite physical disks and actual friction remain unchanged. Native clipping is exploratory and not selected. CPU geometry/profile/P2 checks pass for 1200 samples, but the second fit is continuously incompatible under the declared per-component law. Native generated ray remains continuum unresolved. No policy throughput or goal completion is claimed.",
    "files":entries,"external_files":external}
path = destination/"manifest.json"
path.write_text(json.dumps(manifest,indent=2)+"\n")
for entry in entries:
    assert hashlib.sha256(Path(entry["original"]).read_bytes()).hexdigest() == entry["sha256"]
    assert hashlib.sha256((destination/entry["artifact"]).read_bytes()).hexdigest() == entry["sha256"]
for entry in external:
    assert hashlib.sha256(Path(entry["original"]).read_bytes()).hexdigest() == entry["sha256"]
print(json.dumps({"artifacts":len(entries),"bytes":sum(e["bytes"] for e in entries),"external_artifacts":len(external),
    "manifest_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}),flush=True)
