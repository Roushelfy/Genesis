import json
from pathlib import Path
from unittest.mock import patch
from subprocess import CalledProcessError
from research.rigid_stress.qipc_comparison.artifacts import runtime_identity
rows=[]
for failure in (FileNotFoundError("no nvidia-smi"), CalledProcessError(6,["nvidia-smi"])):
    with patch("subprocess.check_output",side_effect=failure):
        result=runtime_identity(require_gpu_inventory=False)
        assert result["nvidia_smi_visible_inventory"] is None and result["gpu_inventory_error"]
        rows.append({"case":type(failure).__name__,"cpu_optional_records_absence":True})
        try: runtime_identity()
        except type(failure): rows.append({"case":type(failure).__name__,"gpu_required_still_raises":True})
        else: raise AssertionError("Required GPU inventory silently accepted failure")
Path(__file__).with_suffix(".json").write_text(json.dumps({"passed":True,"cases":rows,"scope":"Stdlib mocked metadata controls only; no numerical workload on login."},indent=2)+"\n")
print("Metadata controls passed:4")
