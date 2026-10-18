"""Small metadata helpers independent of either simulator's import tree."""

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def file_id(path):
    path = Path(path)
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": hasher.hexdigest()}


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def runtime_identity():
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURM_ARRAY_JOB_ID",
                "SLURM_ARRAY_TASK_ID",
                "CUDA_VISIBLE_DEVICES",
                "GENESIS_IMAGE_VER",
            )
        },
        "nvidia_smi_visible_inventory": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,uuid,pci.bus_id,driver_version,memory.total", "--format=csv"], text=True
        ),
    }
