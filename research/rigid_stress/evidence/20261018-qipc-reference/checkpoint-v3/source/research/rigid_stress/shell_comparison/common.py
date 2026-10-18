"""Inputs and receipts shared by the bounded shell comparison."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

import genesis as gs


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def vars_json(args):
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def identity():
    return {
        "analysis_import": __file__,
        "genesis_import": gs.__file__,
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cpu_models": sorted(
            {
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            }
        ),
        "host": platform.node(),
        "job": os.environ.get("SLURM_JOB_ID"),
        "array_job": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "array_task": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "packages": {
            key: importlib.metadata.version(key)
            for key in (
                "quadrants",
                "torch",
                "numpy",
                "scipy",
                "trimesh",
                "pydantic",
            )
        },
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"], text=True
        ).strip()
        if gs.backend == gs.cuda
        else None,
    }


def file_identity(path):
    payload = Path(path).read_bytes()
    return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def grid_sheet(path, nx, ny, length=0.2, width=0.2):
    x, y = np.meshgrid(np.linspace(-0.5, 0.5, nx + 1) * length, np.linspace(-0.5, 0.5, ny + 1) * width, indexing="ij")
    vertices = np.stack((x.reshape(-1), y.reshape(-1), np.zeros(x.size)), axis=-1)
    faces = []
    for i in range(nx):
        for j in range(ny):
            a, b = i * (ny + 1) + j, (i + 1) * (ny + 1) + j
            c, d = b + 1, a + 1
            faces.extend(((a, b, c), (a, c, d)) if (i + j) % 2 == 0 else ((a, b, d), (b, c, d)))
    mesh = trimesh.Trimesh(vertices, np.array(faces), process=False)
    mesh.export(path)
    return mesh


def synchronize():
    if gs.backend == gs.cuda:
        torch.cuda.synchronize()
