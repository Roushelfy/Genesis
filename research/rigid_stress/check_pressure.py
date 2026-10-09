"""Compare device pad mapping and full stress recovery to the FP64 CPU oracle."""

import argparse
import json
from pathlib import Path

import cupy as cp
import numpy as np
from threadpoolctl import threadpool_limits

from .cpu import ContactBatch, EggConfig, EggRecoveryCPU
from .device_pressure import PadPressureGPU
from .oracle import reference
from .sparse_gpu import EggRecoveryGPU


def compare(model, contacts, rows, label):
    for sampling in ("scan", "grid"):
        compare_sampling(model, contacts, rows, label, sampling)


def compare_sampling(model, contacts, rows, label, sampling):
    expected_loads = model.map_contacts(contacts)
    mapper = PadPressureGPU(model.surface, anchor_to_surface=model.mapper.anchor_to_surface, sampling=sampling)
    device = mapper.map(
        *[cp.asarray(a) for a in (contacts.position_m, contacts.force_n, contacts.radius_m, contacts.inward_normal)],
        cp.asarray(contacts.friction),
        cp.asarray(contacts.valid),
        contacts.source_epsilon,
    )
    assert cp.all(device.is_accepted).item(), cp.asnumpy(device.contact_status)
    actual_loads = cp.asnumpy(device.nodal_force_n)
    np.testing.assert_allclose(
        cp.asnumpy(device.footprint_center_m)[contacts.valid],
        expected_loads.footprint_center_m[contacts.valid],
        atol=1e-12,
        rtol=1e-10,
    )
    np.testing.assert_allclose(actual_loads, expected_loads.nodal_force_n, atol=1e-11, rtol=1e-8)
    nodal = actual_loads.reshape(-1, 3, model.environments).transpose(2, 0, 1)
    actual_force = nodal.sum(axis=1)
    actual_moment = np.cross(model.fem.xyz[None], nodal).sum(axis=1)
    np.testing.assert_allclose(actual_force, expected_loads.resultant_force_n, atol=1e-10, rtol=1e-8)
    np.testing.assert_allclose(actual_moment, expected_loads.input_moment_nm, atol=1e-11, rtol=1e-8)
    rhs = model.compatible_rhs(expected_loads.nodal_force_n)
    expected = model.recover(rhs)
    gpu = EggRecoveryGPU(model.fem, model.environments)
    actual = gpu.recover(gpu.compatible_rhs(device.nodal_force_n))
    assert cp.all(actual.is_accepted).item()
    actual_peaks = cp.asnumpy(actual.peak_pa)
    peak_error = np.abs(actual_peaks - expected.peak_pa)
    peak_relative = peak_error / np.maximum(expected.peak_pa, 1)
    assert peak_relative.max() <= 1e-4
    rows.append(
        {
            "label": label,
            "sampling": sampling,
            "environments": model.environments,
            "contact_count": contacts.valid.sum(axis=1).tolist(),
            "nodal_load_difference_max_n": float(np.abs(actual_loads - expected_loads.nodal_force_n).max()),
            "moment_error_max_nm": float(np.linalg.norm(actual_moment - expected_loads.input_moment_nm, axis=1).max()),
            "peak_error_max_pa": float(peak_error.max()),
            "peak_error_relative_max": float(peak_relative.max()),
            "full_residual_relative_max": float(cp.asnumpy(actual.relative_residual).max()),
            "mapping_diagnostics_max": cp.asnumpy(device.contact_diagnostics).max(axis=(0, 1)).tolist(),
            "force_line_anchoring": model.mapper.anchor_to_surface,
            "maximum_anchor_shift_m": float(
                np.linalg.norm(expected_loads.footprint_center_m - contacts.position_m, axis=2)[contacts.valid].max()
            ),
        }
    )
    print(json.dumps(rows[-1]), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--force-line", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    with threadpool_limits(limits=1):
        model = EggRecoveryCPU(environments=8, config=EggConfig(level=args.level), history=0, direct=True)
        rng = np.random.default_rng(814722)
        position = np.zeros((8, 8, 3))
        force = np.zeros_like(position)
        normal = np.zeros_like(position)
        radii = rng.uniform(0.004, 0.009, (8, 8))
        friction = rng.uniform(0.3, 1.2, (8, 8))
        valid = np.arange(8)[None] < np.arange(8)[:, None]
        for i_env, i_contact in np.argwhere(valid):
            face = rng.integers(len(model.fem.outer_faces))
            center, outward, tangent, _ = reference.surface_frame(model.fem, face, rng.dirichlet([2, 2, 2]))
            position[i_env, i_contact] = center
            normal[i_env, i_contact] = -outward
            force[i_env, i_contact] = -2 * outward + 2 * friction[i_env, i_contact] * tangent
        contacts = ContactBatch(position, force, radii, valid, friction, normal)
        compare(model, contacts, rows, "arbitrary_saturated_pad_wrenches")
        model.mapper.anchor_to_surface = True
        shifted = position + 0.008 * force / np.maximum(np.linalg.norm(force, axis=2)[..., None], 1e-30)
        compare(
            model,
            ContactBatch(shifted, force, radii, valid, friction, normal),
            rows,
            "same_wrenches_eight_mm_inside_reference",
        )
        compare(
            model,
            ContactBatch(position, force, np.full_like(radii, 0.05), valid, friction, normal),
            rows,
            "full_surface_radius",
        )
        mapper = PadPressureGPU(model.surface, sampling="grid")
        invalid_force = force.copy()
        invalid_force[1, 0] = -normal[1, 0]
        rejected = mapper.map(*[cp.asarray(a) for a in (position, invalid_force, radii, normal, friction, valid)])
        assert not rejected.is_accepted[1].item()
        assert rejected.contact_status[1, 0].item() == 1
        zero = mapper.map(*[cp.asarray(a) for a in (position, np.zeros_like(force), radii, normal, friction, valid)])
        assert cp.all(zero.is_accepted).item()
        assert cp.max(cp.abs(zero.nodal_force_n)).item() <= 1e-30
        if args.snapshot is not None:
            snapshot = np.load(args.snapshot)
            quat = snapshot["quat"].astype(float)
            quat /= np.linalg.norm(quat, axis=1)[:, None]
            normal = snapshot["contact_normal_world"].astype(float)
            vector = quat[:, None, 1:]
            normal -= 2 * np.cross(vector, quat[:, None, :1] * normal - np.cross(vector, normal))
            normal *= np.where(np.einsum("bci,bci->bc", normal, snapshot["force_n"]) >= 0, 1, -1)[..., None]
            live_model = EggRecoveryCPU(
                environments=len(quat),
                config=EggConfig(level=args.level),
                history=0,
                direct=True,
                anchor_to_surface=args.force_line,
            )
            live = ContactBatch(
                snapshot["position_m"],
                snapshot["force_n"],
                snapshot["radius_m"],
                snapshot["valid"],
                snapshot["friction"],
                normal,
                np.finfo(np.float32).eps,
            )
            compare(live_model, live, rows, f"actual_sliding_step_{int(snapshot['step'])}")
    report = {
        "passed": True,
        "scope": "Full-shell GPU pad mapping and direct recovery vs CPU FP64 for identical input contact wrenches",
        "live_end_to_end_tested": False,
        "physical_mesh_converged": False,
        "tensile_rejected": True,
        "zero_load_exact": True,
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
