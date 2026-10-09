"""Untimed native GPU gauge and live-adapter pose covariance QA with explicit synthetic fixtures."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cupy as cp
import numpy as np
import torch
from threadpoolctl import threadpool_limits

import genesis as gs
from genesis.utils.geom import transform_by_quat
from research.rigid_stress.cpu import ContactBatch, EggConfig, EggRecoveryCPU
from research.rigid_stress.cudss import SharedCuDSSFactor
from research.rigid_stress.device_pressure import PadPressureGPU
from research.rigid_stress.live import RigidContactAdapter, RigidSnapshot
from research.rigid_stress.mechanics import gauge_rows
from research.rigid_stress.oracle import reference
from research.rigid_stress.sparse_gpu import EggRecoveryGPU


@dataclass
class ContactFixture:
    contacts: dict[str, torch.Tensor]
    geom_start: int = 10
    geom_end: int = 12

    def get_contacts(self, is_padded: bool) -> dict[str, torch.Tensor]:
        assert is_padded
        return self.contacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gs.init(backend=gs.gpu, precision="64", logging_level="warning")
    rng = np.random.default_rng(912681)
    b, c = 7, 6
    with threadpool_limits(limits=1):
        model = EggRecoveryCPU(b, EggConfig(level=2), history=0, direct=True, anchor_to_surface=True)
        fem = model.fem
        positions, forces, normals = (np.zeros((b, c, 3)) for _ in range(3))
        valid = np.arange(c)[None] < np.arange(b)[:, None]
        radii = rng.uniform(0.007, 0.010, (b, c))
        friction = np.full((b, c), 0.7)
        for i, j in zip(*np.nonzero(valid)):
            face = int(rng.integers(len(fem.outer_faces)))
            point, normal, tangent, _ = reference.surface_frame(fem, face, rng.dirichlet([2, 2, 2]))
            positions[i, j] = point
            normals[i, j] = -normal
            forces[i, j] = -normal * rng.uniform(0.2, 2) + tangent * 0.03
        omega = rng.uniform(-4, 4, (b, 3))
        omega[0] = 0
        gravity = np.tile([0.0, 0.0, -9.81], (b, 1))
        contacts = ContactBatch(positions, forces, radii, valid, friction, normals)
        cpu_mapped = model.map_contacts(contacts)
        cpu_rhs = model.compatible_rhs(cpu_mapped.nodal_force_n + fem.mr[:, :3] @ gravity.T, omega)
        expected = model.recover(cpu_rhs)
        rows = []
        for layout in ("F", "C"):
            gpu = EggRecoveryGPU(
                fem, b, factor_backend="cudss", inertia="quadratic", body_products="fused", sparse_layout=layout
            )
            alternate_pins = gauge_rows(fem.r, np.arange(fem.ndof)[::-1])
            assert set(alternate_pins) != set(fem.pins)
            alternate_free = np.setdiff1d(np.arange(fem.ndof), alternate_pins)
            alternate_factor = SharedCuDSSFactor(fem.k[alternate_free][:, alternate_free], b)
            rhs_device = cp.asarray(cpu_rhs)
            alternate_u = cp.zeros_like(rhs_device, order="F")
            alternate_u[cp.asarray(alternate_free)] = alternate_factor.solve(rhs_device[alternate_free])
            alternate_peak = cp.asnumpy(gpu.peak(alternate_u))
            alternate_residual = cp.linalg.norm(rhs_device - gpu.apply_stiffness(alternate_u), axis=0)
            rhs_norm = cp.linalg.norm(rhs_device, axis=0)
            assert cp.all(alternate_residual <= cp.maximum(1e-11, 1e-6 * rhs_norm)).item()
            gauge_peak_error = abs(alternate_peak - expected.peak_pa) / np.maximum(expected.peak_pa, 1)
            assert gauge_peak_error.max() < 1e-8
            canonical_error = np.linalg.norm(
                fem.canonical(cp.asnumpy(alternate_u)) - fem.canonical(expected.displacement_m), axis=0
            ) / np.maximum(np.linalg.norm(fem.canonical(expected.displacement_m), axis=0), 1e-12)
            assert canonical_error[np.linalg.norm(cpu_rhs, axis=0) > 1e-5].max() < 1e-6
            mapper = PadPressureGPU(model.surface, anchor_to_surface=True, sampling="grid", scatter="atomic")

            def tensor(a: np.ndarray) -> torch.Tensor:
                return torch.as_tensor(a, dtype=torch.float64, device=gs.device)

            for pose in range(5):
                quaternion = rng.normal(size=(b, 4))
                quaternion /= np.linalg.norm(quaternion, axis=1)[:, None]
                translation = tensor(rng.uniform(-0.5, 0.5, (b, 3)))
                quat = tensor(quaternion)
                world_point = transform_by_quat(tensor(positions), quat[:, None]) + translation[:, None]
                world_force = transform_by_quat(tensor(forces), quat[:, None])
                world_normal = transform_by_quat(tensor(normals), quat[:, None])
                egg_a = torch.as_tensor(rng.integers(2, size=(b, c)).astype(bool), device=gs.device)
                fixture = ContactFixture({
                    "geom_a": torch.where(egg_a, 10, 20),
                    "position": world_point,
                    "force_a": torch.where(egg_a[..., None], world_force, -world_force),
                    "force_b": torch.where(egg_a[..., None], -world_force, world_force),
                    "normal": torch.where(egg_a[..., None], world_normal, -world_normal),
                    "friction": tensor(friction),
                    "valid_mask": torch.as_tensor(valid, device=gs.device),
                })
                adapter = RigidContactAdapter(fixture, b, (0.0, 0.0, -9.81))
                adapter.gravity = transform_by_quat(tensor(gravity), quat)
                adapter.snapshot = RigidSnapshot(
                    translation, quat, transform_by_quat(tensor(omega), quat), torch.zeros_like(translation)
                )
                local = adapter.contacts()
                for actual, truth in ((local.position_m, positions), (local.force_n, forces),
                                      (local.inward_normal, normals), (local.omega_rad_s, omega),
                                      (local.gravity_m_s2, gravity)):
                    np.testing.assert_allclose(actual.cpu().numpy(), truth, atol=1e-14, rtol=1e-12)
                mapped = mapper.map(
                    cp.from_dlpack(local.position_m), cp.from_dlpack(local.force_n), cp.asarray(radii),
                    cp.from_dlpack(local.inward_normal), cp.from_dlpack(local.friction), cp.from_dlpack(local.is_valid)
                )
                assert cp.all(mapped.is_accepted).item()
                rhs = gpu.compatible_rhs(
                    mapped.nodal_force_n + gpu.gravity_load(cp.from_dlpack(local.gravity_m_s2)),
                    cp.from_dlpack(local.omega_rad_s),
                )
                np.testing.assert_allclose(cp.asnumpy(rhs), cpu_rhs, atol=1e-11, rtol=1e-7)
                out = gpu.recover(rhs)
                assert cp.all(out.is_accepted).item()
                peak_error = abs(cp.asnumpy(out.peak_pa) - expected.peak_pa) / np.maximum(expected.peak_pa, 1)
                assert peak_error.max() < 1e-4
                rows.append({
                    "layout": layout, "pose": pose, "environments": b,
                    "peak_relative_error_max": float(peak_error.max()),
                    "complete_rhs_entry_error_max_n": float(abs(cp.asnumpy(rhs) - cpu_rhs).max()),
                    "gauge_peak_relative_error_max": float(gauge_peak_error.max()),
                    "gauge_canonical_displacement_relative_error_max": float(canonical_error.max()),
                    "gauge_full_absolute_residual_max_n": float(alternate_residual.max().item()),
                })
                print(json.dumps(rows[-1]), flush=True)
    report = {"passed": True, "scope": "Untimed synthetic GPU gauge/pose covariance QA; no actual grasp or FPS",
              "layouts": ["F", "C"], "dofs": fem.ndof, "tetrahedra": fem.ne,
              "cpu_oracle": "Same-mesh FP64 full direct", "cases": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
