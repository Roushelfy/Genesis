"""Measure actual complete warmup grasps separately from the timed simulator workload."""

import torch

import genesis as gs
from genesis.utils.geom import transform_by_quat

from .panda_scene import PandaEggScene


class PandaTrajectoryMetrics:
    """Accumulate device quality diagnostics; transfer only the final small summary.

    Counts use the final contact snapshot of each scene step. Stress observations independently cover every substep.
    Keeping this audit in the complete warmup prevents quality instrumentation from changing benchmark scope.
    """

    def __init__(self, scene: PandaEggScene) -> None:
        self.scene = scene
        capacity = scene.egg.get_contacts(is_padded=True)["valid_mask"].shape[1]
        self.count_histogram = torch.zeros(capacity + 1, device=gs.device, dtype=torch.int64)
        self.hold_seen = torch.zeros(scene.config.environments, device=gs.device, dtype=torch.bool)
        self.hold_failed = torch.zeros_like(self.hold_seen)
        self.egg_a_count = torch.zeros((), device=gs.device, dtype=torch.int64)
        self.tangential_max_n = torch.zeros((), device=gs.device, dtype=gs.tc_float)
        self.radius_sum_m = torch.zeros_like(self.tangential_max_n)
        self.radius_min_m = torch.full_like(self.tangential_max_n, torch.inf)
        self.radius_max_m = torch.zeros_like(self.tangential_max_n)
        self.friction_sum = torch.zeros_like(self.tangential_max_n)
        self.friction_min = torch.full_like(self.tangential_max_n, torch.inf)
        self.friction_max = torch.zeros_like(self.tangential_max_n)
        self.steps = 0

    def update(self) -> None:
        scene = self.scene
        contacts = scene.egg.get_contacts(is_padded=True)
        valid = contacts["valid_mask"]
        count = valid.sum(dim=1)
        self.count_histogram += torch.bincount(count, minlength=len(self.count_histogram))
        egg_a = (contacts["geom_a"] >= scene.egg.geom_start) & (contacts["geom_a"] < scene.egg.geom_end)
        self.egg_a_count += (egg_a & valid).sum()
        force = torch.where(egg_a[..., None], contacts["force_a"], contacts["force_b"])
        normal = torch.where(egg_a[..., None], contacts["normal"], -contacts["normal"])
        tangent = force - (force * normal).sum(dim=-1)[..., None] * normal
        self.tangential_max_n = torch.maximum(
            self.tangential_max_n, torch.where(valid, torch.linalg.vector_norm(tangent, dim=-1), 0).max()
        )
        radius = scene.radius_m.expand_as(valid)
        self.radius_sum_m += torch.where(valid, radius, 0).sum()
        self.radius_min_m = torch.minimum(self.radius_min_m, torch.where(valid, radius, torch.inf).min())
        self.radius_max_m = torch.maximum(self.radius_max_m, torch.where(valid, radius, 0).max())
        friction = contacts["friction"]
        self.friction_sum += torch.where(valid, friction, 0).sum()
        self.friction_min = torch.minimum(self.friction_min, torch.where(valid, friction, torch.inf).min())
        self.friction_max = torch.maximum(self.friction_max, torch.where(valid, friction, 0).max())
        opposite = torch.where(egg_a, contacts["link_b"], contacts["link_a"])
        robot_contact = valid & (opposite >= scene.robot.link_start) & (opposite < scene.robot.link_end)
        position = scene.egg.get_pos() + transform_by_quat(scene.com_body, scene.egg.get_quat())
        hold = (scene.phase >= 0.5) & (scene.phase < 0.6)
        self.hold_seen |= hold
        self.hold_failed |= hold & ((position[:, 2] <= 0.11) | ~robot_contact.any(dim=1))
        self.steps += 1

    def report(self) -> dict:
        histogram = self.count_histogram.cpu().tolist()
        total = sum(count * frequency for count, frequency in enumerate(histogram))
        largest_observed = max((count for count, frequency in enumerate(histogram) if frequency), default=0)
        success = (self.hold_seen & ~self.hold_failed).cpu().tolist()
        return {
            "scope": "Complete warmup; quality instrumentation excluded from steady-state timing",
            "contact_sampling": "Final contact snapshot of every scene step",
            "steps": self.steps,
            "environment_steps": self.steps * self.scene.config.environments,
            "contact_slot_capacity": len(histogram) - 1,
            "contact_count_histogram": histogram[: largest_observed + 1],
            "contact_events": total,
            "egg_as_a_contacts": self.egg_a_count.item(),
            "egg_as_b_contacts": total - self.egg_a_count.item(),
            "tangential_force_max_n": self.tangential_max_n.item(),
            "contact_radius_min_m": self.radius_min_m.item() if total else None,
            "contact_radius_max_m": self.radius_max_m.item() if total else None,
            "contact_radius_mean_m": self.radius_sum_m.item() / total if total else None,
            "contact_friction_min": self.friction_min.item() if total else None,
            "contact_friction_max": self.friction_max.item() if total else None,
            "contact_friction_mean": self.friction_sum.item() / total if total else None,
            "grasp_success": success,
            "grasp_success_fraction": sum(success) / len(success),
            "grasp_criterion": "All sampled hold phases [0.5,0.6) have COM z>0.11 m and at least one robot contact",
        }
