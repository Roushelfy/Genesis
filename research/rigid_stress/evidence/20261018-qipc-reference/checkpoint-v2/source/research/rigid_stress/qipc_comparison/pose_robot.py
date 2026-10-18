"""Original rigid Franka link geometries driven through QIPC's existing pose DBC, without joint feedback."""

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from qipc import RigidPoseDBC
from qipc._src.native.solver import GlobalVertexManager


@dataclass
class PoseRobot:
    links: list
    ignored_pairs: set
    names: tuple
    body_indices: torch.Tensor | None = None
    position_views: tuple = ()
    quaternion_views: tuple = ()
    target_positions: torch.Tensor | None = None
    target_quaternions: torch.Tensor | None = None
    global_to_rigid: torch.Tensor | None = None

    @classmethod
    def create(cls, scene, descriptor, stiffness):
        links = []
        for original in descriptor.links:
            geometry = original.geometry
            RigidPoseDBC().apply_to(geometry, stiffness=(stiffness, stiffness), fix_tol=0.0, is_constrained=True)
            links.append(scene.geometries.create(original.name, geometry))
        names = tuple(slot.name for slot in links)
        ignored = set()
        for joint in descriptor.joints:
            geometry = joint.geometry
            left_id = geometry.vertices["left_slot_id"].cpu()[0].item()
            right_id = geometry.vertices["right_slot_id"].cpu()[0].item()
            left = descriptor._scene.geometries[left_id].name
            right = descriptor._scene.geometries[right_id].name
            ignored.add(frozenset((left, right)))
        return cls(links, ignored, names)

    def allocate_targets(self, scene, motion):
        names = motion["link_names"].tolist()
        source_indices = [names.index(name.rsplit("/", 1)[-1]) for name in self.names]
        indices = [slot.geometry.meta["rigid_body_offset"].cpu()[0].item() for slot in self.links]
        rotation = Rotation.from_quat(
            motion["link_quaternion_wxyz"][:, source_indices][:, :, [1, 2, 3, 0]].reshape(-1, 4)
        )
        rotation = rotation.as_matrix().reshape(len(motion["time_s"]), len(indices), 3, 3)
        frame = scene.rigid_body.host("frame_rot")[indices]
        center = scene.rigid_body.host("frame_com")[indices]
        positions = motion["link_origin_world_m"][:, source_indices] + np.einsum("tbij,bj->tbi", rotation, center)
        material_rotation = rotation @ frame
        quaternions = Rotation.from_matrix(material_rotation.reshape(-1, 3, 3)).as_quat()[:, [3, 0, 1, 2]]
        self.body_indices = torch.tensor(indices, device="cuda")
        self.target_positions = torch.from_numpy(positions).to(device="cuda")
        self.target_quaternions = torch.from_numpy(quaternions.reshape(len(positions), len(indices), 4)).to(
            device="cuda"
        )
        self.position_views = tuple(slot.geometry.instances["aim_position"].gpu() for slot in self.links)
        self.quaternion_views = tuple(slot.geometry.instances["aim_quaternion"].gpu() for slot in self.links)
        # Pinned original-main SimEngine::init publishes [ABD | RIGID | FEM]
        # body ids. This prescribed scene has no ABD or topology-changing
        # clusters; verify the complete rigid prefix against its public owner
        # view rather than relying on integration-only layout bindings.
        assert scene.affine_body.n_bodies == 0
        local = scene.sim_systems[GlobalVertexManager].body_id_view()
        rigid_owners = scene.rigid_body.body_id
        assert torch.equal(local[: rigid_owners.numel()], rigid_owners)
        assert torch.all(local[rigid_owners.numel() :] >= scene.rigid_body.n_bodies).item()
        self.global_to_rigid = torch.where((local >= 0) & (local < scene.rigid_body.n_bodies), local, -1)
        initial_position_error, initial_rotation_error = self.tracking(scene, 0)
        assert initial_position_error < 1e-12 and initial_rotation_error < 1e-11

    def push(self, sample):
        for index, (position, quaternion) in enumerate(zip(self.position_views, self.quaternion_views)):
            position[0] = self.target_positions[sample, index]
            quaternion[0] = self.target_quaternions[sample, index]

    def tracking(self, scene, sample):
        current = scene.rigid_body.t[self.body_indices]
        position = torch.linalg.norm(current - self.target_positions[sample], dim=1).max().item()
        current_q = scene.rigid_body.quat[self.body_indices]
        target_q = self.target_quaternions[sample]
        sign = torch.where((current_q * target_q).sum(dim=1) > 0, 1, -1)
        difference = torch.linalg.norm(current_q - sign[:, None] * target_q, dim=1)
        angle = (4 * torch.asin((difference / 2).clamp(max=1))).max().item()
        return position, angle

    def boundary_power(self, scene, contact):
        ids = self.global_to_rigid[contact.global_vertex_ids]
        valid = ids >= 0
        ids = ids[valid]
        positions = contact.global_positions_m[contact.global_vertex_ids[valid]]
        velocity = scene.rigid_body.v[ids] + torch.cross(
            scene.rigid_body.omega[ids], positions - scene.rigid_body.t[ids], dim=1
        )
        return -(contact.global_forces_n[valid] * velocity).sum().item()


def pose_contact_table(scene, robot, floor, resistance):
    floor_element = scene.contact_tabular.create("floor")
    floor_element.apply_to(floor)
    elements = []
    for slot in robot.links:
        element = scene.contact_tabular.create(slot.name)
        element.apply_to(slot.geometry)
        elements.append(element)
        fixed = bool(slot.geometry.instances["is_fixed"].cpu()[0])
        scene.contact_tabular.insert(element, floor_element, friction_rate=0.1, resistance=resistance, enable=not fixed)
    for left, first in enumerate(elements):
        for right in range(left, len(elements)):
            ignored = frozenset((robot.names[left], robot.names[right])) in robot.ignored_pairs
            scene.contact_tabular.insert(
                first, elements[right], friction_rate=0.1, resistance=resistance, enable=not ignored
            )
