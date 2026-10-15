"""Construct the actual original initial geometry and its rigid ground support."""

import json
from pathlib import Path

import ipctk
import numpy as np
from scipy import sparse

from .audit_franka_reference import rotation
from .franka_contact_reference import RigidFingerContactSystem, build_from_capture
from .franka_mujoco_reference import PandaSkeleton
from .predicted_wide_franka import PredictedWideContact
from .skfem_reference import SkfemElasticReference
from .summarize_fine_references import closed_source


def build_original(capture, protocol_run, mjcf, dt, dhat):
    closed_source(capture)
    closed_source(protocol_run)
    assert int((protocol_run / "returncode.txt").read_text()) == 0
    protocol = json.loads((protocol_run / "results/original_protocol.json").read_text())
    state = protocol["initial_state"]
    report = json.loads((capture / "results/summary.json").read_text())
    seed, last = build_from_capture(capture, dhat)
    origin = np.array(state["egg_com_world_m"])
    body_rotation = rotation(state["egg_quaternion_wxyz"])
    with np.load(capture / "source/examples/rigid/assets/hollow_egg/level1/elastic.npz") as saved:
        vertices = saved["vertices"] @ body_rotation.T + np.array(state["egg_origin_world_m"]) - origin
        body = SkfemElasticReference(vertices, saved["tetrahedra"], 1e10, 0.3, 2000)
    np.testing.assert_allclose(body.total_mass, last["egg_mass_kg"], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(body.com, 0, rtol=0, atol=1e-14)
    robot = PandaSkeleton(mjcf, dt)
    last_q = np.array(last["robot_joint_position"])
    last_q[7:] = 0.04
    robot.set_state(last_q, np.zeros(9))
    local, faces = [], []
    for index, (name, ids) in enumerate(zip(["left_finger", "right_finger"], seed.finger_nodes)):
        body_id = robot.body_ids[name]
        local.append(
            (seed.rest[ids] + seed.world_coordinate_origin_m - robot.data.xpos[body_id])
            @ robot.data.xmat[body_id].reshape(3, 3)
        )
        np.testing.assert_allclose(local[-1][-1], 0, rtol=0, atol=1e-12)
        selected = seed.faces[seed.groups[seed.faces[:, 0]] == index + 1]
        faces.append(selected - ids[0])
    robot.set_state(np.array(state["joint_position"]), np.array(state["joint_velocity"]))
    vertices, centres, axes = [], [], []
    for name, points in zip(["left_finger", "right_finger"], local):
        body_id = robot.body_ids[name]
        matrix = robot.data.xmat[body_id].reshape(3, 3)
        world = points @ matrix.T + robot.data.xpos[body_id] - origin
        vertices.append(world[:-1])
        centres.append(world[-1])
        axes.append(matrix @ [0, 1, 0])
    static = RigidFingerContactSystem(body, vertices, faces, axes, seed.masses, centres, dhat, seed.friction)
    static.world_coordinate_origin_m = origin
    # A single large triangle is locally identical to the infinite original plane.
    # Every accepted ground stencil is required to lie strictly inside it.
    ground_world = np.array([[-1.35, -1.0, 0.0], [2.65, -1.0, 0.0], [0.65, 3.0, 0.0]])
    first = len(static.rest)
    static.ground_nodes = np.arange(first, first + 3)
    static.rest = np.vstack((static.rest, ground_world - origin))
    static.faces = np.vstack((static.faces, static.ground_nodes))
    static.groups = np.r_[static.groups, np.full(3, 3)]
    static.stiffness = sparse.block_diag((static.stiffness, sparse.csc_matrix((9, 9))), format="csc")
    edges = np.unique(np.sort(static.faces[:, [[0, 1], [0, 2], [1, 2]]].reshape(-1, 2)), axis=0)
    static.mesh = ipctk.CollisionMesh.build_from_full_mesh(static.rest, edges, static.faces)
    static.mesh.can_collide = ipctk.make_vertex_patches_filter(static.groups[static.mesh.to_full_vertex_id()])
    assert not ipctk.has_intersections(static.mesh, static.mesh.vertices(static.rest))
    frame = dict(last)
    frame.update(
        robot_joint_position=state["joint_position"],
        robot_joint_velocity=state["joint_velocity"],
        egg_position_m=state["egg_origin_world_m"],
        egg_world_com_m=state["egg_com_world_m"],
        egg_quaternion_wxyz=state["egg_quaternion_wxyz"],
        egg_velocity_m_s=[0, 0, 0],
        egg_angular_velocity_rad_s=state["egg_angular_velocity_world_rad_s"],
    )
    return static, frame, report, protocol


class GroundedOriginalContact(PredictedWideContact):
    def kinematics(self, q):
        jacobian, positions = super().kinematics(q)
        ground = self.contact.rest[self.contact.ground_nodes]
        return np.vstack((jacobian, np.zeros((3 * len(ground), 9)))), np.vstack((positions, ground))


def contact_groups(system, full):
    normal, current = system.contact.normal_collisions(full)
    ids = system.contact.mesh.to_full_vertex_id()
    groups, ground_points = [], []
    for index in range(len(normal)):
        collision = normal[index]
        vertices = np.array(collision.vertex_ids(system.contact.mesh.edges, system.contact.mesh.faces))
        selected = vertices[vertices >= 0]
        group = set(system.contact.groups[ids[selected]])
        assert group in ({0, 1}, {0, 2}, {0, 3}), group
        groups.append(int(max(group)))
        if 3 in group:
            point = current[selected][system.contact.groups[ids[selected]] == 0][0]
            # Barycentric coordinates in the horizontal support triangle.
            ground = system.contact.rest[system.contact.ground_nodes, :2]
            bary = np.linalg.solve(np.vstack((ground.T, np.ones(3))), np.r_[point[:2], 1.0])
            assert bary.min() > 0.05
            ground_points.append(np.asarray(point, dtype=float).tolist())
        del collision
    return groups, ground_points
