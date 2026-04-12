"""
Replay exported G1 kimono dancing simulations using Genesis.

Loads USD (``.usdc``) trajectories containing:
- Rigid robot link transforms (42 links, per-frame 4x4 matrices)
- Deformable cloth vertex positions (kimono pieces, per-frame)

The robot is loaded as a URDF entity.  Per-frame joint angles (``qpos``)
are computed from the USD world transforms via inverse kinematics on the
URDF kinematic chain.

Available trajectories:
  d0_lower, d0_no_lower, d1_no_lower, d2_lower, d2_no_lower,
  d4_lower, d4_no_lower, d5_lower, d5_no_lower

Usage:
    python ipc_costume_replay.py                              # GUI (d0_no_lower)
    python ipc_costume_replay.py --trajectory d5_lower        # different trajectory
    python ipc_costume_replay.py --render                     # render video (LuisaRender)
    python ipc_costume_replay.py --render --nyx               # render video (Nyx)
    python ipc_costume_replay.py --start-frame 100            # start from frame 100
    python ipc_costume_replay.py --no-gui                     # headless verification
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from _replay_common import TrajectoryReplay

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACK_DIR = _REPO_ROOT / "DemoAssets" / "robot_cloth" / "dancing_export_pack"
_KIMONO_DIR = _REPO_ROOT / "DemoAssets" / "kimono"

AVAILABLE_TRAJECTORIES = [
    "d0_lower",
    "d0_no_lower",
    "d1_no_lower",
    "d2_lower",
    "d2_no_lower",
    "d4_lower",
    "d4_no_lower",
    "d5_lower",
    "d5_no_lower",
]

URDF_PATH = str(_REPO_ROOT / "DemoAssets" / "g1_robot" / "assets" / "g1_29dof_rev_1_0.urdf")

# Cloth PBR materials from the Sketchfab kimono GLTF.
# (base_color_rgb, roughness)
CLOTH_MATERIALS = {
    "kimono_outer": ((0.020, 0.031, 0.040), 0.82),
    "kimono_inner_upper": ((0.080, 0.112, 0.137), 0.98),
    "kimono_inner_lower": ((0.020, 0.031, 0.040), 0.98),
    "kimono_belt": ((0.014, 0.017, 0.019), 0.35),
}

# Map from USD link names (from GLB filenames) to Genesis URDF link names.
# Most are identical; a few have _rev_1_0 suffixes in USD.
_USD_TO_URDF_LINK = {
    "waist_yaw_link_rev_1_0": "waist_yaw_link",
    "waist_roll_link_rev_1_0": "waist_roll_link",
    "torso_link_rev_1_0": "torso_link",
}


# ── USD loading ──────────────────────────────────────────────────────────────


def _usd_name_to_link(usd_name: str) -> str:
    """``pelvis_glb`` -> ``pelvis``, ``torso_link_rev_1_0_glb`` -> ``torso_link``."""
    stripped = re.sub(r"_glb(_\d+)?$", "", usd_name)
    return _USD_TO_URDF_LINK.get(stripped, stripped)


def _usd_cloth_to_piece_name(usd_name: str) -> str:
    """``wearing_kimono_belt_0`` -> ``kimono_belt``."""
    return re.sub(r"^wearing_", "", re.sub(r"_\d+$", "", usd_name))


def load_usd_trajectory(usdc_path: Path) -> dict:
    """Load rigid transforms and cloth positions from a USD file.

    Returns dict with:
        frame_count: int
        link_transforms: dict[urdf_link_name -> (T, 4, 4) ndarray]  (column-vector)
        cloth: dict[usd_name -> { "positions": (T, N, 3), "faces": (M, 3) }]
    """
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usdc_path))
    root_name = usdc_path.stem
    root_prim = stage.GetPrimAtPath(f"/{root_name}")

    link_transforms = {}
    cloth_data = {}
    frame_count = 0

    for child in root_prim.GetChildren():
        if child.GetName() == "rigid":
            for link_prim in child.GetChildren():
                xf = UsdGeom.Xformable(link_prim)
                ops = xf.GetOrderedXformOps()
                if not ops:
                    continue
                ts = ops[0].GetTimeSamples()
                frame_count = max(frame_count, len(ts))

                link_name = _usd_name_to_link(link_prim.GetName())
                # Skip duplicate sub-meshes (e.g. left_knee_link_glb_1)
                if link_name in link_transforms:
                    continue

                # USD is row-vector convention, transpose to column-vector
                transforms = np.zeros((len(ts), 4, 4), dtype=np.float64)
                for i, t in enumerate(ts):
                    m = np.array(ops[0].Get(t))
                    transforms[i] = m.T
                link_transforms[link_name] = transforms

        elif child.GetName() == "deformable":
            for cloth_prim in child.GetChildren():
                mesh = UsdGeom.Mesh(cloth_prim)
                pts_attr = mesh.GetPointsAttr()
                ts = pts_attr.GetTimeSamples()
                frame_count = max(frame_count, len(ts))

                face_indices = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
                faces = face_indices.reshape(-1, 3)

                first_pts = np.array(pts_attr.Get(ts[0]), dtype=np.float64)
                positions = np.zeros((len(ts), len(first_pts), 3), dtype=np.float64)
                for i, t in enumerate(ts):
                    positions[i] = np.array(pts_attr.Get(t), dtype=np.float64)

                cloth_data[cloth_prim.GetName()] = {
                    "positions": positions,
                    "faces": faces,
                }

    return {
        "frame_count": frame_count,
        "link_transforms": link_transforms,
        "cloth": cloth_data,
    }


# ── IK: compute qpos from world transforms ──────────────────────────────────


def _extract_joint_angle(R_rel: np.ndarray, axis: np.ndarray) -> float:
    """Extract rotation angle about a given axis from a rotation matrix."""
    rot = Rotation.from_matrix(R_rel)
    rotvec = rot.as_rotvec()
    # Project onto joint axis
    angle = np.dot(rotvec, axis)
    return angle


def build_ik_chain(urdf_path: str) -> list[dict]:
    """Parse URDF to build the IK chain for computing qpos.

    Returns a list of joint dicts, one per revolute joint, ordered by Genesis
    qs_idx.  Each dict has: name, parent_link, child_link, origin (4x4), axis.
    """
    from genesis.ext.urdfpy.urdf import URDF

    urdf = URDF.load(urdf_path)
    chain = []
    for j in urdf.joints:
        if j.joint_type == "fixed":
            continue
        origin = j.origin if j.origin is not None else np.eye(4)
        chain.append(
            {
                "name": j.name,
                "parent_link": j.parent,
                "child_link": j.child,
                "origin": origin.copy(),
                "axis": np.array(j.axis, dtype=np.float64),
            }
        )
    return chain


def _build_fixed_joint_map(urdf_path: str) -> dict[str, tuple[str, np.ndarray]]:
    """Map fixed-joint children to their parent link and static offset.

    Returns {child_link_name: (parent_link_name, T_parent_to_child)}.
    """
    from genesis.ext.urdfpy.urdf import URDF

    urdf = URDF.load(urdf_path)
    fixed_map = {}
    for j in urdf.joints:
        if j.joint_type == "fixed":
            origin = j.origin if j.origin is not None else np.eye(4)
            fixed_map[j.child] = (j.parent, origin.copy())
    return fixed_map


def compute_qpos_from_transforms(
    link_transforms: dict[str, np.ndarray],
    ik_chain: list[dict],
    fixed_joint_map: dict[str, tuple[str, np.ndarray]],
    frame: int,
) -> np.ndarray:
    """Compute joint angles for one frame from link world transforms.

    For each revolute joint:
      T_child = T_parent @ T_joint_origin @ R(q, axis)
      => R(q, axis) = (T_parent @ T_joint_origin)^-1 @ T_child
      => q = project rotation onto joint axis
    """
    qpos = np.zeros(len(ik_chain), dtype=np.float64)

    def _get_link_tf(link_name: str) -> np.ndarray | None:
        """Get link world transform, resolving fixed-joint chains."""
        if link_name in link_transforms:
            tf = link_transforms[link_name]
            return tf[frame] if frame < len(tf) else tf[-1]
        # Fixed joint child — compute from parent
        if link_name in fixed_joint_map:
            parent_name, offset = fixed_joint_map[link_name]
            parent_tf = _get_link_tf(parent_name)
            if parent_tf is not None:
                return parent_tf @ offset
        return None

    for i, jinfo in enumerate(ik_chain):
        T_parent = _get_link_tf(jinfo["parent_link"])
        T_child = _get_link_tf(jinfo["child_link"])
        if T_parent is None or T_child is None:
            continue

        # T_child = T_parent @ T_joint_origin @ R(q, axis)
        T_joint_frame = T_parent @ jinfo["origin"]
        T_rel = np.linalg.inv(T_joint_frame) @ T_child
        R_rel = T_rel[:3, :3]
        qpos[i] = _extract_joint_angle(R_rel, jinfo["axis"])

    return qpos


# ── Main ─────────────────────────────────────────────────────────────────────


class CostumeReplay(TrajectoryReplay):
    name = "costume"
    cam_pos = (2.0, -1.5, 0.8)
    cam_lookat = (0.0, 0.0, 0.2)
    cam_fov = 45

    def add_args(self, parser):
        parser.add_argument(
            "--trajectory",
            type=str,
            default="d0_no_lower",
            choices=AVAILABLE_TRAJECTORIES,
        )

    def load_trajectory(self) -> int:
        usdc_path = _PACK_DIR / f"{self.args.trajectory}.usdc"
        if not usdc_path.exists():
            raise FileNotFoundError(f"USD file not found: {usdc_path}")

        print(f"[replay] Loading {usdc_path.name} ...")
        self._traj = load_usd_trajectory(usdc_path)
        self._ik_chain = build_ik_chain(URDF_PATH)
        self._fixed_joint_map = _build_fixed_joint_map(URDF_PATH)

        frame_count = self._traj["frame_count"]
        print(
            f"[replay] {frame_count} frames, "
            f"{len(self._traj['link_transforms'])} rigid links, "
            f"{len(self._traj['cloth'])} cloth"
        )
        return frame_count

    def build_scene(self, scene):
        import genesis as gs

        # Ground
        scene.add_entity(gs.morphs.Plane())

        # Robot (URDF)
        self._robot = scene.add_entity(
            gs.morphs.URDF(file=URDF_PATH, pos=(0.0, 0.0, 0.0), fixed=True, collision=False),
            surface=gs.surfaces.BSDF(),
        )

        # Cloth entities
        cloth_data = self._traj["cloth"]
        for usd_name in cloth_data:
            piece_name = _usd_cloth_to_piece_name(usd_name)
            obj_path = _KIMONO_DIR / f"{piece_name}.obj"
            if not obj_path.exists():
                print(f"[warn] cloth mesh not found: {obj_path.name}")
                continue
            # Prefer GLB (has PBR material baked in), fall back to OBJ
            glb_path = _KIMONO_DIR / f"{piece_name}.glb"
            mesh_path = glb_path if glb_path.exists() else obj_path
            ent = scene.add_entity(
                morph=gs.morphs.Mesh(file=str(mesh_path)),
                material=gs.materials.FEM.Cloth(E=1e4, rho=200.0, thickness=1e-4),
                surface=gs.surfaces.BSDF(),
                name=usd_name,
            )
            self._fem_entities[usd_name] = ent

    def post_build(self):
        # Build Genesis joint name -> qs_idx mapping
        self._joint_name_to_qs_idx: dict[str, int] = {}
        for joint in self._robot.joints:
            if joint.n_qs > 0:
                self._joint_name_to_qs_idx[joint.name] = joint.qs_idx_local[0]

    def apply_frame(self, scene, frame_idx):
        link_transforms = self._traj["link_transforms"]
        cloth_data = self._traj["cloth"]
        frame_count = self._traj["frame_count"]

        if frame_idx < 0 or frame_idx >= frame_count:
            return

        # Compute qpos from USD transforms via IK
        qpos_urdf_order = compute_qpos_from_transforms(
            link_transforms,
            self._ik_chain,
            self._fixed_joint_map,
            frame_idx,
        )

        # Reorder from URDF joint order to Genesis qs_idx order
        qpos = self._robot.get_qpos()
        # With n_envs>=1, qpos has shape (B, n_dofs); squeeze for scalar indexing.
        batched = qpos.ndim > 1
        qpos_flat = qpos[0] if batched else qpos
        for j_idx, jinfo in enumerate(self._ik_chain):
            gs_idx = self._joint_name_to_qs_idx.get(jinfo["name"])
            if gs_idx is not None:
                qpos_flat[gs_idx] = qpos_urdf_order[j_idx]
        if batched:
            qpos[0] = qpos_flat

        # Set pelvis (base) position from USD
        if "pelvis" in link_transforms and frame_idx < link_transforms["pelvis"].shape[0]:
            tf = link_transforms["pelvis"][frame_idx]
            pos = tf[:3, 3]
            self._robot.set_pos(pos)
            rot = Rotation.from_matrix(tf[:3, :3])
            xyzw = rot.as_quat()
            self._robot.set_quat(np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]]))

        self._robot.set_qpos(qpos, zero_velocity=True)

        # Cloth
        for usd_name, ent in self._fem_entities.items():
            cdata = cloth_data[usd_name]
            if frame_idx < cdata["positions"].shape[0]:
                ent.set_position(cdata["positions"][frame_idx])


if __name__ == "__main__":
    CostumeReplay().run()
