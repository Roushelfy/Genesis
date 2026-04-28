"""
Replay exported cooking simulation sequences using the standard replay framework.

Loads ``meta.json`` from a sequence directory exported by
``cook_with_teleop/scripts/replay_cook.py --export-seq``, then sets up:

- Rigid parts (pan, spatula, broccoli) with per-frame 4×4 transforms
- FEM cloth (tomato slices) with per-frame vertex positions
- FEM rope (noodles, split per strand) with per-frame vertex positions
- Optional robot (MARVIN) with qpos + base pose

Usage
-----
    python examples/IPC_Solver/replay_robot_cook_traj.py --seq-dir DemoAssets/cook_with_teleop/seq_v1

Interactive viewer (default)
    --loop                  Loop replay continuously
    --speed FLOAT           Playback speed multiplier (default: 1.0)
                            Keyboard: , = half speed  . = double speed
    --start-frame N         Start from frame N (BACKSPACE also resets here)
    --end-frame N           Stop at frame N exclusive (default: last frame)
    --camera-traj MODE      Drive camera automatically: surround | full | ego | custom
                            K key   = log current frame + camera pose to stdout
                            [ / ]   = jump to prev/next custom keyframe (pauses)

Live preview (interactive mode only)
    --preview               Open an OpenCV window with live Luisa-rendered view
    --preview --nyx         Add a second window with Nyx render alongside
    --preview-spp N         Samples-per-pixel for the preview camera (default: 64)
    --no-raytracer          Suppress Luisa renderer:
                              with --nyx  -> Nyx-only preview window
                              without     -> no preview window at all

Render to video
    --render                Record a video (Luisa by default)
    --render --nyx          Record with Nyx renderer
    --render --camera-traj MODE  Use a camera trajectory for the render
    --spp N                 Samples-per-pixel for the render camera (default: 256)
    --save-frames           Also save each frame as a PNG alongside the video

Shared camera / render options
    --res W H               Resolution for viewer, preview, and render cameras
                            (default: 1920 1080)
    --nyx                   Use Nyx renderer (preview or render)
    --dof                   Enable depth-of-field (thinlens model)
    --aperture F            Aperture f-number for DOF (default: 1.4, lower = shallower)
    --focus-dist METRES     Focus distance (default: auto from cam_pos -> lookat)
    --focal-len METRES      Focal length (default: 0.05 = 50 mm)
    --exposure EV           Exposure stops (default: 0.0)
    --tone-mapping MODE     Tone mapping operator: none | aces | uncharted2

Sequence
    --seq-dir PATH          Sequence directory containing meta.json
                            (default: DemoAssets/cook_with_teleop/seq)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from _replay_common import TrajectoryReplay

_REPO = Path(__file__).resolve().parents[2]
_DEMO = _REPO / "DemoAssets"
_COOK = _DEMO / "cook_with_teleop"

_DEFAULT_SEQ = _COOK / "seq"

TABLE_GLB    = str(_DEMO / "trashbag" / "work_table.glb")
STOVETOP_GLB = str(_COOK / "stovetop.glb")

# Camera pose — matches the hanger_sharpa scene (same table height / work area)
_CAM_POS    = (1.5122, -0.767, 2.0931)
_CAM_LOOKAT = (0.838, -0.3497, 1.5337)
_CAM_FOV    = 40

# Sphere-light rig — three-light key/fill/rim setup (from ipc_robot_cook.py)
_LIGHTS = [
    # key: warm, above-left, casts shadows across hand/pan
    {"pos": (0.5,   1.1,  2.4),  "radius": 0.2,  "color": (1.0, 0.97, 0.92), "intensity": 50.0},
    # fill: cool, soft, opposite side
    {"pos": (0.5,  -1.8,  4.2),  "radius": 1.0,  "color": (0.48, 0.52, 0.6), "intensity": 1.0},
    # rim: hard, behind, separates subjects from background
    {"pos": (-0.8, -3.0,  0.5),  "radius": 0.25, "color": (0.8, 0.88, 1.0),  "intensity": 150.0},
]
_NYX_RADIUS_SCALE    = 1.0
_NYX_INTENSITY_SCALE = 0.2

# Per-object fallback colors (used when object name has no named preset)
_ENTITY_COLORS = {
    "pan":     (0.6, 0.6, 0.65, 1.0),
    "spatula": (0.5, 0.5, 0.55, 1.0),
}
_BROC_COLOR   = (0.2, 0.6, 0.15, 1.0)
_TOMATO_COLOR = (0.85, 0.15, 0.1, 1.0)
_NOODLE_COLOR = (0.92, 0.86, 0.55, 1.0)


# ---------------------------------------------------------------------------
# Mesh helpers (ported from ipc_robot_cook.py)
# ---------------------------------------------------------------------------

def _euler_xyz_deg_to_quat_wxyz(rx_deg, ry_deg, rz_deg):
    """Convert XYZ-Euler (degrees) to a (w, x, y, z) quaternion."""
    r = Rotation.from_euler("XYZ", [rx_deg, ry_deg, rz_deg], degrees=True)
    xyzw = r.as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float32)


def _tf_to_pos_quat(tf: np.ndarray):
    """Extract (pos, quat_wxyz) from a 4×4 homogeneous transform."""
    pos = tf[:3, 3].copy()
    xyzw = Rotation.from_matrix(tf[:3, :3]).as_quat()
    return pos, np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)


def _apply_tf_to_verts(verts: np.ndarray, tf: np.ndarray) -> np.ndarray:
    return (tf[:3, :3] @ verts.T).T + tf[:3, 3]


def _write_obj(path: Path, verts: np.ndarray, faces: np.ndarray | None = None,
               edges: np.ndarray | None = None):
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        if faces is not None:
            for tri in faces:
                f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
        if edges is not None:
            for e in edges:
                f.write(f"l {e[0]+1} {e[1]+1}\n")


def _read_obj(path: Path):
    verts, faces, edges = [], [], []
    with open(path) as f:
        for raw in f:
            if raw.startswith("v "):
                verts.append([float(x) for x in raw.split()[1:4]])
            elif raw.startswith("f "):
                faces.append([int(x.split("/")[0]) - 1 for x in raw.split()[1:4]])
            elif raw.startswith("l "):
                idx = [int(x) - 1 for x in raw.split()[1:]]
                for a, b in zip(idx, idx[1:]):
                    edges.append([a, b])
    return (np.array(verts, dtype=np.float64),
            np.array(faces, dtype=np.int32) if faces else None,
            np.array(edges, dtype=np.int32) if edges else None)


def _prepare_rigid_world_mesh(seq_dir: Path, name: str, tf0: np.ndarray) -> Path:
    """Transform a local-space mesh.obj into frame-0 world positions (cached)."""
    world_path = seq_dir / name / "mesh_world.obj"
    if world_path.exists():
        return world_path
    local_verts, local_faces, _ = _read_obj(seq_dir / name / "mesh.obj")
    world_verts = _apply_tf_to_verts(local_verts, tf0)
    _write_obj(world_path, world_verts, faces=local_faces)
    print(f"[mesh] {name}: wrote world-space mesh ({len(local_verts)} verts)")
    return world_path


def _detect_noodle_strands(verts: np.ndarray, edges: np.ndarray):
    """Detect individual noodle strands via connectivity flood fill."""
    n_verts = len(verts)
    if edges is None or len(edges) == 0:
        return [(0, n_verts)]
    adj: dict[int, set[int]] = {i: set() for i in range(n_verts)}
    for e in edges:
        a, b = int(e[0]), int(e[1])
        adj[a].add(b)
        adj[b].add(a)
    visited = [False] * n_verts
    strands = []
    for start in range(n_verts):
        if visited[start]:
            continue
        chain, queue = [], [start]
        visited[start] = True
        while queue:
            v = queue.pop(0)
            chain.append(v)
            for nb in adj[v]:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
        chain.sort()
        strands.append((chain[0], len(chain)))
    return strands


def _split_noodle_meshes(seq_dir: Path, n_total_verts: int):
    """Split the combined noodle line mesh into per-strand OBJ files.

    Returns (list_of_paths, verts_per_noodle).
    """
    noodle_dir = seq_dir / "noodles"
    verts, _, edges = _read_obj(noodle_dir / "mesh.obj")
    strands = _detect_noodle_strands(verts, edges)
    n_noodles = len(strands)
    vpn = strands[0][1] if strands else n_total_verts

    first_path = noodle_dir / "noodle_0.obj"
    if first_path.exists():
        old_v, _, _ = _read_obj(first_path)
        if len(old_v) != vpn:  # stale split — regenerate
            for i in range(500):
                p = noodle_dir / f"noodle_{i}.obj"
                if p.exists():
                    p.unlink()
                else:
                    break

    if not first_path.exists():
        for i, (v_start, v_count) in enumerate(strands):
            nv = verts[v_start:v_start + v_count]
            ne = np.array([[j, j + 1] for j in range(v_count - 1)], dtype=np.int32)
            _write_obj(noodle_dir / f"noodle_{i}.obj", nv, edges=ne)

    paths = [noodle_dir / f"noodle_{i}.obj" for i in range(n_noodles)]
    print(f"[mesh] split {n_noodles} noodle meshes ({vpn} verts each)")
    return paths, vpn


# ---------------------------------------------------------------------------
# Replay class
# ---------------------------------------------------------------------------

class RobotCookReplay(TrajectoryReplay):
    """Replay exported cooking simulation sequences."""

    name       = "robot_cook"
    cam_pos    = _CAM_POS
    cam_lookat = _CAM_LOOKAT
    cam_fov    = _CAM_FOV

    def add_args(self, parser):
        parser.add_argument(
            "--seq-dir",
            type=str,
            default=str(_DEFAULT_SEQ),
            help="Sequence directory containing meta.json "
                 "(default: DemoAssets/cook_with_teleop/seq)",
        )

    def load_trajectory(self) -> int:
        self._seq_dir = Path(self.args.seq_dir)
        meta_path = self._seq_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.json not found in {self._seq_dir}")
        self._meta = json.loads(meta_path.read_text(encoding="utf-8"))
        n_frames = self._meta["frame_count"]

        # FPS from simulation dt.  The cook export rate is typically 10× slower
        # than the sim rate, so divide by 10 to get the effective replay fps.
        dt = self._meta.get("dt", 0.005)
        self.fps = min(int(1.0 / dt / 10), 60)

        # Load per-object arrays, split by type
        rigid_raw: dict[str, np.ndarray] = {}
        fem_raw:   dict[str, np.ndarray] = {}
        self._rod_data: dict[str, np.ndarray] = {}
        for name, info in self._meta["objects"].items():
            npy_path = self._seq_dir / info["data"]
            if not npy_path.exists():
                print(f"[warn] {npy_path} not found, skipping {name}")
                continue
            arr = np.load(str(npy_path))
            if info["type"] == "rigid":
                rigid_raw[name] = arr
            elif info["type"] == "rod":
                self._rod_data[name] = arr
            else:
                fem_raw[name] = arr

        # Convert 4×4 rigid transforms to (N, 7) pos+quat so the base class
        # apply_frame can drive them with set_pos / set_quat.
        # Each mesh is pre-transformed to frame-0 world space; T_rel = T_i @ T0⁻¹
        # gives the incremental pose from frame 0 to frame i.
        self._rigid_T0: dict[str, np.ndarray] = {}
        self._rigid_data = {}
        for name, tfs in rigid_raw.items():
            T0 = tfs[0]
            T0_inv = np.linalg.inv(T0)
            self._rigid_T0[name] = T0
            poses = np.zeros((len(tfs), 7), dtype=np.float32)
            for i, tf in enumerate(tfs):
                pos, quat = _tf_to_pos_quat(tf @ T0_inv)
                poses[i, :3] = pos
                poses[i, 3:] = quat
            self._rigid_data[name] = poses

        # FEM data: vertex positions (N, n_verts, 3)
        self._fem_data = fem_raw

        # Optional robot
        self._robot_seq: dict | None = None
        robot_npz = self._seq_dir / "robot.npz"
        if robot_npz.exists():
            data = np.load(str(robot_npz), allow_pickle=True)
            base_rpy = data["base_rpy_deg"].astype(np.float64)
            self._robot_seq = {
                "qpos":      data["qpos"].astype(np.float32),
                "urdf":      str(data["urdf"]).replace("\\", "/"),
                "base_pos":  data["base_pos"].astype(np.float32),
                "base_quat": _euler_xyz_deg_to_quat_wxyz(*base_rpy),
            }
            self._joint_qpos = self._robot_seq["qpos"]

        print(f"[replay] {n_frames} frames  dt={dt:.4f}s  fps={self.fps}")
        print(f"  rigid:  {list(self._rigid_data)}")
        print(f"  fem:    {list(self._fem_data)}")
        print(f"  rods:   {list(self._rod_data)}")
        if self._robot_seq is not None:
            print(f"  robot:  {Path(self._robot_seq['urdf']).name}  "
                  f"qpos={self._robot_seq['qpos'].shape}")
        return n_frames

    def build_scene(self, scene):
        import genesis as gs

        seq_dir = self._seq_dir

        # Ground — dark box so the surface color override works
        scene.add_entity(
            gs.morphs.Box(size=(20.0, 20.0, 0.02), pos=(0.0, 0.0, -0.143), fixed=True),
            surface=gs.surfaces.Plastic(color=(0.02, 0.02, 0.02)),
        )

        # Work table (same GLB + surface as trashbag_sharpa)
        scene.add_entity(
            gs.morphs.Mesh(
                file=TABLE_GLB,
                pos=(0.55, 0.0, -0.09),
                scale=(1.14, 1.0, 1.47),
                fixed=True,
                file_meshes_are_zup=True,
                convexify=False,
            ),
            surface=gs.surfaces.BSDF(
                roughness=0.45,
                metallic=0.0,
                ior=1.45,
            ),
            vis_mode="visual",
        )

        # Stovetop (scale/position TBD — tune with --preview)
        scene.add_entity(
            gs.morphs.Mesh(
                file=STOVETOP_GLB,
                pos=(0.55, 0.05, 0.683),
                euler=(0, 0, -90),
                scale=1.0,
                fixed=True,
                convexify=False,
            ),
            vis_mode="visual",
        )

        # Rigid entities — meshes pre-transformed to frame-0 world space
        self._rigid_entities = {}
        for name in self._rigid_data:
            mesh_path = seq_dir / name / "mesh.obj"
            if not mesh_path.exists():
                print(f"[warn] mesh.obj not found for rigid '{name}', skipping")
                continue
            world_mesh = _prepare_rigid_world_mesh(seq_dir, name, self._rigid_T0[name])
            color = _ENTITY_COLORS.get(name)
            if color is None:
                color = _BROC_COLOR if name.startswith("broc") else (0.7, 0.7, 0.7, 1.0)
            self._rigid_entities[name] = scene.add_entity(
                morph=gs.morphs.Mesh(file=str(world_mesh), fixed=True, collision=False),
                material=gs.materials.Rigid(),
                surface=gs.surfaces.Default(color=color),
                name=name,
            )

        # FEM cloth entities (e.g. tomato slices)
        self._fem_entities = {}
        for name in self._fem_data:
            mesh_path = seq_dir / name / "mesh.obj"
            if not mesh_path.exists():
                print(f"[warn] mesh.obj not found for FEM '{name}', skipping")
                continue
            color = _TOMATO_COLOR if name.startswith("tomato") else (0.7, 0.7, 0.7, 1.0)
            self._fem_entities[name] = scene.add_entity(
                morph=gs.morphs.Mesh(file=str(mesh_path)),
                material=gs.materials.FEM.Cloth(E=1e4, nu=0.45, rho=1000.0),
                surface=gs.surfaces.Default(color=color),
                name=name,
            )

        # FEM rope entities (noodles, split per strand)
        self._noodle_entities: list = []
        self._noodle_vpn: int = 0
        self._noodle_rod_key: str | None = None
        for rod_name, rod_arr in self._rod_data.items():
            n_total_verts = rod_arr.shape[1]
            noodle_meshes, vpn = _split_noodle_meshes(seq_dir, n_total_verts)
            self._noodle_vpn = vpn
            self._noodle_rod_key = rod_name
            for i, mp in enumerate(noodle_meshes):
                self._noodle_entities.append(scene.add_entity(
                    morph=gs.morphs.Mesh(file=str(mp)),
                    material=gs.materials.FEM.Rope(E=1e6, rho=100.0, thickness=0.001),
                    surface=gs.surfaces.Default(color=_NOODLE_COLOR),
                    name=f"noodle_{i}",
                ))
            print(f"[scene] {len(noodle_meshes)} noodle FEM.Rope entities ({vpn} verts each)")
            break  # only one rod group expected

        # Optional robot
        if self._robot_seq is not None:
            urdf_path = _REPO / self._robot_seq["urdf"]
            print(f"[robot] Loading {urdf_path.name} at base_pos={self._robot_seq['base_pos']}")
            self._robot = scene.add_entity(
                morph=gs.morphs.URDF(
                    file=str(urdf_path),
                    fixed=True,
                    collision=False,
                    pos=tuple(self._robot_seq["base_pos"].tolist()),
                    quat=tuple(self._robot_seq["base_quat"].tolist()),
                ),
                vis_mode="visual",
            )

    def apply_frame(self, scene, frame_idx: int) -> None:
        # Standard handling: robot qpos, rigid pos/quat, FEM positions
        super().apply_frame(scene, frame_idx)
        # Noodle strands — rod FEM data is one combined array, split by strand
        if self._noodle_rod_key and self._noodle_entities:
            rod_arr = self._rod_data[self._noodle_rod_key]
            if frame_idx < rod_arr.shape[0]:
                frame_pos = rod_arr[frame_idx]
                vpn = self._noodle_vpn
                for ni, ent in enumerate(self._noodle_entities):
                    ent.set_position(frame_pos[ni * vpn: (ni + 1) * vpn])

    def make_renderer(self):
        import genesis as gs
        from genesis.options.renderers import SphereLight

        return gs.renderers.RayTracer(
            logging_level="warning",
            tracing_depth=32,
            env_surface=gs.surfaces.Emission(
                emissive_texture=gs.textures.ColorTexture(color=(0.01, 0.01, 0.01)),
            ),
            env_radius=100.0,
            env_euler=(0, 0, 20),
            lights=[
                SphereLight(
                    pos=l["pos"],
                    radius=l["radius"],
                    color=l["color"],
                    intensity=l["intensity"],
                )
                for l in _LIGHTS
            ],
        )

    def nyx_lights(self):
        return [
            {
                "type":      "point",
                "pos":       l["pos"],
                "radius":    float(l["radius"])    * _NYX_RADIUS_SCALE,
                "color":     l["color"],
                "intensity": float(l["intensity"]) * _NYX_INTENSITY_SCALE,
            }
            for l in _LIGHTS
        ]

    def custom_camera_keyframes(self):
        # Use K key in interactive mode to capture good shots, then paste here.
        # Format: (frame, pos, lookat[, up[, ease_in[, ease_out]]])
        return []


if __name__ == "__main__":
    RobotCookReplay().run()
