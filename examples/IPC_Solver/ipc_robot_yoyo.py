"""
Replay exported yoyo simulation sequences.

Loads OBJ meshes for the static rest shapes and per-frame NPY files
(transforms for rigid bodies, positions for the string, joint angles)
exported by ``IPC-Samples/python/Yoyo/urdf_controller_main.py``.

Usage:
    python ipc_robot_yoyo.py                # GUI playback
    python ipc_robot_yoyo.py --no-gui       # headless verification
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_YOYO_DIR = _REPO_ROOT / "IPC-Samples" / "python" / "Yoyo"
_ASSET_DIR = _YOYO_DIR / "results" / "v3"
_SEQ_DIR = _ASSET_DIR / "seq"


def _parse_obj(path: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    edges: list[list[int]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                idx = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                for k in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[k], idx[k + 1]])
            elif parts[0] == "l":
                idx = [int(p) - 1 for p in parts[1:]]
                for k in range(len(idx) - 1):
                    edges.append([idx[k], idx[k + 1]])
    v = np.array(verts, dtype=np.float64) if verts else np.zeros((0, 3))
    f = np.array(faces, dtype=np.int32) if faces else None
    e = np.array(edges, dtype=np.int32) if edges else None
    return v, f, e


def _discover_frames(seq_dir: Path) -> list[int]:
    """Find all exported frame numbers by scanning one subdirectory."""
    for sub in sorted(seq_dir.iterdir()):
        if sub.is_dir():
            frames = sorted(
                int(p.stem) for p in sub.glob("*.npy") if p.stem.isdigit()
            )
            if frames:
                return frames
    return []


def _apply_transform(verts: np.ndarray, tf: np.ndarray) -> np.ndarray:
    R = tf[:3, :3]
    t = tf[:3, 3]
    return verts @ R.T + t



def _load_all_meshes_from_seq(seq_dir: Path) -> dict[str, dict]:
    """Discover and load mesh.obj from each subdirectory of seq_dir."""
    meshes: dict[str, dict] = {}
    for sub in sorted(seq_dir.iterdir()):
        if not sub.is_dir():
            continue
        obj_path = sub / "mesh.obj"
        if not obj_path.exists():
            continue
        name = sub.name
        verts, faces, edges = _parse_obj(obj_path)
        meshes[name] = {"verts": verts, "faces": faces, "edges": edges, "name": name}
    return meshes


def run_gui(frames: list[int]) -> None:
    import polyscope as ps
    from polyscope import imgui

    ps.init()
    ps.set_up_dir("z_up")

    all_meshes = _load_all_meshes_from_seq(_SEQ_DIR)
    meshes: dict[str, dict] = {}
    for name, entry in all_meshes.items():
        verts, faces, edges = entry["verts"], entry["faces"], entry["edges"]
        is_string = "string" in name
        if is_string and edges is not None and len(edges) > 0:
            cn = ps.register_curve_network(name, verts, edges, radius=0.0004)
            entry["ps_curve"] = cn
        elif faces is not None and len(faces) > 0:
            sm = ps.register_surface_mesh(name, verts, faces)
            entry["ps_surf"] = sm
        meshes[name] = entry

    state = {"frame_idx": 0, "playing": False}

    def _apply_frame(fidx: int) -> None:
        if fidx < 0 or fidx >= len(frames):
            return
        frame = frames[fidx]

        for name, entry in meshes.items():
            npy_path = _SEQ_DIR / name / f"{frame}.npy"
            if not npy_path.exists():
                continue
            data = np.load(str(npy_path), allow_pickle=True)
            if "ps_curve" in entry:
                entry["ps_curve"].update_node_positions(data.reshape(-1, 3))
            elif "ps_surf" in entry:
                if "string" in name:
                    entry["ps_surf"].update_vertex_positions(data.reshape(-1, 3))
                else:
                    tf = data.reshape(4, 4)
                    entry["ps_surf"].update_vertex_positions(
                        _apply_transform(entry["verts"], tf)
                    )

    _apply_frame(0)

    def on_update() -> None:
        imgui.Text("=== Yoyo Replay ===")
        imgui.Text(f"Frames: {len(frames)}  ({frames[0]}..{frames[-1]})")
        imgui.Separator()

        changed, idx = imgui.SliderInt("Frame", state["frame_idx"], 0, len(frames) - 1)
        if changed:
            state["frame_idx"] = idx
            _apply_frame(idx)

        if imgui.Button("Play / Pause"):
            state["playing"] = not state["playing"]

        if state["playing"]:
            state["frame_idx"] = min(state["frame_idx"] + 1, len(frames) - 1)
            _apply_frame(state["frame_idx"])
            if state["frame_idx"] >= len(frames) - 1:
                state["playing"] = False

        joints_path = _SEQ_DIR / "joints" / f"{frames[state['frame_idx']]}.npy"
        if joints_path.exists():
            jdata = np.load(str(joints_path), allow_pickle=True).item()
            if isinstance(jdata, dict) and imgui.TreeNode("Joint Angles"):
                for jn, jv in sorted(jdata.items()):
                    imgui.Text(f"  {jn}: {jv:.4f}")
                imgui.TreePop()

    ps.set_user_callback(on_update)
    ps.show()


def run_no_gui(frames: list[int]) -> None:
    print(f"[no-gui] {len(frames)} frames available ({frames[0]}..{frames[-1]})")

    all_meshes = _load_all_meshes_from_seq(_SEQ_DIR)
    for name, entry in all_meshes.items():
        print(f"[mesh] {name}: {entry['verts'].shape[0]} verts (from mesh.obj)")

    test_frames = [frames[0], frames[len(frames) // 2], frames[-1]]
    for f in test_frames:
        missing = []
        for sub in sorted(_SEQ_DIR.iterdir()):
            if not sub.is_dir():
                continue
            npy = sub / f"{f}.npy"
            if not npy.exists() and sub.name != "joints":
                missing.append(sub.name)
        joints_path = _SEQ_DIR / "joints" / f"{f}.npy"
        j_ok = joints_path.exists()
        status = "OK" if not missing and j_ok else f"MISSING: {missing}, joints={'OK' if j_ok else 'MISS'}"
        print(f"[check] frame {f}: {status}")

    print("[no-gui] Done.")


def main() -> None:
    global _SEQ_DIR

    parser = argparse.ArgumentParser(description="Replay exported yoyo simulation.")
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--seq-dir", type=str, default=str(_SEQ_DIR))
    args = parser.parse_args()

    _SEQ_DIR = Path(args.seq_dir)

    if not _SEQ_DIR.exists():
        print(f"[error] Sequence directory not found: {_SEQ_DIR}")
        return

    frames = _discover_frames(_SEQ_DIR)
    if not frames:
        print(f"[error] No frames found in {_SEQ_DIR}")
        return

    print(f"[replay] Found {len(frames)} frames in {_SEQ_DIR}")

    if args.no_gui:
        run_no_gui(frames)
    else:
        run_gui(frames)


if __name__ == "__main__":
    main()
