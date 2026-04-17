"""
Live-reload preview of fixed meshes from parts/fixed/ in Polyscope.
Falls back to parts/<name>.obj for any part not yet fixed.
Watches the fixed/ folder for changes and auto-reloads on add/modify.

Usage:  python view_fixed_parts.py
"""

from pathlib import Path
import json
import os
import numpy as np
import polyscope as ps
from polyscope import imgui

PARTS_DIR = Path(__file__).parent / "assets"
FIXED_DIR = PARTS_DIR / "fixed"
JSON_PATH = PARTS_DIR / "transforms.json"

COLORS = [
    (0.85, 0.32, 0.30),
    (0.30, 0.68, 0.85),
    (0.40, 0.82, 0.45),
    (0.92, 0.72, 0.20),
    (0.65, 0.40, 0.85),
    (0.90, 0.55, 0.25),
    (0.35, 0.80, 0.75),
    (0.80, 0.45, 0.65),
    (0.55, 0.55, 0.55),
]


def load_obj(filepath: Path):
    verts, faces = [], []
    with open(filepath) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:4]]
                faces.append(idx)
    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int32)


def get_file_stamp(path: Path):
    """Return (mtime, size) or None if file doesn't exist."""
    try:
        st = path.stat()
        return (st.st_mtime, st.st_size)
    except FileNotFoundError:
        return None


def resolve_source(obj_name: str):
    fixed = FIXED_DIR / obj_name
    orig = PARTS_DIR / obj_name
    if fixed.exists():
        return fixed, "fixed"
    return orig, "original"


def load_all_parts(meta, part_names):
    """Load all meshes and register in polyscope. Returns stamp dict."""
    stamps = {}
    for i, name in enumerate(part_names):
        obj_name = meta[name]["obj"]
        src, tag = resolve_source(obj_name)
        verts, faces = load_obj(src)
        color = COLORS[i % len(COLORS)]

        sm = ps.register_surface_mesh(name, verts, faces)
        sm.set_color(color)
        sm.set_smooth_shade(True)
        sm.set_edge_width(0.5)

        stamps[name] = {
            "path": src,
            "tag": tag,
            "stamp": get_file_stamp(src),
            "fixed_stamp": get_file_stamp(FIXED_DIR / obj_name),
        }
    return stamps


def check_and_reload(meta, part_names, stamps):
    """Check for changes and reload only the parts that changed."""
    changed = []
    for i, name in enumerate(part_names):
        obj_name = meta[name]["obj"]
        src, tag = resolve_source(obj_name)
        cur_stamp = get_file_stamp(src)
        cur_fixed_stamp = get_file_stamp(FIXED_DIR / obj_name)

        prev = stamps[name]
        source_changed = (cur_stamp != prev["stamp"])
        fixed_appeared = (cur_fixed_stamp != prev["fixed_stamp"])

        if source_changed or fixed_appeared or (src != prev["path"]):
            try:
                verts, faces = load_obj(src)
            except Exception as e:
                print(f"  [!] {name}: failed to reload ({e})")
                continue

            color = COLORS[i % len(COLORS)]
            sm = ps.register_surface_mesh(name, verts, faces)
            sm.set_color(color)
            sm.set_smooth_shade(True)
            sm.set_edge_width(0.5)

            stamps[name] = {
                "path": src,
                "tag": tag,
                "stamp": cur_stamp,
                "fixed_stamp": cur_fixed_stamp,
            }
            changed.append((name, tag, len(verts), len(faces)))

    return changed


def main():
    with open(JSON_PATH, encoding="utf-8") as f:
        meta = json.load(f)

    part_names = [k for k in meta if k != "_meta"]

    FIXED_DIR.mkdir(parents=True, exist_ok=True)

    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_window_size(1600, 1000)

    print("Loading parts ...")
    stamps = load_all_parts(meta, part_names)
    for name, info in stamps.items():
        print(f"  {name:<20s}  {info['tag']:<10s}  {info['path'].name}")
    print(f"\nWatching {FIXED_DIR} for changes ...")

    ps.reset_camera_to_home_view()

    frame_counter = 0
    POLL_INTERVAL = 30  # check every ~30 frames (~0.5s at 60fps)

    def on_update():
        nonlocal frame_counter
        frame_counter += 1

        imgui.TextUnformatted("Live-reload: watching parts/fixed/")
        imgui.Separator()

        for name in part_names:
            info = stamps[name]
            tag_color = (0.3, 0.9, 0.4, 1.0) if info["tag"] == "fixed" else (0.7, 0.7, 0.7, 1.0)
            imgui.TextColored(tag_color, f"[{info['tag']:>8s}]")
            imgui.SameLine()
            imgui.TextUnformatted(f" {name}")

        if imgui.Button("Force Reload All"):
            print("  [Force reload]")
            load_all_parts(meta, part_names)
            for k, v in stamps.items():
                v["stamp"] = get_file_stamp(v["path"])
                v["fixed_stamp"] = get_file_stamp(FIXED_DIR / meta[k]["obj"])

        if frame_counter % POLL_INTERVAL == 0:
            changed = check_and_reload(meta, part_names, stamps)
            for name, tag, nv, nf in changed:
                print(f"  [Reloaded] {name:<20s}  {tag:<10s}  {nv} verts  {nf} tris")

    ps.set_user_callback(on_update)
    ps.show()


if __name__ == "__main__":
    main()
