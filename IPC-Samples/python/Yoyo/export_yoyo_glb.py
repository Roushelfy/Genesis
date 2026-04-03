"""
Export yoyo rigid parts as individual GLB files with PBR metallic materials.

Matches the partition used by IPC simulation and Genesis replay:
- yoyo-ball.glb       (two shells + axle, one ABD body)
- bearing_outer.glb    (outer bearing ring)
- bearing_spheres.glb  (8 ball bearings as one mesh)
- bearing_sphere_N.glb (individual bearing spheres, N=0..7)

The string is FEM (line mesh) and loaded separately — not exported here.

Each half-shell is cleanly split into dome (anodized body) and rim (polished
aluminum edge) by a horizontal plane cut, giving perfectly clean edge loops.

Usage:
    python export_yoyo_glb.py
"""

from pathlib import Path

import numpy as np
import trimesh
import trimesh.intersections
from PIL import Image, ImageDraw, ImageFilter
from trimesh.visual.material import PBRMaterial

RESULTS_DIR = Path(__file__).resolve().parents[3] / "DemoAssets" / "yoyo" / "v3"
TEX_SIZE = 1024

# ── Material definitions ──

PART_MATERIALS = {
    "bearing_outer": ([0.78, 0.78, 0.82, 1.0], 1.0, 0.10),
    "bearing_spheres": ([0.86, 0.88, 0.90, 1.0], 1.0, 0.05),
}

# Shell dome: anodized metal
SHELL_DOME_METALLIC = 0.9
SHELL_DOME_ROUGHNESS = 0.1

# Shell rim: polished raw aluminum edge
SHELL_RIM_COLOR = [0.88, 0.89, 0.91, 1.0]
SHELL_RIM_METALLIC = 0.95
SHELL_RIM_ROUGHNESS = 0.05

# Fraction of shell z-range from base where the rim-to-dome cut is made
RIM_CUT_FRAC = 0.30


def make_pbr_visual(mesh, base_color, metallic, roughness, name):
    """Assign a PBR metallic-roughness material to a mesh."""
    mat = PBRMaterial(
        baseColorFactor=base_color,
        metallicFactor=metallic,
        roughnessFactor=roughness,
        name=name,
    )
    mesh.visual = trimesh.visual.TextureVisuals(material=mat)


def _gen_dome_texture(size=TEX_SIZE):
    """Clean teal anodize with swept lines for rotation visibility."""
    base = (70, 150, 175)
    img = Image.new("RGB", (size, size), base)
    draw = ImageDraw.Draw(img)
    cx, cy = size / 2, size / 2

    # Concentric rings: subtle CNC turning marks
    pixels = np.array(img, dtype=np.float32)
    y, x = np.mgrid[:size, :size]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    ring_pattern = np.sin(r * 0.8) * 6
    pixels += ring_pattern[:, :, None]
    pixels = np.clip(pixels, 0, 255).astype(np.uint8)
    img = Image.fromarray(pixels)

    # Swept arc lines for rotation visibility (smooth anti-aliased lines, no dots)
    draw = ImageDraw.Draw(img)
    arc_r_inner = size * 0.20
    arc_r_outer = size * 0.40
    n_lines = 5
    for li in range(n_lines):
        r_frac = arc_r_inner + (arc_r_outer - arc_r_inner) * li / (n_lines - 1)
        # Draw a smooth arc from -30° to +90°
        pts = []
        for angle_deg in range(0, 121):
            alpha = 1.0 - (angle_deg / 120.0) ** 0.6
            angle = np.radians(angle_deg - 30)
            px = cx + r_frac * np.cos(angle)
            py = cy - r_frac * np.sin(angle)
            pts.append((px, py))
        # Draw the arc as connected line segments
        color = (220, 245, 250)
        width = max(4, int(6 * (1.0 - li / n_lines)))
        for k in range(len(pts) - 1):
            draw.line([pts[k], pts[k + 1]], fill=color, width=width)

    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))

    # Circular mask
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, size, size], fill=255)
    bg = Image.new("RGB", (size, size), base)
    return Image.composite(img, bg, mask)


def _polar_uv(verts):
    """Compute polar UV mapping from vertex (x, y) positions."""
    r_max = np.sqrt(verts[:, 0] ** 2 + verts[:, 1] ** 2).max()
    if r_max < 1e-10:
        r_max = 1.0
    u = 0.5 + verts[:, 0] / (2 * r_max)
    v = 0.5 + verts[:, 1] / (2 * r_max)
    return np.column_stack([u, v])


# GLB spec uses Y-up. Our OBJ meshes are Z-up. Rotate (x,y,z) → (x,z,-y)
# so that Genesis's Y-up→Z-up auto-conversion reproduces the original coords.
_ZUP_TO_YUP = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float64,
)


def _to_yup(mesh):
    """Convert a Z-up mesh to Y-up for GLB export."""
    mesh.apply_transform(_ZUP_TO_YUP)
    return mesh


def _apply_shell_materials(shell_mesh, shell_name, dome_tex):
    """Cut a half-shell into dome + rim and apply materials. Returns list of (geom_name, mesh)."""
    outward_sign = 1.0 if shell_name == "top" else -1.0
    z_min = shell_mesh.vertices[:, 2].min()
    z_max = shell_mesh.vertices[:, 2].max()
    z_range = z_max - z_min

    if outward_sign > 0:
        z_cut = z_min + z_range * RIM_CUT_FRAC
        dome = trimesh.intersections.slice_mesh_plane(shell_mesh, [0, 0, 1], [0, 0, z_cut])
        rim = trimesh.intersections.slice_mesh_plane(shell_mesh, [0, 0, -1], [0, 0, z_cut])
    else:
        z_cut = z_max - z_range * RIM_CUT_FRAC
        dome = trimesh.intersections.slice_mesh_plane(shell_mesh, [0, 0, -1], [0, 0, z_cut])
        rim = trimesh.intersections.slice_mesh_plane(shell_mesh, [0, 0, 1], [0, 0, z_cut])

    result = []
    if dome is not None and len(dome.faces) > 0:
        uvs = _polar_uv(dome.vertices)
        mat = PBRMaterial(
            baseColorTexture=dome_tex,
            metallicFactor=SHELL_DOME_METALLIC,
            roughnessFactor=SHELL_DOME_ROUGHNESS,
            name=f"{shell_name}_dome",
        )
        dome.visual = trimesh.visual.TextureVisuals(uv=uvs, material=mat)
        result.append((f"{shell_name}_dome", dome))

    if rim is not None and len(rim.faces) > 0:
        make_pbr_visual(rim, SHELL_RIM_COLOR, SHELL_RIM_METALLIC, SHELL_RIM_ROUGHNESS, f"{shell_name}_rim")
        result.append((f"{shell_name}_rim", rim))

    return result


def _classify_ball_parts(ball_mesh):
    """Split ball mesh into shells, rings, axle, and hub.

    Returns dict with keys: top_shell, bottom_shell, top_ring, bottom_ring, axle, hub.
    Each value is a trimesh or None.
    """
    parts = ball_mesh.split()
    parts.sort(key=lambda c: c.centroid[2])

    result = {
        "top_shell": None,
        "bottom_shell": None,
        "top_ring": None,
        "bottom_ring": None,
        "axle": None,
        "hub": None,
    }

    for p in parts:
        n = len(p.vertices)
        z = p.centroid[2]
        r_max = np.sqrt(p.vertices[:, 0] ** 2 + p.vertices[:, 1] ** 2).max()

        if n > 1000:
            # Large part = shell
            result["top_shell" if z > 0 else "bottom_shell"] = p
        elif r_max > 0.005:
            # Small part with large radius = ring disc
            result["top_ring" if z > 0 else "bottom_ring"] = p
        elif p.extents[2] > 0.005:
            # Elongated along z = axle shaft
            result["axle"] = p
        else:
            # Compact center piece = hub
            result["hub"] = p

    return result


INTERNAL_MATERIALS = {
    "top_ring": ([0.55, 0.51, 0.39, 1.0], 1.0, 0.08),  # Polished brass
    "bottom_ring": ([0.55, 0.51, 0.39, 1.0], 1.0, 0.08),  # Polished brass
    "axle": ([0.47, 0.45, 0.43, 1.0], 1.0, 0.15),  # Brushed steel
    "hub": ([0.63, 0.60, 0.57, 1.0], 1.0, 0.10),  # Nickel
}


def export_ball_glb(ball_mesh, output_path):
    """Export full yoyo ball (all parts combined) as a single GLB."""
    scene = trimesh.Scene()
    dome_tex = _gen_dome_texture()
    classified = _classify_ball_parts(ball_mesh)

    for key in ("top_shell", "bottom_shell"):
        shell = classified[key]
        if shell is not None:
            half = "top" if "top" in key else "bottom"
            for geom_name, mesh in _apply_shell_materials(shell, half, dome_tex):
                scene.add_geometry(_to_yup(mesh), geom_name=geom_name)

    for key in ("top_ring", "bottom_ring", "axle", "hub"):
        part = classified[key]
        if part is not None:
            color, metallic, roughness = INTERNAL_MATERIALS[key]
            make_pbr_visual(part, color, metallic, roughness, key)
            scene.add_geometry(_to_yup(part), geom_name=key)

    scene.export(str(output_path))
    return scene


def export_ball_halves(ball_mesh, output_dir):
    """Export each yoyo ball sub-part as a separate GLB for exploded view.

    Produces: yoyo-top_shell.glb, yoyo-bottom_shell.glb,
              yoyo-top_ring.glb, yoyo-bottom_ring.glb,
              yoyo-axle.glb, yoyo-hub.glb
    """
    dome_tex = _gen_dome_texture()
    classified = _classify_ball_parts(ball_mesh)
    exported = []

    # Shells (with dome/rim materials)
    for key in ("top_shell", "bottom_shell"):
        shell = classified[key]
        if shell is None:
            continue
        shell_scene = trimesh.Scene()
        half = "top" if "top" in key else "bottom"
        for geom_name, mesh in _apply_shell_materials(shell, half, dome_tex):
            shell_scene.add_geometry(_to_yup(mesh), geom_name=geom_name)
        out_path = output_dir / f"yoyo-{key}.glb"
        shell_scene.export(str(out_path))
        exported.append((key, out_path))

    # Internal parts (each as its own GLB)
    for key in ("top_ring", "bottom_ring", "axle", "hub"):
        part = classified[key]
        if part is None:
            continue
        color, metallic, roughness = INTERNAL_MATERIALS[key]
        make_pbr_visual(part, color, metallic, roughness, key)
        out_path = output_dir / f"yoyo-{key}.glb"
        _to_yup(part).export(str(out_path))
        exported.append((key, out_path))

    return exported


def export_simple_glb(mesh_path, output_path, material_key):
    """Export a single mesh with a flat PBR material, converted to Y-up for GLB."""
    mesh = trimesh.load(str(mesh_path), force="mesh")
    color, metallic, roughness = PART_MATERIALS[material_key]
    make_pbr_visual(mesh, color, metallic, roughness, material_key)
    _to_yup(mesh).export(str(output_path))
    return mesh


def main():
    ball = trimesh.load(str(RESULTS_DIR / "yoyo-ball.obj"), force="mesh")

    # Full ball (for ipc_robot_yoyo replay)
    ball_out = RESULTS_DIR / "yoyo-ball.glb"
    s = export_ball_glb(ball, ball_out)
    print(f"Exported {ball_out.name}: {len(ball.vertices)} verts")
    for name in s.geometry:
        print(f"  {name}: {len(s.geometry[name].faces)} faces")

    # Split halves + internals (for ipc_show_yoyo exploded view)
    halves = export_ball_halves(ball, RESULTS_DIR)
    for name, path in halves:
        print(f"Exported {path.name} ({name})")

    bo_out = RESULTS_DIR / "bearing_outer.glb"
    bo = export_simple_glb(RESULTS_DIR / "bearing_outer.obj", bo_out, "bearing_outer")
    print(f"Exported {bo_out.name}: {len(bo.vertices)} verts")

    bs_out = RESULTS_DIR / "bearing_spheres.glb"
    bs = export_simple_glb(RESULTS_DIR / "bearing_spheres.obj", bs_out, "bearing_spheres")
    print(f"Exported {bs_out.name}: {len(bs.vertices)} verts")

    for i in range(8):
        sp_path = RESULTS_DIR / f"bearing_sphere_{i}.obj"
        if sp_path.exists():
            sp_out = RESULTS_DIR / f"bearing_sphere_{i}.glb"
            sp = export_simple_glb(sp_path, sp_out, "bearing_spheres")
            print(f"Exported {sp_out.name}: {len(sp.vertices)} verts")

    print("\nString (yoyo-string.obj) is FEM — loaded separately, not exported as GLB.")


if __name__ == "__main__":
    main()
