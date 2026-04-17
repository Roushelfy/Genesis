"""
Build GLB visual meshes for G1 robot from segmented sub-meshes.

Each link's material assignment is defined in LINK_MATERIALS below.
Supports three modes:
  - "solid": entire mesh gets one material
  - "split": segmented parts are grouped by material

Usage:
    python build_glb.py                # build all
    python build_glb.py --link head_link  # build one
    python build_glb.py --update-urdf  # also update URDF visual refs
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

import trimesh

MESH_DIR = Path(__file__).parent / "meshes"
SEG_DIR = Path(__file__).parent / "meshes_segmented"
URDF_PATH = Path(__file__).parent / "g1_29dof_rev_1_0.urdf"


# ── Materials ──────────────────────────────────────────────────────────────────


@dataclass
class PBR:
    name: str
    color: tuple[float, float, float]
    metallic: float = 0.0
    roughness: float = 0.5
    emissive: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_trimesh(self) -> trimesh.visual.material.PBRMaterial:
        r, g, b = self.color
        return trimesh.visual.material.PBRMaterial(
            name=self.name,
            baseColorFactor=[int(r * 255), int(g * 255), int(b * 255), 255],
            emissiveFactor=list(self.emissive),
            metallicFactor=self.metallic,
            roughnessFactor=self.roughness,
        )


BLACK = PBR("black", color=(0.05, 0.05, 0.05), metallic=0.3, roughness=0.2)
SILVER = PBR("silver", color=(0.9, 0.9, 0.9), metallic=0.8, roughness=0.45)
CYAN_GLOW = PBR("cyan_glow", color=(0.039, 0.820, 0.820), metallic=0.0, roughness=0.3, emissive=(0.039, 0.820, 0.820))
LOGO_WHITE = PBR("logo_white", color=(0.95, 0.95, 0.95), metallic=0.0, roughness=0.3)


# ── Per-link material config ──────────────────────────────────────────────────


@dataclass
class SolidLink:
    """Entire STL gets one material."""

    mat: PBR


@dataclass
class SplitLink:
    """Segmented parts grouped by material. Keys are part indices."""

    groups: list[tuple[PBR, set[int]]]
    default: PBR = field(default_factory=lambda: SILVER)


LINK_MATERIALS: dict[str, SolidLink | SplitLink] = {
    # ── Head (segmented: cyan glow on parts 03, 06) ──
    "head_link": SplitLink(
        groups=[(CYAN_GLOW, {3, 6})],
        default=BLACK,
    ),
    # ── Solid black ──
    "pelvis": SolidLink(BLACK),
    "pelvis_contour_link": SolidLink(BLACK),
    "left_hip_pitch_link": SolidLink(BLACK),
    "right_hip_pitch_link": SolidLink(BLACK),
    "left_ankle_pitch_link": SolidLink(BLACK),
    "right_ankle_pitch_link": SolidLink(BLACK),
    "left_ankle_roll_link": SolidLink(BLACK),
    "right_ankle_roll_link": SolidLink(BLACK),
    "left_rubber_hand": SolidLink(BLACK),
    "right_rubber_hand": SolidLink(BLACK),
    "waist_yaw_link_rev_1_0": SolidLink(BLACK),
    "waist_roll_link_rev_1_0": SolidLink(BLACK),
    # ── Solid silver ──
    "torso_link_rev_1_0": SolidLink(SILVER),
    "left_shoulder_pitch_link": SolidLink(SILVER),
    "right_shoulder_pitch_link": SolidLink(SILVER),
    "left_shoulder_roll_link": SolidLink(SILVER),
    "right_shoulder_roll_link": SolidLink(SILVER),
    "left_shoulder_yaw_link": SolidLink(SILVER),
    "right_shoulder_yaw_link": SolidLink(SILVER),
    "left_wrist_roll_link": SolidLink(SILVER),
    "right_wrist_roll_link": SolidLink(SILVER),
    "left_wrist_pitch_link": SolidLink(SILVER),
    "right_wrist_pitch_link": SolidLink(SILVER),
    "left_wrist_yaw_link": SolidLink(SILVER),
    "right_wrist_yaw_link": SolidLink(SILVER),
    "left_hip_roll_link": SolidLink(SILVER),
    "right_hip_roll_link": SolidLink(SILVER),
    # ── Logo ──
    "logo_link": SolidLink(LOGO_WHITE),
    # ── Split: hip_yaw_link ──
    "left_hip_yaw_link": SplitLink(
        groups=[(BLACK, {1, 5, 7, 8, 10, 15, 18})],
        default=SILVER,
    ),
    "right_hip_yaw_link": SplitLink(
        groups=[(BLACK, {1, 5, 7, 8, 10, 15, 18})],
        default=SILVER,
    ),
    # ── Split: knee_link ──
    "left_knee_link": SplitLink(
        groups=[(BLACK, {5, 7, 17, 19, 26, 27, 28, 29, 31, 45})],
        default=SILVER,
    ),
    "right_knee_link": SplitLink(
        groups=[(BLACK, {5, 7, 17, 19, 25, 26, 27, 28, 29, 44})],
        default=SILVER,
    ),
    # ── Split: elbow_link ──
    "left_elbow_link": SplitLink(
        groups=[(BLACK, {1, 3, 6})],
        default=SILVER,
    ),
    "right_elbow_link": SplitLink(
        groups=[(BLACK, {1, 3, 6})],
        default=SILVER,
    ),
}


# ── Build functions ───────────────────────────────────────────────────────────


def build_solid(link_name: str, mat: PBR) -> Path:
    stl_path = MESH_DIR / f"{link_name}.STL"
    mesh = trimesh.load(str(stl_path))
    mesh.visual = trimesh.visual.TextureVisuals(material=mat.to_trimesh())
    scene = trimesh.Scene()
    scene.add_geometry(mesh, node_name=link_name, geom_name=link_name)
    out_path = MESH_DIR / f"{link_name}.glb"
    scene.export(str(out_path), file_type="glb")
    return out_path


def build_split(link_name: str, cfg: SplitLink) -> Path:
    seg_dir = SEG_DIR / link_name
    part_files = sorted(seg_dir.glob(f"{link_name}_part*.obj"))

    # Map part index → material
    part_to_mat: dict[int, PBR] = {}
    for mat, indices in cfg.groups:
        for idx in indices:
            part_to_mat[idx] = mat

    # Group meshes by material
    mat_meshes: dict[str, list[trimesh.Trimesh]] = {}
    for pf in part_files:
        idx = int(pf.stem.split("part")[1])
        mesh = trimesh.load(str(pf))
        mat = part_to_mat.get(idx, cfg.default)
        mat_meshes.setdefault(mat.name, (mat, []))[1].append(mesh)

    scene = trimesh.Scene()
    for mat_name, (mat, meshes) in mat_meshes.items():
        combined = trimesh.util.concatenate(meshes)
        combined.visual = trimesh.visual.TextureVisuals(material=mat.to_trimesh())
        geom_name = f"{link_name}_{mat_name}"
        scene.add_geometry(combined, node_name=geom_name, geom_name=geom_name)

    out_path = MESH_DIR / f"{link_name}.glb"
    scene.export(str(out_path), file_type="glb")
    return out_path


def build_link(link_name: str) -> Path:
    cfg = LINK_MATERIALS[link_name]
    if isinstance(cfg, SolidLink):
        return build_solid(link_name, cfg.mat)
    else:
        return build_split(link_name, cfg)


def update_urdf(links: list[str]) -> None:
    with open(URDF_PATH, "r") as f:
        content = f.read()

    for link in links:
        old = f'<mesh filename="meshes/{link}.STL"/>'
        idx = content.find(old)
        if idx < 0:
            # Already updated to .glb
            continue
        bs = content.rfind("<visual>", 0, idx)
        be = content.find("</visual>", idx)
        if bs < 0 or be < 0:
            continue
        vb = content[bs : be + len("</visual>")]
        nb = vb.replace(f"{link}.STL", f"{link}.glb")
        nb = re.sub(r'\s*<material name="[^"]*"/>\s*', "\n", nb)
        content = content[:bs] + nb + content[be + len("</visual>") :]

    with open(URDF_PATH, "w") as f:
        f.write(content)


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--link", type=str, default=None, help="Build single link")
    parser.add_argument("--update-urdf", action="store_true", help="Update URDF visual refs")
    args = parser.parse_args()

    if args.link:
        links = [args.link]
    else:
        links = list(LINK_MATERIALS.keys())

    for link_name in links:
        out = build_link(link_name)
        cfg = LINK_MATERIALS[link_name]
        label = cfg.mat.name if isinstance(cfg, SolidLink) else "split"
        print(f"  {link_name:35s} [{label:10s}] -> {out.name}")

    if args.update_urdf:
        update_urdf(links)
        print(f"\n  URDF updated: {URDF_PATH.name}")

    print(f"\n  {len(links)} GLBs built.")


if __name__ == "__main__":
    main()
