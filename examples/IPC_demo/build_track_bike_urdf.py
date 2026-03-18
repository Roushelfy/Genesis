#!/usr/bin/env python3
"""
Build an articulated track bike URDF from the Track_Bike.glb mesh.

Splits the bike into:
  - bike_base.{glb,obj}:       frame, wheels, fork, seat, handlebars, etc.
  - combined_crank.{glb,obj}:  crank arms + bracket + pedals (recentered)
  - front_sprocket.obj:        scaled sprocket-20teeth mesh (centered at origin)
  - rear_sprocket.obj:         scaled sprocket-8teeth mesh  (centered at origin)
  - track_bike.urdf:           articulated robot definition

GLB files are used for visuals (preserves materials/colors from the original
Blender model). OBJ files are used for collision geometry.

Articulation:
  bike_base (base_link, fixed)
    +-- front_sprocket (continuous, Z axis at crankset center)
    |     +-- combined_crank (fixed)
    +-- rear_sprocket (continuous, Z axis at fixed_gear center)
    |     +-- rear_wheel (fixed)
    +-- front_wheel (continuous, Z axis at front hub center)

Usage:
    python examples/IPC_demo/build_track_bike_urdf.py
"""

import os
from pathlib import Path

import numpy as np
import trimesh

# ─── Paths ───────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_ASSETS = REPO_ROOT / "DemoAssets"
RIGID_IPC_ROOT = DEMO_ASSETS / "rigid-ipc"
GLB_PATH = DEMO_ASSETS / "track_bike" / "Track_Bike.glb"
SPROCKET_20_PATH = RIGID_IPC_ROOT / "meshes" / "507-movements" / "227-chain-pully" / "sprocket-20teeth.obj"
SPROCKET_8_PATH = RIGID_IPC_ROOT / "meshes" / "507-movements" / "227-chain-pully" / "sprocket-8teeth.obj"
OUTPUT_DIR = DEMO_ASSETS / "track_bike"


# ─── Helpers ─────────────────────────────────────────────────────────────────


def collect_world_meshes(scene, top_nodes):
    """Collect all geometry under the given top-level node names.

    Returns a list of trimesh.Trimesh objects with vertices in world coords.
    """
    meshes = []
    for node_name in top_nodes:
        # The node itself may have geometry
        try:
            T, geom_name = scene.graph.get(node_name)
            if geom_name and geom_name in scene.geometry:
                g = scene.geometry[geom_name].copy()
                g.apply_transform(T)
                meshes.append(g)
        except ValueError:
            pass

        # Check children
        children = scene.graph.transforms.children.get(node_name, [])
        for child in children:
            try:
                T, geom_name = scene.graph.get(child)
                if geom_name and geom_name in scene.geometry:
                    g = scene.geometry[geom_name].copy()
                    g.apply_transform(T)
                    meshes.append(g)
            except ValueError:
                pass
    return meshes


def collect_subscene(scene, top_nodes):
    """Build a trimesh.Scene containing only the given top-level nodes.

    Preserves original graph structure, transforms, and materials.
    """
    sub = trimesh.Scene()
    for node_name in top_nodes:
        try:
            T, geom_name = scene.graph.get(node_name)
            if geom_name and geom_name in scene.geometry:
                sub.add_geometry(scene.geometry[geom_name], node_name=node_name, transform=T)
        except ValueError:
            pass

        children = scene.graph.transforms.children.get(node_name, [])
        for child in children:
            try:
                T, geom_name = scene.graph.get(child)
                if geom_name and geom_name in scene.geometry:
                    sub.add_geometry(scene.geometry[geom_name], node_name=child, transform=T)
            except ValueError:
                pass
    return sub


def get_center(scene, top_nodes):
    """Get the bounding-box center of all geometry under the given nodes."""
    meshes = collect_world_meshes(scene, top_nodes)
    all_verts = np.vstack([m.vertices for m in meshes])
    return (all_verts.min(axis=0) + all_verts.max(axis=0)) / 2


def save_as_glb(scene_or_meshes, path, translate=None):
    """Save a trimesh.Scene as GLB, optionally translating all geometry."""
    if isinstance(scene_or_meshes, trimesh.Scene):
        sub = scene_or_meshes
    else:
        sub = trimesh.Scene()
        for i, m in enumerate(scene_or_meshes):
            sub.add_geometry(m, node_name=f"part_{i}")

    if translate is not None:
        # Apply translation to all geometry transforms in the scene
        T = np.eye(4)
        T[:3, 3] = translate
        sub.apply_transform(T)

    sub.export(path)
    n_geoms = len(sub.geometry)
    print(f"  Saved {path} ({n_geoms} geometries)")


def save_as_obj(meshes, path):
    """Concatenate multiple trimesh.Trimesh objects into a single OBJ file."""
    combined = trimesh.util.concatenate(meshes)
    combined.export(path, file_type="obj")
    print(f"  Saved {path} ({len(meshes)} parts, verts={len(combined.vertices)}, faces={len(combined.faces)})")


def save_mesh_as_obj(mesh, path):
    """Save a single trimesh.Trimesh as OBJ."""
    mesh.export(path, file_type="obj")
    print(f"  Saved {path} (verts={len(mesh.vertices)}, faces={len(mesh.faces)})")


def save_mesh_as_glb(mesh, path, base_color=(0.05, 0.05, 0.05, 1.0), metallic=1.0, roughness=0.3):
    """Save a single trimesh.Trimesh as GLB with a PBR metallic material."""
    material = trimesh.visual.material.PBRMaterial(
        baseColorFactor=base_color,
        metallicFactor=metallic,
        roughnessFactor=roughness,
    )
    mesh.visual = trimesh.visual.TextureVisuals(material=material)
    scene = trimesh.Scene()
    scene.add_geometry(mesh, node_name="part_0")
    scene.export(path)
    print(f"  Saved {path} (verts={len(mesh.vertices)}, faces={len(mesh.faces)})")


def box_inertia(mass, sx, sy, sz):
    """Inertia tensor of a uniform box with side lengths sx, sy, sz."""
    return np.diag(
        [
            mass / 12.0 * (sy**2 + sz**2),
            mass / 12.0 * (sx**2 + sz**2),
            mass / 12.0 * (sx**2 + sy**2),
        ]
    )


def inertial_from_mesh(mesh, mass):
    """Compute <inertial> parameters from a trimesh mesh and a target mass.

    Uses a box approximation from the bounding box extents.
    Returns (mass, com, ixx, iyy, izz) — off-diagonal terms assumed zero
    for axis-aligned bounding box approximation.
    """
    bounds = mesh.bounds
    extents = bounds[1] - bounds[0]
    com = (bounds[0] + bounds[1]) / 2.0
    I = box_inertia(mass, *extents)
    return mass, com, I[0, 0], I[1, 1], I[2, 2]


def inertial_xml(mass, com, ixx, iyy, izz):
    """Format an <inertial> URDF element."""
    return (
        f"    <inertial>\n"
        f'      <mass value="{mass:.6f}"/>\n'
        f'      <origin xyz="{com[0]:.6f} {com[1]:.6f} {com[2]:.6f}"/>\n'
        f'      <inertia ixx="{ixx:.8f}" ixy="0" ixz="0"'
        f' iyy="{iyy:.8f}" iyz="0" izz="{izz:.8f}"/>\n'
        f"    </inertial>"
    )


def write_urdf(
    path,
    crankset_center,
    fixed_gear_center,
    rear_wheel_center,
    front_wheel_center,
    sprocket_axis,
    front_sprocket_mesh,
    crank_mesh,
    rear_sprocket_mesh,
    rear_wheel_meshes,
    front_wheel_meshes,
    base_meshes,
):
    """Generate the track_bike.urdf file with inertial properties."""

    cx, cy, cz = crankset_center
    fx, fy, fz = fixed_gear_center
    rwx, rwy, rwz = rear_wheel_center
    fwx, fwy, fwz = front_wheel_center
    ax, ay, az = sprocket_axis

    # Approximate masses (kg) for a real track bike
    base_mass, base_com, base_ixx, base_iyy, base_izz = inertial_from_mesh(trimesh.util.concatenate(base_meshes), 6.0)
    fs_mass, fs_com, fs_ixx, fs_iyy, fs_izz = inertial_from_mesh(front_sprocket_mesh, 0.3)
    crank_mass, crank_com, crank_ixx, crank_iyy, crank_izz = inertial_from_mesh(crank_mesh, 0.5)
    rs_mass, rs_com, rs_ixx, rs_iyy, rs_izz = inertial_from_mesh(rear_sprocket_mesh, 0.1)
    rw_mass, rw_com, rw_ixx, rw_iyy, rw_izz = inertial_from_mesh(trimesh.util.concatenate(rear_wheel_meshes), 0.8)
    fw_mass, fw_com, fw_ixx, fw_iyy, fw_izz = inertial_from_mesh(trimesh.util.concatenate(front_wheel_meshes), 0.8)

    base_inertial = inertial_xml(base_mass, base_com, base_ixx, base_iyy, base_izz)
    fs_inertial = inertial_xml(fs_mass, fs_com, fs_ixx, fs_iyy, fs_izz)
    crank_inertial = inertial_xml(crank_mass, crank_com, crank_ixx, crank_iyy, crank_izz)
    rs_inertial = inertial_xml(rs_mass, rs_com, rs_ixx, rs_iyy, rs_izz)
    rw_inertial = inertial_xml(rw_mass, rw_com, rw_ixx, rw_iyy, rw_izz)
    fw_inertial = inertial_xml(fw_mass, fw_com, fw_ixx, fw_iyy, fw_izz)

    urdf = f"""\
<?xml version="1.0" ?>
<robot name="track_bike">

  <!-- ====== Base link: frame, fork, seat, handlebars, etc. ====== -->
  <link name="bike_base">
{base_inertial}
    <visual>
      <geometry>
        <mesh filename="bike_base.glb"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <mesh filename="bike_base.obj"/>
      </geometry>
    </collision>
  </link>

  <!-- ====== Front sprocket (replaces Crankset) ====== -->
  <link name="front_sprocket">
{fs_inertial}
    <visual>
      <geometry>
        <mesh filename="front_sprocket.glb"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <mesh filename="front_sprocket.obj"/>
      </geometry>
    </collision>
  </link>

  <joint name="front_sprocket_joint" type="continuous">
    <parent link="bike_base"/>
    <child link="front_sprocket"/>
    <origin xyz="{cx} {cy} {(cz + fz) / 2}" rpy="0 0 0"/>
    <axis xyz="{ax} {ay} {az}"/>
  </joint>

  <!-- ====== Combined crank (arms + bracket + pedals), fixed to front sprocket ====== -->
  <link name="combined_crank">
{crank_inertial}
    <visual>
      <geometry>
        <mesh filename="combined_crank.glb"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <mesh filename="combined_crank.obj"/>
      </geometry>
    </collision>
  </link>

  <joint name="combined_crank_joint" type="fixed">
    <parent link="front_sprocket"/>
    <child link="combined_crank"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>

  <!-- ====== Rear sprocket (replaces Fixed_Gear) ====== -->
  <link name="rear_sprocket">
{rs_inertial}
    <visual>
      <geometry>
        <mesh filename="rear_sprocket.glb"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <mesh filename="rear_sprocket.obj"/>
      </geometry>
    </collision>
  </link>

  <joint name="rear_sprocket_joint" type="continuous">
    <parent link="bike_base"/>
    <child link="rear_sprocket"/>
    <origin xyz="{fx} {fy} {(cz + fz) / 2}" rpy="0 0 0"/>
    <axis xyz="{ax} {ay} {az}"/>
  </joint>

  <!-- ====== Rear wheel, fixed to rear sprocket (spins with it) ====== -->
  <link name="rear_wheel">
{rw_inertial}
    <visual>
      <geometry>
        <mesh filename="rear_wheel.glb"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <mesh filename="rear_wheel.obj"/>
      </geometry>
    </collision>
  </link>

  <joint name="rear_wheel_joint" type="fixed">
    <parent link="rear_sprocket"/>
    <child link="rear_wheel"/>
    <origin xyz="0 0 {-(cz + fz) / 2}" rpy="0 0 0"/>
  </joint>

  <!-- ====== Front wheel, revolute on bike_base (spins freely) ====== -->
  <link name="front_wheel">
{fw_inertial}
    <visual>
      <geometry>
        <mesh filename="front_wheel.glb"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <mesh filename="front_wheel.obj"/>
      </geometry>
    </collision>
  </link>

  <joint name="front_wheel_joint" type="continuous">
    <parent link="bike_base"/>
    <child link="front_wheel"/>
    <origin xyz="{fwx} {fwy} 0" rpy="0 0 0"/>
    <axis xyz="{ax} {ay} {az}"/>
  </joint>

</robot>
"""
    with open(path, "w") as f:
        f.write(urdf)
    print(f"  Saved {path}")


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # ── Load scene ──
    print("\nLoading GLB scene...")
    scene = trimesh.load(str(GLB_PATH))

    # All top-level node names (children of "world")
    all_top_nodes = list(scene.graph.transforms.children.get("world", []))
    print(f"  Top-level nodes ({len(all_top_nodes)}): {sorted(all_top_nodes)}")

    # ── Define part groups ──

    # Parts to delete entirely (not in any output mesh)
    delete_only = {"Chain", "Crankset", "Fixed_Gear"}

    # Parts for combined_crank (also removed from base)
    # Including Pedal_L/R since they are physically on the crank arms
    crank_parts = {"Crank_arm_L", "Crank_arm_R", "Bracket", "Pedal_L", "Pedal_R"}

    # Rear wheel parts: fixed to rear sprocket so they spin together
    rear_wheel_parts = {"Hub_back", "Hubs_001", "Rims_001", "Spokes_001", "Tires_Back_001"}

    # Front wheel parts: revolute joint to bike_base (spins freely)
    front_wheel_parts = {"Hub_front", "Hubs", "Rims_002", "Spokes_002", "Tires_Back_002"}

    # Everything else goes in bike_base
    excluded_from_base = delete_only | crank_parts | rear_wheel_parts | front_wheel_parts
    base_parts = [n for n in all_top_nodes if n not in excluded_from_base]

    print(f"\n  bike_base parts ({len(base_parts)}): {sorted(base_parts)}")
    print(f"  combined_crank parts ({len(crank_parts)}): {sorted(crank_parts)}")
    print(f"  rear_wheel parts ({len(rear_wheel_parts)}): {sorted(rear_wheel_parts)}")
    print(f"  front_wheel parts ({len(front_wheel_parts)}): {sorted(front_wheel_parts)}")
    print(f"  deleted parts ({len(delete_only)}): {sorted(delete_only)}")

    # ── Get reference positions before splitting ──

    crankset_center = get_center(scene, ["Crankset"])
    fixed_gear_center = get_center(scene, ["Fixed_Gear"])
    rear_wheel_center = get_center(scene, list(rear_wheel_parts))
    front_wheel_center = get_center(scene, list(front_wheel_parts))
    # Sprocket/wheel rotation axis: Z in mesh coords (the lateral / axle direction)
    sprocket_axis = np.array([0.0, 0.0, 1.0])

    print(f"\n  Crankset center:    {np.round(crankset_center, 5)}")
    print(f"  Fixed_Gear center:  {np.round(fixed_gear_center, 5)}")
    print(f"  Rear wheel center:  {np.round(rear_wheel_center, 5)}")
    print(f"  Front wheel center: {np.round(front_wheel_center, 5)}")

    # ── 1. Export bike_base (GLB for visual, OBJ for collision) ──

    print("\nExporting bike_base...")
    base_sub = collect_subscene(scene, base_parts)
    save_as_glb(base_sub, os.path.join(OUTPUT_DIR, "bike_base.glb"))
    base_meshes = collect_world_meshes(scene, base_parts)
    save_as_obj(base_meshes, os.path.join(OUTPUT_DIR, "bike_base.obj"))

    # ── 2. Export combined_crank (recentered to crankset axis) ──

    print("\nExporting combined_crank (recentered to crankset axis)...")
    crank_sub = collect_subscene(scene, crank_parts)
    save_as_glb(crank_sub, os.path.join(OUTPUT_DIR, "combined_crank.glb"), translate=-crankset_center)
    crank_meshes = collect_world_meshes(scene, crank_parts)
    for m in crank_meshes:
        m.vertices -= crankset_center
    save_as_obj(crank_meshes, os.path.join(OUTPUT_DIR, "combined_crank.obj"))

    # ── 3. Export rear_wheel (recentered to rear axle) ──

    print("\nExporting rear_wheel (recentered to rear wheel axle)...")
    rear_wheel_sub = collect_subscene(scene, list(rear_wheel_parts))
    save_as_glb(rear_wheel_sub, os.path.join(OUTPUT_DIR, "rear_wheel.glb"), translate=-rear_wheel_center)
    rear_wheel_meshes = collect_world_meshes(scene, list(rear_wheel_parts))
    for m in rear_wheel_meshes:
        m.vertices -= rear_wheel_center
    save_as_obj(rear_wheel_meshes, os.path.join(OUTPUT_DIR, "rear_wheel.obj"))

    # ── 4. Export front_wheel (recentered to front axle) ──

    print("\nExporting front_wheel (recentered to front wheel axle)...")
    front_wheel_sub = collect_subscene(scene, list(front_wheel_parts))
    save_as_glb(front_wheel_sub, os.path.join(OUTPUT_DIR, "front_wheel.glb"), translate=-front_wheel_center)
    front_wheel_meshes = collect_world_meshes(scene, list(front_wheel_parts))
    for m in front_wheel_meshes:
        m.vertices -= front_wheel_center
    save_as_obj(front_wheel_meshes, os.path.join(OUTPUT_DIR, "front_wheel.obj"))

    # ── 5. Export front_sprocket (scaled sprocket-20teeth) ──

    print("\nExporting front_sprocket.obj...")
    sprocket_20 = trimesh.load(str(SPROCKET_20_PATH), force="mesh")

    # Compute scale: match the crankset outer radius
    crankset_meshes = collect_world_meshes(scene, ["Crankset"])
    crankset_verts = np.vstack([m.vertices for m in crankset_meshes])
    dxy = crankset_verts[:, :2] - crankset_center[:2]
    crankset_r = np.linalg.norm(dxy, axis=1).max()

    dxy_20 = sprocket_20.vertices[:, :2]
    sprocket_20_r = np.linalg.norm(dxy_20, axis=1).max()

    scale = crankset_r / sprocket_20_r
    print(f"  Crankset outer radius: {crankset_r:.5f}")
    print(f"  sprocket-20teeth radius: {sprocket_20_r:.4f}")
    print(f"  Scale factor: {scale:.6f}")

    # sprocket-20teeth is already centered at origin; just scale
    sprocket_20.vertices *= scale
    save_mesh_as_obj(sprocket_20, os.path.join(OUTPUT_DIR, "front_sprocket.obj"))
    # Dark metallic GLB for visual
    save_mesh_as_glb(sprocket_20.copy(), os.path.join(OUTPUT_DIR, "front_sprocket.glb"))

    # ── 6. Export rear_sprocket (scaled sprocket-8teeth, same scale) ──

    print("\nExporting rear_sprocket.obj (same scale)...")
    sprocket_8 = trimesh.load(str(SPROCKET_8_PATH), force="mesh")

    # sprocket-8teeth is NOT centered at origin; recenter first, then scale
    s8_center = (sprocket_8.bounds[0] + sprocket_8.bounds[1]) / 2
    sprocket_8.vertices -= s8_center
    sprocket_8.vertices *= scale
    s8_r_scaled = np.linalg.norm(sprocket_8.vertices[:, :2], axis=1).max()
    print(f"  sprocket-8teeth scaled radius: {s8_r_scaled:.5f}")
    save_mesh_as_obj(sprocket_8, os.path.join(OUTPUT_DIR, "rear_sprocket.obj"))
    # Dark metallic GLB for visual
    save_mesh_as_glb(sprocket_8.copy(), os.path.join(OUTPUT_DIR, "rear_sprocket.glb"))

    # ── 7. Generate URDF ──

    print("\nGenerating track_bike.urdf...")
    # Combined crank mesh (already recentered) for inertial computation
    crank_combined = trimesh.util.concatenate(crank_meshes)

    write_urdf(
        os.path.join(OUTPUT_DIR, "track_bike.urdf"),
        crankset_center=crankset_center,
        fixed_gear_center=fixed_gear_center,
        rear_wheel_center=rear_wheel_center,
        front_wheel_center=front_wheel_center,
        sprocket_axis=sprocket_axis,
        front_sprocket_mesh=sprocket_20,
        crank_mesh=crank_combined,
        rear_sprocket_mesh=sprocket_8,
        rear_wheel_meshes=rear_wheel_meshes,
        front_wheel_meshes=front_wheel_meshes,
        base_meshes=base_meshes,
    )

    print(f"\nDone! Output in: {OUTPUT_DIR}")
    print("\nTo load in Genesis:")
    print("  scene.add_entity(gs.morphs.URDF(")
    print(f'      file="{os.path.join(OUTPUT_DIR, "track_bike.urdf")}",')
    print("      fixed=True,")
    print("  ))")


if __name__ == "__main__":
    main()
