"""Convert broccoli.glb to broccoli.obj using trimesh."""
import os
import sys

try:
    import trimesh
except ImportError:
    print("trimesh not installed. Run: pip install trimesh")
    sys.exit(1)

here = os.path.dirname(os.path.abspath(__file__))
glb_path = os.path.join(here, "..", "broccoli.glb")
obj_path = os.path.join(here, "..", "broccoli.obj")

scene = trimesh.load(glb_path)

if isinstance(scene, trimesh.Scene):
    mesh = trimesh.util.concatenate(
        [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
    )
else:
    mesh = scene

print(f"Vertices: {len(mesh.vertices)}")
print(f"Faces:    {len(mesh.faces)}")

mesh.export(obj_path, file_type="obj")
print(f"Saved to {obj_path}")
