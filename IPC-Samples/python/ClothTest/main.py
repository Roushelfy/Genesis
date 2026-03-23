from pathlib import Path

import numpy as np
from uipc import Engine, Logger, Scene, Timer, World, view
from uipc.constitution import DiscreteShellBending, ElasticModuli2D, StrainLimitingBaraffWitkinShell
from uipc.geometry import SimplicialComplexIO, ground, label_surface
from uipc.unit import MPa
from asset_dir import AssetDir

def _read_obj_triangles(path: Path):
    vertices = []
    faces = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                refs = line.split()[1:]
                if len(refs) < 3:
                    continue
                idx = [int(token.split("/")[0]) - 1 for token in refs]
                # Triangulate polygon faces.
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def _write_obj_triangles(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for v in vertices:
            f.write(f"v {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}\n")
        for tri in faces:
            f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")


def _report_and_collect_bad_original_indices(issue_container, level: str):
    bad_triangle_indices = set()
    bad_vertex_indices = set()
    target_geo_keys = {"close_mesh", "intersected_mesh"}
    items = issue_container.items() if hasattr(issue_container, "items") else []
    for issue_id, msg in items:
        if msg is None:
            continue
        print(f"[sanity-{level}] id={int(issue_id)} name={msg.name()}")
        print(f"[sanity-{level}] message: {msg.message()}")
        try:
            if not hasattr(msg, "geometries"):
                continue
            geometries = msg.geometries()
            geo_items = geometries.items() if hasattr(geometries, "items") else []
            for geo_key, geo_val in geo_items:
                if geo_val is None:
                    continue
                print(f"[sanity-{level}] geometry key={geo_key}")
                print(geo_val)
                if str(geo_key) not in target_geo_keys:
                    continue

                # Collect bad triangle original indices.
                tri_attr = geo_val.triangles().find("sanity_check/original_index")
                if tri_attr is None:
                    print(f"[sanity-fix] geometry={geo_key} missing sanity_check/original_index")
                else:
                    tri_ids = np.asarray(view(tri_attr)).reshape(-1)
                    valid = tri_ids[tri_ids >= 0].astype(np.int64)
                    if valid.size > 0:
                        bad_triangle_indices.update(valid.tolist())
                    print(
                        f"[sanity-fix] geometry={geo_key} "
                        f"triangle original_index count={valid.size}"
                    )

                # close_mesh may contain only vertices/edges, no triangles.
                # Remove source faces touching bad source vertices from close_mesh.
                if str(geo_key) == "close_mesh":
                    v_attr = geo_val.vertices().find("sanity_check/original_index")
                    if v_attr is not None:
                        v_ids = np.asarray(view(v_attr)).reshape(-1)
                        v_valid = v_ids[v_ids >= 0].astype(np.int64)
                        if v_valid.size > 0:
                            bad_vertex_indices.update(v_valid.tolist())
                        print(
                            f"[sanity-fix] geometry={geo_key} "
                            f"vertex original_index count={v_valid.size}"
                        )
                print(
                    f"[sanity-fix] collected totals: "
                    f"triangles={len(bad_triangle_indices)} vertices={len(bad_vertex_indices)}"
                )
        except Exception as exc:
            print(f"[sanity-fix] failed to collect original_index: {exc}")
    return bad_triangle_indices, bad_vertex_indices


def _generate_fixed_obj(src_obj: Path, dst_obj: Path, bad_triangle_indices, bad_vertex_indices):
    vertices, faces = _read_obj_triangles(src_obj)
    if vertices.size == 0 or faces.size == 0:
        print(f"[sanity-fix] source mesh empty, skip: {src_obj}")
        return False

    if not bad_triangle_indices and not bad_vertex_indices:
        print("[sanity-fix] no bad original_index found.")
        return False

    keep = []
    removed_by_triangle = 0
    removed_by_vertex = 0
    tri_out_of_range = 0
    v_out_of_range = 0
    face_count = int(faces.shape[0])
    for face_idx, tri in enumerate(faces):
        if face_idx in bad_triangle_indices:
            removed_by_triangle += 1
            continue
        if (tri[0] in bad_vertex_indices) or (tri[1] in bad_vertex_indices) or (tri[2] in bad_vertex_indices):
            removed_by_vertex += 1
            continue
        keep.append(tri)

    for idx in bad_triangle_indices:
        if idx < 0 or idx >= face_count:
            tri_out_of_range += 1

    vertex_count = int(vertices.shape[0])
    for idx in bad_vertex_indices:
        if idx < 0 or idx >= vertex_count:
            v_out_of_range += 1

    if tri_out_of_range > 0:
        print(f"[sanity-fix] warning: {tri_out_of_range} triangle original_index values out of range.")
    if v_out_of_range > 0:
        print(f"[sanity-fix] warning: {v_out_of_range} vertex original_index values out of range.")

    removed_total = removed_by_triangle + removed_by_vertex
    if removed_total == 0:
        print("[sanity-fix] removed 0 triangles by original_index.")
        return False

    kept_faces = np.asarray(keep, dtype=np.int32)
    used = np.unique(kept_faces.reshape(-1))
    remap = -np.ones(vertices.shape[0], dtype=np.int32)
    remap[used] = np.arange(used.shape[0], dtype=np.int32)
    compact_vertices = vertices[used]
    compact_faces = remap[kept_faces]

    _write_obj_triangles(dst_obj, compact_vertices, compact_faces)
    print(
        f"[sanity-fix] removed {removed_total} triangles "
        f"(by triangle={removed_by_triangle}, by close-vertex={removed_by_vertex}), "
        f"kept {compact_faces.shape[0]} triangles, wrote: {dst_obj}"
    )
    return True


Logger.set_level(Logger.Level.Warn)
Timer.enable_all()

this_dir = Path(__file__).resolve().parent
workspace = AssetDir.output_path(__file__)

# Cloth asset and constitution parameters
cloth_obj_path = (this_dir / "data" / "cloth_6.obj").resolve()
young_modulus = 8.0e3
poisson_ratio = 0.45
mass_density = 200.0
thickness = 5.0e-6
bending_stiffness = 37.0

if not cloth_obj_path.exists():
    raise FileNotFoundError(f"Cloth mesh not found: {cloth_obj_path}")

engine = Engine("cuda", str(workspace))
world = World(engine)

config = Scene.default_config()
config["dt"] = 0.005
config["contact"]["enable"] = True
config["contact"]["friction"]["enable"] = True
config["contact"]["d_hat"] = 0.001
config["newton"]["semi_implicit"] = True
config["newton"]["velocity_tol"] = 1
config["newton"]["transrate_tol"] = 10
scene = Scene(config)

scene.contact_tabular().default_model(0.5, 1000.0 * MPa)
cloth_contact = scene.contact_tabular().create("cloth")
ground_contact = scene.contact_tabular().create("ground")
scene.contact_tabular().insert(cloth_contact, cloth_contact, 0.05, 10.0 * MPa, enable=True)
scene.contact_tabular().insert(cloth_contact, ground_contact, 0.5, 1000.0 * MPa, enable=True)

io = SimplicialComplexIO()
cloth_mesh = io.read(str(cloth_obj_path))
label_surface(cloth_mesh)

shell = StrainLimitingBaraffWitkinShell()
bending = DiscreteShellBending()
shell.apply_to(
    cloth_mesh,
    moduli=ElasticModuli2D.youngs_poisson(young_modulus, poisson_ratio),
    mass_density=mass_density,
    thickness=thickness,
)
bending.apply_to(cloth_mesh, bending_stiffness=bending_stiffness)
cloth_contact.apply_to(cloth_mesh)

cloth_obj = scene.objects().create("cloth")
cloth_obj.geometries().create(cloth_mesh)

ground_obj = scene.objects().create("ground")
ground_mesh = ground(-0.8)
ground_contact.apply_to(ground_mesh)
ground_obj.geometries().create(ground_mesh)

world.init(scene)

checker = world.sanity_checker()
sanity_result = checker.check()
checker.report()
print(f"[sanity] result={sanity_result}")
err_tri, err_vtx = _report_and_collect_bad_original_indices(checker.errors(), "error")
warn_tri, warn_vtx = _report_and_collect_bad_original_indices(checker.warns(), "warning")

result_name = str(sanity_result).lower()
if "success" not in result_name:
    bad_triangle_indices = set()
    bad_vertex_indices = set()
    bad_triangle_indices.update(err_tri)
    bad_triangle_indices.update(warn_tri)
    bad_vertex_indices.update(err_vtx)
    bad_vertex_indices.update(warn_vtx)
    fixed_obj_path = cloth_obj_path.with_name(f"{cloth_obj_path.stem}_fixed.obj")
    _generate_fixed_obj(cloth_obj_path, fixed_obj_path, bad_triangle_indices, bad_vertex_indices)
else:
    print("[sanity] check success, skip fixed obj generation.")

# no-gui batch simulation
total_steps = 200
for step in range(total_steps):
    if not world.is_valid():
        print(f"[no-gui] world became invalid at step {step}, stop advancing.")
        break
    world.advance()
    world.retrieve()
    world.dump()
    if (step + 1) % 50 == 0 or (step + 1) == total_steps:
        print(f"[no-gui] advanced {step + 1}/{total_steps} steps")

Timer.report()
print("[no-gui] done.")
