import numpy as np
import polyscope as ps
from polyscope import imgui

from uipc import view
from uipc import Logger, Animation, Timer, SceneIO
from uipc import Vector3, Transform
from uipc import builtin
from uipc import Engine, World, Scene
from uipc.geometry import (
    SimplicialComplex, SimplicialComplexSlot, SimplicialComplexIO,
    linemesh, label_surface, ground, mesh_partition,
    apply_transform, closest_vertex_triangle_pairs,
    merge, trimesh,
)
from uipc.constitution import (
    AffineBodyConstitution,
    SoftTransformConstraint,
    SoftVertexTriangleStitch,
    ElasticModuli,
    HookeanSpring,
    KirchhoffRodBending,
)
from uipc.unit import MPa, GPa
from uipc.gui import SceneGUI

from asset_dir import AssetDir

Logger.set_level(Logger.Level.Warn)
Timer.enable_all()

workspace = AssetDir.output_path(__file__)
engine = Engine("cuda", workspace)
world = World(engine)

# ========================  Scene Configuration  ========================
config = Scene.default_config()
config["dt"] = 0.005
config["contact"]["d_hat"] = 0.001
config["contact"]["enable"] = True
config["contact"]["friction"]["enable"] = True
config["newton"]["semi_implicit"] = True
config["newton"]["velocity_tol"] = 0.1
config["newton"]["transrate_tol"] = 10
config["newton"]["max_iter"] = 5
config["linear_system"]["tol_rate"] = 1e-4
# config["linear_system"]["solver"] = "linear_pcg"
# config["extras"]["debug"]["dump_surface"] = True
print(config)
scene = Scene(config)

scene.contact_tabular().default_model(0.00, 10.0 * MPa)
default_element = scene.contact_tabular().default_element()
nut_element = scene.contact_tabular().create("nut")
ground_element = scene.contact_tabular().create("ground")
scene.contact_tabular().insert(nut_element, ground_element, 0.5, 1.0 * GPa, enable=True)

abd = AffineBodyConstitution()
stc = SoftTransformConstraint()
svts = SoftVertexTriangleStitch()
hs = HookeanSpring()
krb = KirchhoffRodBending()

# ========================  Brush Dimensions  ========================
bristle_length = 0.09
bristle_n_edges = 15
bristle_n_verts = bristle_n_edges + 1
bristle_thickness = 0.0005
bristle_spring_moduli = 1.0 * MPa
bristle_bending_stiffness = 1e8

bristle_grid_nx = 15
bristle_grid_nz = 18
bristle_spacing = bristle_thickness * 6

box_width = bristle_grid_nx * bristle_spacing
box_depth = bristle_grid_nz * bristle_spacing
box_height = 0.018

# Handle above brush head, centered on head (same local space as head)
handle_height = 0.03
handle_half_width = box_width * 0.3   # half-width from center
handle_half_depth = box_depth * 0.3   # half-depth from center

bristle_box_gap = 0.002
brush_lift = 0.0001
box_bottom_y = bristle_length + bristle_box_gap + brush_lift
box_center_y = box_bottom_y + box_height / 2.0
bristle_top_y = box_bottom_y - bristle_box_gap

brush_start_x = 0
sweep_amplitude = 0.09

# ========================  1. Brush Head (Box) + Handle  ========================
def process_surface(sc: SimplicialComplex):
    label_surface(sc)
    return sc

pre_transform = Transform.Identity()
pre_transform.scale(Vector3.Values([box_width, box_height, box_depth]))

io = SimplicialComplexIO(pre_transform)
box_mesh = io.read(f"{AssetDir.trimesh_path()}/cube.obj")
box_mesh = process_surface(box_mesh)

pre_transform = Transform.Identity()
pre_transform.translate(Vector3.Values([0.0, handle_height * 2, 0.0]))
pre_transform.scale(Vector3.Values([box_width/2.0, handle_height * 4, box_depth/2.0]))
io = SimplicialComplexIO(pre_transform)
handle_mesh = io.read(f"{AssetDir.trimesh_path()}/cube.obj")
handle_mesh = process_surface(handle_mesh)

# Merge head + handle into one mesh (same affine body / transform)
brush_mesh = merge([box_mesh, handle_mesh])

abd.apply_to(brush_mesh, 100.0 * MPa)
stc.apply_to(brush_mesh, np.array([10.0, 10.0]))
default_element.apply_to(brush_mesh)

t = Transform.Identity()
t.translate(Vector3.Values([
    brush_start_x + box_width / 2.0,
    box_center_y,
    box_depth / 2.0,
]))
view(brush_mesh.transforms())[0] = t.matrix()

brush_obj = scene.objects().create("brush")
box_geo_slot, box_rest_geo_slot = brush_obj.geometries().create(brush_mesh)

# ========================  2. Bristles (Rods)  ========================
bristles_obj = scene.objects().create("bristles")

bristle_slots = []
bristle_top_positions = []

for ix in range(bristle_grid_nx):
    for iz in range(bristle_grid_nz):
        top_x = brush_start_x + (ix + 0.5) * bristle_spacing
        top_z = (iz + 0.5) * bristle_spacing
        top_y = bristle_top_y

        bristle_top_positions.append([top_x, top_y, top_z])

        Vs = np.zeros((bristle_n_verts, 3), dtype=np.float32)
        for i in range(bristle_n_verts):
            frac = i / bristle_n_edges
            Vs[i] = [top_x, top_y - frac * bristle_length, top_z]

        Es = np.array([[j, j + 1] for j in range(bristle_n_edges)], dtype=np.int32)
        mesh = linemesh(Vs, Es)
        label_surface(mesh)

        hs.apply_to(mesh, bristle_spring_moduli, thickness=bristle_thickness, mass_density=100)
        krb.apply_to(mesh, bristle_bending_stiffness)
        default_element.apply_to(mesh)
        mesh_partition(mesh)

        geo_slot, rest_geo_slot = bristles_obj.geometries().create(mesh)
        bristle_slots.append((geo_slot, rest_geo_slot))

bristle_top_positions = np.array(bristle_top_positions, dtype=np.float64)

# ========================  3. VT Stitch (bristles -> box)  ========================
box_world = apply_transform(brush_mesh)[0]

stitch_obj = scene.objects().create("stitch")

for geo_slot, rest_geo_slot in bristle_slots:
    bristle_sc = geo_slot.geometry()
    pairs = closest_vertex_triangle_pairs(bristle_sc, box_world, max_distance=0.01)

    stitch_geo = svts.create_geometry(
        (geo_slot, box_geo_slot),
        (rest_geo_slot, box_rest_geo_slot),
        pairs,
        ElasticModuli.youngs_poisson(120e2, 0.49),
        min_separate_distance=bristle_box_gap * 0.5,
    )
    stitch_obj.geometries().create(stitch_geo)

ground_height = -0.005
# ========================  4. Nuts  ========================
nut_scale = 0.004
# nut_scale = 0.02
nut_y = ground_height + nut_scale / 2.0 + 0.002
nut_positions = [
    [0.07, nut_y, 0.015],
    [0.08, nut_y, 0.045],
    [0.12, nut_y, 0.030],
]

nut_pre = Transform.Identity()
nut_pre.scale(nut_scale)
nut_io = SimplicialComplexIO(nut_pre)
nut_base_mesh = nut_io.read(f"{AssetDir.trimesh_path()}/screw-and-nut/nut-big.obj")
# nut_base_mesh = nut_io.read(f"{AssetDir.trimesh_path()}/cube.obj")
nut_base_mesh = process_surface(nut_base_mesh)

nuts_obj = scene.objects().create("nuts")

for pos in nut_positions:
    nut_mesh = nut_base_mesh.copy()
    abd.apply_to(nut_mesh, 10.0 * MPa)
    nut_element.apply_to(nut_mesh)
    nt = Transform.Identity()
    nt.translate(Vector3.Values(pos))
    view(nut_mesh.transforms())[0] = nt.matrix()
    nuts_obj.geometries().create(nut_mesh)

# ========================  5. Ground  ========================
ground_obj = scene.objects().create("ground")
g = ground(ground_height)
ground_element.apply_to(g)
ground_obj.geometries().create(g)

# ========================  6. Animation  ========================
animator = scene.animator()

def brush_animation(info: Animation.UpdateInfo):
    geo_slot: SimplicialComplexSlot = info.geo_slots()[0]
    rest_geo_slot: SimplicialComplexSlot = info.rest_geo_slots()[0]
    geo = geo_slot.geometry()
    rest_geo = rest_geo_slot.geometry()

    is_constrained = geo.instances().find(builtin.is_constrained)
    view(is_constrained)[0] = 1

    current_t = info.dt() * info.frame()
    x_offset = sweep_amplitude * (1 - np.cos(2.0 * np.pi * current_t / 2.0))

    rest_transform = Transform(rest_geo.transforms().view()[0])
    rest_pos = rest_transform.translation()

    aim = Transform.Identity()
    aim.translate(Vector3.Values([
        rest_pos[0][0] + x_offset,
        rest_pos[1][0],
        rest_pos[2][0],
    ]))

    aim_transform = geo.instances().find(builtin.aim_transform)
    view(aim_transform)[0] = aim.matrix()

animator.insert(brush_obj, brush_animation)

# ========================  7. Initialize & GUI  ========================
world.init(scene)

ps.init()
ps.set_ground_plane_height(ground_height)
sgui = SceneGUI(scene)
sio = SceneIO(scene)
sgui.register()
sgui.set_edge_width(1)

n_bristles = len(bristle_slots)
stitch_edges = np.array([[2 * i, 2 * i + 1] for i in range(n_bristles)], dtype=np.int32)

def build_stitch_nodes():
    nodes = np.zeros((2 * n_bristles, 3), dtype=np.float64)
    for idx, (geo_slot, _) in enumerate(bristle_slots):
        bristle_geo = geo_slot.geometry()
        top_pos = bristle_geo.positions().view()[0].flatten()
        nodes[2 * idx] = top_pos
        nodes[2 * idx + 1] = [top_pos[0], box_bottom_y, top_pos[2]]
    return nodes

stitch_net = ps.register_curve_network("stitch", build_stitch_nodes(), stitch_edges, radius=0.0008)
sio.write_surface(f"{workspace}/brush_stitch_{world.frame()}.obj")

run = False
recover_frame = 0

def on_update():
    global run
    global recover_frame

    imgui.Text(f"Frame: {world.frame()}")

    if imgui.Button("Run & Stop"):
        run = not run

    if run:
        world.advance()
        world.retrieve()
        world.dump()
        stitch_net.update_node_positions(build_stitch_nodes())
        sio.write_surface(f"{workspace}/brush_stitch_{world.frame()}.obj")
        Timer.report()
        sgui.update()
    
    changed, value = imgui.InputInt("Recover Frame", recover_frame)
    if changed:
        recover_frame = value
    
    if imgui.Button("Recover"):
        world.recover(recover_frame)
        world.retrieve()
        sgui.update()
        stitch_net.update_node_positions(build_stitch_nodes())
    
    if imgui.Button("ONCE_ADVANCE"):
        world.advance()
        world.retrieve()
        world.dump()
        sgui.update()
        stitch_net.update_node_positions(build_stitch_nodes())
        Timer.report()
    
    if world.frame() >= 1000:
        run = False

ps.set_user_callback(on_update)
ps.show()
