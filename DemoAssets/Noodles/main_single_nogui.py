import numpy as np

from uipc import view
from uipc import Logger
from uipc import Engine, World, Scene
from uipc.geometry import linemesh, label_surface, ground, mesh_partition

from uipc.constitution import HookeanSpring, KirchhoffRodBending

from asset_dir import AssetDir

Logger.set_level(Logger.Level.Info)

workspace = AssetDir.output_path(__file__)
engine = Engine("cuda", workspace)
world = World(engine)

config = Scene.default_config()
config["dt"] = 0.005
config["contact"]["d_hat"] = 0.001
config["contact"]["friction"]["enable"] = True
config["newton"]["semi_implicit"] = True
config["newton"]["velocity_tol"] = 0.1
config["linear_system"]["tol_rate"] = 1e-4
scene = Scene(config)

hs = HookeanSpring()
krb = KirchhoffRodBending()
scene.contact_tabular().default_model(0.05, -1)
default_element = scene.contact_tabular().default_element()

bending_stiffness = 1e4

thickness = 0.001
n_edges = 50
n_verts = n_edges + 1
y_length = 0.2

noodles_obj = scene.objects().create("noodles")
grid_nx, grid_nz = 1, 1
spacing = 0.005

geo_slots = []

for ix in range(grid_nx):
    for iz in range(grid_nz):
        x = ix * spacing
        z = iz * spacing
        Vs = np.zeros((n_verts, 3), dtype=np.float32)
        for i in range(n_verts):
            Vs[i] = [x, i * (y_length / n_edges), z]
        Es = np.array([[i, i + 1] for i in range(n_edges)], dtype=np.int32)
        mesh = linemesh(Vs, Es)
        label_surface(mesh)
        hs.apply_to(mesh, thickness=thickness)
        krb.apply_to(mesh, bending_stiffness)
        default_element.apply_to(mesh)
        mesh_partition(mesh)
        slot, rest_slot = noodles_obj.geometries().create(mesh)
        geo_slots.append(slot)

ground_height = -y_length / 1.5
ground_obj = scene.objects().create("ground")
g = ground(ground_height)
ground_obj.geometries().create(g)

world.init(scene)

print(f"Noodle: {n_verts} vertices, {n_edges} edges, thickness={thickness}")
print(f"Ground at y={ground_height}")
print(f"d_hat={config['contact']['d_hat']}")
print()

for frame in range(1, 201):
    try:
        world.advance()
        world.retrieve()
    except Exception as e:
        print(f"Frame {frame}: EXCEPTION: {e}")
        break

    geo = geo_slots[0].geometry()
    pos = geo.positions().view()

    min_y = min(p[1][0] for p in pos)
    max_y = max(p[1][0] for p in pos)

    min_dist_sq = float('inf')
    min_pair = (-1, -1)
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            pi = pos[i].flatten()
            pj = pos[j].flatten()
            d2 = float(np.sum((pi - pj) ** 2))
            if d2 < min_dist_sq:
                min_dist_sq = d2
                min_pair = (i, j)

    print(f"Frame {frame:3d}: y=[{min_y:.6f}, {max_y:.6f}]  "
          f"min_d2={min_dist_sq:.2e} between v{min_pair[0]},v{min_pair[1]}")

    if min_dist_sq < 1e-10:
        pi = pos[min_pair[0]].flatten()
        pj = pos[min_pair[1]].flatten()
        print(f"  WARNING: near-zero distance!")
        print(f"  v{min_pair[0]} = [{pi[0]:.8f}, {pi[1]:.8f}, {pi[2]:.8f}]")
        print(f"  v{min_pair[1]} = [{pj[0]:.8f}, {pj[1]:.8f}, {pj[2]:.8f}]")

print("\nDone.")
