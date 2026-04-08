import sys
import numpy as np

from uipc import Logger, Timer
from uipc import Engine, World, Scene
from uipc.geometry import linemesh, label_surface, ground, mesh_partition
from uipc.constitution import HookeanSpring, KirchhoffRodBending
from uipc.stats import SimulationStats
from uipc import profile

from asset_dir import AssetDir

solver_name = sys.argv[1] if len(sys.argv) > 1 else "linear_pcg"
n_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 200

Logger.set_level(Logger.Level.Warn)

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
config["linear_system"]["solver"] = solver_name
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
grid_nx, grid_nz = 40, 40
spacing = 0.005

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
        noodles_obj.geometries().create(mesh)

ground_height = -y_length / 1.5
ground_obj = scene.objects().create("ground")
g = ground(ground_height)
ground_obj.geometries().create(g)

world.init(scene)

output_dir = f"bench_results/{solver_name}"
result = profile.run(world, num_frames=n_frames, name=solver_name, output_dir=output_dir)

print(result["summary"])

stats = result["stats"]
stats.summary_report(
    output_dir=f"{output_dir}/report",
    workspace=workspace,
)
