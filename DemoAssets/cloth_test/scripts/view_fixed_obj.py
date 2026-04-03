import argparse
from pathlib import Path

import numpy as np
import polyscope as ps
from polyscope import imgui
from uipc import Engine, Logger, Scene, Timer, World
from uipc.constitution import DiscreteShellBending, ElasticModuli2D, StrainLimitingBaraffWitkinShell
from uipc.geometry import SimplicialComplexIO, ground, label_surface, mesh_partition
from uipc.gui import SceneGUI
from uipc.unit import MPa
from asset_dir import AssetDir


def read_obj_triangles(path: Path):
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
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def main():
    this_dir = Path(__file__).resolve().parent
    default_obj = this_dir / "data" / "cloth_6_fixed.obj"
    default_workspace = AssetDir.output_path(__file__)

    parser = argparse.ArgumentParser(description="Run IPC GUI simulation with fixed cloth OBJ.")
    parser.add_argument("--obj", type=Path, default=default_obj, help="Path to OBJ file.")
    parser.add_argument("--workspace", type=Path, default=default_workspace, help="IPC output workspace.")
    parser.add_argument("--dt", type=float, default=0.005, help="Simulation time step.")
    parser.add_argument("--thickness", type=float, default=5.0e-6, help="Cloth shell thickness.")
    parser.add_argument("--young", type=float, default=8.0e3, help="Young's modulus.")
    parser.add_argument("--poisson", type=float, default=0.45, help="Poisson ratio.")
    parser.add_argument("--density", type=float, default=200.0, help="Mass density.")
    parser.add_argument("--bending", type=float, default=37.0, help="Bending stiffness.")
    parser.add_argument("--ground-y", type=float, default=0.0, help="Ground height.")
    args = parser.parse_args()

    obj_path = args.obj.resolve()
    if not obj_path.exists():
        raise FileNotFoundError(f"OBJ not found: {obj_path}")

    v, f = read_obj_triangles(obj_path)
    if v.size == 0 or f.size == 0:
        raise RuntimeError(f"Invalid OBJ mesh (empty vertices/faces): {obj_path}")

    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    Logger.set_level(Logger.Level.Warn)
    Timer.enable_all()

    print(f"[ipc-gui] loading: {obj_path}")
    print(f"[ipc-gui] vertices={v.shape[0]} triangles={f.shape[0]}")
    print(f"[ipc-gui] workspace: {workspace}")

    engine = Engine("cuda", str(workspace))
    world = World(engine)

    config = Scene.default_config()
    config["dt"] = float(args.dt)
    config["contact"]["enable"] = True
    config["contact"]["friction"]["enable"] = True
    config["contact"]["d_hat"] = 0.001
    config["newton"]["semi_implicit"] = False
    config["newton"]["velocity_tol"] = 0.5
    config["newton"]["transrate_tol"] = 10
    config["linear_system"]["tol_rate"] = 1e-5
    scene = Scene(config)

    scene.contact_tabular().default_model(0.5, 1000.0 * MPa)
    cloth_contact = scene.contact_tabular().create("cloth")
    ground_contact = scene.contact_tabular().create("ground")
    scene.contact_tabular().insert(cloth_contact, cloth_contact, 0.05, 10.0 * MPa, enable=True)
    scene.contact_tabular().insert(cloth_contact, ground_contact, 0.5, 1000.0 * MPa, enable=True)

    io = SimplicialComplexIO()
    cloth_mesh = io.read(str(obj_path))
    label_surface(cloth_mesh)

    shell = StrainLimitingBaraffWitkinShell()
    bending = DiscreteShellBending()
    shell.apply_to(
        cloth_mesh,
        moduli=ElasticModuli2D.youngs_poisson(float(args.young), float(args.poisson)),
        mass_density=float(args.density),
        thickness=float(args.thickness),
    )
    bending.apply_to(cloth_mesh, bending_stiffness=float(args.bending))
    cloth_contact.apply_to(cloth_mesh)
    mesh_partition(cloth_mesh)

    cloth_obj = scene.objects().create("cloth_fixed")
    cloth_obj.geometries().create(cloth_mesh)

    ground_obj = scene.objects().create("ground")
    ground_mesh = ground(float(args.ground_y))
    ground_contact.apply_to(ground_mesh)
    ground_obj.geometries().create(ground_mesh)

    world.init(scene)

    ps.init()
    gui = SceneGUI(scene)
    gui.register()

    running = False

    def on_update():
        nonlocal running
        if imgui.Button("Run / Pause"):
            running = not running
        imgui.SameLine()
        if imgui.Button("Step"):
            world.advance()
            world.retrieve()
            world.dump()
            gui.update()
        imgui.Text(f"Frame: {world.frame()}")

        if running:
            world.advance()
            world.retrieve()
            world.dump()
            gui.update()

    ps.set_user_callback(on_update)
    ps.show()


if __name__ == "__main__":
    main()
