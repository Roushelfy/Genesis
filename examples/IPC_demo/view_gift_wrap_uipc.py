"""
Standalone libuipc gift_wrap scene — no Genesis, pure uipc API.

Identical scene setup to view_gift_wrap.py for comparison:
  - Cloth ribbon (StrainLimitingBaraffWitkinShell + DiscreteShellBending)
  - Non-fixed rigid box (AffineBody)
  - Ground plane at z=0
  - Contact: d_hat=0.0001, friction=0.02, resistance=1e8

Usage:
    LD_PRELOAD=/usr/local/cuda-12.9/targets/x86_64-linux/lib/libcublas.so.12 \
        python examples/IPC_demo/view_gift_wrap_uipc.py [--no-viewer] [--steps N] \
            [--use-al] [--no-box] [--no-ribbon]
"""

import argparse
import os
from pathlib import Path

import numpy as np

import uipc
from uipc import Logger, Timer, Transform, Vector3, AngleAxis, view, builtin
from uipc.constitution import (
    AffineBodyConstitution,
    DiscreteShellBending,
    ElasticModuli2D,
    StrainLimitingBaraffWitkinShell,
)
from uipc.core import Engine, World, Scene
from uipc.geometry import SimplicialComplexIO, ground, label_surface

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_ASSETS = REPO_ROOT / "DemoAssets" / "gift_wrap"
RIBBON_MESH = DEMO_ASSETS / "ribbon_reordered_fine.obj"
BOX_MESH = DEMO_ASSETS / "box.obj"


def main():
    parser = argparse.ArgumentParser(description="libuipc gift wrap (standalone)")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--use-al", action="store_true", help="Use AL-IPC contact constitution")
    parser.add_argument("--no-box", action="store_true", help="Remove the rigid box")
    parser.add_argument("--no-ribbon", action="store_true", help="Remove the cloth ribbon")
    parser.add_argument("--no-ground", action="store_true", help="Replace ground plane with a fixed floor box")
    args = parser.parse_args()

    Timer.enable_all()
    Logger.set_level(Logger.Level.Info)

    workspace = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "gift_wrap_uipc")
    os.makedirs(workspace, exist_ok=True)

    engine = Engine("cuda", workspace)
    world = World(engine)

    config = Scene.default_config()
    config["dt"] = 0.01
    config["contact"]["d_hat"] = 0.0001
    config["contact"]["constitution"] = "al-ipc" if args.use_al else "ipc"
    config["gravity"] = [[0.0], [0.0], [-9.8]]
    config["newton"]["velocity_tol"] = 0.5
    config["newton"]["transrate_tol"] = 10
    config["newton"]["min_iter"] = 2
    print("Config:", config)
    # if args.use_al:
    #     config["contact"]["al-ipc"]["toi_threshold"] = 0.1
    #     # Match Genesis auto mu_scale: corrected for box+ribbon mass ratio
    #     # box mass=19.17, ribbon max vertex mass=2.82e-6, ratio=6.8M
    #     # mu_scale = default * target/actual = 5e6 * 2.82e-6 / 19.17 = 0.736
    #     if not args.no_box and not args.no_ribbon:
    #         config["contact"]["al-ipc"]["mu_scale"] = 0.7356699607168192
    print("Config:", config)
    scene = Scene(config)

    # Global contact model: friction=0.02, resistance=1e8
    scene.contact_tabular().default_model(0.02, 1e8)

    slbws = StrainLimitingBaraffWitkinShell()
    dsb = DiscreteShellBending()
    abd = AffineBodyConstitution()
    cloth_moduli = ElasticModuli2D.youngs_poisson(20e3, 0.4)

    # Cloth ribbon
    # Y-up mesh → Z-up: rotate 90° around X
    if not args.no_ribbon:
        ribbon_obj = scene.objects().create("ribbon")
        pre = Transform.Identity()
        pre.rotate(AngleAxis(np.pi / 2, Vector3.UnitX()))
        io = SimplicialComplexIO(pre)
        ribbon_mesh = io.read(str(RIBBON_MESH))
        label_surface(ribbon_mesh)
        slbws.apply_to(ribbon_mesh, moduli=cloth_moduli, mass_density=200.0, thickness=0.0001)
        dsb.apply_to(ribbon_mesh, bending_stiffness=40.0)
        ribbon_obj.geometries().create(ribbon_mesh)

    # Non-fixed rigid box
    # Same Y-up → Z-up rotation
    if not args.no_box:
        box_obj = scene.objects().create("box")
        pre = Transform.Identity()
        pre.rotate(AngleAxis(np.pi / 2, Vector3.UnitX()))
        io = SimplicialComplexIO(pre)
        box_mesh = io.read(str(BOX_MESH))
        label_surface(box_mesh)
        abd.apply_to(box_mesh, 100e6, mass_density=1000.0)
        box_obj.geometries().create(box_mesh)

    # Ground: either half-plane or fixed ABD floor box
    if args.no_ground:
        # Fixed floor box as ground replacement (to test if half-plane causes the crash)
        floor_obj = scene.objects().create("floor_box")
        pre = Transform.Identity()
        # Scale cube to a flat slab, then shift down so top face is at z=0
        pre.scale(2.0)
        trimesh_path = REPO_ROOT / "DemoAssets" / "assets" / "sim_data" / "trimesh"
        io = SimplicialComplexIO(pre)
        floor_mesh = io.read(str(trimesh_path / "cube.obj"))
        # Shift the mesh down so the top face is at z=0
        positions = view(floor_mesh.positions())
        positions[:, 2] -= float(np.max(positions[:, 2]))
        label_surface(floor_mesh)
        abd.apply_to(floor_mesh, 100e6, mass_density=1000.0)
        is_fixed = floor_mesh.instances().find(builtin.is_fixed)
        view(is_fixed)[:] = 1
        floor_obj.geometries().create(floor_mesh)
    else:
        ground_obj = scene.objects().create("ground")
        g = ground(0.0, Vector3.UnitZ())
        ground_obj.geometries().create(g)

    world.init(scene)

    if not args.no_viewer:
        import polyscope as ps
        from polyscope import imgui
        from uipc.gui import SceneGUI

        ps.init()
        ps.set_ground_plane_height(0.0)
        sgui = SceneGUI(scene, "split")
        sgui.register()
        sgui.set_edge_width(1.0)

        run = False

        def on_update():
            nonlocal run
            if imgui.Button("run & stop"):
                run = not run
            if run:
                world.advance()
                world.retrieve()
                sgui.update()

        ps.set_user_callback(on_update)
        ps.show()
    else:
        for step in range(args.steps):
            world.advance()
            world.retrieve()
            if step % 50 == 0:
                print(f"  Step {step}")
        print("Done.")


if __name__ == "__main__":
    main()
