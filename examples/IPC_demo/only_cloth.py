"""Run cloth-only IPC scene with Sweater-like solver config."""

from pathlib import Path

import genesis as gs

gs.init(backend=gs.cpu)

REPO_ROOT = Path(__file__).resolve().parents[2]
CLOTH_DIR = REPO_ROOT / "DemoAssets" / "Cloth"
CLOTH_OBJ = CLOTH_DIR / "wearing_cloth_0_init-shape.obj"

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.005),
    fem_options=gs.options.FEMOptions(
        use_rigid_compatible_transform=False,
    ),
    coupler_options=gs.options.IPCCouplerOptions(
        contact_enable=True,
        contact_friction_enable=True,
        contact_d_hat=0.001,
        newton_tolerance=0.1,
        newton_translation_tolerance=10.0,
        linear_system_tolerance=1e-4,
        # newton_semi_implicit_enable=True,
        enable_rigid_rigid_contact=False,
        ignore_end_effector_check=True,
    ),
    show_viewer=True,
)

scene.add_entity(
    gs.morphs.Plane(),
    material=gs.materials.Rigid(coup_type="ipc_only"),
    name="ground_plane",
)

scene.add_entity(
    surface=gs.surfaces.Plastic(
        opacity=0.8,
    ),
    morph=gs.morphs.Mesh(
        file=str(CLOTH_OBJ),
        scale=1.0,
        pos=(0.0, 0.0, 1.0),
    ),
    material=gs.materials.FEM.Cloth(
        E=8e3,
        nu=0.45,
        rho=200,
        thickness=0.0001,
        bending_stiffness=37.0,
        friction_mu=0.5,
    ),
    name="cloth",
)

scene.build()

while True:
    scene.step()
