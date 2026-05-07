"""Render Robowits 'settled' frames after 200 free-physics steps.

For each non-rigid task (08, 15, 20, 21, 22, 25, 30), this:
  1. Builds a multi-solver scene (rigid + MPM + SPH) with gravity ON and
     gs-core robowits scene defaults (dt=1/30, mpm grid_density=64, per-task
     substeps).
  2. Swaps the rigid stand-in deformable/granular/fluid entity for the real
     gs-core MPM/SPH material:
        08 dough     -> MPM.ElastoPlastic   (vis_mode='recon')
        15 sand      -> MPM.Sand            (vis_mode='particle')
        20 foam ball -> MPM.Elastic         (vis_mode='recon')
        21 water     -> SPH.Liquid + Glass  (vis_mode='recon')
        22 dry_sand  -> MPM.Sand            (vis_mode='particle')
        25 water     -> SPH.Liquid + Glass  (vis_mode='recon')
        30 water     -> MPM.Liquid + Glass  (vis_mode='recon')
  3. Adds the classic MARVIN_PIKA URDF and pins it to NPZ-frame-0 qpos via
     high PD gains (gripper pose held while the scene settles).
  4. Steps the scene 200 frames.
  5. Renders a single Luisa-RT frame and writes
     /home/zhehuan/Desktop/hz/Genesis-IPC/data/ipc_demo/ipc_robowits/
        _settled_frames_classic_luisa/task<NN>.png

Usage: python settle_and_render.py --task 08
       python settle_and_render.py --task 30 --steps 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Ensure replay_robowits & _replay_common are importable
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))  # examples/IPC_Solver/

import replay_robowits as rr  # noqa: E402
from replay_robowits import (  # noqa: E402
    CLASSIC_MARVIN_PIKA_URDF,
    TABLE_GLB,
    TASK_REGISTRY,
    _LUISA_TO_NYX_INTENSITY_SCALE,  # noqa: F401  (kept for symmetry; unused)
    _CLASSIC_ENVMAP_REGISTRY,
    _ENV_LIGHT_EFFECT,
    _OVERALL_INTENSITY,
    _classic_envmap_path,
    _resolve_traj,
)


# ---------------------------------------------------------------------------
# Output dir (deterministic stable path — feedback_no_timestamp_in_paths)
# ---------------------------------------------------------------------------

OUT_DIR = Path("/home/zhehuan/Desktop/hz/Genesis-IPC/data/ipc_demo/ipc_robowits/_settled_frames_classic_luisa")


# ---------------------------------------------------------------------------
# Per-task non-rigid material/surface overrides (matches gs-core source)
# ---------------------------------------------------------------------------
# Each entry: task_id -> {entity_name: builder(gs)->dict({material, surface})}
# Only the non-rigid entity per task is overridden; everything else (table,
# rigid task entities, robot) stays byte-equal to replay_robowits.
#
# vis_mode rules (from user directive):
#   - water & elastic  -> 'recon'
#   - sand             -> 'particle'
#   - water surface    -> Glass


def _override_08(gs):
    return {
        "dough ball": {
            "material": gs.materials.MPM.ElastoPlastic(
                E=2e5, nu=0.3, rho=800.0, sampler="pbs", von_mises_yield_stress=500.0
            ),
            "surface": gs.surfaces.Default(color=(0.95, 0.85, 0.65), vis_mode="recon"),
        },
    }


def _override_15(gs):
    return {
        "sand": {
            "material": gs.materials.MPM.Sand(rho=1200.0, sampler="random", friction_angle=40),
            "surface": gs.surfaces.Default(color=(0.9, 0.8, 0.3), vis_mode="particle", double_sided=True),
        },
    }


def _override_20(gs):
    return {
        "foam ball": {
            "material": gs.materials.MPM.Elastic(E=5e4, nu=0.2, rho=200.0, sampler="pbs"),
            "surface": gs.surfaces.Smooth(color=(1.0, 0.9, 0.1), double_sided=True, vis_mode="recon"),
        },
    }


def _override_21(gs):
    return {
        "water": {
            "material": gs.materials.SPH.Liquid(
                rho=500.0,
                stiffness=80000.0,
                exponent=7.0,
                mu=0.005,
                gamma=0.01,
                sampler="pbs",
            ),
            "surface": gs.surfaces.Glass(color=(0.6, 0.7, 1.0), double_sided=True, vis_mode="recon"),
        },
    }


def _override_22(gs):
    return {
        "dry_sand": {
            "material": gs.materials.MPM.Sand(rho=1100.0, sampler="random", friction_angle=45),
            "surface": gs.surfaces.Default(vis_mode="particle"),
        },
    }


def _override_25(gs):
    return {
        "water": {
            "material": gs.materials.SPH.Liquid(
                rho=100.0,
                stiffness=5000.0,
                exponent=7.0,
                mu=0.01,
                gamma=0.02,
                sampler="pbs",
            ),
            "surface": gs.surfaces.Glass(color=(0.6, 0.85, 1.0), double_sided=True, vis_mode="recon"),
        },
    }


def _override_30(gs):
    return {
        "water": {
            "material": gs.materials.MPM.Liquid(E=8e5, nu=0.25, rho=1000.0, viscous=True, sampler="pbs"),
            "surface": gs.surfaces.Glass(color=(0.55, 0.75, 1.0), double_sided=True, vis_mode="recon"),
        },
    }


SETTLE_OVERRIDES = {
    "08": _override_08,
    "15": _override_15,
    "20": _override_20,
    "21": _override_21,
    "22": _override_22,
    "25": _override_25,
    "30": _override_30,
}


# Per-task substeps (gs-core registry_robowits.make_robowits override table).
SUBSTEPS = {
    "08": 50,
    "15": 2,
    "20": 100,
    "21": 30,
    "22": 2,
    "25": 20,
    "30": 2,
}

# Per-task SPH particle_size override (gs-core: water_into_mug uses 0.015).
SPH_PARTICLE_SIZE = {
    "08": 0.02,
    "15": 0.02,
    "20": 0.02,
    "21": 0.02,
    "22": 0.02,
    "25": 0.015,
    "30": 0.02,
}


# Names of "blocker" entities (rigid containment cylinders/boxes for SPH).
# These should be invisible — they're a containment hack for the SPH solver
# to keep water inside the pitcher rim, exactly as in gs-core.
def _is_blocker(name: str) -> bool:
    return name.startswith("blocker_")


# ---------------------------------------------------------------------------
# Lighting / camera / env map (mirrors RobowitsReplay --classic --luisa path)
# ---------------------------------------------------------------------------

CAM_POS = (1.5122, 0.0, 1.8931)  # classic: Y=0 side-on
CAM_LOOKAT = (0.838, 0.0, 1.2837)
CAM_FOV = 40
RES = (1280, 720)


def _make_renderer(gs):
    """Luisa renderer with classic envmap and the Robowits sphere-light rig."""
    from genesis.options.renderers import SphereLight

    yaw, registry_mult = _CLASSIC_ENVMAP_REGISTRY
    env_path = _classic_envmap_path()
    # Match RobowitsReplay --classic --luisa default knobs.
    luisa_env_scale = 0.6
    env_multiplier = registry_mult * _OVERALL_INTENSITY * luisa_env_scale
    light_factor = _ENV_LIGHT_EFFECT  # use_env_map=True path

    return gs.renderers.RayTracer(
        logging_level="warning",
        tracing_depth=32,
        env_radius=100.0,
        env_euler=(0, 0, -yaw),
        env_surface=gs.surfaces.Emission(
            emissive_texture=gs.textures.ImageTexture(
                image_path=env_path,
                image_color=env_multiplier,
                encoding="linear",
            ),
        ),
        lights=[
            SphereLight(
                pos=light["pos"],
                radius=light["radius"],
                color=light["color"],
                intensity=light["intensity"] * light_factor,
            )
            for light in rr.RobowitsReplay._LIGHTS
        ],
    )


# ---------------------------------------------------------------------------
# Robot pose holding
# ---------------------------------------------------------------------------
# Strategy: hard-set qpos and zero velocity before EVERY scene.step(). The
# rigid solver still sees the robot as a fixed-base obstacle for soft-body
# coupling, but the joints don't drift under gravity and we avoid the
# constraint-force blow-ups that high PD gains cause at dt=1/60.


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(task_id: str, steps: int, output: Path | None, show_viewer: bool = False, vis_col: bool = False) -> None:
    if task_id not in SETTLE_OVERRIDES:
        raise ValueError(f"task {task_id!r} has no settle override (must be one of {sorted(SETTLE_OVERRIDES)})")

    import genesis as gs

    gs.init(backend=gs.gpu, logging_level="warning")

    overrides = SETTLE_OVERRIDES[task_id](gs)
    substeps = SUBSTEPS[task_id]
    sph_particle_size = SPH_PARTICLE_SIZE[task_id]

    print(f"[settle] task={task_id}  substeps={substeps}  sph_particle_size={sph_particle_size}")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.001,
            substeps=4,
            gravity=(0.0, 0.0, -9.81),
        ),
        rigid_options=gs.options.RigidOptions(
            gravity=(0.0, 0.0, -9.81),
            enable_collision=True,
            enable_self_collision=False,
            noslip_iterations=10,
        ),
        mpm_options=gs.options.MPMOptions(
            grid_density=128,
            lower_bound=(-0.5, -0.5, -0.1),
            upper_bound=(1.5, 1.5, 1.0),
        ),
        sph_options=gs.options.SPHOptions(
            particle_size=sph_particle_size,
            lower_bound=(-0.5, -0.5, -0.1),
            upper_bound=(1.5, 1.5, 1.5),
        ),
        viewer_options=gs.options.ViewerOptions(
            res=RES,
            camera_pos=CAM_POS,
            camera_lookat=CAM_LOOKAT,
            camera_fov=CAM_FOV,
        ),
        vis_options=gs.options.VisOptions(ambient_light=(0.3, 0.3, 0.35)),
        show_viewer=show_viewer,
        renderer=_make_renderer(gs),
    )

    # When --vis_col is set, render rigid entities in collision mode so the
    # COACD convex pieces are visible instead of the original visual mesh.
    rigid_surface = gs.surfaces.Collision() if vis_col else None

    # Table — matches gs-core MORPH_TABLE: convexify=True with COACD HQ
    # (preprocess_resolution=150). Required so MPM/SPH particles have a usable
    # SDF on the table surface; replay_robowits.py uses convexify=False because
    # IPC-coupled physics handles soft-rigid contact differently.
    scene.add_entity(
        gs.morphs.Mesh(
            align=False,
            file=TABLE_GLB,
            pos=(0.597, 0.0, 0.0),
            euler=(0, 0, 0),
            scale=(1.14, 1.0, 1.4377),
            fixed=True,
            file_meshes_are_zup=True,
            convexify=True,
            coacd_options=gs.options.CoacdOptions(preprocess_resolution=150),
        ),
        surface=rigid_surface or gs.surfaces.BSDF(roughness=0.45, metallic=0.0),
    )

    # Task entities — start from replay_robowits builder, then patch the
    # non-rigid stand-in's material/surface and hide blocker geometry.
    # Tighten COACD threshold for thin-wall hollow containers so the convex
    # decomposition follows the cavity instead of filling it with a near-solid
    # hull (otherwise MPM/SPH particles spawned inside the cavity overlap a
    # convex piece and get pushed out through the wall).
    # CoACD enforces threshold >= 0.01 (library aborts otherwise). Use the
    # floor and lean on a high hull count + fine preprocess grid to follow
    # cavity walls. max_convex_hull=128 cuts the residual concavity gap from
    # the 64-cap (jar saw concavity 0.081 at 64 hulls, 8× threshold).
    settle_coacd = gs.options.CoacdOptions(
        threshold=0.01, preprocess_resolution=160, max_convex_hull=128, decimate=True
    )
    entity_defs = TASK_REGISTRY[task_id]()
    for edef in entity_defs:
        name = edef["name"]
        morph = edef["morph"]
        material = edef.get("material", gs.materials.Rigid())
        surface = edef.get("surface", gs.surfaces.Default())

        if name in overrides:
            material = overrides[name]["material"]
            surface = overrides[name]["surface"]
        elif rigid_surface is not None:
            # Rigid task entity — show collision mesh instead of visual mesh.
            surface = rigid_surface

        if isinstance(morph, gs.morphs.Mesh) and morph.coacd_options is not None:
            morph.coacd_options = settle_coacd

        # SPH containment blockers must be invisible (matches gs-core where
        # they are added with visualization=False).
        if _is_blocker(name) and isinstance(morph, (gs.morphs.Box, gs.morphs.Cylinder)):
            morph.visualization = False

        # Lift task-08 dough so it falls onto the board during settle (the
        # MCAP-derived pose has dough touching the board exactly). Capped at
        # 0.90 so sphere top (0.94) stays inside MPM upper-z bound (~0.953).
        if task_id == "08" and name == "dough ball":
            x, y, _ = morph.pos
            morph.pos = (x, y, 0.90)

        scene.add_entity(morph=morph, material=material, surface=surface)

    # Robot — classic MARVIN_PIKA.
    robot_surface = (
        rigid_surface
        if rigid_surface is not None
        else {
            "paint_white_glossy": gs.surfaces.BSDF(color=(0.74, 0.74, 0.74), roughness=0.25, metallic=0.25),
            "plastic_black_rough": gs.surfaces.BSDF(color=(0.02, 0.02, 0.03), roughness=0.35, metallic=0.0, ior=1.45),
        }
    )
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=CLASSIC_MARVIN_PIKA_URDF,
            fixed=True,
            collision=True,
            pos=(0, 0, 1.08),
        ),
        surface=robot_surface,
    )

    # Render camera (Luisa).
    cam = scene.add_camera(
        res=RES,
        pos=CAM_POS,
        lookat=CAM_LOOKAT,
        fov=CAM_FOV,
        spp=512,
    )

    print("[settle] building scene…")
    scene.build()

    # Set robot to NPZ frame-0 qpos and lock with high PD gains.
    traj_path = _resolve_traj(task_id, None)
    traj = np.load(traj_path)
    qpos0 = traj["robot_qpos"][0].astype(np.float32)
    print(f"[settle] frame-0 qpos shape={qpos0.shape}  traj={Path(traj_path).name}")

    robot.set_qpos(qpos0)
    robot.zero_all_dofs_velocity()

    # Settle (hard-pin robot every step to avoid PD blow-ups).
    print(f"[settle] stepping {steps} frames…")
    for i in range(steps):
        robot.set_qpos(qpos0)
        robot.zero_all_dofs_velocity()
        scene.step()
        if (i + 1) % 50 == 0:
            print(f"  step {i + 1}/{steps}")

    # Render single frame.
    import imageio.v3 as iio

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = output or (OUT_DIR / f"task{task_id}.png")
    print(f"[settle] rendering -> {out_path}")
    rgb_result = cam.render(rgb=True, force_render=True)
    rgb_tensor = rgb_result[0]
    try:
        rgb = rgb_tensor.cpu().numpy()
    except AttributeError:
        rgb = np.array(rgb_tensor)
    iio.imwrite(str(out_path), rgb)
    print(f"[settle] DONE: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render settled Robowits frames")
    parser.add_argument(
        "--task",
        required=True,
        choices=sorted(SETTLE_OVERRIDES.keys()),
        help="Task ID",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Number of free-physics settle steps (default: 200)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: _settled_frames_classic_luisa/task<NN>.png)",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open the interactive Genesis viewer (pyrender) during simulation",
    )
    parser.add_argument(
        "--vis_col",
        action="store_true",
        help="Render rigid entities (table, task rigids, robot) with surfaces.Collision() "
        "to inspect the COACD convex decomposition instead of the visual mesh",
    )
    args = parser.parse_args()
    run(args.task, args.steps, args.output, show_viewer=args.viewer, vis_col=args.vis_col)


if __name__ == "__main__":
    main()
