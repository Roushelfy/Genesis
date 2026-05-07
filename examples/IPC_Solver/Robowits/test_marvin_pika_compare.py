"""Render archive vs current marvin_pika URDF (one at a time).

Loads a single robot — either the archive marvin_pika with a per-material BSDF
override (paint_white_glossy → r=0.25 m=0.25, plastic_black_rough …) or the
current marvin_pika that has those values baked into the GLBs.

If override + bake are equivalent, the two should render identically. With
--nyx they currently do NOT, because nyx_scene_exporter drops the dict override
on URDF entities and falls back to whatever the underlying GLB has baked in.
Use this script to demonstrate / regress that behaviour.

Examples
--------
    # Current (no override, baked GLB) — Luisa
    python test_marvin_pika_compare.py --luisa --output current_luisa.png

    # Archive + override — Luisa
    python test_marvin_pika_compare.py --archive --luisa --output archive_luisa.png

    # Same pair under Nyx
    python test_marvin_pika_compare.py --nyx --output current_nyx.png
    python test_marvin_pika_compare.py --archive --nyx --output archive_nyx.png

    # Archive WITHOUT override (baseline — shows raw archive GLB)
    python test_marvin_pika_compare.py --archive --no-override --nyx
"""

import argparse
import sys
from pathlib import Path

import numpy as np

import genesis as gs
from genesis.options.renderers import SphereLight

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "examples" / "IPC_Solver"))
from _replay_common import marvin_urdf  # noqa: E402

ARCHIVE_URDF = str(_REPO / "data" / "archive_marvin_demoassets" / "marvin_robot" / "urdf" / "marvin_pika.urdf")

# Mirror the lighting rig used by Robowits classic mode (replay_robowits.py):
# small_empty_room_1 polyhaven EXR + 3 sphere lights, dampened by 0.1 when the
# env map is contributing.
_OVERALL_INTENSITY = 0.5
_ENV_LIGHT_EFFECT = 0.1
_LUISA_TO_NYX_INTENSITY_SCALE = 0.3
_ROBOWITS_LIGHTS = [
    {"pos": (0.5, 1.1, 2.4), "radius": 0.2, "color": (1.0, 0.97, 0.92), "intensity": 50.0},
    {"pos": (0.5, -1.8, 4.2), "radius": 1.0, "color": (0.48, 0.52, 0.6), "intensity": 1.0},
    {"pos": (-0.8, -3.0, 0.5), "radius": 0.25, "color": (0.8, 0.88, 1.0), "intensity": 150.0},
]


def archive_surface_override():
    """Per-material BSDF override that pre-e8b0feef applied to archive marvin GLBs.

    These values were later baked into the new marvin_pika GLBs (commit
    e8b0feef), so archive+override and current+no-override SHOULD render the
    same. They don't, in nyx — which is the point of this test.
    """
    return {
        "paint_white_glossy": gs.surfaces.BSDF(
            color=(0.74, 0.74, 0.74),
            roughness=0.25,
            metallic=0.25,
        ),
        "plastic_black_rough": gs.surfaces.BSDF(
            color=(0.02, 0.02, 0.03),
            roughness=0.35,
            metallic=0.0,
            ior=1.45,
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--nyx", action="store_true", help="Render with Nyx")
    backend.add_argument("--luisa", action="store_true", help="Render with LuisaRender (default)")
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Load the archive marvin_pika URDF (with BSDF override). Default loads the current one.",
    )
    parser.add_argument(
        "--no-override",
        action="store_true",
        help="Skip the BSDF override on the archive robot (baseline test)",
    )
    parser.add_argument("--res", type=int, nargs=2, default=(960, 720))
    parser.add_argument("--spp", type=int, default=1024)
    parser.add_argument(
        "--envmap",
        type=str,
        default=None,
        help="Optional EXR/HDR env map (defaults to the polyhaven small_empty_room cached locally)",
    )
    args = parser.parse_args()

    use_nyx = args.nyx
    if not use_nyx and not args.luisa:
        args.luisa = True

    gs.init(backend=gs.gpu, logging_level="warning")

    scene_kwargs = dict(
        sim_options=gs.options.SimOptions(dt=1 / 60),
        show_viewer=False,
    )
    env_map_rotation = 0.0
    if not use_nyx:
        scene_kwargs["renderer"] = gs.renderers.RayTracer(
            tracing_depth=32,
            env_radius=100.0,
            env_euler=(0.0, 0.0, env_map_rotation),
            env_surface=gs.surfaces.Emission(
                emissive_texture=gs.textures.ImageTexture(
                    image_path=_resolve_envmap(args),
                    image_color=_OVERALL_INTENSITY,
                    encoding="linear",
                ),
            ),
            lights=[
                SphereLight(
                    pos=l["pos"],
                    radius=l["radius"],
                    color=l["color"],
                    intensity=l["intensity"] * _ENV_LIGHT_EFFECT,
                )
                for l in _ROBOWITS_LIGHTS
            ],
        )

    scene = gs.Scene(**scene_kwargs)

    scene.add_entity(
        gs.morphs.Plane(),
        surface=gs.surfaces.Rough(color=(0.0, 0.0, 0.0)),
    )

    robot_pos = (0.0, 0.0, 1.08)

    if args.archive:
        label = "archive"
        urdf_file = ARCHIVE_URDF
        surface = None if args.no_override else archive_surface_override()
    else:
        label = "current"
        urdf_file = marvin_urdf("marvin_pika")
        surface = None

    scene.add_entity(
        gs.morphs.URDF(file=urdf_file, fixed=True, pos=robot_pos),
        surface=surface,
    )

    cam_pos = (2.0, 0.0, 2.2)
    cam_lookat = (0.0, 0.0, 1.1)
    cam_fov = 45.0

    if use_nyx:
        from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions
        from gs_nyx_plugin.nyx_camera_sensor import NyxCameraSensor  # noqa: F401
        import gs_nyx.nyx_py_renderer as npr
        import gs_nyx.nyx_py_sdk as ap

        env_map = ap.EnvironmentMapAsset()
        env_map.texture = _resolve_envmap(args)
        env_map.rotation = env_map_rotation
        env_map.multiplier = _OVERALL_INTENSITY

        nyx_lights = [
            {
                "type": "point",
                "pos": l["pos"],
                "radius": float(l["radius"]),
                "color": l["color"],
                "intensity": float(l["intensity"]) * _ENV_LIGHT_EFFECT * _LUISA_TO_NYX_INTENSITY_SCALE,
            }
            for l in _ROBOWITS_LIGHTS
        ]

        cam = scene.add_sensor(
            NyxCameraOptions(
                res=tuple(args.res),
                pos=cam_pos,
                lookat=cam_lookat,
                fov=cam_fov,
                spp=args.spp,
                denoise=True,
                render_mode=npr.ERenderMode.RefPathTracer,
                env_maps=(env_map,),
                lights=nyx_lights,
            )
        )
    else:
        cam = scene.add_camera(
            res=tuple(args.res),
            pos=cam_pos,
            lookat=cam_lookat,
            fov=cam_fov,
            spp=args.spp,
        )

    print(
        f"[compare] backend={'nyx' if use_nyx else 'luisa'}, robot={label}, "
        f"override={'no' if (args.no_override or not args.archive) else 'yes'}",
        flush=True,
    )
    scene.build()
    scene.step()

    output_file = f"./data/ipc_demo/ipc_robowits/marvin_pika_compare_{use_nyx}_{args.archive}_{args.no_override}.png"
    if use_nyx:
        data = cam.read()
        rgb = data.rgb.cpu().numpy()
        gs.tools.save_img_arr(rgb, output_file)
    else:
        rgb = cam.render()[0]
        if isinstance(rgb, np.ndarray):
            gs.tools.save_img_arr(rgb, output_file)
        else:
            gs.tools.save_img_arr(np.asarray(rgb), output_file)
    print(f"[compare] saved {output_file}")


def _resolve_envmap(args) -> str:
    if args.envmap:
        return args.envmap
    cache_dir = _REPO / "DemoAssets" / "_polyhaven_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / "small_empty_room_1_4k.exr"
    if not target.exists():
        import urllib.request

        url = "https://dl.polyhaven.org/file/ph-assets/HDRIs/exr/4k/small_empty_room_1_4k.exr"
        print(f"[compare] downloading {url} -> {target}", flush=True)
        urllib.request.urlretrieve(url, target)
    return str(target)


if __name__ == "__main__":
    main()
