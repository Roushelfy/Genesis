"""Render frame 0 of the orbit and showcase views as PNGs.

By default, both PNGs land in ``data/ipc_demo/ipc_yoyo_v4_separated/``:

    orbit_first_frame.png       - frame 0 of the bullet-time orbit camera (Nyx)
    showcase_first_frame.png    - the exploded-parts showcase, robot hidden (Nyx)

Usage:
    python examples/IPC_Solver/yoyo/render_first_frames.py
    python examples/IPC_Solver/yoyo/render_first_frames.py --out-dir <path>
    python examples/IPC_Solver/yoyo/render_first_frames.py --phase orbit
    python examples/IPC_Solver/yoyo/render_first_frames.py --phase showcase

The two renders share Nyx GPU state poorly, so the top-level invocation spawns
a fresh subprocess per phase. The same script handles both spawn and worker
roles via ``--phase``.

Nyx needs the local Vulkan SDK on ``LD_LIBRARY_PATH``; if ``VULKAN_SDK`` is
not already set, this script auto-points it at the team's standard install at
``/home/qq/Desktop/github/jaehoon_version/1.4.321.0/x86_64`` (skipped if that
directory is missing).
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
# _replay_common / _yoyo_common / replay_yoyo_traj live one level up in
# examples/IPC_Solver/, so the spawned workers can find them.
sys.path.insert(0, str(SCRIPT_DIR.parent))
DEFAULT_OUT = REPO_ROOT / "data/ipc_demo/ipc_yoyo_v4_separated"
DEFAULT_VULKAN_SDK = "/home/qq/Desktop/github/jaehoon_version/1.4.321.0/x86_64"


def _nyx_env() -> dict:
    """Return os.environ + Vulkan SDK paths Nyx needs."""
    env = os.environ.copy()
    if "VULKAN_SDK" not in env and Path(DEFAULT_VULKAN_SDK).is_dir():
        env["VULKAN_SDK"] = DEFAULT_VULKAN_SDK
        env["LD_LIBRARY_PATH"] = f"{DEFAULT_VULKAN_SDK}/lib:" + env.get("LD_LIBRARY_PATH", "")
        env["PATH"] = f"{DEFAULT_VULKAN_SDK}/bin:" + env.get("PATH", "")
    return env


def _save_first_frame(mp4: Path, out: Path) -> None:
    import imageio
    out.parent.mkdir(parents=True, exist_ok=True)
    reader = imageio.get_reader(str(mp4))
    imageio.imwrite(str(out), reader.get_data(0))
    reader.close()
    print(f"saved {out}")


BLACK_EXR = REPO_ROOT / "DemoAssets/textures/pure_black.exr"


def _force_black_env_map(args):
    """Replace ``--dark-bg``'s light-grey env-map with pure_black.exr.

    Monkeypatches gs_nyx_plugin's EnvironmentMapAsset so every instance whose
    texture is set to g_warm_light_gray_02.exr (the dark-bg default) is
    redirected to pure_black.exr with a 0× multiplier — effectively a pure
    black background for Nyx.
    """
    import gs_nyx.nyx_py_sdk as ap

    _OriginalEnv = ap.EnvironmentMapAsset

    class _BlackEnv(_OriginalEnv):
        def __setattr__(self, key, value):
            if key == "texture" and "g_warm_light_gray_02.exr" in str(value):
                value = str(BLACK_EXR)
                super().__setattr__("multiplier", 0.0)
            super().__setattr__(key, value)

    ap.EnvironmentMapAsset = _BlackEnv


def _render_orbit(out_dir: Path) -> None:
    sys.path.insert(0, str(SCRIPT_DIR))
    os.chdir(REPO_ROOT)
    sys.argv = ["render_bullet_time_frames.py", "--mode", "orbit", "--render", "--nyx", "--dark-bg"]

    import render_bullet_time_frames as RB
    RB.compute_needed_frames = lambda n: [0]

    _force_black_env_map(None)

    renderer = RB.BulletTimeRenderer()
    renderer.run()

    mp4 = REPO_ROOT / f"data/ipc_demo/ipc_{renderer.name}/{renderer.name}_v4_nyx.mp4"
    _save_first_frame(mp4, out_dir / "orbit_first_frame.png")


def _render_showcase(out_dir: Path) -> None:
    sys.path.insert(0, str(SCRIPT_DIR))
    os.chdir(REPO_ROOT)
    sys.argv = ["render_yoyo_v4_frame0.py", "--render", "--robot-pass", "hidden", "--dark-bg", "--nyx"]

    import render_yoyo_v4_frame0 as RY

    # Limit to a single rendered frame.
    original_load = RY.YoyoV4Showcase.load_trajectory

    def patched_load(self):
        original_load(self)
        return 1

    RY.YoyoV4Showcase.load_trajectory = patched_load

    _force_black_env_map(None)

    renderer = RY.YoyoV4Showcase()
    renderer.run()

    mp4 = (
        REPO_ROOT
        / f"data/ipc_demo/ipc_{renderer.name}/ipc_{renderer.name}_v4_nyx.mp4"
    )
    _save_first_frame(mp4, out_dir / "showcase_first_frame.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out-dir", default=str(DEFAULT_OUT),
        help=f"Where to write the two PNGs (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--phase", choices=("orbit", "showcase"),
        help="Render only one phase. Internal flag used by the spawned worker.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    if args.phase == "orbit":
        _render_orbit(out_dir)
        return
    if args.phase == "showcase":
        _render_showcase(out_dir)
        return

    here = Path(__file__).resolve()
    for phase in ("orbit", "showcase"):
        cmd = [sys.executable, str(here), "--phase", phase, "--out-dir", str(out_dir)]
        print(f">>> {' '.join(cmd)}")
        subprocess.run(cmd, env=_nyx_env(), check=True)


if __name__ == "__main__":
    main()
