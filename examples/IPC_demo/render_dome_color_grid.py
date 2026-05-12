"""
Render a metallic × color matrix of the yoyo showcase frame 0 with true-black
background. Rows = dome metallicFactor (0.3, 0.6, 0.9); columns = dome base
color (teal, cyan, gold, rose_red, firefly_green, sapphire_blue).

For each cell, regenerate the yoyo-{top,bottom}_shell_logo.glb pair with the
chosen (color, metallic) combo, run render_yoyo_v4_frame0.py at frame 0 with
--robot-pass hidden --dark-bg, save the PNG to
data/tests/dome_colors/m{metallic}/{color}.png, and assemble a 3x6 contact
sheet at data/tests/dome_colors/_matrix.png.

Run from the repo root:
    LD_PRELOAD=/usr/local/cuda-12.9/.../libcublas.so.12 \\
        python examples/IPC_demo/render_dome_color_grid.py
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import trimesh
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "DemoAssets/yoyo/scripts"))
import export_yoyo_glb as E  # noqa: E402

V3 = REPO / "DemoAssets/yoyo/v3"
OUT = REPO / "data/tests/dome_colors"
OUT.mkdir(parents=True, exist_ok=True)
FRAMES_DIR = REPO / "data/ipc_demo/ipc_yoyo_v4_showcase_hidden/ipc_yoyo_v4_showcase_hidden_v4_nyx_frames"


def rebuild_shells(base_rgb: tuple[int, int, int], logo_img: Image.Image, metallic: float) -> None:
    """Regenerate yoyo-{top,bottom}_shell_logo.glb with the given dome color
    and metallicFactor. Roughness is left at the export-module default."""
    E.SHELL_DOME_METALLIC = metallic
    ball = trimesh.load(str(V3 / "yoyo-ball.obj"), force="mesh")
    classified = E._classify_ball_parts(ball)
    dome_tex = E._gen_dome_texture(base=base_rgb)
    for half_key in ("top_shell", "bottom_shell"):
        shell = classified[half_key]
        if shell is None:
            continue
        half = "top" if "top" in half_key else "bottom"
        scene = trimesh.Scene()
        for geom_name, mesh in E._apply_shell_materials(shell, half, dome_tex, logo_img):
            scene.add_geometry(E._to_yup(mesh), geom_name=geom_name)
        out_path = V3 / f"yoyo-{half_key}_logo.glb"
        scene.export(str(out_path))


def render_frame0(out_png: Path) -> bool:
    """Run render_yoyo_v4_frame0.py for 1 frame, copy frame 0 PNG to out_png."""
    if FRAMES_DIR.exists():
        shutil.rmtree(FRAMES_DIR)
    env = os.environ.copy()
    env["LD_PRELOAD"] = "/usr/local/cuda-12.9/targets/x86_64-linux/lib/libcublas.so.12"
    cmd = [
        sys.executable,
        "-X",
        "faulthandler",
        "examples/IPC_Solver/yoyo/render_yoyo_v4_frame0.py",
        "--render",
        "--nyx",
        "--dark-bg",
        "--robot-pass",
        "hidden",
        "--save-frames",
        "--end-frame",
        "1",
    ]
    r = subprocess.run(cmd, env=env, cwd=str(REPO), capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [render failed] {r.stderr.splitlines()[-3:]}")
        return False
    pngs = sorted(FRAMES_DIR.glob("*.png")) if FRAMES_DIR.exists() else []
    if not pngs:
        print(f"  [no frames found in {FRAMES_DIR}]")
        return False
    shutil.copy(pngs[0], out_png)
    return True


def matrix_sheet(
    grid: list[list[Path | None]],
    row_labels: list[str],
    col_labels: list[str],
    out_path: Path,
    cell_w: int = 480,
    cell_h: int = 240,
) -> None:
    from PIL import ImageDraw, ImageFont

    pad = 32
    rows = len(grid)
    cols = len(grid[0]) if grid else 0
    W = pad + cell_w * cols
    H = pad + cell_h * rows
    sheet = Image.new("RGB", (W, H), (15, 15, 15))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    for ci, lbl in enumerate(col_labels):
        x = pad + ci * cell_w + cell_w // 2
        draw.text((x, pad // 4), lbl, fill=(220, 220, 220), font=font, anchor="mt")
    for ri, lbl in enumerate(row_labels):
        y = pad + ri * cell_h + cell_h // 2
        draw.text((pad // 4, y), lbl, fill=(220, 220, 220), font=font, anchor="lm")
    for ri in range(rows):
        for ci in range(cols):
            p = grid[ri][ci]
            if not p or not p.exists():
                continue
            img = Image.open(p).convert("RGB")
            scale = min(cell_w / img.width, cell_h / img.height)
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size, Image.LANCZOS)
            x = pad + ci * cell_w + (cell_w - img.width) // 2
            y = pad + ri * cell_h + (cell_h - img.height) // 2
            sheet.paste(img, (x, y))
    sheet.save(out_path)


METALLIC_LEVELS = [0.3, 0.6, 0.9, 1.0]


def main() -> None:
    logo_img = Image.open(REPO / "DemoAssets/yoyo/logo_centered.png").convert("RGB")
    colors = list(E.DOME_COLOR_PRESETS.items())
    grid: list[list[Path | None]] = [[None] * len(colors) for _ in METALLIC_LEVELS]
    for ri, m in enumerate(METALLIC_LEVELS):
        for ci, (name, rgb) in enumerate(colors):
            cell_dir = OUT / f"m{m:.1f}"
            cell_dir.mkdir(parents=True, exist_ok=True)
            out_png = cell_dir / f"{name}.png"
            if out_png.exists():
                print(f"\n=== metallic={m}  color={name} (cached) ===")
                grid[ri][ci] = out_png
                continue
            print(f"\n=== metallic={m}  color={name} {rgb} ===")
            rebuild_shells(rgb, logo_img, metallic=m)
            if render_frame0(out_png):
                print(f"  -> {out_png}")
                grid[ri][ci] = out_png

    # Restore canonical defaults so the repo's shells aren't left mutated.
    rebuild_shells(E.DOME_COLOR_PRESETS["teal"], logo_img, metallic=0.9)
    print("\n[restored shells to teal + metallic=0.9 default]")

    matrix_path = OUT / "_matrix.png"
    matrix_sheet(
        grid,
        row_labels=[f"metallic={m}" for m in METALLIC_LEVELS],
        col_labels=[name for name, _ in colors],
        out_path=matrix_path,
    )
    print(f"\nmatrix sheet -> {matrix_path}")


if __name__ == "__main__":
    main()
