"""
ABD cube -> shell plastic crease demo.

The demo builds:
- a thin shell sheet with `NeoHookeanShell + PlasticDiscreteShellBending`
- animated diagonal corner constraints on the sheet
- a driven ABD cube with `SoftTransformConstraint`
- a ground plane below the sheet

The sequence is:
1. hold one sheet corner fixed and move the opposite corner along an upper arc,
2. press the sheet from above using the flat bottom face of an ABD cube,
3. lift the cube away and inspect the remaining crease.

Run:
    python python/examples/abd_cube_shell_plastic_crease_demo.py

Controls:
    - `run / pause`: toggle playback
    - `step`: advance one frame

The simulation auto-pauses after the cycle so the residual fold can be inspected.
"""

import argparse
import os
import sys
import numpy as np

try:
    import polyscope as ps
    import polyscope.imgui as psim
except ModuleNotFoundError as exc:
    raise SystemExit("This example requires `polyscope`. Install it with `pip install polyscope`.") from exc

try:
    from uipc import Logger, Matrix4x4, Engine, World, Scene, SceneIO, Animation, view, builtin
    from uipc.geometry import (
        SimplicialComplex,
        SimplicialComplexIO,
        trimesh,
        ground,
        mesh_partition,
        label_surface,
        label_triangle_orient,
        flip_inward_triangles,
    )
except ImportError as exc:
    raise SystemExit(
        "This example requires the libuipc Python bindings (`uipc._native.pyuipc`). "
        "Build/install the Python package before running it."
    ) from exc

sys.path.append(os.path.dirname(__file__))
from asset_dir import AssetDir


SHEET_RESOLUTION = 40
SHEET_SIZE = 1.6
MESH_PARTITION_SIZE = 16

SHELL_THICKNESS = 5e-5
SHELL_DENSITY = 1200.0
SHELL_YOUNG = 1.5e9
SHELL_POISSON = 0.3
SHELL_BENDING_STIFFNESS = 1.2e6
SHELL_STRAIN_YIELD_THRESHOLD = 0.05
SHELL_STRAIN_HARDENING_MODULUS = 0.0
SHELL_STRESS_YIELD_STRESS = 4.2e5
SHELL_STRESS_HARDENING_MODULUS = 0.0


CUBE_SCALE = 0.6
CUBE_Y_HIGH = 1.55
CUBE_Y_LOW = 0.33
GROUND_Y = -0.01
CORNER_TARGET_GAP = 0.22

GATHER_FRAMES = 480
GATHER_HOLD_FRAMES = 40
PRESS_FRAMES = 180
PRESS_HOLD_FRAMES = 220
RELEASE_FRAMES = 160
SETTLE_FRAMES = 180
TOTAL_FRAMES = GATHER_FRAMES + GATHER_HOLD_FRAMES + PRESS_FRAMES + PRESS_HOLD_FRAMES + RELEASE_FRAMES + SETTLE_FRAMES


def process_closed_surface(sc: SimplicialComplex) -> SimplicialComplex:
    label_surface(sc)
    label_triangle_orient(sc)
    return flip_inward_triangles(sc)


def make_sheet_mesh(resolution: int, size: float) -> SimplicialComplex:
    xs = np.linspace(-0.5 * size, 0.5 * size, resolution)
    zs = np.linspace(-0.5 * size, 0.5 * size, resolution)

    vertices = []
    triangles = []

    for z in zs:
        for x in xs:
            vertices.append([x, 0.0, z])

    for j in range(resolution - 1):
        for i in range(resolution - 1):
            v00 = j * resolution + i
            v10 = v00 + 1
            v01 = v00 + resolution
            v11 = v01 + 1
            triangles.append([v00, v10, v11])
            triangles.append([v00, v11, v01])

    sheet = trimesh(np.asarray(vertices, dtype=np.float64), np.asarray(triangles, dtype=np.int32))
    label_surface(sheet)
    return sheet


def cube_transform(center_y: float) -> Matrix4x4:
    transform = Matrix4x4.Identity()
    transform[0:3, 3] = np.array([0.0, center_y, 0.0], dtype=np.float64)
    return transform


def smooth_lerp(a: float, b: float, t: float) -> float:
    t = np.clip(t, 0.0, 1.0)
    s = 0.5 - 0.5 * np.cos(np.pi * t)
    return a + (b - a) * s


def motion_schedule(frame: int) -> tuple[float, float, str]:
    if frame < GATHER_FRAMES:
        arc_alpha = smooth_lerp(0.0, 1.0, frame / max(GATHER_FRAMES - 1, 1))
        return arc_alpha, CUBE_Y_HIGH, "gather"

    frame -= GATHER_FRAMES
    if frame < GATHER_HOLD_FRAMES:
        return 1.0, CUBE_Y_HIGH, "gather-hold"

    frame -= GATHER_HOLD_FRAMES
    if frame < PRESS_FRAMES:
        cube_y = smooth_lerp(CUBE_Y_HIGH, CUBE_Y_LOW, frame / max(PRESS_FRAMES - 1, 1))
        return 1.0, cube_y, "press"

    frame -= PRESS_FRAMES
    if frame < PRESS_HOLD_FRAMES:
        return 1.0, CUBE_Y_LOW, "press-hold"

    frame -= PRESS_HOLD_FRAMES
    if frame < RELEASE_FRAMES:
        cube_y = smooth_lerp(CUBE_Y_LOW, CUBE_Y_HIGH, frame / max(RELEASE_FRAMES - 1, 1))
        return 1.0, cube_y, "release"

    return 1.0, CUBE_Y_HIGH, "inspect"


def upper_arc_point(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    alpha = np.clip(alpha, 0.0, 1.0)
    chord = end - start
    chord_len = np.linalg.norm(chord)
    if chord_len < 1.0e-8:
        return end.copy()

    center = 0.5 * (start + end)
    radius = 0.5 * chord_len
    u = (start - center) / radius

    v = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    theta = np.pi * alpha
    return center + radius * (np.cos(theta) * u + np.sin(theta) * v)


def build_demo(bending_model: str = "stress"):
    try:
        from uipc.constitution import (
            AffineBodyConstitution,
            SoftTransformConstraint,
            SoftPositionConstraint,
            NeoHookeanShell,
            StrainPlasticDiscreteShellBending,
            StressPlasticDiscreteShellBending,
            ElasticModuli2D,
        )
    except ImportError as exc:
        raise SystemExit(
            "This example requires newer libuipc Python bindings with "
            "`NeoHookeanShell`, `StrainPlasticDiscreteShellBending`, "
            "`StressPlasticDiscreteShellBending`, `SoftTransformConstraint`, "
            "`SoftPositionConstraint`, and `ElasticModuli2D`."
        ) from exc

    if bending_model not in {"strain", "stress"}:
        raise SystemExit(f"Unknown bending model '{bending_model}'. Available: 'strain', 'stress'.")

    Logger.set_level(Logger.Level.Warn)

    workspace = AssetDir.output_path(__file__)
    engine = Engine("cuda", workspace)
    world = World(engine)

    config = Scene.default_config()
    config["dt"] = 0.01
    config["gravity"] = [[0.0], [0.0], [0.0]]
    config["contact"]["enable"] = True
    config["contact"]["friction"]["enable"] = False
    config["contact"]["constitution"] = "ipc"
    config["line_search"]["max_iter"] = 12
    config["linear_system"]["tol_rate"] = 1.0e-3
    scene = Scene(config)

    scene.contact_tabular().default_model(0.2, 1.0e9)
    default_contact = scene.contact_tabular().default_element()

    sheet_object = scene.objects().create("sheet")
    cube_object = scene.objects().create("press_cube")

    shell = NeoHookeanShell()
    abd = AffineBodyConstitution()
    stc = SoftTransformConstraint()
    spc = SoftPositionConstraint()

    if bending_model == "strain":
        plastic_bending = StrainPlasticDiscreteShellBending()
        bending_constitution_name = "StrainPlasticDiscreteShellBending"
        bending_yield_label = "yield threshold"
        bending_yield_value = SHELL_STRAIN_YIELD_THRESHOLD
        bending_hardening_value = SHELL_STRAIN_HARDENING_MODULUS
    else:
        plastic_bending = StressPlasticDiscreteShellBending()
        bending_constitution_name = "StressPlasticDiscreteShellBending"
        bending_yield_label = "yield stress"
        bending_yield_value = SHELL_STRESS_YIELD_STRESS
        bending_hardening_value = SHELL_STRESS_HARDENING_MODULUS

    sheet = make_sheet_mesh(SHEET_RESOLUTION, SHEET_SIZE)
    # Enable the MAS preconditioner on the shell solve.
    mesh_partition(sheet, MESH_PARTITION_SIZE)
    moduli = ElasticModuli2D.youngs_poisson(SHELL_YOUNG, SHELL_POISSON)
    shell.apply_to(sheet, moduli, SHELL_DENSITY, SHELL_THICKNESS)
    if bending_model == "strain":
        plastic_bending.apply_to(
            sheet, SHELL_BENDING_STIFFNESS, SHELL_STRAIN_YIELD_THRESHOLD, SHELL_STRAIN_HARDENING_MODULUS
        )
    else:
        plastic_bending.apply_to(
            sheet, SHELL_BENDING_STIFFNESS, SHELL_STRESS_YIELD_STRESS, SHELL_STRESS_HARDENING_MODULUS
        )
    spc.apply_to(sheet, 50.0)
    default_contact.apply_to(sheet)

    sheet_rest_positions = np.array(view(sheet.positions()), copy=True).reshape(-1, 3)
    sheet_slot = sheet_object.geometries().create(sheet)[0]

    # Build a unit cube [-0.5, 0.5]^3 scaled by CUBE_SCALE (replaces cube.msh file).
    _cube_s = CUBE_SCALE / 2.0
    _cube_verts = np.array(
        [
            [-_cube_s, -_cube_s, -_cube_s],
            [_cube_s, -_cube_s, -_cube_s],
            [_cube_s, _cube_s, -_cube_s],
            [-_cube_s, _cube_s, -_cube_s],
            [-_cube_s, -_cube_s, _cube_s],
            [_cube_s, -_cube_s, _cube_s],
            [_cube_s, _cube_s, _cube_s],
            [-_cube_s, _cube_s, _cube_s],
        ],
        dtype=np.float64,
    )
    _cube_faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [2, 6, 7],
            [2, 7, 3],
            [0, 3, 7],
            [0, 7, 4],
            [1, 5, 6],
            [1, 6, 2],
        ],
        dtype=np.int32,
    )
    cube = trimesh(_cube_verts, _cube_faces)
    cube = process_closed_surface(cube)
    abd.apply_to(cube, 2.0e7)
    stc.apply_to(cube, np.array([3600.0, 120.0], dtype=np.float64))
    default_contact.apply_to(cube)

    view(cube.transforms())[0] = cube_transform(CUBE_Y_HIGH)
    cube_slot = cube_object.geometries().create(cube)[0]

    ground_object = scene.objects().create("ground")
    ground_object.geometries().create(ground(GROUND_Y))

    corner_a = 0
    corner_b = SHEET_RESOLUTION * SHEET_RESOLUTION - 1
    corner_c = SHEET_RESOLUTION - 1
    corner_d = SHEET_RESOLUTION * (SHEET_RESOLUTION - 1)
    rest_a = sheet_rest_positions[corner_a].copy()
    rest_b = sheet_rest_positions[corner_b].copy()
    rest_c = sheet_rest_positions[corner_c].copy()
    rest_d = sheet_rest_positions[corner_d].copy()
    corner_d_end = rest_c

    def animate_sheet(info: Animation.UpdateInfo):
        geo = info.geo_slots()[0].geometry()
        is_constrained = view(geo.vertices().find(builtin.is_constrained))
        aim_position = view(geo.vertices().find(builtin.aim_position))

        is_constrained[:] = 0
        is_constrained[corner_a] = 1
        is_constrained[corner_b] = 1
        is_constrained[corner_c] = 1
        is_constrained[corner_d] = 1

        arc_alpha, _, _ = motion_schedule(max(info.frame() - 1, 0))
        target_a = rest_a.reshape(3, 1)
        target_b = rest_b.reshape(3, 1)
        target_c = rest_c.reshape(3, 1)
        target_d = upper_arc_point(rest_d, corner_d_end, arc_alpha).reshape(3, 1)
        aim_position[corner_a] = target_a
        aim_position[corner_b] = target_b
        aim_position[corner_c] = target_c
        aim_position[corner_d] = target_d

    def animate_cube(info: Animation.UpdateInfo):
        geo = info.geo_slots()[0].geometry()
        is_constrained = view(geo.instances().find(builtin.is_constrained))
        aim_transform = view(geo.instances().find(builtin.aim_transform))

        is_constrained[:] = 0
        is_constrained[0] = 1

        _, center_y, _ = motion_schedule(max(info.frame() - 1, 0))
        aim_transform[0] = cube_transform(center_y)

    scene.animator().insert(sheet_object, animate_sheet)
    scene.animator().insert(cube_object, animate_cube)

    world.init(scene)

    return {
        "engine": engine,
        "world": world,
        "scene": scene,
        "scene_io": SceneIO(scene),
        "sheet_slot": sheet_slot,
        "cube_slot": cube_slot,
        "sheet_rest_positions": sheet_rest_positions,
        "moving_corner": corner_d,
        "target_corner": corner_c,
        "bending_model": bending_model,
        "bending_constitution_name": bending_constitution_name,
        "bending_yield_label": bending_yield_label,
        "bending_yield_value": bending_yield_value,
        "bending_hardening_value": bending_hardening_value,
    }


def run_demo(bending_model: str = "stress"):
    state = build_demo(bending_model)
    world = state["world"]
    scene_io = state["scene_io"]
    sheet_slot = state["sheet_slot"]
    sheet_rest_positions = state["sheet_rest_positions"]
    moving_corner = state["moving_corner"]
    target_corner = state["target_corner"]
    bending_constitution_name = state["bending_constitution_name"]
    bending_yield_label = state["bending_yield_label"]
    bending_yield_value = state["bending_yield_value"]
    bending_hardening_value = state["bending_hardening_value"]

    ps.init()
    ps.set_ground_plane_mode("none")

    surface = scene_io.simplicial_surface()
    mesh = ps.register_surface_mesh(
        "abd_cube_shell_plastic_crease",
        surface.positions().view().reshape(-1, 3),
        surface.triangles().topo().view().reshape(-1, 3),
    )
    mesh.set_edge_width(1.0)

    ui_state = {"run": True}

    def update_visual_mesh():
        merged = scene_io.simplicial_surface()
        mesh.update_vertex_positions(merged.positions().view().reshape(-1, 3))

    def sheet_metrics() -> tuple[float, float, float]:
        current_positions = np.array(view(sheet_slot.geometry().positions()), copy=True).reshape(-1, 3)
        displacement = current_positions - sheet_rest_positions
        max_disp = float(np.linalg.norm(displacement, axis=1).max())
        crease_depth = float(np.max(sheet_rest_positions[:, 1] - current_positions[:, 1]))
        corner_gap = float(np.linalg.norm(current_positions[moving_corner] - current_positions[target_corner]))
        return max_disp, crease_depth, corner_gap

    def advance_one_frame():
        if world.frame() >= TOTAL_FRAMES:
            ui_state["run"] = False
            return

        world.advance()
        if not world.is_valid():
            ui_state["run"] = False
            return

        world.retrieve()
        update_visual_mesh()

        if world.frame() >= TOTAL_FRAMES:
            ui_state["run"] = False

    def on_update():
        if psim.Button("run / pause"):
            ui_state["run"] = not ui_state["run"]

        psim.SameLine()
        if psim.Button("step"):
            advance_one_frame()

        if ui_state["run"]:
            advance_one_frame()

        frame = min(world.frame(), TOTAL_FRAMES)
        arc_alpha, cube_y, phase = motion_schedule(frame)
        max_disp, crease_depth, corner_gap = sheet_metrics()

        psim.Separator()
        psim.Text("Fix one corner, arc the opposite corner, then press with a flat ABD cube")
        psim.Text(f"Frame: {world.frame()} / {TOTAL_FRAMES}")
        psim.Text(f"Phase: {phase}")
        psim.Text(f"Arc progress: {arc_alpha:.3f}")
        psim.Text(f"Cube target Y: {cube_y:+.3f}")
        psim.Text(f"Ground Y: {GROUND_Y:+.3f}")
        psim.Text(f"Bending model: {bending_constitution_name}")
        psim.Text(f"{bending_yield_label.capitalize()}: {bending_yield_value:.3g}")
        psim.Text(f"Hardening modulus: {bending_hardening_value:.3g}")
        psim.Text(f"Max sheet displacement: {max_disp:.4f}")
        psim.Text(f"Residual crease depth: {crease_depth:.4f}")
        psim.Text(f"Diagonal corner gap: {corner_gap:.4f}")

        if world.frame() >= TOTAL_FRAMES:
            psim.Text("Press cycle finished. Rotate the view to inspect the remaining fold.")

    ps.set_user_callback(on_update)
    ps.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ABD cube -> shell plastic crease demo.")
    parser.add_argument(
        "--bending-model",
        choices=("strain", "stress"),
        default="stress",
        dest="bending_model",
        help="Plastic bending model to use. Default: 'stress'.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(bending_model=args.bending_model)
