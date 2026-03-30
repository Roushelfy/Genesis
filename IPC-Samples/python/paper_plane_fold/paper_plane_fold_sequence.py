"""
Asset-aware paper-plane fold sequence demo.

This script keeps the fold-step / dump / auto-resume framework while adapting
to boundary-to-boundary creases from
``paper_plane_fold.paper_plane_2_boundary_crease_debug_demo``.

Built-in sequences (--sequence):
- 'short' (default): steps 1, 2, c01, c03, 7, 8
- 'full': all 8 boundary steps

Default: paper_plane_2_coarse.obj, --sequence short, --bending-model strain, --from-start

Run:
    python python/examples/paper_plane_fold/paper_plane_fold_sequence.py
    python python/examples/paper_plane_fold/paper_plane_fold_sequence.py --sequence full --no-from-start
    python python/examples/paper_plane_fold/paper_plane_fold_sequence.py --bending-model stress
    python python/examples/paper_plane_fold/paper_plane_fold_sequence.py assets/sim_data/trimesh/paper_plane_2_fine.obj
    python python/examples/paper_plane_fold/paper_plane_fold_sequence.py --start-after-step 2
    python python/examples/paper_plane_fold/paper_plane_fold_sequence.py --overhead-big-cube
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

try:
    from uipc import Logger, Matrix4x4, Engine, World, Scene, Animation, view, builtin
except ImportError as exc:
    raise SystemExit(
        "This example requires the libuipc Python bindings (`uipc._native.pyuipc`). "
        "Build/install the Python package before running it."
    ) from exc

_FOLD_DIR = os.path.dirname(__file__)
if _FOLD_DIR not in sys.path:
    sys.path.insert(0, _FOLD_DIR)

from paper_plane_helpers import (  # noqa: E402
    # Constants
    MESH_PARTITION_SIZE,
    SHELL_THICKNESS,
    SHELL_DENSITY,
    SHELL_YOUNG,
    SHELL_POISSON,
    SHELL_BENDING_STIFFNESS,
    SHELL_STRAIN_YIELD_THRESHOLD,
    SHELL_STRAIN_HARDENING_MODULUS,
    SHELL_STRESS_YIELD_STRESS,
    SHELL_STRESS_HARDENING_MODULUS,
    STRONG_SPC_STRENGTH,
    WEAK_SPC_STRENGTH,
    GROUND_Y,
    PLANE_LIFT_Y,
    DT,
    CONTACT_D_HAT,
    NEWTON_TOL_RATE,
    STRONG_FIX_CURRENT_Y,
    CUBE_SCALE,
    CUBE_HOVER_Y,
    CUBE_PRESS_Y,
    CUBE_STC_STRENGTH,
    OVERHEAD_CUBE_SCALE,
    OVERHEAD_CUBE_HOVER_Y,
    OVERHEAD_CUBE_PRESS_Y,
    FOLD_MOVE_FRAMES,
    FOLD_HOLD_FRAMES,
    INSPECT_FRAMES,
    DEFAULT_ASSET_FILENAME,
    # Dataclasses
    StepSpec,
    PressSpec,
    GlobalFlipSpec,
    FoldStep,
    AssetContext,
    NamedVertex,
    MoverAction,
    PressAction,
    MotionState,
    GlobalFlipAction,
    # Functions used in build_demo / run_demo
    load_asset_context,
    validate_asset_context,
    demo_workspace,
    compute_cube_setup,
    compile_step,
    total_sequence_frames,
    completed_step_end_frames,
    resume_manifest_template,
    maybe_recover_world,
    maybe_dump_completed_step,
    sequence_schedule,
    motion_schedule,
    mover_position,
    global_flip_position,
    recover_global_flip_start_positions,
    cube_transform,
    process_closed_surface,
    orient_mesh_for_ground,
    step_duration,
    next_step_index_for_frame,
    build_curve_overlay,
    build_pair_overlay,
    fill_cloud_positions,
    frame_obj_directory,
    write_obj_frame,
    crease_vertex_ids,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Asset-aware paper-plane fold sequence demo.")
    parser.add_argument("asset_filename", nargs="?", default=None)
    parser.add_argument(
        "--from-start",
        action="store_true",
        default=True,
        dest="from_start",
        help="Start from scratch (default). Use --no-from-start to enable auto-resume.",
    )
    parser.add_argument("--no-from-start", action="store_false", dest="from_start")
    parser.add_argument(
        "--start-after-step",
        type=int,
        default=None,
        dest="start_after_step",
        help="Resume from the dump after the given 1-based step index. Overrides auto-resume.",
    )
    parser.add_argument(
        "--overhead-big-cube",
        action="store_true",
        dest="overhead_big_cube",
        help="Use a large overhead cube (edge length 2) that only presses vertically.",
    )
    parser.add_argument(
        "--export-obj-every-frame",
        action="store_true",
        dest="export_obj_every_frame",
        help="Export the current paper mesh as one OBJ per frame under the demo workspace.",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default="short",
        dest="sequence",
        help="Named fold sequence to use. Options: 'full' (all 8 steps), 'short' (steps 1,2,c01,c03,7,8).",
    )
    parser.add_argument(
        "--bending-model",
        choices=("strain", "stress"),
        default="stress",
        dest="bending_model",
        help="Plastic bending model to use. Default: 'stress'.",
    )
    return parser.parse_args(argv)


def build_step_specs(context: AssetContext, sequence: str = "full") -> tuple[StepSpec, ...]:
    boundary_step6_crease = (
        "C04_UPPER_SHORT"
        if "paper_plane_5" in Path(context.asset_filename).stem.lower()
        else "C01_C04_X_TO_C03_C04_X"
    )
    step1 = StepSpec(
        name="step1_lt_dp2_fold",
        crease_name="LT_DP2",
        mover_labels=("LEFT_TOP",),
        strong_fix_labels=("LEFT_TOP", "RIGHT_BOTTOM"),
        current_pose_fix_labels=(),
        strong_fix_crease_names=(),
        strong_fix_segment_labels=(),
        press=PressSpec(),
    )
    step2 = StepSpec(
        name="step2_tr_dn1_fold",
        crease_name="TR_DN1",
        mover_labels=("RIGHT_TOP",),
        strong_fix_labels=("RIGHT_TOP", "LEFT_BOTTOM"),
        current_pose_fix_labels=(),
        strong_fix_crease_names=(),
        strong_fix_segment_labels=(),
        press=PressSpec("stamped"),
    )
    step3 = StepSpec(
        name="step3_lr_h3_fold",
        crease_name="LR_H3",
        mover_labels=("P08A", "P08B", "P12A", "P08A_TO_LR_H3_MID","RIGHT_TOP", "LEFT_TOP",),
        strong_fix_labels=("P07B", "P07A", "LEFT_BOTTOM", "RIGHT_BOTTOM"),
        current_pose_fix_labels=(),
        strong_fix_crease_names=(),
        strong_fix_segment_labels=(),
        press=PressSpec(),
    )
    step4 = StepSpec(
        name="step4_p09_half_fold",
        crease_name="P09_C02_TO_B",
        mover_labels=("P06B", "P06B_TO_P09_C02_TO_B_MID"),
        strong_fix_labels=(
            "LEFT_BOTTOM",
            "RIGHT_BOTTOM",
            "P09_C02_X_TO_P08_P13_X_MID",
            "P06A",
            "P08_P13_X"
        ),
        # strong_fix_crease_names=("P09_C02_TO_B",),
        # strong_fix_segment_labels=(("P09_C02_X", "P08_P13_X"),),
        strong_fix_segment_labels=(),
        press=PressSpec(),
    )
    step5 = StepSpec(
        name="step5_p13_half_fold",
        crease_name="P13_A_TO_C02",
        mover_labels=("P06A", "P06A_TO_P13_A_TO_C02_MID"),
        strong_fix_labels=(
            "LEFT_BOTTOM",
            "RIGHT_BOTTOM",
            "P09_P12_X",
            "P13_C02_X_TO_P09_P12_X_MID",
            "P06B",
        ),
        strong_fix_segment_labels=(),
        strong_fix_crease_names=("P13_A_TO_C02",),
        press=PressSpec(),
    )
    step6 = StepSpec(
        name="step6_c04_upper_fold",
        crease_name=boundary_step6_crease,
        mover_labels=("P08A",),
        strong_fix_labels=("P06A", "P06B", "C04_C02_X"),
        strong_fix_segment_labels=(),
        press=PressSpec("stamped"),
    )
    step7 = StepSpec(
        name="step7_bottom_edge_fold",
        crease_name="BOTTOM_EDGE",
        mover_labels=("P13_C02_X_TO_P09_P12_X_MID","P08A","P06A", "P06A_TO_P13_A_TO_C02_MID","P06B","P08A", "P08B", "P12A", "P08A_TO_LR_H3_MID","LEFT_TOP","RIGHT_TOP","P13A","P13B","P14B",),
        strong_fix_labels=(),
        strong_fix_segment_labels=(),
        press=None,
    )
    step8 = StepSpec(
        name="step8_Mid_edge_fold",
        crease_name="BT_V2",
        mover_labels=("LEFT_BOTTOM","P13A","P06A",),
        strong_fix_labels=("P12B",),
        strong_fix_crease_names=(),
        strong_fix_segment_labels=(),
        press=PressSpec("stamped"),
    )
    step_c01_c04_to_p01a = StepSpec(
        name="step_c01_c04_to_p01a_fold",
        crease_name="C01_C04_X_TO_P01A",
        mover_labels=("P05A", "P06A", "P07A", "P13A", "LEFT_BOTTOM"),
        strong_fix_labels=("P04A",),
        strong_fix_segment_labels=(),
        press=PressSpec("stamped"),
    )
    step_c03_c04_to_p03a = StepSpec(
        name="step_c03_c04_to_p03a_fold",
        crease_name="C03_C04_X_TO_P03A",
        mover_labels=("P05B", "P06B", "P07B", "P13B", "RIGHT_BOTTOM"),
        strong_fix_labels=("P04B","LEFT_BOTTOM",),
        strong_fix_segment_labels=(),
        press=PressSpec("stamped"),
    )
    step_c03_c04_to_p03a_no_stamp = StepSpec(
        name="step_c03_c04_to_p03a_fold",
        crease_name="C03_C04_X_TO_P03A",
        mover_labels=("P05B", "P06B", "P07B", "P13B", "RIGHT_BOTTOM"),
        strong_fix_labels=("P04B","LEFT_BOTTOM","P05A",),
        strong_fix_segment_labels=(),
        press=None,
    )
    all_steps = (step1, step2, step3, step4, step5, step6, step7, step8)
    if sequence == "full":
        return all_steps
    if sequence == "short":
        return (step1, step2,  step7, step8,step_c01_c04_to_p01a,step7, step_c03_c04_to_p03a,step_c03_c04_to_p03a_no_stamp,)
    raise SystemExit(f"Unknown sequence '{sequence}'. Available: 'full', 'short'.")


def build_demo(
    asset_filename: str | None = None,
    *,
    from_start: bool = False,
    start_after_step: int | None = None,
    overhead_big_cube: bool = False,
    sequence: str = "full",
    bending_model: str = "strain",
) -> dict[str, object]:
    try:
        from uipc import SceneIO
        from uipc.geometry import (
            SimplicialComplexIO,
            ground,
            mesh_partition,
            label_surface,
            label_triangle_orient,
            flip_inward_triangles,
        )
        from uipc.constitution import (
            AffineBodyConstitution,
            NeoHookeanShell,
            StrainPlasticDiscreteShellBending,
            StressPlasticDiscreteShellBending,
            SoftTransformConstraint,
            SoftPositionConstraint,
            ElasticModuli2D,
        )
    except ImportError as exc:
        raise SystemExit(
            "This example requires newer libuipc Python bindings with "
            "`NeoHookeanShell`, `StrainPlasticDiscreteShellBending`, "
            "`StressPlasticDiscreteShellBending`, `SoftTransformConstraint`, "
            "`SoftPositionConstraint`, `ElasticModuli2D`, `SimplicialComplexIO`, "
            "and `mesh_partition`."
        ) from exc

    Logger.set_level(Logger.Level.Warn)

    if bending_model not in {"strain", "stress"}:
        raise SystemExit(f"Unknown bending model '{bending_model}'. Available: 'strain', 'stress'.")

    context = load_asset_context(asset_filename)
    validate_asset_context(context)
    workspace = demo_workspace(context.asset_filename, sequence, bending_model)

    engine = Engine("cuda", workspace)
    world = World(engine)

    config = Scene.default_config()
    config["dt"] = DT
    config["gravity"] = [[0.0], [0.0], [0.0]]
    config["contact"]["enable"] = True
    config["contact"]["d_hat"] = CONTACT_D_HAT
    config["contact"]["friction"]["enable"] = False
    config["contact"]["constitution"] = "ipc"

    config["linear_system"]["tol_rate"] = NEWTON_TOL_RATE
    scene = Scene(config)

    scene.contact_tabular().default_model(0.0, 1.0e9)
    default_contact = scene.contact_tabular().default_element()

    plane_object = scene.objects().create("paper_plane")
    cube_object = scene.objects().create("press_cube")
    ground_object = scene.objects().create("ground")

    abd = AffineBodyConstitution()
    shell = NeoHookeanShell()
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

    plane = context.plane
    diagnostics = context.diagnostics
    base_named_vertices = context.named_vertices
    cube_scale, home_xyz, cube_hover_y, cube_press_y, cube_vertical_only = compute_cube_setup(
        diagnostics,
        overhead_big_cube,
    )
    step_specs = build_step_specs(context, sequence)

    mesh_partition(plane, MESH_PARTITION_SIZE)
    moduli = ElasticModuli2D.youngs_poisson(SHELL_YOUNG, SHELL_POISSON)
    shell.apply_to(plane, moduli, SHELL_DENSITY, SHELL_THICKNESS)
    if bending_model == "strain":
        plastic_bending.apply_to(
            plane,
            SHELL_BENDING_STIFFNESS,
            SHELL_STRAIN_YIELD_THRESHOLD,
            SHELL_STRAIN_HARDENING_MODULUS,
        )
    else:
        plastic_bending.apply_to(
            plane,
            SHELL_BENDING_STIFFNESS,
            SHELL_STRESS_YIELD_STRESS,
            SHELL_STRESS_HARDENING_MODULUS,
        )
    spc.apply_to(plane, STRONG_SPC_STRENGTH)
    default_contact.apply_to(plane)

    rest_positions = np.array(view(plane.positions()), copy=True).reshape(-1, 3)
    schedule_steps = tuple(
        compile_step(
            step_spec,
            rest_positions,
            base_named_vertices,
            diagnostics,
            home_xyz,
            cube_hover_y,
            cube_press_y,
            cube_vertical_only,
        )
        for step_spec in step_specs
    )
    total_frames = total_sequence_frames(schedule_steps)
    step_end_frames = completed_step_end_frames(schedule_steps)
    manifest_template = resume_manifest_template(
        context.asset_filename,
        step_specs,
        schedule_steps,
        total_frames,
        sequence,
        bending_model,
    )

    plane_slot = plane_object.geometries().create(plane)[0]
    ground_object.geometries().create(ground(GROUND_Y))

    cube_slot = None
    if any(step.press is not None for step in schedule_steps):
        pre = Matrix4x4.Identity()
        pre[0, 0] = cube_scale
        pre[1, 1] = cube_scale
        pre[2, 2] = cube_scale
        io = SimplicialComplexIO(pre)
        cube = io.read(context.cube_mesh_path)
        cube = process_closed_surface(cube, label_surface, label_triangle_orient, flip_inward_triangles)
        abd.apply_to(cube, 2.0e7)
        stc.apply_to(cube, CUBE_STC_STRENGTH)
        default_contact.apply_to(cube)
        first_press = next(step.press for step in schedule_steps if step.press is not None)
        view(cube.transforms())[0] = cube_transform(home_xyz, first_press.cube_yaw_radians)
        cube_slot = cube_object.geometries().create(cube)[0]

    runtime: dict[str, object] = {
        "step_index": None,
        "resolved_step": None,
        "motion": None,
        "dumped_step_indices": set(),
        "current_pose_fix_bindings": {},
        "global_step_start_positions": {},
        "resolved_step_cache": {},
    }

    def resolve_runtime_step(step_index: int, positions: np.ndarray) -> FoldStep:
        cache = runtime["resolved_step_cache"]
        if step_index not in cache:
            step_spec = step_specs[step_index]
            compile_positions = positions
            if step_spec.global_flip is not None:
                group_name = step_spec.global_flip.group_name or step_spec.name
                global_step_start_positions = runtime["global_step_start_positions"]
                if group_name not in global_step_start_positions:
                    provisional_step = compile_step(
                        step_spec,
                        positions,
                        base_named_vertices,
                        diagnostics,
                        home_xyz,
                        cube_hover_y,
                        cube_press_y,
                        cube_vertical_only,
                    )
                    if provisional_step.global_flip is None:
                        raise AssertionError(f"{step_spec.name} failed to resolve global flip state")
                    global_step_start_positions[group_name] = recover_global_flip_start_positions(
                        positions,
                        provisional_step.global_flip,
                    )
                compile_positions = global_step_start_positions[group_name]
            cache[step_index] = compile_step(
                step_spec,
                compile_positions,
                base_named_vertices,
                diagnostics,
                home_xyz,
                cube_hover_y,
                cube_press_y,
                cube_vertical_only,
            )
        return cache[step_index]

    def step_state_for_frame(frame: int, positions: np.ndarray | None = None) -> tuple[FoldStep, MotionState]:
        step_index, local_frame = sequence_schedule(frame, schedule_steps)
        if positions is None:
            positions = np.array(view(plane_slot.geometry().positions()), copy=True).reshape(-1, 3)
        resolved_step = resolve_runtime_step(step_index, positions)
        motion = motion_schedule(local_frame, schedule_steps, step_index, resolved_step.press, home_xyz)
        runtime["step_index"] = step_index
        runtime["resolved_step"] = resolved_step
        runtime["motion"] = motion
        return resolved_step, motion

    def animate_plane(info: Animation.UpdateInfo):
        geo = info.geo_slots()[0].geometry()
        is_constrained = view(geo.vertices().find(builtin.is_constrained))
        aim_position = view(geo.vertices().find(builtin.aim_position))
        strength_ratio = view(geo.vertices().find("strength_ratio"))
        current_xyz = np.array(view(geo.positions()), copy=True).reshape(-1, 3)
        frame = max(info.frame() - 1, 0)
        resolved_step, motion = step_state_for_frame(frame, current_xyz)
        step_index = int(runtime["step_index"])
        current_pose_fix_bindings = runtime["current_pose_fix_bindings"]
        global_step_start_positions = runtime["global_step_start_positions"]
        if resolved_step.free_run_frames > 0:
            is_constrained[:] = 0
            strength_ratio[:] = 0.0
            return
        if resolved_step.global_flip is not None:
            group_name = resolved_step.global_flip.group_name
            if group_name not in global_step_start_positions:
                global_step_start_positions[group_name] = recover_global_flip_start_positions(
                    current_xyz,
                    resolved_step.global_flip,
                )
            start_positions = global_step_start_positions[group_name]
            is_constrained[:] = 0
            strength_ratio[:] = 0.0
            for vertex_id in resolved_step.global_flip.mover_vertex_ids:
                target_xyz = global_flip_position(start_positions[vertex_id], resolved_step.global_flip, motion)
                is_constrained[vertex_id] = 1
                strength_ratio[vertex_id] = STRONG_SPC_STRENGTH
                aim_position[vertex_id] = target_xyz.reshape(3, 1)
            return
        if step_index not in current_pose_fix_bindings:
            bindings: dict[int, np.ndarray] = {}
            for action in (*resolved_step.weak_anchors, *resolved_step.strong_fixes):
                if not action.lock_current_pose:
                    continue
                target_xyz = np.array(current_xyz[action.vertex_id], copy=True)
                if action.current_pose_y is not None:
                    target_xyz[1] = action.current_pose_y
                bindings[action.vertex_id] = target_xyz
            current_pose_fix_bindings[step_index] = bindings
        step_bindings = current_pose_fix_bindings[step_index]
        is_constrained[:] = 0
        strength_ratio[:] = 0.0
        for anchor in resolved_step.weak_anchors:
            is_constrained[anchor.vertex_id] = 1
            strength_ratio[anchor.vertex_id] = WEAK_SPC_STRENGTH
            target_xyz = step_bindings.get(anchor.vertex_id, anchor.xyz)
            aim_position[anchor.vertex_id] = target_xyz.reshape(3, 1)
        for fixed in resolved_step.strong_fixes:
            is_constrained[fixed.vertex_id] = 1
            strength_ratio[fixed.vertex_id] = STRONG_SPC_STRENGTH
            target_xyz = step_bindings.get(fixed.vertex_id, fixed.xyz)
            aim_position[fixed.vertex_id] = target_xyz.reshape(3, 1)
        for mover in resolved_step.movers:
            is_constrained[mover.vertex_id] = 1
            strength_ratio[mover.vertex_id] = STRONG_SPC_STRENGTH
            aim_position[mover.vertex_id] = mover_position(mover, motion.fold_alpha).reshape(3, 1)

    def animate_cube(info: Animation.UpdateInfo):
        geo = info.geo_slots()[0].geometry()
        is_constrained = view(geo.instances().find(builtin.is_constrained))
        aim_transform = view(geo.instances().find(builtin.aim_transform))
        is_constrained[:] = 0
        is_constrained[0] = 1
        frame = max(info.frame() - 1, 0)
        current_xyz = np.array(view(plane_slot.geometry().positions()), copy=True).reshape(-1, 3)
        resolved_step, motion = step_state_for_frame(frame, current_xyz)
        cube_yaw = resolved_step.press.cube_yaw_radians if resolved_step.press is not None else 0.0
        aim_transform[0] = cube_transform(motion.cube_center_xyz, cube_yaw)

    scene.animator().insert(plane_object, animate_plane)
    if cube_slot is not None:
        scene.animator().insert(cube_object, animate_cube)

    world.init(scene)
    resume_info = maybe_recover_world(
        world,
        workspace,
        manifest_template,
        schedule_steps,
        from_start,
        start_after_step,
    )
    completed_step_index = resume_info["last_completed_step_index"]
    if completed_step_index is not None:
        runtime["dumped_step_indices"] = set(range(int(completed_step_index) + 1))

    max_crease_nodes = max(len(step.crease_vertex_ids) for step in schedule_steps)
    max_movers = max(len(step.movers) for step in schedule_steps)
    max_targets = max(len(step.targets) for step in schedule_steps)
    max_weak_anchors = max(len(step.weak_anchors) for step in schedule_steps)
    max_strong_fixes = max(len(step.strong_fixes) for step in schedule_steps)
    max_press_path_points = max(max(len(step.press.path_labels), 2) if step.press is not None else 2 for step in schedule_steps)

    return {
        "engine": engine,
        "scene": scene,
        "world": world,
        "scene_io": SceneIO(scene),
        "plane_slot": plane_slot,
        "cube_slot": cube_slot,
        "diagnostics": diagnostics,
        "context": context,
        "base_named_vertices": base_named_vertices,
        "step_specs": step_specs,
        "schedule_steps": schedule_steps,
        "runtime": runtime,
        "total_frames": total_frames,
        "workspace": workspace,
        "bending_model": bending_model,
        "bending_constitution_name": bending_constitution_name,
        "bending_yield_label": bending_yield_label,
        "bending_yield_value": bending_yield_value,
        "bending_hardening_value": bending_hardening_value,
        "step_end_frames": step_end_frames,
        "resume_info": resume_info,
        "manifest_template": manifest_template,
        "cube_scale": cube_scale,
        "cube_vertical_only": cube_vertical_only,
        "cube_home_xyz": np.array(home_xyz, copy=True),
        "cube_hover_y": cube_hover_y,
        "cube_press_y": cube_press_y,
        "step_state_for_frame": step_state_for_frame,
        "max_crease_nodes": max_crease_nodes,
        "max_movers": max_movers,
        "max_targets": max_targets,
        "max_weak_anchors": max_weak_anchors,
        "max_strong_fixes": max_strong_fixes,
        "max_press_path_points": max_press_path_points,
    }


def run_demo(
    asset_filename: str | None = None,
    *,
    from_start: bool = False,
    start_after_step: int | None = None,
    overhead_big_cube: bool = False,
    export_obj_every_frame: bool = False,
    sequence: str = "full",
    bending_model: str = "strain",
):
    try:
        import polyscope as ps
        import polyscope.imgui as psim
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "This example requires `polyscope`. Install it with `pip install polyscope`."
        ) from exc

    state = build_demo(
        asset_filename,
        from_start=from_start,
        start_after_step=start_after_step,
        overhead_big_cube=overhead_big_cube,
        sequence=sequence,
        bending_model=bending_model,
    )
    world = state["world"]
    scene_io = state["scene_io"]
    plane_slot = state["plane_slot"]
    cube_slot = state["cube_slot"]
    diagnostics = state["diagnostics"]
    context = state["context"]
    base_named_vertices = state["base_named_vertices"]
    step_specs = state["step_specs"]
    schedule_steps = state["schedule_steps"]
    runtime = state["runtime"]
    total_frames = state["total_frames"]
    workspace = state["workspace"]
    bending_model = state["bending_model"]
    bending_constitution_name = state["bending_constitution_name"]
    bending_yield_label = state["bending_yield_label"]
    bending_yield_value = state["bending_yield_value"]
    bending_hardening_value = state["bending_hardening_value"]
    step_end_frames = state["step_end_frames"]
    resume_info = state["resume_info"]
    manifest_template = state["manifest_template"]
    cube_scale = state["cube_scale"]
    cube_vertical_only = state["cube_vertical_only"]
    cube_home_xyz = state["cube_home_xyz"]
    cube_hover_y = state["cube_hover_y"]
    cube_press_y = state["cube_press_y"]
    max_crease_nodes = state["max_crease_nodes"]
    max_movers = state["max_movers"]
    max_targets = state["max_targets"]
    max_weak_anchors = state["max_weak_anchors"]
    max_strong_fixes = state["max_strong_fixes"]
    max_press_path_points = state["max_press_path_points"]
    obj_frame_dir = frame_obj_directory(workspace) if export_obj_every_frame else None

    def visible_surface():
        if cube_vertical_only:
            return plane_slot.geometry()
        return scene_io.simplicial_surface()

    def export_current_frame_obj(frame_index: int) -> None:
        if obj_frame_dir is None:
            return
        plane_geo = plane_slot.geometry()
        vertices = np.array(view(plane_geo.positions()), copy=True).reshape(-1, 3)
        triangles = np.array(view(plane_geo.triangles().topo()), copy=True).reshape(-1, 3)
        extra_meshes = []
        if cube_slot is not None:
            cube_geo = cube_slot.geometry()
            cube_local = np.array(view(cube_geo.positions()), copy=True).reshape(-1, 3)
            cube_tris = np.array(view(cube_geo.triangles().topo()), copy=True).reshape(-1, 3)
            xform = np.array(view(cube_geo.transforms())[0], copy=True)
            rot = xform[:3, :3]
            trans = xform[:3, 3]
            cube_world = (cube_local @ rot.T) + trans.reshape(1, 3)
            extra_meshes.append((cube_world, cube_tris))
        output_path = os.path.join(obj_frame_dir, f"frame_{frame_index:06d}.obj")
        write_obj_frame(output_path, vertices, triangles, extra_meshes)

    ps.init()
    ps.set_ground_plane_mode("none")

    surface = visible_surface()
    mesh = ps.register_surface_mesh(
        "paper_plane_fold_sequence",
        surface.positions().view().reshape(-1, 3),
        surface.triangles().topo().view().reshape(-1, 3),
    )
    mesh.set_edge_width(1.0)

    crease_overlay = build_curve_overlay(max_crease_nodes)
    crease_curve = ps.register_curve_network(
        "active_crease",
        crease_overlay["positions"],
        crease_overlay["edges"],
    )
    crease_curve.set_radius(0.0035, relative=False)
    crease_curve.set_color((0.92, 0.55, 0.12))

    pair_overlay = build_pair_overlay(max_movers)
    pair_curve = ps.register_curve_network(
        "active_pairs",
        pair_overlay["positions"],
        pair_overlay["edges"],
    )
    pair_curve.set_radius(0.0025, relative=False)
    pair_curve.set_color((0.15, 0.55, 0.95))

    press_curve = ps.register_curve_network(
        "press_path",
        np.zeros((max_press_path_points, 3), dtype=np.float64),
        np.asarray([[i, i + 1] for i in range(max_press_path_points - 1)], dtype=np.int32),
    )
    press_curve.set_radius(0.0025, relative=False)
    press_curve.set_color((0.85, 0.85, 0.20))

    weak_anchor_cloud = ps.register_point_cloud(
        "weak_crease_anchors",
        np.zeros((max_weak_anchors, 3), dtype=np.float64),
    )
    weak_anchor_cloud.set_color((0.90, 0.35, 0.15))

    strong_fix_cloud = ps.register_point_cloud("strong_fixes", np.zeros((max_strong_fixes, 3), dtype=np.float64))
    strong_fix_cloud.set_color((0.10, 0.65, 0.25))

    mover_cloud = ps.register_point_cloud("movers", np.zeros((max_movers, 3), dtype=np.float64))
    mover_cloud.set_color((0.15, 0.45, 0.92))

    target_cloud = ps.register_point_cloud("symmetry_targets", np.zeros((max_targets, 3), dtype=np.float64))
    target_cloud.set_color((0.12, 0.78, 0.78))

    axis_cloud = ps.register_point_cloud("axis_origins", np.zeros((max_movers, 3), dtype=np.float64))
    axis_cloud.set_color((0.55, 0.20, 0.72))

    ui_state = {"run": True}

    step_state_for_frame = state["step_state_for_frame"]

    def current_positions() -> np.ndarray:
        return np.array(view(plane_slot.geometry().positions()), copy=True).reshape(-1, 3)


    def update_visuals():
        surface = visible_surface()
        mesh.update_vertex_positions(surface.positions().view().reshape(-1, 3))

        frame = min(world.frame(), total_frames)
        current = current_positions()
        resolved_step, _ = step_state_for_frame(frame, current)

        crease_nodes = np.zeros((max_crease_nodes, 3), dtype=np.float64)
        active_crease = current[np.array(resolved_step.crease_vertex_ids, dtype=np.int32)]
        crease_nodes[: len(active_crease)] = active_crease
        for index in range(len(active_crease), max_crease_nodes):
            crease_nodes[index] = active_crease[-1]
        crease_curve.update_node_positions(crease_nodes)

        mover_positions = [current[mover.vertex_id] for mover in resolved_step.movers]
        weak_anchor_positions = [current[anchor.vertex_id] for anchor in resolved_step.weak_anchors]
        strong_fix_positions = [current[fixed.vertex_id] for fixed in resolved_step.strong_fixes]
        target_positions = [target.xyz for target in resolved_step.targets]
        axis_positions = [mover.center_xyz for mover in resolved_step.movers]

        mover_cloud.update_point_positions(fill_cloud_positions(mover_positions, max_movers))
        weak_anchor_cloud.update_point_positions(fill_cloud_positions(weak_anchor_positions, max_weak_anchors))
        strong_fix_cloud.update_point_positions(fill_cloud_positions(strong_fix_positions, max_strong_fixes))
        target_cloud.update_point_positions(fill_cloud_positions(target_positions, max_targets))
        axis_cloud.update_point_positions(fill_cloud_positions(axis_positions, max_movers))

        pair_nodes = np.zeros((2 * max_movers, 3), dtype=np.float64)
        filler = mover_positions[0] if mover_positions else np.zeros(3, dtype=np.float64)
        pair_nodes[:] = filler
        for pair_index, mover in enumerate(resolved_step.movers):
            pair_nodes[2 * pair_index] = current[mover.vertex_id]
            pair_nodes[2 * pair_index + 1] = mover.target_xyz
        pair_curve.update_node_positions(pair_nodes)

        press_nodes = np.zeros((max_press_path_points, 3), dtype=np.float64)
        if resolved_step.press is not None:
            path_xyzs = list(resolved_step.press.path_xyzs)
            press_nodes[: len(path_xyzs)] = np.asarray(path_xyzs, dtype=np.float64)
            for index in range(len(path_xyzs), max_press_path_points):
                press_nodes[index] = path_xyzs[-1]
        press_curve.update_node_positions(press_nodes)

    def advance_one_frame():
        if world.frame() >= total_frames:
            ui_state["run"] = False
            return
        world.advance()
        if not world.is_valid():
            ui_state["run"] = False
            return
        world.retrieve()
        maybe_dump_completed_step(
            world,
            world.frame(),
            workspace,
            step_specs,
            step_end_frames,
            manifest_template,
            runtime,
            resume_info,
            schedule_steps,
        )
        update_visuals()
        export_current_frame_obj(world.frame())
        if world.frame() >= total_frames:
            ui_state["run"] = False

    def on_update():
        if psim.Button("run / pause"):
            ui_state["run"] = not ui_state["run"]
        psim.SameLine()
        if psim.Button("step"):
            advance_one_frame()
        if ui_state["run"]:
            advance_one_frame()

        frame = min(world.frame(), total_frames)
        step_index, local_frame = sequence_schedule(frame, schedule_steps)
        resolved_step, motion = step_state_for_frame(frame)

        psim.Separator()
        psim.Text("Paper-plane fold sequence")
        psim.Text(f"Frame: {world.frame()} / {total_frames}")
        psim.Text(f"Asset: {context.asset_filename}")
        psim.Text(f"Asset mode: boundary")
        psim.Text(f"Bending model: {bending_model} ({bending_constitution_name})")
        psim.Text(f"Bending stiffness: {SHELL_BENDING_STIFFNESS:.4e}")
        psim.Text(f"Bending {bending_yield_label}: {bending_yield_value:.4e}")
        psim.Text(f"Bending hardening: {bending_hardening_value:.4e}")
        psim.Text(f"Workspace: {workspace}")
        psim.Text(f"Resume mode: {resume_info['mode']}")
        if resume_info["mode"] == "recovered":
            psim.Text(f"Recovered frame: {resume_info['recovered_frame']}")
        else:
            psim.Text(f"Resume note: {resume_info['skip_reason']}")
        psim.Text(f"Last completed step: {resume_info['last_completed_step_name'] or '-'}")
        psim.Text(f"Next step to run: {resume_info['next_step_name'] or 'inspect'}")
        psim.Text(f"Active step: {motion.step_name}")
        psim.Text(f"Active crease: {resolved_step.crease_name}")
        psim.Text(f"Phase: {motion.phase}")
        psim.Text(f"Fold alpha: {motion.fold_alpha:.3f}")
        psim.Text(f"Contact d_hat: {CONTACT_D_HAT:.4e}")
        psim.Text(f"Ground Y: {GROUND_Y:+.4f}")
        psim.Text(f"Cube mode: {'overhead-big' if cube_vertical_only else 'path-follow'}")
        psim.Text(f"Cube scale: {cube_scale:.3f}")
        psim.Text(
            f"Cube home: ({cube_home_xyz[0]:+.4f}, {cube_home_xyz[1]:+.4f}, {cube_home_xyz[2]:+.4f})"
        )
        if resolved_step.global_flip is not None:
            psim.Text(
                f"Global flip: axis={resolved_step.global_flip.axis_name}, "
                f"lift={resolved_step.global_flip.lift_y:.3f}, "
                f"angle={math.degrees(resolved_step.global_flip.angle_radians):.1f}"
            )
        psim.Text(f"Mover labels: {', '.join(step_specs[step_index].mover_labels)}")
        psim.Text(f"Strong-fix labels: {', '.join(step_specs[step_index].strong_fix_labels)}")

        if resolved_step.press is not None:
            press = resolved_step.press
            psim.Text(f"Press path: {' -> '.join(press.path_labels)}")
            psim.Text(
                f"Cube center: ({motion.cube_center_xyz[0]:+.4f}, "
                f"{motion.cube_center_xyz[1]:+.4f}, {motion.cube_center_xyz[2]:+.4f})"
            )

        psim.Separator()
        psim.Text("Weak crease anchors")
        step_bindings = runtime["current_pose_fix_bindings"].get(step_index, {})
        for anchor in resolved_step.weak_anchors:
            xyz = step_bindings.get(anchor.vertex_id, anchor.xyz)
            x, y, z = xyz
            suffix = " [current]" if anchor.lock_current_pose else ""
            psim.Text(f"{anchor.label:<14s} v{anchor.vertex_id:<5d} ({x:+.4f}, {y:+.4f}, {z:+.4f}){suffix}")

        psim.Separator()
        psim.Text("Strong fixed vertices")
        for fixed in resolved_step.strong_fixes:
            xyz = step_bindings.get(fixed.vertex_id, fixed.xyz)
            x, y, z = xyz
            suffix = " [current]" if fixed.lock_current_pose else ""
            psim.Text(f"{fixed.label:<14s} v{fixed.vertex_id:<5d} ({x:+.4f}, {y:+.4f}, {z:+.4f}){suffix}")

        psim.Separator()
        psim.Text("Movers")
        for mover in resolved_step.movers:
            sx, sy, sz = mover.start_xyz
            tx, ty, tz = mover.target_xyz
            psim.Text(
                f"{mover.label:<12s} v{mover.vertex_id:<5d} -> "
                f"{mover.target_label:<16s} v{mover.target_vertex_id:<5d}"
            )
            psim.Text(
                f"start=({sx:+.4f}, {sy:+.4f}, {sz:+.4f})  "
                f"target=({tx:+.4f}, {ty:+.4f}, {tz:+.4f})"
            )

        if world.frame() >= total_frames:
            psim.Separator()
            psim.Text("Sequence finished. Inspect the residual folds and press marks.")

    update_visuals()
    export_current_frame_obj(world.frame())
    ps.set_user_callback(on_update)
    ps.show()


if __name__ == "__main__":
    args = parse_args()
    run_demo(
        args.asset_filename,
        from_start=args.from_start,
        start_after_step=args.start_after_step,
        overhead_big_cube=args.overhead_big_cube,
        export_obj_every_frame=args.export_obj_every_frame,
        sequence=args.sequence,
        bending_model=args.bending_model,
    )
