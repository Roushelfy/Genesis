from __future__ import annotations

from polyscope import imgui

from wearing_urdf import UrdfForwardAdapter


class WearingGUIController:
    def __init__(self, runtime, urdf_adapter: UrdfForwardAdapter | None) -> None:
        self.runtime = runtime
        self.urdf_adapter = urdf_adapter
        self.joint_cache: dict[str, float] = {}
        if urdf_adapter is not None:
            for jn in urdf_adapter.joint_names():
                self.joint_cache[jn] = 0.0

        self.export_dir_path = str(runtime.output_dir)
        self.save_joint_path = str(runtime.output_dir / "joint_pose.json")
        self.load_joint_path = str(runtime.output_dir / "joint_pose.json")
        self.save_line_mesh_path = str(runtime.output_dir / "skeleton_line.obj")
        self.save_inflation_path = str(runtime.output_dir / "inflation_scaling.json")
        self.load_inflation_path = str(runtime.output_dir / "inflation_scaling.json")

    def _draw_sim_controls(self) -> None:
        if imgui.Button("Run / Stop"):
            self.runtime.toggle_run()
        imgui.SameLine()
        if imgui.Button("Step"):
            self.runtime.step_once(force_sanity_check=True)
        imgui.SameLine()
        if imgui.Button("Sanity Check"):
            self.runtime.run_sanity_check()

    def _draw_save_load_panel(self) -> None:
        if not imgui.CollapsingHeader("Save / Load"):
            return
        imgui.Text("Load Joint / Load Inflation: ok before first Step (scene-only); Save uses frame 0 until sim runs.")
        if imgui.Button("Export rest/init"):
            self.runtime.export_shapes(self.export_dir_path)
        imgui.SameLine()
        changed, new_val = imgui.InputText("Export Dir Path##export_dir", self.export_dir_path)
        if changed:
            self.export_dir_path = new_val

        if imgui.Button("Save Joint"):
            self.runtime.save_joint_json(self.save_joint_path)
        imgui.SameLine()
        changed, new_val = imgui.InputText("Save Joint Path##save_joint_path", self.save_joint_path)
        if changed:
            self.save_joint_path = new_val

        if imgui.Button("Load Joint"):
            loaded_state = self.runtime.load_joint_json(self.load_joint_path)
            if loaded_state is not None:
                for joint_name in self.joint_cache:
                    if joint_name in loaded_state:
                        self.joint_cache[joint_name] = float(loaded_state[joint_name])
        imgui.SameLine()
        changed, new_val = imgui.InputText("Load Joint Path##load_joint_path", self.load_joint_path)
        if changed:
            self.load_joint_path = new_val

        if imgui.Button("Save LineMesh OBJ"):
            self.runtime.save_line_mesh_obj(self.save_line_mesh_path)
        imgui.SameLine()
        changed, new_val = imgui.InputText("Save LineMesh Path##save_line_mesh_path", self.save_line_mesh_path)
        if changed:
            self.save_line_mesh_path = new_val

        if imgui.Button("Save Inflation"):
            self.runtime.save_inflation_json(self.save_inflation_path)
        imgui.SameLine()
        changed, new_val = imgui.InputText("Save Inflation Path##save_inflation_path", self.save_inflation_path)
        if changed:
            self.save_inflation_path = new_val

        if imgui.Button("Load Inflation"):
            self.runtime.load_inflation_json(self.load_inflation_path)
        imgui.SameLine()
        changed, new_val = imgui.InputText("Load Inflation Path##load_inflation_path", self.load_inflation_path)
        if changed:
            self.load_inflation_path = new_val

    def _draw_inflation_panel(self) -> None:
        if not imgui.CollapsingHeader("Inflation Controls"):
            return
        changed_step_global, step_global = imgui.InputFloat(
            "Global Scale Delta/Frame", self.runtime.max_global_scale_delta_per_frame
        )
        if changed_step_global:
            self.runtime.max_global_scale_delta_per_frame = float(max(step_global, 1e-6))

        changed_step_bone, step_bone = imgui.InputFloat(
            "Bone Scale Delta/Frame", self.runtime.max_bone_scale_delta_per_frame
        )
        if changed_step_bone:
            self.runtime.max_bone_scale_delta_per_frame = float(max(step_bone, 1e-6))

        changed_slider, global_scale_slider = imgui.SliderFloat(
            "Global Inflation Slider", self.runtime.target_global_scale, 0.0, 20.0
        )
        if changed_slider:
            self.runtime.set_global_inflation_scale(global_scale_slider)
        changed_input, global_scale_input = imgui.InputFloat("Global Inflation Value", self.runtime.target_global_scale)
        if changed_input:
            self.runtime.set_global_inflation_scale(global_scale_input)
        imgui.Text(f"Global current={self.runtime.global_scale:.4f}, target={self.runtime.target_global_scale:.4f}")

        for i in range(len(self.runtime.bone_scales)):
            bone_name = self.runtime.bone_display_names[i] if i < len(self.runtime.bone_display_names) else f"bone_{i}"
            changed_slider, local_scale_slider = imgui.SliderFloat(
                f"{bone_name} Scale Slider##bone_slider_{i}", self.runtime.target_bone_scales[i], 0.0, 20.0
            )
            if changed_slider:
                self.runtime.set_bone_inflation_scale(i, local_scale_slider)
            changed_input, local_scale_input = imgui.InputFloat(
                f"{bone_name} Scale Value##bone_value_{i}", self.runtime.target_bone_scales[i]
            )
            if changed_input:
                self.runtime.set_bone_inflation_scale(i, local_scale_input)

        if not imgui.CollapsingHeader("Capsule Length Controls"):
            return
        for i in range(len(self.runtime.bone_scales)):
            bone_name = self.runtime.bone_display_names[i] if i < len(self.runtime.bone_display_names) else f"bone_{i}"
            changed_start, start_offset = imgui.SliderFloat(
                f"{bone_name} Length Start Offset##len_start_{i}",
                self.runtime.bone_length_start_offsets[i],
                -1.0,
                1.0,
            )
            changed_end, end_offset = imgui.SliderFloat(
                f"{bone_name} Length End Offset##len_end_{i}",
                self.runtime.bone_length_end_offsets[i],
                -1.0,
                1.0,
            )
            if changed_start or changed_end:
                self.runtime.set_bone_length_offsets(i, start_offset, end_offset)

    def _draw_urdf_panel(self) -> None:
        if self.urdf_adapter is None:
            return
        if not imgui.CollapsingHeader("URDF Forward Control"):
            return
        for joint_name in self.joint_cache:
            changed, angle = imgui.SliderFloat(f"{joint_name}", self.joint_cache[joint_name], -1.57, 1.57)
            if changed:
                self.joint_cache[joint_name] = angle
                self.runtime.set_joint_angle(joint_name, angle)

    def on_update(self) -> None:
        self._draw_sim_controls()
        self._draw_save_load_panel()
        self._draw_inflation_panel()
        self._draw_urdf_panel()
        imgui.Text(self.runtime.latest_message)
        if self.runtime.run_simulation:
            self.runtime.step_once()
