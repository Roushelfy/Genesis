"""Bind actual candidate stress frames to their complete own mechanical histories."""


def admit_candidates(data_root, read, verify, check):
    candidates, failures = {}, []
    for method, group in (("dpr", 2), ("scr", 3), ("shell", 0)):
        relative = (
            f"native-pad-full-source-stability-v1/group{group}/native"
            if method != "shell"
            else "compliant-pad-friction250-history-v1/group0/fine"
        )
        run = data_root / relative
        start, result = read(run / "start.json"), read(run / "result.json")
        spec = start["args"]
        check(
            method + "/declared_physical_case",
            spec["loads"] == [0.1, 0.1]
            and spec["dt"] == 0.001
            and spec["friction"] == 0.3
            and spec["gravity_x"] == 1
            and spec["record_stride"] == 1,
        )
        rows = read(run / "steps.jsonl", journal=True)
        check(
            method + "/complete400_step_journal",
            len(rows) == 400 and [r["step"] for r in rows] == list(range(1, 401)) and result["completed"],
        )
        if method != "shell":
            check(
                method + "/fixed_full_source_configuration",
                spec["method"] == method
                and spec["radius"] == 0.002
                and spec["source_reduction"] == "none"
                and spec["integrator"] == "Euler",
            )
            check(
                method + "/own_history_and_both_windows",
                result["mechanical_history_passed"]
                and all(result["history_checks"].values())
                and len(result["windows"]) == 2
                and all(
                    w["samples"] >= 51 and w["mechanical_window_passed"] and all(w["checks"].values())
                    for w in result["windows"]
                ),
            )
            check(
                method + "/all_original_equations_and_contact_laws",
                all(
                    r["native_prediction_velocity_error_m_per_s"] <= 1e-9
                    and max(
                        r["normal_law_relative_error"],
                        r["friction_law_relative_error"],
                        r["equation_residual_relative"],
                    )
                    <= 1e-7
                    and max(r["angular_equation_residual_nm"], r["frozen_point_moment_defect_nm"])
                    <= 1e-7 * max(r["command_force_n"], 1e-3) * 0.03
                    for r in rows
                ),
            )
            frames = []
            for level in range(2):
                endpoint = next(r for r in reversed(rows) if r["level"] == level)
                stress = endpoint["stress"]
                check(
                    f"{method}/{level}/own_requested_native_admission",
                    stress["native_valid"]
                    and stress["errno"] == 0
                    and all(status == 0 for status in stress["contact_status"])
                    and stress["source_reduction"] == "none",
                )
                field = stress["field"]
                check(
                    f"{method}/{level}/correct_own_frame",
                    field["path"] == str(run / "frames" / f"stress-level{level}.npz"),
                )
                verify(field)
                frames.append(field)
            verify(start["mesh"])
            mesh = start["mesh"]
            # These are preserved failures, never rows with an invented error value.
            failed_run = data_root / f"pad-primary-friction-cost-v1/group{group - 2}/{method}"
            failed = read(failed_run / "result.json")
            check(method + "/every_step_failure_preserved", not failed["completed"])
            failures.append(
                {
                    "method": method,
                    "run": str(failed_run),
                    "completed": failed["completed"],
                    "returned_steps": failed["steps"],
                    "numerical_error_rank_assigned": False,
                }
            )
        else:
            audit = read(run.parent / "audit/result.json")
            selected = [r for r in audit["runs"] if r["result"]["path"] == str(run / "result.json")]
            check("shell/own_audit_binds_correct_run", len(selected) == 1)
            selected = selected[0]
            verify(selected["result"])
            check(
                "shell/both_admitted_windows",
                selected["completed"]
                and len(selected["equilibrium_windows"]) == 2
                and all(
                    w["samples"] >= 51 and w["mechanical_window_passed"] and all(w["checks"].values())
                    for w in selected["equilibrium_windows"]
                ),
            )
            check(
                "shell/all_original_equations_and_contact_laws",
                all(
                    not any(r["solver_failure_flags"])
                    and r["equation_residual_relative"] <= 1e-3
                    and r["physical_pad"]["normal_law_and_transfer_passed"]
                    and r["physical_pad"]["friction"]["lagged_law_passed"]
                    for r in rows
                ),
            )
            frames = []
            for level in range(2):
                expected = str(run / "frames" / f"level{level}.npz")
                chosen = [r["field"] for r in selected["fields"] if r["field"]["path"] == expected]
                check(f"shell/{level}/own_audited_frame", len(chosen) == 1)
                verify(chosen[0])
                endpoint = next(r for r in reversed(rows) if r["level"] == level)
                check(f"shell/{level}/own_stress_valid", endpoint["stress"]["valid"])
                frames.append(chosen[0])
            # This is the matched250um material mesh used by the admitted Shell.
            mesh_path = data_root / "finite-area-isolated-mesh-v1/group0/meshes/land-contact-1.npz"
            mesh = start["mesh"]
            check("shell/declared250um_mesh", mesh["path"] == str(mesh_path))
            verify(mesh)
        candidates[method] = {
            "run": str(run),
            "mesh": mesh,
            "frames": frames,
            "mass_kg": start["mass_kg"],
            "configuration": spec,
            "model_variant": start.get("variant", method),
            "source": start.get("solver_source", start.get("native_source")),
            "observation_scope": "admitted requested endpoint fields; complete own mechanical history",
        }
    return candidates, failures
