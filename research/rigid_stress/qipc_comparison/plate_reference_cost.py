"""Measure original QIPC plate mechanics with sparse complete-field admission."""

from time import perf_counter

import torch

from .artifacts import write_json
from .plates import equilibrium
from .reference import solver_statistics


def measure(args, config, scene, plate, observer, owners, indices):
    assert observer is not None and args.cost_steps > 0 and args.cost_stride > 0
    rows = []
    for repeat in range(-1, args.cost_repeats):
        steps = args.cost_stride if repeat < 0 else args.cost_steps
        caps, newton_caps, observations, core_seconds, observer_seconds = 0, 0, [], 0.0, 0.0
        torch.cuda.synchronize()
        begun = perf_counter()
        for step in range(1, steps + 1):
            observe = step % args.cost_stride == 0 or step == steps
            if observe:
                with torch.cuda.stream(scene.solver.cuda_stream):
                    previous = scene.finite_element.velocities.clone()
            plate.control_dofs_force(args.loads[-1])
            scene.step()
            core_seconds += scene.solver.step_ms / 1000
            stats = solver_statistics(scene)
            caps += stats["pcg_max"] >= config.max_pcg
            newton_caps += stats["newton"] >= config.max_newton
            assert not stats["edge_triangle_intersection_count"] and stats["newton"] < config.max_newton
            if not observe:
                continue
            started = perf_counter()
            state = observer.read_step(previous)
            contact = state.contact
            active = owners[contact.global_vertex_ids]
            reactions = {name: contact.global_forces_n[active == index].sum(dim=0) for name, index in indices.items()}
            observed = {
                "upper_reaction_n": reactions["upper"][2].item(),
                "lower_reaction_n": -reactions["lower"][2].item(),
                "plate_speed_m_per_s": -scene.rigid_body.v[indices["upper"], 2].item(),
                "egg_inertia_resultant_norm_n": torch.linalg.norm(state.inertial_reaction_n.sum(dim=0)).item(),
                "egg_kinetic_j": state.kinetic_j.item(),
                "equation_residual_relative": torch.linalg.norm(state.residual_n).item()
                / max(torch.linalg.norm(contact.fem_forces_n).item(), 1e-12),
                "minimum_det_F": state.field.determinant.min().item(),
            }
            field, vm = state.field.cauchy_pa.cpu(), state.field.von_mises_pa.cpu()
            assert torch.isfinite(field).all() and torch.isfinite(vm).all() and observed["minimum_det_F"] > 0
            observed["checks"] = equilibrium(observed, args.loads[-1])
            observations.append({"step": step, **observed})
            observer_seconds += perf_counter() - started
        torch.cuda.synchronize()
        seconds = perf_counter() - begun
        rows.append(
            {
                "repeat": repeat,
                "warmup": repeat < 0,
                "steps": steps,
                "elapsed_seconds": seconds,
                "aggregate_env_steps_per_second": steps / seconds,
                "latency_ms_per_step": seconds * 1000 / steps,
                "synchronous_engine_seconds": core_seconds,
                "sparse_observation_admission_seconds": observer_seconds,
                "pcg_cap_steps": caps,
                "newton_cap_steps": newton_caps,
                "observations": observations,
                "all_observed_equilibria_passed": all(all(row["checks"].values()) for row in observations),
            }
        )
        write_json(
            args.output / "cost.json",
            {
                "envs": 1,
                "method": "qipc",
                "samples": rows,
                "scope": "Warmed continuously held discrete plate contact; sparse complete stress/VM host transfer and mechanical/equation admission, per-step original solver/ET checks included. No frame file I/O. Engine and observer times are measured separately; this does not qualify unresolved continuum contact fields. PCG caps and failed observations remain explicit, not accepted speedups.",
            },
        )
