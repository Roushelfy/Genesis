"""Independent FP64 oracle on changing native Panda contacts, friction and partial resets."""

import argparse
import json
from pathlib import Path

import numpy as np
import quadrants as qd
from scipy.sparse.linalg import splu
from scipy.spatial.transform import Rotation

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.utils.misc import qd_to_numpy
from research.rigid_stress.mechanics import P2Shell, SurfaceGeometry
from research.rigid_stress.peak_cpu import P2Peak
from research.rigid_stress.wrench import FinitePatchMapper, WrenchPatch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=510000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gs.init(backend=gs.gpu, precision="64", seed=args.seed, logging_level="warning")
    workload = FrankaEgg(args.envs, varied=True, seed=args.seed)
    entry = workload.scene.rigid_solver.stress_recovery.links[0]
    with np.load("examples/rigid/assets/hollow_egg/level1/elastic.npz") as asset:
        oracle = P2Shell(
            asset["vertices"],
            asset["tetrahedra"],
            asset["surface_triangles"],
            entry.model.options.young,
            entry.model.options.poisson,
            entry.model.options.density,
            2,
            factor_backend="none",
        )
    mapper = FinitePatchMapper(SurfaceGeometry(oracle, 10), anchor_to_surface=True, adaptive_integration=True)
    free = np.setdiff1d(np.arange(oracle.ndof), qd_to_numpy(entry.model.info.pins))
    factor = splu(oracle.k[free][:, free].tocsc())
    peak_scan = P2Peak(
        oracle.glambda, oracle.elements, entry.model.options.young / (2 * (1 + entry.model.options.poisson))
    )
    rows = []
    coefficient_range = [float("inf"), -float("inf")]
    for tick in range(args.steps):
        workload.step()
        workload.scene.rigid_solver.check_errno()
        if tick % 50 != 49:
            continue
        arrays = {
            name: qd_to_numpy(value, transpose=True)
            for name, value in (
                ("position", entry.contacts.position),
                ("force", entry.contacts.force),
                ("normal", entry.contacts.normal),
                ("friction", entry.contacts.friction),
                ("radius", entry.contacts.radius),
                ("valid", entry.contacts.valid),
                ("nodal_force", entry.state.force),
                ("displacement", entry.state.displacement),
                ("source_indices", workload.scene.rigid_solver.collider.collider_state.contact_sort_idx),
                ("source_position", workload.scene.rigid_solver.collider.collider_state.contact_data.pos),
                ("source_force", workload.scene.rigid_solver.collider.collider_state.contact_data.force),
                ("source_normal", workload.scene.rigid_solver.collider.collider_state.contact_data.normal),
                ("source_a", workload.scene.rigid_solver.collider.collider_state.contact_data.link_a),
            )
        }
        omega = qd_to_numpy(entry.omega)
        native_peak = qd_to_numpy(entry.state.peak)
        frame_position = qd_to_numpy(entry.contacts.frame_position)
        rotations = Rotation.from_quat(qd_to_numpy(entry.contacts.frame_quaternion)[:, [1, 2, 3, 0]])
        # Rotate the sampled environments over the trajectory rather than always selecting the same rows.
        selected = (np.arange(min(4, args.envs)) * 7 + tick // 50) % args.envs
        for env in selected:
            expected_force = np.zeros((len(oracle.xyz), 3))
            contact_rows = []
            for contact in np.flatnonzero(arrays["valid"][env]):
                position, force, normal = (arrays[name][env, contact] for name in ("position", "force", "normal"))
                mu = arrays["friction"][env, contact]
                source = arrays["source_indices"][env, contact]
                sign = -1 if arrays["source_a"][env, source] == workload.link.idx else 1
                inverse = rotations[env].inv()
                np.testing.assert_allclose(
                    position, inverse.apply(arrays["source_position"][env, source] - frame_position[env]), atol=1e-15
                )
                np.testing.assert_allclose(force, inverse.apply(sign * arrays["source_force"][env, source]), atol=1e-13)
                np.testing.assert_allclose(
                    normal, inverse.apply(-sign * arrays["source_normal"][env, source]), atol=1e-14
                )
                if np.linalg.norm(force) == 0:
                    continue
                coefficient_range = [min(coefficient_range[0], mu), max(coefficient_range[1], mu)]
                mapping = mapper.map(
                    WrenchPatch(position, force, arrays["radius"][env, contact], mu, np.zeros(3), inward_normal=normal)
                )
                expected_force += mapping.nodal_force_n.reshape((-1, 3))
                contact_rows.append(
                    {
                        "mu": float(mu),
                        "force_N": float(np.linalg.norm(force)),
                        "wrench_force_error_N": float(mapping.diagnostics.force_error_n),
                        "wrench_moment_error_Nm": float(mapping.diagnostics.moment_error_nm),
                        "cone_excess_N": float(mapping.diagnostics.maximum_cone_excess_n),
                    }
                )
            np.testing.assert_allclose(arrays["nodal_force"][env], expected_force, rtol=2e-8, atol=1e-12)
            centrifugal = -np.cross(omega[env], np.cross(omega[env], oracle.xyz - oracle.com))
            raw = expected_force.ravel() + oracle.m @ centrifugal.ravel()
            rhs = raw - oracle.mr @ np.linalg.solve(oracle.gram, oracle.r.T @ raw)
            displacement = np.zeros(oracle.ndof)
            displacement[free] = factor.solve(rhs[free])
            recovered = arrays["displacement"][env].ravel()
            residual = np.linalg.norm(oracle.k @ recovered - rhs)
            budget = max(entry.model.options.absolute_tolerance, entry.model.options.tolerance * np.linalg.norm(rhs))
            assert residual <= budget, (tick, env, residual, budget)
            peak = peak_scan(displacement)[0]
            np.testing.assert_allclose(native_peak[env], peak, rtol=1e-4, atol=1e-3)
            rows.append(
                {
                    "tick": tick,
                    "env": int(env),
                    "contacts": contact_rows,
                    "full_gauge_residual_N": float(residual),
                    "residual_budget_N": float(budget),
                    "global_peak_error_Pa": float(abs(native_peak[env] - peak)),
                    "global_peak_reference_Pa": float(peak),
                }
            )
        print("Oracle accepted tick", tick, "samples", len(rows), flush=True)
    qd.sync()
    result = {
        "envs": args.envs,
        "steps": args.steps,
        "seed": args.seed,
        "resets": workload.reset_count,
        "actual_contact_mu_range": coefficient_range,
        "load_model": "finite_pad_adaptive_q10",
        "oracle_rows": rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Accepted", len(rows), "independent same-mesh FP64 samples.")


if __name__ == "__main__":
    main()
