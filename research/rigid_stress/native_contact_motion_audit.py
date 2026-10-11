"""Untimed CPU diagnostics of the exact pre-integration rigid contact frame."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.utils.misc import qd_to_numpy, tensor_to_array


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=510000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gs.init(backend=gs.gpu, precision="64", seed=args.seed, logging_level="warning")
    workload = FrankaEgg(args.envs, varied=True, seed=args.seed)
    solver = workload.scene.rigid_solver
    recovery = solver.stress_recovery
    entry = recovery.links[0]
    original = recovery.recover
    fingers = [workload.robot.get_link(name).idx for name in ("left_finger", "right_finger")]
    initial_quaternion = tensor_to_array(workload.initial_quaternion)
    counts = np.zeros((args.envs, 12), dtype=np.int64)
    maxima = np.zeros((args.envs, 6))
    minimum_mu = np.full(args.envs, np.inf)
    previous_pairs = np.zeros((args.envs, 3), dtype=bool)
    captures = 0

    def observe(substep):
        nonlocal captures, previous_pairs
        original(substep)
        captures += 1
        # recover() runs after constraint force evaluation and before integration.
        # These velocities, COMs and world positions belong to the same frame.
        data = solver.collider.collider_state
        slots = qd_to_numpy(data.contact_sort_idx, transpose=True)
        sources = {
            name: qd_to_numpy(getattr(data.contact_data, name), transpose=True)
            for name in ("pos", "force", "normal", "link_a", "link_b")
        }
        valid = qd_to_numpy(entry.contacts.valid, transpose=True)
        mu = qd_to_numpy(entry.contacts.friction, transpose=True)
        states = {
            name: qd_to_numpy(getattr(solver.dyn_state.links, name), transpose=True)
            for name in ("cd_vel", "cd_ang", "root_COM")
        }
        quaternion = qd_to_numpy(entry.contacts.frame_quaternion)
        phase = tensor_to_array(workload.phase)
        current_pairs = np.zeros_like(previous_pairs)
        for env in range(args.envs):
            loaded = 0
            for contact in np.flatnonzero(valid[env]):
                source = slots[env, contact]
                a, b = (int(sources[name][env, source]) for name in ("link_a", "link_b"))
                assert a == workload.link.idx or b == workload.link.idx
                other = b if a == workload.link.idx else a
                sign = -1 if a == workload.link.idx else 1
                force = sign * sources["force"][env, source]
                magnitude = np.linalg.norm(force)
                if magnitude <= 1e-12:
                    continue
                loaded += 1
                normal = -sign * sources["normal"][env, source]
                np.testing.assert_allclose(np.linalg.norm(normal), 1, rtol=0, atol=1e-12)
                normal_force = force @ normal
                tangent_force = force - normal_force * normal
                cone_excess = np.linalg.norm(tangent_force) - mu[env, contact] * normal_force
                assert normal_force >= -1e-10 and cone_excess <= 1e-8, (env, normal_force, cone_excess)
                velocities = [
                    states["cd_vel"][env, link]
                    + np.cross(states["cd_ang"][env, link], sources["pos"][env, source] - states["root_COM"][env, link])
                    for link in (a, b)
                ]
                relative = velocities[0] - velocities[1]
                tangential_speed = np.linalg.norm(relative - (relative @ normal) * normal)
                group = fingers.index(other) if other in fingers else 2
                current_pairs[env, group] = True
                counts[env, group] += 1
                counts[env, 3] += int(np.linalg.norm(tangent_force) > 1e-8)
                counts[env, 4] += int(tangential_speed > 1e-5)
                counts[env, 5] += int(group < 2 and tangential_speed > 1e-5)
                counts[env, 6] += int(group < 2 and 0.7 <= phase[env] < 0.76 and tangential_speed > 1e-5)
                minimum_mu[env] = min(minimum_mu[env], mu[env, contact])
                maxima[env, 0] = max(maxima[env, 0], tangential_speed)
                maxima[env, 1] = max(maxima[env, 1], np.linalg.norm(tangent_force))
                maxima[env, 2] = max(maxima[env, 2], max(0, cone_excess))
                maxima[env, 3] = max(maxima[env, 3], mu[env, contact])
            counts[env, 7] += int(loaded == 0)
            counts[env, 8] += int(loaded > 0)
            counts[env, 9] += int(phase[env] >= 0.97 and not current_pairs[env, :2].any())
            counts[env, 10] += int(np.count_nonzero(current_pairs[env] & ~previous_pairs[env]))
            counts[env, 11] += int(np.count_nonzero(previous_pairs[env] & ~current_pairs[env]))
            maxima[env, 4] = max(maxima[env, 4], loaded)
            cosine = abs(quaternion[env] @ initial_quaternion[env])
            maxima[env, 5] = max(maxima[env, 5], 2 * np.arccos(np.clip(cosine, 0, 1)))
        previous_pairs = current_pairs

    recovery.recover = observe
    for tick in range(args.steps):
        workload.step()
        solver.check_errno()
        if tick % 100 == 99:
            print("Contact motion accepted tick", tick, flush=True)
    args.output.write_text(
        json.dumps(
            {
                "arguments": vars(args) | {"output": str(args.output)},
                "command": [sys.executable, *sys.argv],
                "source_revision": os.environ["RIGID_STRESS_SOURCE_REVISION"],
                "source_sha256": {
                    str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(
                        [
                            *Path("genesis/engine/solvers/rigid/stress").glob("*.py"),
                            Path("examples/rigid/franka_egg_stress.py"),
                            Path(__file__),
                        ]
                    )
                },
                "captures": captures,
                "reset_environments": workload.reset_count,
                "counter_columns": [
                    "left_loaded_contacts",
                    "right_loaded_contacts",
                    "support_loaded_contacts",
                    "tangential_force_contacts",
                    "sliding_contacts",
                    "sliding_finger_contacts",
                    "weak_grip_sliding_finger_contacts",
                    "no_load_steps",
                    "loaded_steps",
                    "release_without_finger_steps",
                    "pair_establishments",
                    "pair_disappearances",
                ],
                "per_environment_counts": counts.tolist(),
                "totals": counts.sum(axis=0).tolist(),
                "maximum_columns": [
                    "tangential_relative_speed_m_s",
                    "tangential_force_N",
                    "cone_excess_N",
                    "actual_mu",
                    "loaded_contact_count",
                    "rotation_from_initial_rad",
                ],
                "per_environment_maxima": maxima.tolist(),
                "global_maxima": maxima.max(axis=0).tolist(),
                "actual_mu_range": [float(minimum_mu.min()), float(maxima[:, 3].max())],
                "scope_note": "Offline, untimed every-environment diagnostics immediately after stress recovery and before rigid integration. Contact point velocity = cd_vel + cd_ang cross (world_point - root_COM), using the same source contact frame. Sliding threshold 1e-5 m/s is diagnostic, not a numerical acceptance change or grasp success criterion. Establishment/disappearance counts track loaded egg-other-link pairs in left/right/support groups, not unstable manifold slot identities. All environments and transient/release steps are included; no production numerical code is replaced.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
