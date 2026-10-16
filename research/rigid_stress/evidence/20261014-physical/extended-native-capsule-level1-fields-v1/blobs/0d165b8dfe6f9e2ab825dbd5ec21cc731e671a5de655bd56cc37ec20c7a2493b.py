"""Qualify centered point/edge/face barrier derivatives and the original unmollified friction magnitude."""

import argparse
import json
from pathlib import Path

import ipctk
import numpy as np


def scalar_force(distance_squared, dhat, stiffness):
    ratio = distance_squared / dhat**2
    if ratio >= 1:
        return 0.0
    derivative = -2 * (ratio - 1) * np.log(ratio) - (ratio - 1) ** 2 / ratio
    return -2 * np.sqrt(distance_squared) / dhat * stiffness * derivative


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    dhat, stiffness, gap, length = 2.5e-6, 1e6, 1e-6, 1e-3
    barrier = ipctk.BarrierPotential(dhat, stiffness, use_physical_barrier=True)
    empty = np.empty((0, 3), dtype=int)
    fixtures = [
        (
            "vertex_vertex",
            ipctk.VertexVertexNormalCollision(0, 1),
            np.array([[0, 0, 0], [0, 0, gap]]),
            np.empty((0, 2), dtype=int),
            empty,
        ),
        (
            "edge_vertex",
            ipctk.EdgeVertexNormalCollision(0, 2),
            np.array([[-length, 0, 0], [length, 0, 0], [0, 0, gap]]),
            np.array([[0, 1]]),
            empty,
        ),
        (
            "face_vertex",
            ipctk.FaceVertexNormalCollision(0, 3),
            np.array([[-length, -length, 0], [length, -length, 0], [0, length, 0], [0, 0, gap]]),
            np.array([[0, 1], [1, 2], [0, 2]]),
            np.array([[0, 1, 2]]),
        ),
        (
            "edge_edge",
            ipctk.EdgeEdgeNormalCollision(0, 1, 1e-17),
            np.array([[-length, 0, 0], [length, 0, 0], [-length, -0.5e-6, gap], [length, 0.5e-6, gap]]),
            np.array([[0, 1], [2, 3]]),
            empty,
        ),
    ]
    rows = []
    for name, collision, vertices, edges, faces in fixtures:
        collision.weight = 1.25
        assert collision.dmin == 0
        local = collision.dof(vertices, edges, faces)
        ids = np.array(collision.vertex_ids(edges, faces))
        ids = ids[ids >= 0]
        centered = (local.reshape(-1, 3) - local[:3]).reshape(-1)
        distance = collision.compute_distance(centered)
        np.testing.assert_allclose(distance, gap**2, rtol=1e-11, atol=0)
        normal_force = barrier.force_magnitude(distance, collision.dmin)
        analytic_force = scalar_force(gap**2, dhat, stiffness)
        np.testing.assert_allclose(normal_force, analytic_force, rtol=1e-12, atol=1e-8)
        gradient = barrier.gradient(collision, centered)
        np.testing.assert_allclose(gradient.reshape(-1, 3).sum(axis=0), 0, rtol=0, atol=1e-7)
        errors = []
        for index in range(3):
            direction = np.random.default_rng(20261015 + index).normal(size=len(centered))
            direction /= np.linalg.norm(direction)
            derivative = gradient @ direction
            for step in [1e-10, 3e-11]:
                measured = (
                    barrier(collision, centered + step * direction) - barrier(collision, centered - step * direction)
                ) / (2 * step)
                error = abs(measured - derivative) / max(abs(derivative), np.linalg.norm(gradient) * 1e-12)
                assert error <= 1e-6
                errors.append({"direction": index, "step_m": step, "relative_derivative_difference": float(error)})
        body = ids < (1 if name == "vertex_vertex" else 3 if name == "face_vertex" else 2)
        resultant = np.linalg.norm(gradient.reshape(-1, 3)[body].sum(axis=0)) / abs(collision.weight)
        mollifier_ratio = resultant / normal_force
        if name == "edge_edge":
            assert 0 < mollifier_ratio < 0.99
            tangential = ipctk.EdgeEdgeTangentialCollision(collision, centered, barrier)
            np.testing.assert_allclose(tangential.normal_force_magnitude, analytic_force, rtol=1e-12, atol=1e-8)
            assert abs(tangential.normal_force_magnitude - resultant) > 0.1 * analytic_force
            del tangential
        else:
            np.testing.assert_allclose(resultant, analytic_force, rtol=1e-12, atol=1e-8)
        np.savez_compressed(
            args.output / f"{name}.npz",
            vertices=vertices,
            edges=edges,
            faces=faces,
            local_vertex_ids=ids,
            centered_positions_m=centered,
            gradient_N=gradient,
        )
        rows.append(
            {
                "fixture": name,
                "normal_collision_type": type(collision).__name__,
                "squared_distance_m2": float(distance),
                "original_unmollified_normal_force_magnitude": float(normal_force),
                "independent_scalar_force_magnitude": float(analytic_force),
                "weighted_normal_resultant_over_weight": float(resultant),
                "normal_resultant_over_original_friction_magnitude": float(mollifier_ratio),
                "directional_derivatives": errors,
            }
        )
    # Release collision references before their geometry owners at interpreter shutdown
    fixtures.clear()
    del collision
    result = {
        "scope": __doc__,
        "qualified": True,
        "rows": rows,
        "goal_completion": False,
        "limitations": "Four small deterministic stencils verify barrier force derivatives, independent clamped-log scalar magnitude and the toolkit's unmollified edge-edge friction convention. These qualify reference force reconstruction, not a complete collision algorithm, physical contact convergence, an entire dynamic history, model/feedback accuracy or warmed cost.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "qualified_stencil_types": len(rows),
                "edge_edge_mollifier_ratio": rows[-1]["normal_resultant_over_original_friction_magnitude"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
