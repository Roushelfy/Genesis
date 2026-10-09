"""Inspect rejected rigid contacts without changing their force, moment, friction or footprint."""

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .cpu import EggConfig, EggRecoveryCPU, reference
from .wrench import FinitePatchMapper, WrenchFit, WrenchPatch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--quadratures", type=int, nargs="+", default=[10, 16, 24])
    parser.add_argument("--normal-only", action="store_true")
    parser.add_argument("--footprint-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snapshot = np.load(args.snapshot)
    rows = []
    if args.normal_only:
        normal = snapshot["inward_normal"].astype(float)
        normal /= np.maximum(np.linalg.norm(normal, axis=2)[..., None], 1e-30)
        force = snapshot["force_n"].astype(float)
        compression = np.einsum("bci,bci->bc", force, normal)
        tangent = np.linalg.norm(force - compression[..., None] * normal, axis=2)
        magnitude = np.linalg.norm(force, axis=2)
        excess = tangent - snapshot["friction"] * compression
        allowance = 16 * np.finfo(np.float32).eps * magnitude
        for i_env, i_contact in np.argwhere(snapshot["valid"] & (excess > allowance)):
            rows.append(
                {
                    "step": int(snapshot["step"]),
                    "environment": int(i_env),
                    "contact": int(i_contact),
                    "force_n": force[i_env, i_contact].tolist(),
                    "inward_normal": normal[i_env, i_contact].tolist(),
                    "friction": float(snapshot["friction"][i_env, i_contact]),
                    "compression_n": float(compression[i_env, i_contact]),
                    "tangent_n": float(tangent[i_env, i_contact]),
                    "cone_excess_n": float(excess[i_env, i_contact]),
                    "roundoff_allowance_n": float(allowance[i_env, i_contact]),
                }
            )
        report = {"snapshot": str(args.snapshot), "force_cone_violations": rows}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return
    with threadpool_limits(limits=1):
        model = EggRecoveryCPU(config=EggConfig(level=args.level), history=0, direct=True)
        for quadrature in args.quadratures:
            mapper = FinitePatchMapper(reference.SurfaceGeometry(model.fem, quadrature))
            for i_env, i_contact in np.argwhere(snapshot["valid"]):
                force = snapshot["force_n"][i_env, i_contact]
                center = snapshot["position_m"][i_env, i_contact]
                radius = snapshot["radius_m"][i_env, i_contact]
                if args.footprint_only:
                    ids = mapper.tree.query_ball_point(center, radius)
                    distance, _ = mapper.tree.query(center)
                    rows.append(
                        {
                            "step": int(snapshot["step"]),
                            "environment": int(i_env),
                            "contact": int(i_contact),
                            "quadrature": quadrature,
                            "samples": len(ids),
                            "nearest_sample_distance_m": float(distance),
                            "center_m": center.tolist(),
                            "radius_m": float(radius),
                            "force_magnitude_n": float(np.linalg.norm(force)),
                        }
                    )
                    continue
                if np.linalg.norm(force) < 1e-12:
                    continue
                friction = snapshot["friction"][i_env, i_contact]
                quat = snapshot["quat"][i_env].astype(float)
                quat /= np.linalg.norm(quat)
                vector = snapshot["contact_normal_world"][i_env, i_contact].astype(float)
                vector = vector - 2 * np.cross(quat[1:], quat[0] * vector - np.cross(quat[1:], vector))
                vector *= 1 if vector @ force >= 0 else -1
                normal_force = float(vector @ force)
                ids = np.array(mapper.tree.query_ball_point(center, radius, return_sorted=True))
                distance = np.sum((mapper.positions[ids] - center) ** 2, axis=1) / radius**2
                weights = np.exp(-0.5 * distance / 0.45**2) * np.maximum(0, 1 - distance) ** 2 * mapper.weights[ids]
                patch = WrenchPatch(center, force, radius, friction, np.zeros(3))
                for normal_model in ("shell_faces", "rigid_pad_frame"):
                    normals = mapper.normals[ids] if normal_model == "shell_faces" else np.tile(vector, (len(ids), 1))
                    row = {
                        "step": int(snapshot["step"]),
                        "environment": int(i_env),
                        "contact": int(i_contact),
                        "normal_model": normal_model,
                        "quadrature": quadrature,
                        "samples": len(ids),
                        "radius_m": float(radius),
                        "force_n": force.tolist(),
                        "center_m": center.tolist(),
                        "friction": float(friction),
                        "rigid_normal_force_n": normal_force,
                        "rigid_tangent_ratio": float(np.linalg.norm(force - normal_force * vector) / normal_force),
                        "shell_normal_angle_max_deg": float(
                            np.rad2deg(np.arccos(np.clip(mapper.normals[ids] @ vector, -1, 1))).max()
                        ),
                    }
                    try:
                        forces, diagnostics = WrenchFit(mapper.positions[ids], weights, normals, patch).solve()
                        compression = np.einsum("qi,qi->q", forces, mapper.normals[ids])
                        tangent = forces - compression[:, None] * mapper.normals[ids]
                        row.update(
                            accepted=True,
                            force_error_n=diagnostics.force_error_n,
                            moment_error_nm=diagnostics.moment_error_nm,
                            constrained_evaluations=diagnostics.constrained_evaluations,
                            shell_compression_min_n=float(compression.min()),
                            shell_cone_excess_max_n=float(
                                (np.linalg.norm(tangent, axis=1) - friction * compression).max()
                            ),
                        )
                    except ValueError as error:
                        row.update(accepted=False, error=str(error))
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"snapshot": str(args.snapshot), "level": args.level, "cases": rows}, indent=2))


if __name__ == "__main__":
    main()
