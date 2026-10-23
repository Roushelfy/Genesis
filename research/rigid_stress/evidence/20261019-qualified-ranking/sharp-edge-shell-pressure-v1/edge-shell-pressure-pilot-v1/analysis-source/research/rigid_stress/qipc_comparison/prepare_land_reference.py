"""Build a declared smooth polar-land egg as an additional finite-area plate benchmark."""

import argparse
from pathlib import Path

import numpy as np

from research.rigid_stress.validation_meshes import transform_shell

from .artifacts import file_id, runtime_identity, write_json
from .prepare_contact_reference import refine_contact_surface
from .prepare_inputs import make_mesh, mass_properties


def land_surface(unit, flat_radius, blend_radius):
    outer = transform_shell("egg", unit)
    normal = (outer - transform_shell("egg", 0.92 * unit)) / 0.0005
    radius = np.linalg.norm(outer[:, :2], axis=1)
    cap = radius < blend_radius
    pole = np.sign(unit[cap, 2])
    r = radius[cap]
    t = np.clip((r - flat_radius) / (blend_radius - flat_radius), 0, 1)
    blend = t**3 * (10 + t * (-15 + 6 * t))
    derivative = 30 * t**2 * (1 - t) ** 2 / (blend_radius - flat_radius)
    original_height = outer[cap, 2].copy()
    # The original implicit normal gives dz/dr without a singular pole formula.
    original_slope = -np.einsum("vi,vi->v", normal[cap, :2], outer[cap, :2]) / (
        np.maximum(r, np.finfo(float).tiny) * normal[cap, 2]
    )
    slope = blend * original_slope + derivative * (original_height - pole * 0.03)
    outer[cap, 2] = pole * 0.03 + blend * (original_height - pole * 0.03)
    cap_normal = np.column_stack(
        (-slope[:, None] * outer[cap, :2] / np.maximum(r[:, None], np.finfo(float).tiny), np.ones(len(r)))
    )
    normal[cap] = cap_normal * pole[:, None] / np.linalg.norm(cap_normal, axis=1)[:, None]
    return outer, outer - 0.0005 * normal


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--flat-radius", type=float, default=0.002)
    cli.add_argument("--blend-radius", type=float, default=0.004)
    cli.add_argument("--cases", nargs="+", default=["0.0005:0.002:4", "0.00025:0.0015:6", "0.000125:0.001:8"])
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    assert 0 < args.flat_radius < args.blend_radius < 0.01
    args.output.mkdir(parents=True, exist_ok=False)
    # Check analytic normals against independent centered differences of the surface.
    theta = np.r_[np.linspace(0.001, 0.30, 1001), np.linspace(np.pi - 0.30, np.pi - 0.001, 1001)]
    step = 1e-7
    meridian = np.column_stack((np.sin(theta), np.zeros(len(theta)), np.cos(theta)))
    outer, inner = land_surface(meridian, args.flat_radius, args.blend_radius)
    before = np.column_stack((np.sin(theta - step), np.zeros(len(theta)), np.cos(theta - step)))
    after = np.column_stack((np.sin(theta + step), np.zeros(len(theta)), np.cos(theta + step)))
    tangent = (
        land_surface(after, args.flat_radius, args.blend_radius)[0]
        - land_surface(before, args.flat_radius, args.blend_radius)[0]
    )
    inner_tangent = (
        land_surface(after, args.flat_radius, args.blend_radius)[1]
        - land_surface(before, args.flat_radius, args.blend_radius)[1]
    )
    offset_jacobian = np.einsum("vi,vi->v", inner_tangent, tangent) / np.einsum("vi,vi->v", tangent, tangent)
    assert offset_jacobian.min() > 0, "Inward normal offset folds the meridian"
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    normal_error = np.abs(np.einsum("vi,vi->v", tangent, (outer - inner) / 0.0005)).max()
    assert normal_error < 1e-7, normal_error
    rows = []
    for index, specification in enumerate(args.cases):
        minimum, global_size, layers = specification.split(":")
        unit, faces = refine_contact_surface(float(minimum), float(global_size), args.blend_radius)
        outer, inner = land_surface(unit, args.flat_radius, args.blend_radius)
        np.testing.assert_allclose(np.linalg.norm(outer - inner, axis=1), 0.0005, rtol=0, atol=1e-16)
        vertices, tetrahedra, boundary, volumes = make_mesh(outer, inner, faces, int(layers))
        _, counts = np.unique(
            np.sort(boundary[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0, return_counts=True
        )
        assert (counts == 2).all()
        path = args.output / f"land-contact-{index}.npz"
        np.savez_compressed(
            path,
            vertices=vertices,
            tetrahedra=tetrahedra,
            surface_triangles=boundary,
            outer_vertices=outer,
            inner_vertices=inner,
            mid_vertices=(outer + inner) / 2,
            mid_faces=faces,
            material_unit=unit,
        )
        rows.append(
            {
                "mesh": file_id(path),
                "specification": specification,
                "outer_vertices": len(outer),
                "tetrahedra": len(tetrahedra),
                "flat_nodes": int((np.linalg.norm(outer[:, :2], axis=1) <= args.flat_radius).sum()),
                "minimum_volume_m3": float(volumes.min()),
                **mass_properties(vertices, tetrahedra, volumes),
            }
        )
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "flat_radius_m": args.flat_radius,
            "blend_radius_m": args.blend_radius,
            "normal_tangent_dot_error": float(normal_error),
            "minimum_meridian_offset_jacobian": float(offset_jacobian.min()),
            "meshes": rows,
            "physical_reference_qualified": False,
            "geometry": "Full tapered egg with each polar height replaced by sign(z)*30mm inside the flat radius; "
            "quintic smoothstep blends height to the original egg at the blend radius. The surface is C2 across "
            "both joins. Inner surface is a 500um inward offset along the analytic modified normal. "
            "Additional benchmark explicitly authorized by the goal; original smooth-egg results remain separate. "
            "Material coordinates and identical solid/Shell midpoint surfaces are retained for comparison.",
        },
    )


if __name__ == "__main__":
    main()
