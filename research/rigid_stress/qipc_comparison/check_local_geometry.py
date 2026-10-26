"""Bounded native DPR geometry and lifecycle controls for reduced feature spaces.

Run only under Slurm rtx-mid, image 1_26. All elastic assembly, projection,
factorization, pressure fitting, recovery and residual checks are native
Quadrants operations. NumPy authors test meshes and reads geometry/field oracles.
Every executed frame is retained, including rejected spatial approximations.
"""

import argparse
from dataclasses import asdict, dataclass
import os
from pathlib import Path

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import create_contacts, kernel_anchor, kernel_pressure
from genesis.engine.solvers.rigid.stress.data import StressState, create_stress_state
from genesis.engine.solvers.rigid.stress.feature_atlas import FeatureAtlas
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.scatter import create_scatter_workspace, kernel_scatter_faces
from genesis.engine.solvers.rigid.stress.shell_model import ShellModel
from genesis.engine.solvers.rigid.stress.solve import kernel_full_residual
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.utils.array_class import V_VEC
from genesis.utils.misc import qd_to_numpy

from .artifacts import file_id, runtime_identity, write_json
from .native_adaptation_options import (
    add_native_atlas_arguments,
    native_atlas_options,
    validate_native_atlas_arguments,
    weighted_resultant_error,
)
from .check_local_feature_spaces import source_parents
from .voxel_mesh import voxel_mesh


@dataclass(frozen=True)
class SelectionProbe:
    face: int
    local: np.ndarray
    radius: float
    feature: int = -1
    unsupported_face: int = -1
    unsupported_vertex: int = -1
    unsupported_vertex_distance: float = 0.0
    nominal_gap: float = 0.0


@qd.kernel
def kernel_copy_force(source: StressState, target: StressState):
    for i_n, i_b in qd.ndrange(source.force.shape[0], source.force.shape[1]):
        target.force[i_n, i_b] = source.force[i_n, i_b]


@qd.kernel
def kernel_copy_residual_probe(source: StressState, reference: StressState, target: StressState):
    for i_n, i_b in qd.ndrange(source.displacement.shape[0], source.displacement.shape[1]):
        target.displacement[i_n, i_b] = source.displacement[i_n, i_b]
        target.rhs[i_n, i_b] = reference.rhs[i_n, i_b]


def relative(actual, expected):
    delta = (actual - expected).reshape((len(actual), -1))
    expected = expected.reshape((len(expected), -1))
    return np.linalg.norm(delta, axis=1) / np.maximum(np.linalg.norm(expected, axis=1), 1e-30)


def tensor_error(actual, expected, weights):
    multiplicity = np.array([1, 1, 1, 2, 2, 2])
    numerator = np.sum((actual - expected) ** 2 * multiplicity, axis=-1)
    denominator = np.sum(expected**2 * multiplicity, axis=-1)
    return np.sqrt(
        np.sum(numerator * weights, axis=(1, 2)) / np.maximum(np.sum(denominator * weights, axis=(1, 2)), 1e-60)
    )


def tensor_norm(tensor, weights):
    multiplicity = np.array([1, 1, 1, 2, 2, 2])
    return np.sqrt(np.sum(np.sum(tensor**2 * multiplicity, axis=-1) * weights, axis=(1, 2)))


def tensor_peak(tensor):
    vm = np.sqrt(
        np.maximum(
            0,
            0.5
            * (
                (tensor[..., 0] - tensor[..., 1]) ** 2
                + (tensor[..., 1] - tensor[..., 2]) ** 2
                + (tensor[..., 2] - tensor[..., 0]) ** 2
            )
            + 3 * np.sum(tensor[..., 3:] ** 2, axis=-1),
        )
    )
    return vm.reshape((len(vm), -1)).max(axis=1), vm


def surface_point(vertices, face, local):
    if len(face) == 4:
        x, y = local
        weights = 0.25 * np.array([(1 - x) * (1 - y), (1 + x) * (1 - y), (1 + x) * (1 + y), (1 - x) * (1 + y)])
        dx = 0.25 * np.array([y - 1, 1 - y, 1 + y, -1 - y])
        dy = 0.25 * np.array([x - 1, -1 - x, 1 + x, 1 - x])
        tangent, second = dx @ vertices[face], dy @ vertices[face]
    else:
        weights = np.array([1 - local.sum(), *local])
        tangent, second = vertices[face[1]] - vertices[face[0]], vertices[face[2]] - vertices[face[0]]
    normal = np.cross(tangent, second)
    return weights @ vertices[face], normal / np.linalg.norm(normal), tangent / np.linalg.norm(tangent), weights


def complete_boundary(tetrahedra):
    faces, counts = np.unique(
        np.sort(tetrahedra[:, [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]].reshape((-1, 3)), axis=1),
        axis=0,
        return_counts=True,
    )
    return faces[counts == 1]


def shell_neck(path):
    """Author a U with a 0.3 mm neck and 0.8 mm arms, then triangle -> 3 Q4.

    This is an actual nonconvex material domain, with parallel physical shell
    sides and unchanged directors. Neck macro triangles are selected before
    any mechanics is evaluated.
    """
    x, y = np.array([0, 0.0008, 0.0015, 0.0022, 0.003]), np.array([0, 0.0003, 0.0012, 0.0021, 0.003])
    xyz = np.array([(a, b, 0.0) for b in y for a in x])
    triangles = []
    for j in range(len(y) - 1):
        for i in range(len(x) - 1):
            if j == 0 or i in (0, len(x) - 2):
                a = j * len(x) + i
                b, c, d = a + 1, a + len(x) + 1, a + len(x)
                triangles.extend(((a, b, c), (a, c, d)))
    used, inverse = np.unique(triangles, return_inverse=True)
    xyz, triangles = xyz[used], inverse.reshape((-1, 3))
    n_original = len(xyz)
    edges, inverse = np.unique(
        np.sort(triangles[:, ((0, 1), (1, 2), (2, 0))].reshape((-1, 2)), axis=1), axis=0, return_inverse=True
    )
    mids = n_original + inverse.reshape((-1, 3))
    centers = n_original + len(edges) + np.arange(len(triangles))
    quads = np.stack(
        [np.column_stack((triangles[:, i], mids[:, i], centers, mids[:, (i + 2) % 3])) for i in range(3)], axis=1
    ).reshape((-1, 4))
    parents = np.full((n_original + len(edges) + len(triangles), 3), -1, dtype=np.int64)
    parents[:n_original, 0] = np.arange(n_original)
    parents[n_original : n_original + len(edges), :2] = edges
    parents[n_original + len(edges) :] = np.sort(triangles, axis=1)
    xyz = np.concatenate((xyz, xyz[edges].mean(axis=1), xyz[triangles].mean(axis=1)))
    directors = np.broadcast_to([0.0, 0.0, 1.0], xyz.shape).copy()
    np.savez(path, vertices=xyz, quads=quads, directors=directors)
    neck_macros = np.flatnonzero(xyz[triangles, 1].max(axis=1) <= 0.0003 + 1e-14)
    fixed = np.unique(quads.reshape((-1, 3, 4))[neck_macros])
    fixed = fixed[fixed >= n_original]
    return (
        parents,
        xyz,
        fixed,
        dict(
            author="nonconvex U: oriented triangles to three Q4",
            original_nodes=n_original,
            triangle_count=len(triangles),
            quads=len(quads),
            neck_macro_triangles=neck_macros.tolist(),
            neck_rule="all child nodes incident to original triangles at y <= 0.3 mm",
            neck_y_max_m=0.0003,
            complete_physical_sides=True,
        ),
    )


def prepare_asset(args):
    shell = args.stress_backend == "shell"
    if args.geometry == "egg":
        directory = Path(gs.__file__).resolve().parent.parent / f"examples/rigid/assets/hollow_egg/level{args.level}"
        parents, xyz, provenance = source_parents(directory, shell)
        asset = directory / ("shell.npz" if shell else "elastic.npz")
        if not shell:
            with np.load(asset, allow_pickle=False) as original:
                vertices, tets = original["vertices"], original["tetrahedra"]
            asset = args.output / "complete-boundary.npz"
            np.savez(asset, vertices=vertices, tetrahedra=tets, surface_triangles=complete_boundary(tets))
        return asset, parents, xyz, np.empty(0, dtype=np.int64), provenance, 0.0005, 8e-5
    asset = args.output / "neck.npz"
    if shell:
        parents, xyz, fixed, provenance = shell_neck(asset)
        return asset, parents, xyz, fixed, provenance, 8e-5, 3e-5
    vertices, tets, faces = voxel_mesh("u")
    np.savez(asset, vertices=vertices, tetrahedra=tets, surface_triangles=faces)
    edges = np.unique(
        np.sort(tets[:, ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))].reshape((-1, 2)), axis=1), axis=0
    )
    xyz = np.concatenate((vertices, vertices[edges].mean(axis=1)))
    parents = np.full((len(xyz), 3), -1, dtype=np.int64)
    parents[: len(vertices), 0] = np.arange(len(vertices))
    parents[len(vertices) :, :2] = edges
    neck_macros = np.flatnonzero(vertices[tets, 1].max(axis=1) <= 0.001 + 1e-14)
    neck_edges = np.unique(
        np.sort(tets[neck_macros][:, ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))].reshape((-1, 2)), axis=1), axis=0
    )
    fixed = len(vertices) + np.flatnonzero(
        np.array([any(np.array_equal(edge, other) for other in neck_edges) for edge in edges])
    )
    provenance = dict(
        author="voxel_mesh('u'), complete tetrahedral boundary",
        original_nodes=len(vertices),
        neck_macro_tetrahedra=neck_macros.tolist(),
        neck_rule="P2 edge children incident to bottom-bar macro tetrahedra",
        neck_y_max_m=0.001,
        complete_physical_sides=True,
    )
    return asset, parents, xyz, fixed, provenance, 0.0005, 3e-5


class GeometryControl:
    def __init__(self, args, model, atlas, contacts, surface, omega, directory, result):
        self.args, self.model, self.atlas, self.contacts = args, model, atlas, contacts
        self.surface, self.omega, self.directory, self.result = surface, omega, directory, result
        directory.mkdir()
        self.state, self.reference = model.create_state(args.envs, "full"), model.create_state(args.envs, "full")
        self.probe = create_stress_state(model.info.vertices.shape[0], args.envs)
        self.workspace = create_scatter_workspace(args.envs, 128)
        self.vertices = qd_to_numpy(model.load_info.vertices, copy=True)
        self.faces = qd_to_numpy(model.load_info.surface_nodes, copy=True)[:, : 4 if model.n_shell_elements else 3]
        self.normals = qd_to_numpy(surface.info.face_normal, copy=True)
        self.center_local = np.array([0.0, 0.0]) if model.n_shell_elements else np.array([1 / 3, 1 / 3])
        self.reference_options = atlas.options.model_copy(update={"atlas_regions": 0})
        self.boundary = qd_to_numpy(atlas.regions.info.boundary_nodes, copy=True)
        self.base_radius = qd_to_numpy(contacts.radius, copy=True)
        self.support_radii = qd_to_numpy(atlas.regions.info.support_radii, copy=True)
        if model.n_shell_elements:
            self.sample_weights = np.repeat(qd_to_numpy(model.shell.weights), 2, axis=0) / 2
            # Match shell_operators.func_gauss and the output i_q index exactly.
            gauss = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) / np.sqrt(3.0)
            corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
            shape = 0.25 * np.prod(1 + gauss[:, None] * corners[None], axis=-1)
            mids = np.einsum(
                "qa,eac->eqc", shape, qd_to_numpy(model.info.vertices)[::2][qd_to_numpy(model.shell.quads)]
            )
            offset = 0.5 * atlas.options.shell_thickness * qd_to_numpy(model.shell.interpolated_director)
            self.sample_positions = np.stack((mids + offset, mids - offset), axis=1).reshape((-1, 4, 3))
            self.result["sample_layout"] = "native shell: (2 * element + side, Gauss index); (--),(+-),(++),(-+)"
        else:
            self.sample_weights = np.broadcast_to(
                qd_to_numpy(model.info.volumes)[:, None] / 4, (model.n_output_elements, 4)
            )
            self.sample_positions = qd_to_numpy(model.info.vertices)[qd_to_numpy(model.info.elements)[:, :4]]
            self.result["sample_layout"] = "native volume solve.func_peak_impl: tetrahedron corners 0,1,2,3"
        self.field_bindings = (
            ("displacement", self.state.displacement, self.reference.displacement),
            ("stress_tensor", self.state.stress_tensor, self.reference.stress_tensor),
            ("peak", self.state.peak, self.reference.peak),
        )
        if model.n_shell_elements:
            self.field_bindings += (
                ("membrane_resultant", self.state.membrane_resultant, self.reference.membrane_resultant),
                ("bending_resultant", self.state.bending_resultant, self.reference.bending_resultant),
                ("transverse_resultant", self.state.transverse_resultant, self.reference.transverse_resultant),
            )

    def nominal_weights(self, distances):
        """Untimed probe oracle for the declared native graph selector."""
        minimum = distances.min()
        selection = self.atlas.options.atlas_selection
        if selection == "nearest":
            weights = np.zeros_like(distances)
            weights[np.argmin(distances)] = 1.0
            return weights
        if selection == "legacy":
            width = self.atlas.options.atlas_transition
            taper = (distances == minimum).astype(float)
            if width > 0:
                taper = np.maximum(0.0, 1.0 - (distances - minimum) / width)
            return taper * taper * (3.0 - 2.0 * taper)
        scaled = distances / self.support_radii
        weights = np.maximum(1.0 - scaled, 0.0) ** 4 * (1.0 + 4.0 * scaled)
        if selection == "narrow":
            gap = (distances - minimum) / (self.atlas.options.atlas_narrow_fraction * self.support_radii)
            weights *= np.maximum(1.0 - gap, 0.0) ** 4 * (1.0 + 4.0 * gap)
        return weights

    def inputs(self, sites, *, sign=1, valid=None, omega=None, balanced=False):
        n_contacts, n_envs = self.contacts.valid.shape
        positions, forces, inward = (np.zeros((n_contacts, n_envs, 3)) for _ in range(3))
        for i_c, (face, local) in enumerate(sites):
            point, normal, tangent, _ = surface_point(self.vertices, self.faces[face], np.asarray(local))
            if normal @ self.normals[face] < 0:
                normal *= -1
            for i_b in range(n_envs):
                positions[i_c, i_b] = point
                magnitude = 0.01 * (1 + 0.2 * i_b) * (1 if balanced else 1 + 0.13 * i_c)
                friction = 0 if balanced else sign * (0.05 + 0.03 * i_b + 0.02 * i_c)
                forces[i_c, i_b] = magnitude * (-normal + friction * tangent)
                inward[i_c, i_b] = -normal
        if valid is None:
            valid = np.ones((n_contacts, n_envs), dtype=bool)
        if omega is None:
            omega = np.array([[1.3, -0.7, 2.1], [-2.4, 0.9, 1.1]])[np.arange(n_envs) % 2]
        return dict(
            position=positions,
            force=forces,
            normal=inward,
            valid=valid.copy(),
            omega=np.asarray(omega).copy(),
            radius=self.base_radius.copy(),
        )

    def execute(self, name, inputs, *, category="qualification"):
        self.contacts.position.from_numpy(np.ascontiguousarray(inputs["position"]))
        self.contacts.force.from_numpy(np.ascontiguousarray(inputs["force"]))
        self.contacts.normal.from_numpy(np.ascontiguousarray(inputs["normal"]))
        self.contacts.valid.from_numpy(np.ascontiguousarray(inputs["valid"]))
        self.contacts.radius.from_numpy(np.ascontiguousarray(inputs["radius"]))
        self.contacts.friction.fill(0.5)
        self.omega.from_numpy(np.ascontiguousarray(inputs["omega"]))
        kernel_anchor(float(np.finfo(float).eps), self.contacts, self.surface.info)
        kernel_pressure(self.contacts, self.surface.info)
        self.atlas.prepare(self.contacts, self.model.load_info, self.surface.info)
        # This call packs valid nonzero (slot, environment) pairs itself. Do not
        # author a guessed active-pair table or reuse one from an earlier frame.
        kernel_scatter_faces(
            self.contacts,
            self.state,
            self.model.load_info,
            self.surface.info,
            self.workspace,
            True,
            atlas=self.atlas.scatter,
        )
        kernel_copy_force(self.state, self.reference)
        self.model.recover(self.omega, self.reference, self.reference_options, surface_load=True)
        self.atlas.recover(self.omega, self.state, self.atlas.options)
        actual = {name: qd_to_numpy(field, transpose=True, copy=True) for name, field, _ in self.field_bindings}
        expected = {name: qd_to_numpy(field, transpose=True, copy=True) for name, _, field in self.field_bindings}
        raw = qd_to_numpy(self.state.force, transpose=True, copy=True)
        grouped = qd_to_numpy(self.atlas.regions.state.force, copy=True)
        weights = qd_to_numpy(self.atlas.regions.state.weights, copy=True)
        loaded = inputs["valid"] & (np.linalg.norm(inputs["force"], axis=-1) > 1e-30)
        selected = weights * loaded[None]
        load_error = relative(grouped.sum(axis=1).transpose(1, 0, 2), raw[:, self.boundary])
        errors = {name: relative(actual[name], expected[name]).tolist() for name, _, _ in self.field_bindings}
        errors["stress_tensor"] = tensor_error(
            actual["stress_tensor"], expected["stress_tensor"], self.sample_weights
        ).tolist()
        errors["grouped_load"] = load_error.tolist()
        if self.model.n_shell_elements:
            for field, _, _ in self.field_bindings[3:]:
                errors[f"weighted_{field}"] = weighted_resultant_error(
                    np,
                    actual[field],
                    expected[field],
                    qd_to_numpy(self.model.shell.weights),
                    tensor=field != "transverse_resultant",
                ).tolist()
        group_checks = []
        active = qd_to_numpy(self.atlas.state.active, copy=True).astype(bool)
        for i_g, group in enumerate([None, *self.atlas.groups]):
            state = self.atlas.fallback_state if group is None else group.state
            if group is not None:
                group.operator.residual(self.atlas.options.young, state.rhs, state.displacement, state.residual)
                residual = np.linalg.norm(
                    qd_to_numpy(state.residual).transpose(1, 0, 2).reshape((self.args.envs, -1)), axis=1
                )
            else:
                residual = np.sqrt(qd_to_numpy(state.residual_norm_squared))
            rhs_norm = np.linalg.norm(qd_to_numpy(state.rhs, transpose=True).reshape((self.args.envs, -1)), axis=1)
            native_rhs_norm = np.sqrt(qd_to_numpy(state.rhs_norm_squared))
            budget = np.maximum(
                self.atlas.options.absolute_tolerance,
                self.atlas.options.tolerance * rhs_norm,
            )
            valid = qd_to_numpy(state.valid).astype(bool)
            group_checks.append(
                dict(
                    group=i_g,
                    active=active[i_g].tolist(),
                    unscaled_residual_norm=residual.tolist(),
                    independently_read_rhs_norm=rhs_norm.tolist(),
                    native_rhs_norm=native_rhs_norm.tolist(),
                    original_budget=budget.tolist(),
                    valid=valid.tolist(),
                    passed=bool(np.all(~active[i_g] | ((residual <= budget) & valid))),
                )
            )
        kernel_copy_residual_probe(self.state, self.reference, self.probe)
        kernel_full_residual(
            self.atlas.options.young,
            self.atlas.options.tolerance,
            self.atlas.options.absolute_tolerance,
            self.probe,
            self.model.info,
            False,
        )
        mother_relative = np.sqrt(qd_to_numpy(self.probe.residual_norm_squared)) / np.maximum(
            np.sqrt(qd_to_numpy(self.reference.rhs_norm_squared)), 1e-30
        )
        status = qd_to_numpy(self.contacts.status, copy=True)
        force_error = qd_to_numpy(self.contacts.force_error, copy=True)
        moment_error = qd_to_numpy(self.contacts.moment_error, copy=True)
        magnitude = np.linalg.norm(inputs["force"], axis=-1)
        radius = qd_to_numpy(self.contacts.radius, copy=True)
        force_budget, moment_budget = 1e-8 * magnitude, 1e-8 * magnitude * radius
        relative_force_error = np.divide(force_error, magnitude, out=np.zeros_like(force_error), where=loaded)
        relative_moment_error = np.divide(
            moment_error, magnitude * radius, out=np.zeros_like(moment_error), where=loaded
        )
        numerical = bool(
            all(row["passed"] for row in group_checks)
            and qd_to_numpy(self.state.valid).all()
            and qd_to_numpy(self.reference.valid).all()
            and np.all(status[loaded] == 0)
            and np.all(force_error[loaded] <= force_budget[loaded])
            and np.all(moment_error[loaded] <= moment_budget[loaded])
            and np.all(load_error <= 1e-10)
            and np.isfinite(actual["stress_tensor"]).all()
        )
        spatial = max(errors["stress_tensor"]) <= 0.10 and max(errors["peak"]) <= 0.05
        if self.model.n_shell_elements:
            spatial = spatial and all(
                max(errors[prefix + name]) <= 0.10
                for name, _, _ in self.field_bindings[3:]
                for prefix in ("", "weighted_")
            )
        row = dict(
            name=name,
            category=category,
            errors=errors,
            tensor_error_norm=tensor_norm(
                actual["stress_tensor"] - expected["stress_tensor"], self.sample_weights
            ).tolist(),
            reference_tensor_norm=tensor_norm(expected["stress_tensor"], self.sample_weights).tolist(),
            numerical_passed=numerical,
            spatial_passed=bool(spatial),
            source_numerical_checks=group_checks,
            mother_spatial_residual_relative=mother_relative.tolist(),
            loaded=int(loaded.sum()),
            loaded_environments=loaded.any(axis=0).tolist(),
            fine_contributors=int(np.count_nonzero(selected[1:] > 1e-14, axis=0).max()),
            fine_contributors_per_contact=np.count_nonzero(selected[1:] > 1e-14, axis=0).tolist(),
            mother_weight_per_contact=(weights[0] * loaded).tolist(),
            coverage_fallbacks=int((qd_to_numpy(self.atlas.regions.state.fallback) & loaded).sum()),
            maximum_force_error=float(force_error.max()),
            maximum_moment_error=float(moment_error.max()),
            maximum_relative_force_error=float(relative_force_error.max()),
            maximum_relative_moment_error=float(relative_moment_error.max()),
            numerical_fallbacks=qd_to_numpy(self.atlas.state.numerical_fallbacks).tolist(),
        )
        if name.startswith("remote_neck"):
            _, vm = tensor_peak(expected["stress_tensor"])
            maxima = self.sample_positions.reshape((-1, 3))[np.argmax(vm.reshape((self.args.envs, -1)), axis=1)]
            distances = np.linalg.norm(maxima[:, None] - inputs["position"].transpose(1, 0, 2), axis=-1)
            distances[~loaded.T] = np.inf
            nearest = distances.min(axis=1) / float(qd_to_numpy(self.contacts.radius)[0, 0])
            inside = maxima[:, 1] <= self.result["basis_provenance"]["neck_y_max_m"] + 1e-12
            row["neck_witness"] = dict(
                peak_positions_m=maxima.tolist(),
                in_geometric_neck=inside.tolist(),
                nearest_contact_distance_in_radii=nearest.tolist(),
                target_distance_radii=2.0,
                passed=bool(np.any(inside & (nearest > 2))),
            )
        arrays = {f"candidate_{key}": value for key, value in actual.items()}
        arrays.update({f"reference_{key}": value for key, value in expected.items()})
        arrays.update({f"contact_{key}": value for key, value in inputs.items()})
        arrays.update(
            raw_force=raw,
            grouped_force=grouped,
            weights=weights,
            active_groups=active,
            contact_status=status,
            contact_force_error=force_error,
            contact_moment_error=moment_error,
            contact_force_budget=force_budget,
            contact_moment_budget=moment_budget,
            contact_relative_force_error=relative_force_error,
            contact_relative_moment_error=relative_moment_error,
            anchor_face=qd_to_numpy(self.contacts.anchor_face),
            centers=qd_to_numpy(self.contacts.center),
            sample_positions=self.sample_positions,
            sample_weights=self.sample_weights,
        )
        path = self.directory / f"{len(self.result['rows']):03d}-{name}.npz"
        np.savez_compressed(path, **arrays)
        row["frame"] = file_id(path)
        self.result["rows"].append(row)
        write_json(self.directory / "result.json", self.result)
        return actual, row

    def probes(self):
        """Choose test input locations from topology/distances, never from stress."""
        distances = qd_to_numpy(self.atlas.regions.info.distances)
        coverage = qd_to_numpy(self.atlas.regions.info.face_coverage).astype(bool)
        nodes = self.faces.copy()
        if self.model.n_shell_elements:
            nodes = qd_to_numpy(self.model.load_info.load_nodes)[nodes, 0]
        coordinates = [self.center_local]
        if self.model.n_shell_elements:
            coordinates += [np.array([x, y]) for x in (-0.8, 0.8) for y in (-0.8, 0.8)]
        else:
            coordinates += [np.array([0.05, 0.05]), np.array([0.9, 0.05]), np.array([0.05, 0.9])]
        junction = None
        best_gap = np.inf
        aggregation = None
        aggregation_clearance = -np.inf
        bounds_low = qd_to_numpy(self.surface.info.face_bounds_low)
        bounds_high = qd_to_numpy(self.surface.info.face_bounds_high)
        for i_f, face in enumerate(self.faces):
            for local in coordinates:
                *_, shape = surface_point(self.vertices, face, local)
                d = distances[:, nodes[i_f]] @ shape
                order = np.argsort(d)
                if coverage[order[0], i_f]:
                    point, *_ = surface_point(self.vertices, face, local)
                    unsupported = ~coverage[order[0]]
                    delta = np.maximum(bounds_low[unsupported] - point, 0) + np.maximum(
                        point - bounds_high[unsupported], 0
                    )
                    clearance = float(np.linalg.norm(delta, axis=1).min()) if unsupported.any() else 1e30
                    if clearance > aggregation_clearance:
                        aggregation = (i_f, local.copy())
                        aggregation_clearance = clearance
                if len(order) >= 3:
                    gap = d[order[2]] - d[order[0]]
                    if coverage[order[:3], i_f].all() and gap < best_gap:
                        junction, best_gap = (i_f, local.copy()), gap
        if junction is None:
            junction, best_gap = (0, self.center_local.copy()), None
        if aggregation is None:
            aggregation = (0, self.center_local.copy())
        incident = {}
        for i_f, face in enumerate(self.faces):
            for vertex in face:
                incident.setdefault(int(vertex), []).append(i_f)
        footprint = None
        radius = float(qd_to_numpy(self.contacts.radius)[0, 0])
        for i_f, face in enumerate(self.faces):
            for corner, vertex in enumerate(face):
                others = [j for j in incident[int(vertex)] if self.normals[i_f] @ self.normals[j] > 0.8]
                for feature in np.flatnonzero(coverage[:, i_f]):
                    unsupported = [j for j in others if not coverage[feature, j]]
                    if not unsupported:
                        continue
                    if len(face) == 4:
                        local = 0.999 * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]][corner])
                    else:
                        shape = np.full(3, 0.0005)
                        shape[corner] = 0.999
                        local = shape[1:]
                    point, _, _, shape = surface_point(self.vertices, face, local)
                    d = distances[:, nodes[i_f]] @ shape
                    nominal = self.nominal_weights(d)[feature] > 0
                    if nominal and np.linalg.norm(point - self.vertices[vertex]) < 0.5 * radius:
                        footprint = (i_f, local.copy())
                        self.result["partial_footprint_probe"] = dict(
                            face=i_f,
                            feature=int(feature + 1),
                            unsupported_neighbor_face=int(unsupported[0]),
                            shared_physical_vertex=int(vertex),
                            anchor_face_covered=True,
                            distance_to_unsupported_vertex_m=float(np.linalg.norm(point - self.vertices[vertex])),
                            radius_m=radius,
                            locatable=True,
                        )
                        break
                if footprint is not None:
                    break
            if footprint is not None:
                break
        if footprint is None:
            self.result["partial_footprint_probe"] = dict(
                locatable=False, reason="No supported/unsupported adjacent face boundary with active nominal feature."
            )
            footprint = aggregation
        junction, footprint, best_gap = self.targeted_selection_probes(
            junction, footprint, best_gap, distances, coverage, nodes, coordinates
        )
        uncovered = np.flatnonzero(~coverage.any(axis=0))
        _point, _normal, _scale, junction_shape = surface_point(self.vertices, self.faces[junction[0]], junction[1])
        junction_distances = distances[:, nodes[junction[0]]] @ junction_shape
        self.result["junction_probe"] = dict(
            face=int(junction[0]),
            local=junction[1].tolist(),
            third_minus_first_distance_m=best_gap,
            transition_m=self.atlas.options.atlas_transition,
            nominal_three_locatable=bool(np.count_nonzero(self.nominal_weights(junction_distances) > 0) >= 3),
            selection=self.atlas.options.atlas_selection,
        )
        return junction, aggregation, footprint, uncovered

    def targeted_selection_probes(self, junction, footprint, best_gap, distances, coverage, nodes, coordinates):
        """Search actual native whole-footprint selection, without elastic solves.

        Nominal nearest-feature distance alone does not certify availability.
        A separate finite-radius probe also looks beyond shared face vertices;
        fixed coarse neck nodes can remove the old boundary witness entirely.
        """
        radius = float(self.base_radius[0, 0])
        if self.model.n_shell_elements:
            coordinates = [
                np.array([x, y]) for x in (-0.98, -0.5, 0.0, 0.5, 0.98) for y in (-0.98, -0.5, 0.0, 0.5, 0.98)
            ]
        else:
            coordinates = [self.center_local] + [
                np.array([i / 12, j / 12]) for i in range(1, 12) for j in range(1, 12 - i)
            ]
        unsupported_vertices = []
        unsupported_face_ids = []
        for feature in range(len(coverage)):
            face_ids = np.flatnonzero(~coverage[feature])
            vertices, first = np.unique(self.faces[face_ids], return_index=True)
            unsupported_vertices.append(vertices)
            unsupported_face_ids.append(face_ids[first // self.faces.shape[1]])
        junction_candidates, partial_candidates = [], []
        for i_f, face in enumerate(self.faces):
            for local in coordinates:
                point, _, _, shape = surface_point(self.vertices, face, local)
                d = distances[:, nodes[i_f]] @ shape
                nominal = self.nominal_weights(d) > 0
                if np.count_nonzero(nominal & coverage[:, i_f]) >= 3:
                    ordered = np.sort(d)
                    junction_candidates.append(
                        SelectionProbe(i_f, local, radius, nominal_gap=float(ordered[2] - ordered[0]))
                    )
                for feature in np.flatnonzero(nominal & coverage[:, i_f]):
                    vertices = unsupported_vertices[feature]
                    if len(vertices):
                        distance = np.linalg.norm(self.vertices[vertices] - point, axis=1)
                        nearest = np.argmin(distance)
                        if distance[nearest] < 8 * radius:
                            partial_candidates.append(
                                SelectionProbe(
                                    i_f,
                                    local,
                                    max(radius, float(1.05 * distance[nearest])),
                                    int(feature),
                                    int(unsupported_face_ids[feature][nearest]),
                                    int(vertices[nearest]),
                                    float(distance[nearest]),
                                )
                            )
        # Keep a bounded set of the smallest physical footprints. This changes
        # test inputs, never the feature coverage mask or the pressure tolerance.
        partial_candidates.sort(key=lambda probe: probe.radius)
        nominal_candidate_count = len(junction_candidates)
        if len(junction_candidates) > 640:
            closest = sorted(junction_candidates, key=lambda probe: probe.nominal_gap)[:128]
            indices = np.linspace(0, len(junction_candidates) - 1, 512, dtype=int)
            junction_candidates = closest + [junction_candidates[index] for index in indices]
        candidates = junction_candidates + partial_candidates[:64]
        observed_weights, observed_status = [], []
        selected_junction, selected_partial = None, None
        junction_score = -1.0
        for begin in range(0, len(candidates), 4):
            batch = candidates[begin : begin + 4]
            padded = batch + [batch[-1]] * (4 - len(batch))
            inputs = self.inputs([(probe.face, probe.local) for probe in padded])
            self.contacts.position.from_numpy(np.ascontiguousarray(inputs["position"]))
            self.contacts.force.from_numpy(np.ascontiguousarray(inputs["force"]))
            self.contacts.normal.from_numpy(np.ascontiguousarray(inputs["normal"]))
            self.contacts.valid.from_numpy(np.ascontiguousarray(inputs["valid"]))
            self.contacts.friction.fill(0.5)
            self.contacts.radius.from_numpy(np.array([[probe.radius] * self.args.envs for probe in padded]))
            kernel_anchor(float(np.finfo(float).eps), self.contacts, self.surface.info)
            self.atlas.prepare(self.contacts, self.model.load_info, self.surface.info)
            weights = qd_to_numpy(self.atlas.regions.state.weights, copy=True)
            status = qd_to_numpy(self.contacts.status, copy=True)
            for slot, probe in enumerate(batch):
                value = weights[:, slot, 0]
                observed_weights.append(value.copy())
                observed_status.append(int(status[slot, 0]))
                if status[slot, 0] == 0:
                    contributors = np.count_nonzero(value[1:] > 1e-14)
                    if probe.feature < 0 and contributors >= 3:
                        score = np.sort(value[1:])[-3]
                        if score > junction_score:
                            selected_junction, junction_score = probe, score
                    if probe.feature >= 0 and value[0] > 0 and selected_partial is None:
                        selected_partial = probe
        self.contacts.radius.from_numpy(self.base_radius)
        artifact = self.directory / "selection-probes.npz"
        np.savez_compressed(
            artifact,
            faces=np.array([probe.face for probe in candidates]),
            local=np.array([probe.local for probe in candidates]),
            radius=np.array([probe.radius for probe in candidates]),
            partial_feature=np.array([probe.feature for probe in candidates]),
            weights=np.array(observed_weights),
            status=np.array(observed_status),
        )
        self.result["targeted_selection_search"] = dict(
            probe_count=len(candidates),
            nominal_junction_candidates=nominal_candidate_count,
            executed_junction_candidates=len(junction_candidates),
            partial_candidates=len(partial_candidates[:64]),
            maximum_radius_multiplier=8.0,
            native_three_locatable=selected_junction is not None,
            native_partial_locatable=selected_partial is not None,
            artifact=file_id(artifact),
        )
        if selected_junction is not None:
            junction = (selected_junction.face, selected_junction.local.copy())
            *_, shape = surface_point(self.vertices, self.faces[junction[0]], junction[1])
            ordered = np.sort(distances[:, nodes[junction[0]]] @ shape)
            best_gap = float(ordered[2] - ordered[0])
        if selected_partial is not None:
            footprint = (selected_partial.face, selected_partial.local.copy())
            self.result["partial_footprint_probe"] = dict(
                face=selected_partial.face,
                feature=selected_partial.feature + 1,
                unsupported_neighbor_face=selected_partial.unsupported_face,
                unsupported_physical_vertex=selected_partial.unsupported_vertex,
                anchor_face_covered=True,
                distance_to_unsupported_vertex_m=selected_partial.unsupported_vertex_distance,
                radius_m=selected_partial.radius,
                locatable=True,
                source="native targeted finite-footprint input search",
            )
        return junction, footprint, best_gap

    def suite(self):
        junction, aggregation, footprint, uncovered = self.probes()
        centers = self.vertices[self.faces].mean(axis=1)
        first = aggregation[0]
        opposite = int(np.argmin(self.normals @ self.normals[first]))
        radial = np.einsum("ij,ij->i", centers - self.vertices.mean(axis=0), self.normals)
        inner = np.flatnonzero(radial < 0)
        inner_face = int(inner[len(inner) // 2]) if len(inner) else opposite
        sites = [junction, aggregation, (opposite, self.center_local.copy()), (inner_face, self.center_local.copy())]
        self.result["material_sides"] = dict(
            first_face=int(first),
            opposite_face=opposite,
            inner_face=inner_face,
            authored_inner_faces=int(len(inner)),
            opposite_normal_dot=float(self.normals[first] @ self.normals[opposite]),
        )
        for step, shift in enumerate((-0.015, 0.0, 0.015)):
            moved = [(face, local + np.array([shift, 0.0])) for face, local in sites]
            self.execute(f"moving_asymmetric_{step}", self.inputs(moved))
        self.execute("friction_reversal", self.inputs(sites, sign=-1))
        baseline, _ = self.execute("permutation_reference", self.inputs(sites))
        permuted = self.inputs(sites)
        for key in ("position", "force", "normal", "valid"):
            permuted[key] = permuted[key][[2, 0, 3, 1]].copy()
        actual, row = self.execute("slot_permutation", permuted)
        row["permutation_tensor_error"] = tensor_error(
            actual["stress_tensor"], baseline["stress_tensor"], self.sample_weights
        ).tolist()
        row["permutation_peak_error"] = relative(actual["peak"], baseline["peak"]).tolist()
        row["permutation_passed"] = max(row["permutation_tensor_error"] + row["permutation_peak_error"]) < 1e-8
        footprint_sites = [footprint, *sites[1:]]
        footprint_inputs = self.inputs(footprint_sites)
        if self.result["partial_footprint_probe"]["locatable"]:
            footprint_inputs["radius"][0] = self.result["partial_footprint_probe"]["radius_m"]
        _, footprint_row = self.execute("partial_footprint", footprint_inputs)
        self.result["partial_footprint_probe"]["observed_mother_weight"] = footprint_row["mother_weight_per_contact"][0]
        valid = np.ones(self.contacts.valid.shape, dtype=bool)
        valid[:, -1] = False
        zero_spin = np.zeros((self.args.envs, 3))
        partial, partial_row = self.execute(
            "partial_environment_release", self.inputs(sites, valid=valid, omega=zero_spin)
        )
        partial_row["released_environment_exact_zero"] = bool(np.count_nonzero(partial["stress_tensor"][-1]) == 0)
        released, release_row = self.execute(
            "release_all", self.inputs(sites, valid=np.zeros_like(valid), omega=zero_spin)
        )
        release_row["exact_zero"] = bool(all(np.count_nonzero(value) == 0 for value in released.values()))
        restored, restored_row = self.execute("restore_after_release", self.inputs(sites))
        restored_row["restore_tensor_error"] = tensor_error(
            restored["stress_tensor"], baseline["stress_tensor"], self.sample_weights
        ).tolist()
        self.result["aggregation"] = self.aggregation(aggregation)
        distant = self.inputs([sites[1], sites[1], sites[2], sites[3]], omega=zero_spin, balanced=True)
        distant["valid"][1] = False
        distant["valid"][3] = False
        self.result["opposed_distant_sources"] = self.aggregation(
            aggregation, inputs=distant, label="opposed_distant_sources", require_shared=False
        )
        self.result["continuity"] = [
            self.continuity("junction", junction, sites),
            self.continuity("coverage", footprint, sites),
        ]
        if len(uncovered):
            fallback_sites = [(int(uncovered[len(uncovered) // 2]), self.center_local.copy())] * 4
            _, fallback = self.execute(
                "intentional_mother_coverage", self.inputs(fallback_sites), category="intentional_fallback"
            )
            fallback["expected_mother_only"] = bool(np.all(np.array(fallback["mother_weight_per_contact"]) > 1 - 1e-12))
            self.result["intentional_fallback"] = dict(locatable=True, passed=fallback["expected_mother_only"])
        else:
            self.result["intentional_fallback"] = dict(
                locatable=False,
                reason="Every surface face belongs to at least one fine feature; no fabricated fallback mask was used.",
            )
        if self.args.geometry == "neck":
            self.remote_neck(centers)
        rows = self.result["rows"]
        junction_contributors = row_value(rows, "moving_asymmetric_1", "fine_contributors_per_contact")[0]
        self.result["junction_observed_three"] = max(junction_contributors) >= 3
        checks = dict(
            numerical=all(row["numerical_passed"] for row in rows),
            spatial=all(row["spatial_passed"] for row in rows if row["category"] == "qualification"),
            permutation=row_value(rows, "slot_permutation", "permutation_passed"),
            release=row_value(rows, "release_all", "exact_zero")
            and row_value(rows, "partial_environment_release", "released_environment_exact_zero"),
            restore=max(row_value(rows, "restore_after_release", "restore_tensor_error")) < 1e-8,
            aggregation=self.result["aggregation"]["passed"],
            opposed_distant_source_sum=self.result["opposed_distant_sources"]["passed"],
            continuity=all(row["passed"] for row in self.result["continuity"]),
            partial_footprint=bool(
                self.result["partial_footprint_probe"]["locatable"]
                and np.max(footprint_row["mother_weight_per_contact"][0]) > 0
            ),
            junction=not self.result["junction_probe"]["nominal_three_locatable"]
            or self.result["junction_observed_three"],
            reduction=any(
                group.blocks < self.atlas.metadata().mother_blocks for group in self.atlas.metadata().groups[1:]
            ),
        )
        observations = [
            value
            for row in rows
            if row["category"] == "qualification"
            for value, loaded in zip(row["errors"]["stress_tensor"], row["loaded_environments"], strict=True)
            if loaded
        ]
        self.result["mean_qualification_field_error"] = float(np.mean(observations)) if observations else 0.0
        self.result["mean_field_budget"] = 0.05
        self.result["mean_population"] = "loaded qualification environment-fields; releases excluded"
        checks["mean_field"] = self.result["mean_qualification_field_error"] <= 0.05
        if self.args.geometry == "neck":
            checks["remote_neck_witness"] = any(row.get("neck_witness", {}).get("passed", False) for row in rows)
        self.result.update(complete=True, qualification_checks=checks, passed=bool(all(checks.values())))
        write_json(self.directory / "result.json", self.result)

    def aggregation(self, site, *, inputs=None, label="same_feature_aggregation", require_shared=True):
        if inputs is None:
            sites = [(site[0], site[1] + np.array([offset, 0.0])) for offset in (-0.01, -0.003, 0.004, 0.012)]
            inputs = self.inputs(sites)
        combined, row = self.execute(label, inputs)
        group_weights = qd_to_numpy(self.atlas.regions.state.weights, copy=True)[1:]
        shared = np.all((group_weights > 1e-14) | ~inputs["valid"][None], axis=1)
        shared_groups = [(np.flatnonzero(shared[:, env]) + 1).tolist() for env in range(self.args.envs)]
        summed = {key: np.zeros_like(value) for key, value in combined.items() if key != "peak"}
        source_norm = np.zeros(self.args.envs)
        source_count = 0
        for i_c in range(4):
            if not inputs["valid"][i_c].any():
                continue
            one = {key: value.copy() for key, value in inputs.items()}
            one["valid"][:] = False
            one["valid"][i_c] = inputs["valid"][i_c]
            one["omega"][:] = 0
            fields, source_row = self.execute(f"{label}_separate_source_{i_c}", one, category="aggregation_diagnostic")
            source_norm += np.array(source_row["reference_tensor_norm"])
            source_count += 1
            for key in summed:
                summed[key] += fields[key]
        spin = {key: value.copy() for key, value in inputs.items()}
        spin["valid"][:] = False
        fields, spin_row = self.execute(f"{label}_separate_spin", spin, category="aggregation_diagnostic")
        source_norm += np.array(spin_row["reference_tensor_norm"])
        for key in summed:
            summed[key] += fields[key]
        errors = {key: relative(summed[key], combined[key]).tolist() for key in summed}
        peak, _ = tensor_peak(summed["stress_tensor"])
        errors["peak_after_tensor_sum"] = relative(
            peak[:, None], combined["peak"].reshape((self.args.envs, -1))
        ).tolist()
        return dict(
            errors=errors,
            passed=bool(
                max(max(values) for values in errors.values()) < 1e-8
                and (not require_shared or shared.any(axis=0).all())
            ),
            spin_added_once=True,
            source_count=source_count,
            fine_contributors=row["fine_contributors"],
            shared_fine_groups_per_environment=shared_groups,
            combined_tensor_error_norm=row["tensor_error_norm"],
            reference_source_tensor_norm_sum=source_norm.tolist(),
            source_norm_normalized_tensor_error=(
                np.array(row["tensor_error_norm"]) / np.maximum(source_norm, 1e-30)
            ).tolist(),
            cancellation_ratio=(np.array(row["reference_tensor_norm"]) / np.maximum(source_norm, 1e-30)).tolist(),
            near_cancellation_observed=bool(np.any(np.array(row["reference_tensor_norm"]) < 0.25 * source_norm)),
            rule="Native independently recovered source tensors/displacements sum to native grouped recovery; peak follows the tensor sum.",
        )

    def continuity(self, label, site, sites):
        rows = []
        for delta in (1e-4, 1e-5, 1e-6):
            pair = []
            for sign in (-1, 1):
                perturbed = [(site[0], site[1] + np.array([sign * delta, 0.0])), *sites[1:]]
                inputs = self.inputs(perturbed)
                if label == "coverage" and self.result["partial_footprint_probe"]["locatable"]:
                    inputs["radius"][0] = self.result["partial_footprint_probe"]["radius_m"]
                actual, row = self.execute(f"continuity_{label}_{delta:g}_{sign:+d}", inputs)
                pair.append((actual, row))
            jump = tensor_error(pair[1][0]["stress_tensor"], pair[0][0]["stress_tensor"], self.sample_weights)
            rows.append(
                dict(
                    natural_coordinate_delta=delta,
                    full_tensor_relative_jump=jump.tolist(),
                    frame_names=[value[1]["name"] for value in pair],
                )
            )
        jumps = np.array([row["full_tensor_relative_jump"] for row in rows])
        passed = bool(np.all(jumps[1:] <= 0.3 * jumps[:-1] + 1e-8))
        return dict(
            location=label,
            rows=rows,
            passed=passed,
            gate="Each tenfold perturbation decrease reduces the complete tensor jump at least 70%, with 1e-8 numerical floor.",
        )

    def remote_neck(self, centers):
        def nearest(target, direction):
            candidates = np.flatnonzero(self.normals @ np.asarray(direction) > 0.8)
            return int(candidates[np.argmin(np.linalg.norm(centers[candidates] - target, axis=1))])

        if self.model.n_shell_elements:
            left = nearest([0.0004, 0.0026, 0.00004], [0, 0, 1])
            right = nearest([0.0026, 0.0026, -0.00004], [0, 0, -1])
            right_same = nearest([0.0026, 0.0026, 0.00004], [0, 0, 1])
        else:
            left = nearest([0, 0.0026, 0.0005], [-1, 0, 0])
            right = nearest([0.003, 0.0026, 0.0005], [1, 0, 0])
            right_same = nearest([0.002, 0.0026, 0.0005], [-1, 0, 0])
        valid = np.zeros(self.contacts.valid.shape, dtype=bool)
        valid[:2] = True
        for i, second in enumerate((right, right_same)):
            sites = [(left, self.center_local.copy()), (second, self.center_local.copy())] * 2
            self.execute(
                f"remote_neck_{i}", self.inputs(sites, valid=valid, omega=np.zeros((self.args.envs, 3)), balanced=True)
            )


def row_value(rows, name, key):
    return next(row[key] for row in rows if row["name"] == name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stress-backend", choices=("shell", "volume"), required=True)
    parser.add_argument("--geometry", choices=("egg", "neck"), default="egg")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--regions", type=int, default=12)
    parser.add_argument("--core-scale", type=float, default=1.0)
    parser.add_argument("--halo", type=int, default=0)
    parser.add_argument("--transition", type=float, default=None)
    parser.add_argument("--solver", choices=("factor", "inverse", "response"), default="factor")
    parser.add_argument("--shell-bending", action="store_true")
    parser.add_argument("--fixed-neck", action="store_true")
    parser.add_argument("--level", type=int, choices=(1, 2), default=1)
    parser.add_argument("--envs", type=int, default=2)
    parser.add_argument("--quadrature", type=int, default=32)
    parser.add_argument("--radius", type=float, default=None)
    add_native_atlas_arguments(parser)
    args = parser.parse_args()
    validate_native_atlas_arguments(parser, args, args.regions)
    if args.fixed_neck and args.geometry != "neck":
        parser.error("--fixed-neck requires --geometry neck")
    if args.envs < 2 or args.regions < 1:
        parser.error("Use at least two environments and one feature region")
    assert os.environ.get("SLURM_JOB_ID"), "Scientific execution requires a scheduled Slurm job."
    assert os.environ.get("SLURM_JOB_PARTITION") == "rtx-mid", "Use the established rtx-mid partition."
    assert os.environ.get("GENESIS_IMAGE_VER") == "1_26", "Use the captured image 1_26 environment."
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(
        complete=False,
        passed=False,
        identity=runtime_identity(),
        stress_backend=args.stress_backend,
        geometry=args.geometry,
        variants=[],
        tensor_budget=0.10,
        peak_budget=0.05,
        load_budget=1e-10,
        force_budget=1e-8,
        moment_budget=1e-8,
        metric="Area/volume-weighted full symmetric tensor Frobenius norm; peak scanned over all native samples.",
        spatial_error_separate_from_source_numerical_acceptance=True,
    )
    write_json(args.output / "result.json", result)
    gs.init(backend=gs.cuda, precision="64", logging_level="warning")
    asset, parents, xyz, fixed, provenance, thickness, radius = prepare_asset(args)
    radius = radius if args.radius is None else args.radius
    transition = (0.003 if args.geometry == "egg" else 0.00015) if args.transition is None else args.transition
    contacts = create_contacts(4, args.envs, radius, hertz_profile=True)
    omega = V_VEC(3, dtype=gs.qd_float, shape=(args.envs,))
    options = gs.options.RigidStressOptions.dpr(
        mesh=asset,
        backend=args.stress_backend,
        output_mode="full",
        method="direct",
        surface_inverse=False,
        factor_scaling=True,
        shell_thickness=thickness,
        pressure_profile="hertz",
        quadrature=args.quadrature,
        atlas_regions=args.regions,
        **native_atlas_options(args),
        atlas_core_scale=args.core_scale,
        atlas_halo=args.halo,
        atlas_transition=transition,
        atlas_solver=args.solver,
        atlas_shell_bending=args.shell_bending,
        inverse_max_bytes=512 * 1024 * 1024,
    )
    assert options.tolerance == 1e-8 and options.absolute_tolerance == 1e-11
    model = ShellModel(options) if args.stress_backend == "shell" else StressModel(options)
    bpn = 2 if model.n_shell_elements else 1
    np.testing.assert_allclose(qd_to_numpy(model.info.vertices)[::bpn], xyz, rtol=1e-13, atol=1e-15)
    surface = StressSurface(args.quadrature, model.load_info)
    result.update(
        asset=file_id(asset),
        basis_provenance=provenance,
        physical_nodes=len(xyz),
        mother_blocks=model.info.vertices.shape[0],
        radius_m=radius,
        transition_m=transition,
        complete_boundary=True,
    )
    variants = [("unfixed", np.empty(0, dtype=np.int64))]
    if args.fixed_neck:
        variants.append(("fixed_neck", fixed))
    for name, selected in variants:
        basis_path = args.output / f"basis-{name}.npz"
        np.savez(basis_path, parent_nodes=parents, fixed_nodes=selected)
        # Volume parentage is derived natively; atlas_basis may still carry the
        # geometric fixed-node list. Do not pass shell parent rows to volume.
        if not model.n_shell_elements:
            np.savez(basis_path, fixed_nodes=selected)
        current_options = options.model_copy(update={"atlas_basis": basis_path})
        atlas = FeatureAtlas(model, contacts, current_options)
        current = dict(
            complete=False,
            passed=False,
            name=name,
            rows=[],
            atlas=asdict(atlas.metadata()),
            options=current_options.model_dump(mode="json"),
            basis=file_id(basis_path),
            basis_provenance=provenance,
            fixed_physical_nodes=selected.tolist(),
        )
        # Preserve explicit 1/2/3 parent provenance even for volume assets whose
        # public atlas loader derives that parentage from its tetrahedra.
        np.savez(
            args.output / f"topology-{name}.npz",
            parent_nodes=atlas.basis.parent_nodes,
            fixed_nodes=selected,
            authored_parent_nodes=parents,
            physical_vertices=xyz,
            coarse_nodes=atlas.basis.coarse_nodes,
            **{f"local_nodes_{i}": nodes for i, nodes in enumerate(atlas.regions.local_nodes)},
        )
        control = GeometryControl(args, model, atlas, contacts, surface, omega, args.output / name, current)
        control.suite()
        result["variants"].append(current)
        write_json(args.output / "result.json", result)
    if args.fixed_neck:
        result["fixed_neck_comparison"] = [
            dict(
                name=variant["name"],
                passed=variant["passed"],
                remote_neck_rows=[
                    dict(
                        name=row["name"],
                        spatial_passed=row["spatial_passed"],
                        errors=row["errors"],
                        neck_witness=row["neck_witness"],
                    )
                    for row in variant["rows"]
                    if "neck_witness" in row
                ],
            )
            for variant in result["variants"]
        ]
    result.update(
        complete=True,
        passed=result["variants"][-1]["passed"],
        qualification_variant=result["variants"][-1]["name"],
        failure_policy="All full frames are saved before qualification fails; unfixed-neck control remains a separately reported baseline.",
    )
    write_json(args.output / "result.json", result)
    assert result["passed"], (
        "Geometry/lifecycle or 10% tensor / 5% peak qualification failed; retained full frames explain the rejection."
    )


if __name__ == "__main__":
    main()
