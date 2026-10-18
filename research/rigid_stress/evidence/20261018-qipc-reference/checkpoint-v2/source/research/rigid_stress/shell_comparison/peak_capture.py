"""Retain the raw peak's field, primitive, material/world location and sample time."""

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from genesis.utils.misc import tensor_to_array


@dataclass
class PeakCapture:
    shell_model: bool
    faces: np.ndarray
    material_vertices: np.ndarray
    values: np.ndarray
    steps: np.ndarray
    phases: np.ndarray
    primitive: np.ndarray
    world: np.ndarray
    material: np.ndarray
    fields: dict = field(default_factory=dict)

    @classmethod
    def allocate(cls, args, workload, shell_model):
        if shell_model:
            faces = tensor_to_array(workload.egg.get_faces())[0]
            initial = tensor_to_array(workload.initial_vertices)
            positions = tensor_to_array(workload.initial_position)
            with np.load(args.conditions) as bank:
                quaternions = bank["quaternion"][np.arange(args.envs) % len(bank["position"])]
            matrices = Rotation.from_quat(quaternions[:, [1, 2, 3, 0]]).as_matrix()
            material = np.einsum("bji,bnj->bni", matrices, initial - positions[:, None])
        else:
            with np.load(workload.link.stress_options.mesh) as mesh:
                faces = mesh["tetrahedra"]
                material = mesh["vertices"]
        return cls(
            shell_model,
            faces,
            material,
            np.full(args.envs, -np.inf),
            np.full(args.envs, -1),
            np.zeros(args.envs),
            np.zeros(args.envs, dtype=np.int64),
            np.zeros((args.envs, 3)),
            np.zeros((args.envs, 3)),
        )

    def record(self, workload, step, phase, values):
        selected = np.flatnonzero(values > self.values)
        if not len(selected):
            return
        if self.shell_model:
            primitives = tensor_to_array(workload.observation.locations)
            tensors = tensor_to_array(workload.observation.tensors)
            vertices = tensor_to_array(workload.egg.get_verts_pos())
        else:
            tensor, vm = workload.link.get_stress_field(copy=False)
            tensors = tensor_to_array(tensor)
            primitives = tensor_to_array(vm).reshape(len(values), -1).argmax(axis=1)
            positions = tensor_to_array(workload.egg.get_pos())
            quaternions = tensor_to_array(workload.egg.get_quat())
            rotations = Rotation.from_quat(quaternions[:, [1, 2, 3, 0]])
        for env in selected:
            primitive = int(primitives[env])
            if self.shell_model:
                face, layer = divmod(primitive, 3)
                triangle = vertices[env, self.faces[face]]
                normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
                normal /= np.linalg.norm(normal)
                point = triangle.mean(axis=0) + (layer - 1) * 0.00025 * normal
                material_triangle = self.material_vertices[env, self.faces[face]]
                material_normal = np.cross(
                    material_triangle[1] - material_triangle[0], material_triangle[2] - material_triangle[0]
                )
                material_normal /= np.linalg.norm(material_normal)
                reference_point = material_triangle.mean(axis=0) + (layer - 1) * 0.00025 * material_normal
                self.fields[f"peak_field_env{env}"] = tensors[:, env].copy()
                self.fields[f"peak_vertices_env{env}"] = vertices[env].copy()
            else:
                element, corner = divmod(primitive, 4)
                reference_point = self.material_vertices[self.faces[element, corner]]
                point = rotations[env].apply(reference_point) + positions[env]
                self.fields[f"peak_field_env{env}"] = tensors[env].copy()
                self.fields[f"peak_link_position_env{env}"] = positions[env].copy()
                self.fields[f"peak_link_quaternion_env{env}"] = quaternions[env].copy()
            self.values[env] = values[env]
            self.steps[env] = step
            self.phases[env] = phase[env]
            self.primitive[env] = primitive
            self.world[env] = point
            self.material[env] = reference_point

    def payload(self):
        return {
            "raw_peak_pa": self.values,
            "raw_peak_sample_step": self.steps,
            "raw_peak_time_s": (self.steps + 1) * 0.01,
            "raw_peak_phase": self.phases,
            "raw_peak_primitive": self.primitive,
            "raw_peak_world_position_m": self.world,
            "raw_peak_material_position_m": self.material,
            **self.fields,
        }
