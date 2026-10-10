"""Native positive quadrature of an affine exterior P2 surface."""

from dataclasses import dataclass
from typing import ClassVar

import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V_MAT, V_VEC, DataKind, V

from .data import StressInfo


@dataclass(frozen=True)
class StressSurfaceInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    gauss: qd.Tensor
    gauss_weights: qd.Tensor
    positions: qd.Tensor
    weights: qd.Tensor
    shape: qd.Tensor
    face_origin: qd.Tensor
    face_dual: qd.Tensor
    face_normal: qd.Tensor


class StressSurface:
    def __init__(self, quadrature: int, stress_info: StressInfo):
        n_faces = stress_info.surface_nodes.shape[0]
        n_quadrature = quadrature * quadrature
        self.info = StressSurfaceInfo(
            gauss=V(dtype=gs.qd_float, shape=(quadrature,)),
            gauss_weights=V(dtype=gs.qd_float, shape=(quadrature,)),
            positions=V_VEC(3, dtype=gs.qd_float, shape=(n_faces * n_quadrature,)),
            weights=V(dtype=gs.qd_float, shape=(n_faces * n_quadrature,)),
            shape=V_VEC(6, dtype=gs.qd_float, shape=(n_quadrature,)),
            face_origin=V_VEC(3, dtype=gs.qd_float, shape=(n_faces,)),
            face_dual=V_MAT(2, 3, dtype=gs.qd_float, shape=(n_faces,)),
            face_normal=V_VEC(3, dtype=gs.qd_float, shape=(n_faces,)),
        )
        kernel_gauss(self.info)
        kernel_surface(stress_info, self.info)


@qd.kernel
def kernel_gauss(surface_info: StressSurfaceInfo):
    for i in range(surface_info.gauss.shape[0]):
        n = surface_info.gauss.shape[0]
        x = -qd.cos(3.141592653589793 * (i + 0.75) / (n + 0.5))
        derivative = gs.qd_float(0.0)
        for _ in range(32):
            previous, value = gs.qd_float(0.0), gs.qd_float(1.0)
            for j in range(n):
                old = value
                value = ((2 * j + 1) * x * value - j * previous) / (j + 1)
                previous = old
            derivative = n * (x * value - previous) / (x * x - 1.0)
            x -= value / derivative
        surface_info.gauss[i] = 0.5 * (x + 1.0)
        surface_info.gauss_weights[i] = 1.0 / ((1.0 - x * x) * derivative * derivative)


@qd.kernel
def kernel_surface(stress_info: StressInfo, surface_info: StressSurfaceInfo):
    for i_f in range(stress_info.surface_nodes.shape[0]):
        i_0 = stress_info.surface_nodes[i_f, 0]
        i_1 = stress_info.surface_nodes[i_f, 1]
        i_2 = stress_info.surface_nodes[i_f, 2]
        x = stress_info.vertices[i_0]
        first, second = stress_info.vertices[i_1] - x, stress_info.vertices[i_2] - x
        gram = qd.Matrix([[first.dot(first), first.dot(second)], [second.dot(first), second.dot(second)]])
        surface_info.face_origin[i_f] = x
        surface_info.face_dual[i_f] = gram.inverse() @ qd.Matrix.rows([first, second])
        normal = first.cross(second).normalized()
        com = qd.Vector([stress_info.mass_properties[1 + i_a] for i_a in qd.static(range(3))])
        com /= stress_info.mass_properties[0]
        if normal.dot(x + (first + second) / 3.0 - com) < 0.0:
            normal = -normal
        surface_info.face_normal[i_f] = normal
        for i_q in range(surface_info.shape.shape[0]):
            n = surface_info.gauss.shape[0]
            u, v = surface_info.gauss[i_q // n], surface_info.gauss[i_q % n]
            bary = qd.Vector([1.0 - u, u * (1.0 - v), u * v])
            i_sample = i_f * surface_info.shape.shape[0] + i_q
            surface_info.positions[i_sample] = x + bary[1] * first + bary[2] * second
            surface_info.weights[i_sample] = (
                first.cross(second).norm()
                * u
                * surface_info.gauss_weights[i_q // n]
                * surface_info.gauss_weights[i_q % n]
            )
    for i_q in range(surface_info.shape.shape[0]):
        n = surface_info.gauss.shape[0]
        u, v = surface_info.gauss[i_q // n], surface_info.gauss[i_q % n]
        bary = qd.Vector([1.0 - u, u * (1.0 - v), u * v])
        surface_info.shape[i_q] = qd.Vector(
            [
                bary[0] * (2.0 * bary[0] - 1.0),
                bary[1] * (2.0 * bary[1] - 1.0),
                bary[2] * (2.0 * bary[2] - 1.0),
                4.0 * bary[0] * bary[1],
                4.0 * bary[0] * bary[2],
                4.0 * bary[1] * bary[2],
            ]
        )
