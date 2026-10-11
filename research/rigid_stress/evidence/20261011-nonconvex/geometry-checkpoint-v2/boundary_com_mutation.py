"""Deliberately restore the invalid COM normal rule to check regression sensitivity."""
import quadrants as qd
import genesis as gs
from genesis.engine.solvers.rigid.stress.data import StressInfo
from genesis.engine.solvers.rigid.stress.surface import StressSurface, StressSurfaceInfo

@qd.kernel
def kernel_com_normals(stress_info: StressInfo, surface_info: StressSurfaceInfo):
    for i_f in range(surface_info.face_normal.shape[0]):
        x = surface_info.face_origin[i_f]
        edges = surface_info.face_edges[i_f]
        normal = edges[0, :].cross(edges[1, :]).normalized()
        com = qd.Vector([stress_info.mass_properties[1 + a] for a in qd.static(range(3))])
        com /= stress_info.mass_properties[0]
        if normal.dot(x + (edges[0, :] + edges[1, :]) / 3.0 - com) < 0.0:
            normal = -normal
        surface_info.face_normal[i_f] = normal

original_init = StressSurface.__init__

def mutated_init(self, quadrature, stress_info):
    original_init(self, quadrature, stress_info)
    kernel_com_normals(stress_info, self.info)

StressSurface.__init__ = mutated_init
