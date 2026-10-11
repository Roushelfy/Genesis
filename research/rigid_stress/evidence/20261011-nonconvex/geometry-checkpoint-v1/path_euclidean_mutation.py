"""Replace intrinsic distance by a Euclidean radius while retaining path diagnostics."""
import quadrants as qd
from genesis.engine.solvers.rigid.stress import paths
from genesis.engine.solvers.rigid.stress.data import StressInfo
from genesis.engine.solvers.rigid.stress.paths import StressPathInfo
from genesis.engine.solvers.rigid.stress.surface import StressSurfaceInfo

native_surface_distance = paths.func_surface_distance

@qd.func
def mutated_surface_distance(
    i_anchor: int,
    i_target: int,
    anchor: qd.types.vector(3),
    target: qd.types.vector(3),
    stress_info: StressInfo,
    surface_info: StressSurfaceInfo,
    path_info: StressPathInfo,
):
    distance, straight = native_surface_distance(i_anchor, i_target, anchor, target, stress_info, surface_info, path_info)
    return (target - anchor).norm(), straight

paths.func_surface_distance = mutated_surface_distance
