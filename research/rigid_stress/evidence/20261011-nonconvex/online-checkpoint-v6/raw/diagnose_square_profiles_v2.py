"""Locate native/independent CPU profile differences without relaxing acceptance."""
import argparse
import json
from pathlib import Path

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import create_contacts, StressContactState
from genesis.engine.solvers.rigid.stress.data import StressInfo
from genesis.engine.solvers.rigid.stress.footprint import (
    create_footprints, create_surface_fan_tractions, kernel_footprint_ports,
    kernel_surface_fan_profiles, func_footprint_point_distance, StressFootprintState)
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.paths import StressSurfacePaths, StressPathInfo
from genesis.engine.solvers.rigid.stress.provenance import create_sources, kernel_resolve_sources
from genesis.engine.solvers.rigid.stress.surface import StressSurface, StressSurfaceInfo
from genesis.engine.solvers.rigid.stress.traction import (
    create_traction_info, create_traction_samples, kernel_traction_samples, func_traction_geometry, StressTractionState, StressTractionInfo)
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V
from genesis.utils.misc import qd_to_numpy
from tests.rigid.test_stress_footprint import cpu_intrinsic_distances
from tests.rigid.test_stress_surface_fan import cpu_surface_fan_geometry


@qd.kernel
def kernel_queries(distances: qd.Tensor, straight: qd.Tensor, contacts: StressContactState,
                   footprints: StressFootprintState, tractions: StressTractionState, model: StressInfo,
                   surface: StressSurfaceInfo, paths: StressPathInfo, info: StressTractionInfo):
    for i_q in range(distances.shape[0]):
        if i_q < tractions.sample_count[0, 0]:
            position, _ = func_traction_geometry(i_q, 0, 0, tractions, info)
            i_sample = tractions.sample_indices[i_q, 0, 0]
            face = i_sample // (3 * surface.gauss.shape[0] ** 2)
            distance, is_straight = func_footprint_point_distance(
                face, 0, 0, position, contacts, footprints, model, surface, paths)
            distances[i_q] = distance
            straight[i_q] = is_straight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--tick', type=int, default=147)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    gs.init(backend=gs.cpu if args.backend == 'cpu' else gs.cuda, precision='64', logging_level='warning')
    mesh = Path('examples/rigid/assets/hollow_egg/level1/elastic.npz')
    model = StressModel(RigidStressOptions(mesh=mesh, method='direct'))
    surface = StressSurface(10, model.info)
    paths = StressSurfacePaths(model.info, 8)
    info = create_traction_info(surface.info)
    with np.load(Path('tests/rigid/assets') / f'nonconvex_mesh64_tick{args.tick}.npz') as fixture:
        anchor, point, force = (fixture[key] for key in ('anchor', 'position', 'force'))
        radius, friction = (float(fixture[key]) for key in ('radius', 'friction'))
    contacts, sources = create_contacts(1, 1, radius), create_sources(1, 1)
    contacts.valid.fill(True)
    contacts.position.from_numpy(point[None, None])
    contacts.force.from_numpy(force[None, None])
    contacts.friction.fill(friction)
    sources.geom.fill(7)
    sources.pair.fill(11)
    sources.witness.from_numpy(anchor[None, None])
    kernel_resolve_sources(1e-8, float(np.finfo(gs.np_float).eps), contacts, sources, model.info, surface.info)
    footprints = create_footprints(paths.info.positions.shape[0], 1, 1)
    tractions = create_surface_fan_tractions(surface.info, 1, 1, coordinate_power=2)
    kernel_footprint_ports(contacts, sources, footprints, surface.info, paths.info)
    kernel_surface_fan_profiles(contacts, footprints, tractions, model.info, surface.info, paths.info, info)
    samples = create_traction_samples(tractions)
    kernel_traction_samples(samples, tractions, surface.info, info)
    count = int(qd_to_numpy(tractions.sample_count, transpose=True).item())
    indices = qd_to_numpy(tractions.sample_indices, transpose=True).reshape(-1)[:count]
    actual_profile = qd_to_numpy(tractions.profile, transpose=True).reshape(-1)[:count]
    actual_positions = qd_to_numpy(samples.positions, transpose=True).reshape(-1, 3)[:count]
    actual_faces = qd_to_numpy(samples.faces, transpose=True).reshape(-1)[:count]
    actual_anchor = qd_to_numpy(contacts.center, transpose=True).reshape(3)
    anchor_face = qd_to_numpy(contacts.anchor_face, transpose=True).reshape(1)
    distances = V(dtype=gs.qd_float, shape=(count,))
    straight = V(dtype=gs.qd_bool, shape=(count,))
    kernel_queries(distances, straight, contacts, footprints, tractions, model.info, surface.info, paths.info, info)
    actual_distances = qd_to_numpy(distances, transpose=True)
    is_straight = qd_to_numpy(straight, transpose=True)
    with np.load(mesh) as asset:
        vertices, faces = (asset[key] for key in ('vertices', 'surface_triangles'))
    references = {}
    arrays = {'indices': indices, 'actual_profile': actual_profile, 'native_distance': actual_distances,
              'actual_positions': actual_positions, 'actual_anchor': actual_anchor, 'fixture_anchor': anchor,
              'actual_faces': actual_faces, 'native_is_straight': is_straight}
    path_args = (vertices, faces, qd_to_numpy(paths.info.positions, transpose=True),
                 qd_to_numpy(paths.info.face_ports, transpose=True).T, qd_to_numpy(paths.info.starts, transpose=True),
                 qd_to_numpy(paths.info.neighbors, transpose=True))
    for name, source_anchor in [('fixture', anchor), ('resolved', actual_anchor)]:
        positions, weights, cpu_faces, _ = cpu_surface_fan_geometry(source_anchor, vertices, faces, 10, 2)
        distance = cpu_intrinsic_distances(source_anchor[None], anchor_face, positions[indices], cpu_faces[indices], *path_args)[0]
        profile = weights[indices] * np.exp(-0.5 * (distance / (0.45 * radius)) ** 2)
        profile *= np.maximum(1 - (distance / radius) ** 2, 0) ** 2
        arrays[name + '_distance'] = distance
        arrays[name + '_profile'] = profile
        differences = np.abs(actual_profile - profile) > 2e-20 + 2e-11 * np.abs(profile)
        references[name] = {'bad_count': int(differences.sum()), 'max_absolute_profile_difference': float(np.abs(actual_profile-profile).max()),
            'details': [{'index': int(index), 'face': int(face), 'profile': float(a), 'reference': float(b),
                         'native_distance': float(d), 'cpu_distance': float(e), 'native_is_straight': bool(s)}
                        for index, face, a, b, d, e, s in zip(indices[differences], actual_faces[differences], actual_profile[differences],
                                                             profile[differences], actual_distances[differences], distance[differences], is_straight[differences])]}
    native_point_cpu_distance = cpu_intrinsic_distances(actual_anchor[None], anchor_face, actual_positions, actual_faces, *path_args)[0]
    arrays['native_points_cpu_distance'] = native_point_cpu_distance
    references['native_points_max_distance_difference'] = float(np.abs(actual_distances-native_point_cpu_distance).max())
    np.savez_compressed(args.output / 'profiles.npz', **arrays)
    result = {'tick': args.tick, 'backend': args.backend, 'radius': radius, 'anchor_delta_m': float(np.linalg.norm(anchor-actual_anchor)),
              'samples': count, 'references': references}
    (args.output / 'diagnosis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
