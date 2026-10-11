"""Offline candidate: retain all plane-pair inputs and test an explicitly shared union of their finite intrinsic supports.

This changes the support law and is not selected production behavior. Each
constituent disk keeps its original radius. Test both the original individual
wrenches on this union and the complete aggregate force/moment, without clipping.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import create_contacts
from genesis.engine.solvers.rigid.stress.footprint import create_surface_fan_tractions, create_footprints, kernel_surface_fan_profiles, kernel_footprint_ports
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.paths import StressSurfacePaths
from genesis.engine.solvers.rigid.stress.provenance import create_sources
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.engine.solvers.rigid.stress.traction import create_traction_info, create_traction_samples, create_tractions, kernel_fit_tractions, kernel_traction_samples, StressTractionInfo
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V_VEC
from genesis.utils.misc import qd_to_numpy
from diagnose_native_surface_fans_v11 import cone_lp
from traction_union_candidate_v1 import kernel_fit_tractions as kernel_fit_union_tractions

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
root = Path(__file__).parent
gs.init(backend=gs.cpu, precision='64', logging_level='warning')
model = StressModel(RigidStressOptions(mesh=Path('examples/rigid/assets/hollow_egg/level1/elastic.npz'), method='direct'))
surface = StressSurface(10, model.info)
paths = StressSurfacePaths(model.info, 8)
info = create_traction_info(surface.info)
contacts, sources = create_contacts(1, 1, 0.001), create_sources(1, 1)
footprints = create_footprints(paths.info.positions.shape[0], 1, 1)
fans = create_surface_fan_tractions(surface.info, 1, 1)
samples = create_traction_samples(fans)
contacts.valid.fill(True)
sources.status.fill(0)
records = []
for run, tick, env, failed_slot in [('online-square-fan-v1', 710, 5, 0), ('warp-candidate-online-v1', 1160, 2, 1)]:
    with np.load(root / run / 'results/failure' / f'snapshot-{tick}.npz') as saved:
        original = {name: saved[name] for name in saved.files}
    pair = int(original['pair'][env, failed_slot])
    selected = np.flatnonzero(original['valid'][env] & (original['pair'][env] == pair))
    assert len(selected) > 0
    record = {'run': run, 'tick': tick, 'env': env, 'pair': pair, 'original_slots': selected.tolist(),
        'law': 'candidate explicit shared union of finite intrinsic disks; no production selection',
        'original_inputs': [], 'fit': []}
    positions, normals, profiles, faces = [], [], [], []
    for slot in selected:
        point, anchor, force = (original[key][env, slot] for key in ('position', 'anchor', 'force'))
        radius, friction = (float(original[key][env, slot]) for key in ('radius', 'friction'))
        anchor_face = int(original['anchor_face'][env, slot])
        assert anchor_face >= 0 and original['source_status'][env, slot] == 0
        contacts.position.from_numpy(point[None, None])
        contacts.center.from_numpy(anchor[None, None])
        contacts.force.from_numpy(force[None, None])
        contacts.radius.fill(radius)
        contacts.friction.fill(friction)
        contacts.anchor_face.fill(anchor_face)
        sources.anchor_moment.from_numpy(np.cross(point-anchor, force)[None, None])
        kernel_footprint_ports(contacts, sources, footprints, surface.info, paths.info)
        kernel_surface_fan_profiles(contacts, footprints, fans, model.info, surface.info, paths.info, info)
        kernel_traction_samples(samples, fans, surface.info, info)
        count = int(qd_to_numpy(fans.sample_count, transpose=True).item())
        profile = qd_to_numpy(fans.profile, transpose=True).reshape(-1)[:count]
        eligible = profile > 0
        positions.append(qd_to_numpy(samples.positions, transpose=True).reshape(-1, 3)[:count][eligible])
        normals.append(qd_to_numpy(samples.normals, transpose=True).reshape(-1, 3)[:count][eligible])
        profiles.append(profile[eligible])
        assert np.all(np.linalg.norm(positions[-1]-anchor, axis=1) <= radius*(1+64*np.finfo(gs.np_float).eps))
        faces.append(qd_to_numpy(samples.faces, transpose=True).reshape(-1)[:count][eligible])
        record['original_inputs'].append({'slot': int(slot), 'position': point.tolist(), 'anchor': anchor.tolist(),
            'force': force.tolist(), 'radius_m': radius, 'friction': friction, 'samples': int(eligible.sum()),
            'normal': original['normal'][env, slot].tolist(), 'producer_method': int(original['method'][env, slot])})
    positions, normals, profiles, faces = (np.concatenate(values) for values in (positions, normals, profiles, faces))
    shared_info = StressTractionInfo(positions=V_VEC(3, dtype=gs.qd_float, shape=(len(positions),)),
        inward_normals=V_VEC(3, dtype=gs.qd_float, shape=(len(normals),)))
    shared_info.positions.from_numpy(positions)
    shared_info.inward_normals.from_numpy(normals)
    shared = create_tractions(len(positions), 1, 1)
    shared.profile.from_numpy(profiles[:, None, None])
    anchor = original['anchor'][env, failed_slot]
    radius = float(original['radius'][env, failed_slot])
    aggregate_force = original['force'][env, selected].sum(axis=0)
    aggregate_moment = np.cross(original['position'][env, selected]-anchor, original['force'][env, selected]).sum(axis=0)
    frictions = original['friction'][env, selected]
    assert np.all(frictions == frictions[0]), frictions
    cases = [(f'original-slot-{int(slot)}', original['force'][env, slot],
        np.cross(original['position'][env, slot]-anchor, original['force'][env, slot])) for slot in selected]
    cases.append(('aggregate', aggregate_force, aggregate_moment))
    witnesses = {'positions': positions, 'normals': normals, 'profiles': profiles, 'sample_faces': faces,
        'anchor': anchor, 'original_inputs': selected, 'aggregate_force': aggregate_force, 'aggregate_moment': aggregate_moment}
    for label, force, moment in cases:
        friction = float(frictions[0])
        scale = max(np.linalg.norm(force), np.linalg.norm(moment)/radius)
        if scale == 0:
            record['fit'].append({'case': label, 'zero_wrench': True})
            continue
        target = np.r_[force, moment/radius]/scale
        fit = {'case': label, 'target': target.tolist(), 'lp': [], 'native': []}
        relative = (positions-anchor)/radius
        for sides in (32, 128):
            result, fitted = cone_lp(relative, normals, friction, target, sides, False)
            fit['lp'].append(result)
            if fitted is not None:
                witnesses[f'{label}-lp-{sides}'] = fitted
        contacts.center.from_numpy(anchor[None, None])
        contacts.force.from_numpy(force[None, None])
        contacts.radius.fill(radius)
        contacts.friction.fill(friction)
        sources.anchor_moment.from_numpy(moment[None, None])
        for budget in (128, 512):
            kernel_fit_union_tractions(contacts, sources, shared, shared_info, float(np.finfo(gs.np_float).eps), 9e-9, budget)
            fit['native'].append({name: float(qd_to_numpy(getattr(shared, name), transpose=True).item())
                for name in ('status', 'evaluations', 'force_error', 'moment_error')})
            fit['native'][-1]['evaluation_budget'] = budget
            fit['native'][-1]['dual'] = qd_to_numpy(shared.dual, transpose=True).reshape(-1).tolist()
        record['fit'].append(fit)
    np.savez_compressed(args.output / f'{run}-{tick}-support.npz', **witnesses)
    records.append(record)
    print(json.dumps(record), flush=True)
(args.output / 'summary.json').write_text(json.dumps({'scope': __doc__, 'records': records}, indent=2)+'\n')
