"""Bounded CPU cost experiment for the volume feature-mesh atlas.

This benchmarks exact response columns of each already-defined volume FEM,
not response snapshots at selected contact points.  All three force DOFs on
the complete top face are included.  Thus current pressure disks can move
arbitrarily on that face; the model's actual load routine remains in use.

The output is the full symmetric tensor on common physical probes, or its VM
maximum.  It is NOT every continuum point, full quadrature output, robot/GPU
throughput, or an accuracy-matched comparison with a converged continuum.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
from time import perf_counter

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

import numpy as np
import scipy
from threadpoolctl import threadpool_limits, threadpool_info

from volume.model import BoxModel, Contact, make_mesh, HEIGHT
from volume.study import cases as accuracy_cases, common_probes as accuracy_probes


ANCHORS = np.array([[-.003, 0.], [0., 0.], [.003, 0.]])


def json_default(value):
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    raise TypeError(type(value).__name__)


def common_probes(bank='accuracy'):
    """A fixed physical volume-probe bank shared by all compared models."""
    if bank=='accuracy': return accuracy_probes()[0]
    global_grid = np.meshgrid(np.linspace(-.009, .009, 7),
                              np.linspace(-.009, .009, 7),
                              [.001, .005, .009, HEIGHT-.0003], indexing='ij')
    local_grid = np.meshgrid(np.linspace(-.006, .006, 25),
                             np.linspace(-.0024, .0024, 9),
                             [HEIGHT-.0003, HEIGHT-.0012], indexing='ij')
    return np.unique(np.concatenate([np.stack(global_grid).reshape(3, -1),
                                     np.stack(local_grid).reshape(3, -1)], axis=1), axis=1)


def contacts_for_batch(batch, bank='accuracy'):
    """Deterministic asymmetric moving positions, radius and vector loads."""
    if bank=='accuracy':
        source=accuracy_cases()
        return [[Contact(c.center,c.radius,c.force) for c in source[i%len(source)][1]]
                for i in range(batch)]
    bank = []
    for i in range(batch):
        phase = .47*i+.19
        x1 = .0038*np.sin(phase)
        x2 = -.0037*np.cos(.71*i+.33)
        points = [(x1, .00065*np.cos(.39*i)),
                  (x2, -.00055*np.sin(.61*i+.1))]
        env = []
        for k, center in enumerate(points):
            fn = .05*(1+.22*np.sin(.29*i+.6*k))
            radius = .0012*(fn/.05)**(1/3)
            direction = .41*i+1.1*k
            force = (.23*fn*np.cos(direction), .23*fn*np.sin(direction), -fn)
            env.append(Contact(tuple(center), radius, force))
        bank.append(env)
    return bank


def max_vm(field):
    b = field.shape[1]
    s = field.reshape(-1, 6, b)
    vm2 = .5*((s[:, 0]-s[:, 1])**2+(s[:, 1]-s[:, 2])**2+
              (s[:, 2]-s[:, 0])**2)+3*np.sum(s[:, 3:]**2, axis=1)
    return np.sqrt(np.maximum(vm2, 0.)).max(axis=0)


def relative(a, b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-30))


def field_relative(a,b):
    """Same unweighted-probe Frobenius metric as the accuracy suite."""
    d=(a-b).reshape(-1,6,a.shape[1]);v=b.reshape(-1,6,b.shape[1])
    weights=np.array([1.,1.,1.,2.,2.,2.])[None,:,None]
    return float(np.sqrt(np.sum(weights*d*d)/max(np.sum(weights*v*v),1e-30)))


def field_relative_by_environment(a,b):
    d=(a-b).reshape(-1,6,a.shape[1]);v=b.reshape(-1,6,b.shape[1])
    weights=np.array([1.,1.,1.,2.,2.,2.])[None,:,None]
    return np.sqrt(np.sum(weights*d*d,axis=(0,1))/
                   np.maximum(np.sum(weights*v*v,axis=(0,1)),1e-30)).tolist()


def make_loads(models, contacts, atlas, order):
    """Produce current FE nodal loads, grouping by model and active environment."""
    buckets = {}
    info = []
    for env, sources in enumerate(contacts):
        for c in sources:
            key = int(np.argmin(np.linalg.norm(ANCHORS-np.asarray(c.center), axis=1))) if atlas else 0
            f, report = models[key].load(c, order=order)
            info.append(report)
            bucket = buckets.setdefault(key, {})
            bucket[env] = bucket.get(env, 0.) + f
    loads = {}
    for key, bucket in buckets.items():
        indices = np.array(sorted(bucket), dtype=int)
        loads[key] = (indices, np.column_stack([bucket[int(i)] for i in indices]))
    return loads, dict(sources=len(info),
        force_error=max((r['force_relative'] for r in info),default=0.),
        moment_error=max((r['moment_normalized'] for r in info),default=0.),
        active_environment_counts={str(k):len(v[0]) for k,v in loads.items()})


class SurfaceResponse:
    def __init__(self, model, stress, block=48):
        begin = perf_counter()
        # ElementVector P2 uses collocated, interleaved vector components;
        # basis.doflocs identifies all three DOFs without assuming node order.
        self.columns = np.flatnonzero(np.isclose(model.basis.doflocs[2], HEIGHT,
                                                 atol=1e-14, rtol=0.))
        self.matrix = np.empty((stress.shape[0], len(self.columns)))
        for start in range(0, len(self.columns), block):
            c = self.columns[start:start+block]
            force = np.zeros((model.N, len(c)))
            force[c, np.arange(len(c))] = 1.
            self.matrix[:, start:start+len(c)] = stress @ model.solve(force)
        self.build_seconds = perf_counter()-begin
        self.excluded = np.setdiff1d(np.arange(model.N), self.columns)

    def apply(self, force):
        return self.matrix @ force[self.columns]

    def apply_selected(self, force):
        surface = force[self.columns]
        active = np.flatnonzero(np.any(surface != 0., axis=1))
        if not len(active):
            return np.zeros((self.matrix.shape[0], force.shape[1]))
        # Exact zero-work pruning; there is no threshold or learned sparsity.
        return self.matrix[:, active] @ surface[active]

    def outside_support(self, force):
        return float(np.linalg.norm(force[self.excluded]) / max(np.linalg.norm(force), 1e-30))


def apply_lu(models, maps, loads, batch):
    field = np.zeros((maps[0].shape[0], batch))
    for key, (env, force) in loads.items():
        field[:, env] += maps[key] @ models[key].solve(force)
    return field


def apply_response(responses, loads, batch, selected=False):
    field = np.zeros((responses[0].matrix.shape[0], batch))
    for key, (env, force) in loads.items():
        field[:, env] += (responses[key].apply_selected(force) if selected
                         else responses[key].apply(force))
    return field


def measure(methods, repeats):
    """Warm all methods, then reverse the order on alternate repeats."""
    for method in methods.values():
        method()
    samples = {name: [] for name in methods}
    orders = []
    names = list(methods)
    for rep in range(repeats):
        order = names if rep % 2 == 0 else list(reversed(names))
        orders.append(order)
        for name in order:
            begin = perf_counter()
            methods[name]()
            samples[name].append(1000*(perf_counter()-begin))
    return {name: dict(median_ms=float(np.median(vals)),
                       range_ms=[float(min(vals)), float(max(vals))],
                       repeats_ms=vals) for name, vals in samples.items()}, orders


def run(args):
    source_paths = [Path(__file__), PACKAGE/'volume/model.py',PACKAGE/'volume/study.py']
    hashes = {str(p.relative_to(PACKAGE)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in source_paths}
    begin = perf_counter()
    models = []
    for i, anchor in enumerate(ANCHORS):
        mesh, _ = make_mesh([anchor], hmin=args.hmin, core=args.core)
        models.append(BoxModel(mesh, name=f'atlas_{i}'))
    coarse = BoxModel(make_mesh()[0], name='coarse')
    union = BoxModel(make_mesh(ANCHORS, hmin=args.hmin, core=args.core)[0], name='union')
    setup = perf_counter()-begin
    points = common_probes(args.probe_bank)
    maps = [m.stress_map(points) for m in models]
    coarse_map, union_map = coarse.stress_map(points), union.stress_map(points)
    responses = [SurfaceResponse(m, s, block=args.block) for m, s in zip(models, maps)]
    union_response=SurfaceResponse(union,union_map,block=args.block)
    released,_=make_loads(models,[[],[],[]],True,args.order)
    release_field=apply_response(responses,released,3)
    release_checks=dict(no_load_buckets=not released,
                        exact_zero_field=bool(np.array_equal(release_field,np.zeros_like(release_field))),
                        exact_zero_vm=bool(np.array_equal(max_vm(release_field),np.zeros(3))))
    cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
              if line.startswith('model name')),platform.processor())
    result = dict(scope='Single-thread CPU 3D P2 FEM recovery at a fixed bank of common physical probes. Exact top-boundary nodal-response columns of each model; no point-response snapshots. No GPU/robot throughput, full continuum maximum or equal-accuracy claim.',
        cpu=cpu,
        platform=platform.platform(), python=platform.python_version(), numpy=np.__version__,
        scipy=scipy.__version__, threads=1, threadpools=threadpool_info(),
        source_sha256=hashes, hmin_m=args.hmin, core_m=args.core,
        pressure='Finite Hertz-profile disk, true position/radius/force integrated every update. These are prescribed tractions; independent physical Hertz contact has not been solved.',
        probe_bank=args.probe_bank,query_bank=args.query_bank,
        query_bank_note='accuracy: exact 16 accuracy-suite cases, tiled in original order when B>16. smoke: synthetic smooth asymmetric source bank.',
        probes=points.shape[1], output_components='xx,yy,zz,xy,xz,yz at each physical probe; shear unscaled',
        error_metric='Unweighted-probe relative Frobenius tensor norm (shears weighted twice), matching the volume accuracy suite. Not volume-weighted L2. Maximum is over declared samples only.',
        setup_seconds=setup,
        release_checks=release_checks,
        atlas=[dict(model=m.storage(), response_columns=len(r.columns),
                    response_bytes=r.matrix.nbytes, response_build_seconds=r.build_seconds)
               for m,r in zip(models,responses)],
        coarse=coarse.storage(), union=union.storage(),
        union_response=dict(response_columns=len(union_response.columns),
                            response_bytes=union_response.matrix.nbytes,
                            response_build_seconds=union_response.build_seconds),rows=[])
    for batch in args.batches:
        contacts=contacts_for_batch(batch,args.query_bank)
        atlas_loads, admission=make_loads(models,contacts,True,args.order)
        coarse_loads,_=make_loads([coarse],contacts,False,args.order)
        union_loads,_=make_loads([union],contacts,False,args.order)
        field_lu=apply_lu(models,maps,atlas_loads,batch)
        field_response=apply_response(responses,atlas_loads,batch)
        field_selected=apply_response(responses,atlas_loads,batch,selected=True)
        union_field=apply_lu([union],[union_map],union_loads,batch)
        union_response_field=apply_response([union_response],union_loads,batch)
        union_selected_field=apply_response([union_response],union_loads,batch,selected=True)
        coarse_field=apply_lu([coarse],[coarse_map],coarse_loads,batch)
        errors=dict(response_vs_atlas_lu_full_sample_field=field_relative(field_response,field_lu),
            response_vs_atlas_lu_vm_max=relative(max_vm(field_response),max_vm(field_lu)),
            selected_vs_atlas_lu_full_sample_field=field_relative(field_selected,field_lu),
            response_vs_atlas_lu_by_environment=field_relative_by_environment(field_response,field_lu),
            union_response_vs_union_lu=field_relative(union_response_field,union_field),
            union_selected_vs_union_lu=field_relative(union_selected_field,union_field),
            atlas_vs_union_full_sample_field=field_relative(field_lu,union_field),
            coarse_vs_union_full_sample_field=field_relative(coarse_field,union_field),
            max_excluded_load_relative=max(responses[k].outside_support(f) for k,(_,f) in atlas_loads.items()))
        def atlas_transfer(): return make_loads(models,contacts,True,args.order)[0]
        def union_transfer(): return make_loads([union],contacts,False,args.order)[0]
        methods={
            'atlas_reused_lu_full_samples':lambda:apply_lu(models,maps,atlas_loads,batch),
            'atlas_response_full_samples':lambda:apply_response(responses,atlas_loads,batch),
            'atlas_response_selected_full_samples':lambda:apply_response(responses,atlas_loads,batch,selected=True),
            'union_reused_lu_full_samples':lambda:apply_lu([union],[union_map],union_loads,batch),
            'union_response_full_samples':lambda:apply_response([union_response],union_loads,batch),
            'union_response_selected_full_samples':lambda:apply_response([union_response],union_loads,batch,selected=True),
            'coarse_reused_lu_full_samples':lambda:apply_lu([coarse],[coarse_map],coarse_loads,batch),
            'atlas_current_transfer':atlas_transfer,
            'atlas_transfer_plus_lu_full_samples':lambda:apply_lu(models,maps,atlas_transfer(),batch),
            'atlas_transfer_plus_response_full_samples':lambda:apply_response(responses,atlas_transfer(),batch),
            'atlas_transfer_plus_response_selected_full_samples':lambda:apply_response(responses,atlas_transfer(),batch,selected=True),
            'atlas_transfer_plus_response_vm_max':lambda:max_vm(apply_response(responses,atlas_transfer(),batch)),
            'union_transfer_plus_lu_full_samples':lambda:apply_lu([union],[union_map],union_transfer(),batch),
            'union_transfer_plus_response_full_samples':lambda:apply_response([union_response],union_transfer(),batch),
            'union_transfer_plus_response_selected_full_samples':lambda:apply_response([union_response],union_transfer(),batch,selected=True),
        }
        timings,orders=measure(methods,args.repeats)
        passed=(errors['response_vs_atlas_lu_full_sample_field']<1e-9
                and errors['response_vs_atlas_lu_vm_max']<1e-9
                and errors['selected_vs_atlas_lu_full_sample_field']<1e-9
                and errors['union_response_vs_union_lu']<1e-9
                and errors['union_selected_vs_union_lu']<1e-9
                and errors['max_excluded_load_relative']<1e-12
                and admission['force_error']<1e-10 and admission['moment_error']<1e-9)
        active_columns={str(k):int(np.any(f[responses[k].columns]!=0.,axis=1).sum())
                        for k,(_,f) in atlas_loads.items()}
        row=dict(batch=batch,admission=admission,errors=errors,passed=passed,
                 active_surface_input_columns=active_columns,methods=timings,interleaved_orders=orders)
        result['rows'].append(row)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2,default=json_default))
        print(json.dumps(dict(batch=batch,errors=errors,passed=passed,median_ms={k:v['median_ms'] for k,v in timings.items()})),flush=True)
    result['source_unchanged_during_run']=all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p.relative_to(PACKAGE))] for p in source_paths)
    result['all_response_checks_passed']=(all(r['passed'] for r in result['rows'])
                                        and all(release_checks.values())
                                        and result['source_unchanged_during_run'])
    result['process_peak_rss_bytes']=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    args.output.write_text(json.dumps(result,indent=2,default=json_default))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hmin',type=float,default=.0011)
    parser.add_argument('--core',type=float,default=.0022)
    parser.add_argument('--order',type=int,default=24)
    parser.add_argument('--block',type=int,default=48)
    parser.add_argument('--batches',type=int,nargs='+',default=[1,16,64])
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--probe-bank',choices=['accuracy','smoke'],default='accuracy')
    parser.add_argument('--query-bank',choices=['accuracy','smoke'],default='accuracy')
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'results.json')
    with threadpool_limits(limits=1):
        run(parser.parse_args())
