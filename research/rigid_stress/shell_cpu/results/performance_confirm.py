"""Bounded interleaved confirmation after other long CPU jobs finished.

Run from a Genesis checkout containing research/rigid_stress/shell_cpu:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=research/rigid_stress \
    python -m shell_cpu.results.performance_confirm --output /tmp/shell-confirmation.json
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys
import time

PACKAGE_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE_ROOT.parent))
import numpy as np
import scipy
from threadpoolctl import threadpool_limits,threadpool_info
from shell_cpu.core import ShellModel
from shell_cpu.geometry import egg_mesh
from shell_cpu.contacts import ShellDPRTransfer
from shell_cpu.performance import (_contact_bank,_local_vm,_surface_force_operator,
    _global_tensor_basis,_to_global_shell_field,batched_witness_transfer)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('shell_cpu_confirmation.json'))
    args=parser.parse_args()
    source_paths = [PACKAGE_ROOT/name for name in ('core.py','geometry.py','contacts.py','performance.py')]
    hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    with threadpool_limits(limits=1):
        vertices,quads=egg_mesh(4)
        model=ShellModel(vertices,quads,E=1e10,nu=.3,h=.0005)
        model.factorize()
        stress=model.stress_matrix('local').tocsr()
        tensor_basis=_global_tensor_basis(model)
        build_start=time.perf_counter()
        surface_response=stress@model.solve(_surface_force_operator(model,side=1))
        response_build_ms=1000*(time.perf_counter()-build_start)
        contacts=_contact_bank(model,128)
        transfer=ShellDPRTransfer(model)
        loads=np.column_stack([transfer.transfer(c,method='fast')[0] for c in contacts])
        rows=[]
        for batch in (1,128):
            cases=contacts[:batch]
            f=loads[:,:batch]
            sf=f.reshape((-1,6,batch))[:,:3].reshape((-1,batch))
            def scalar_transfer():
                return np.column_stack([transfer.transfer(c,method='fast')[0] for c in cases])
            def batch_transfer():
                return batched_witness_transfer(transfer,cases)[0]
            def batch_surface_full():
                ff=batch_transfer().reshape((-1,6,batch))[:,:3].reshape((-1,batch))
                return surface_response@ff
            methods={
                'reused_lu_full_local':lambda:stress@model.solve(f),
                'column_loop_lu_full_local':lambda:np.column_stack([stress@model.solve(f[:,i]) for i in range(batch)]),
                'compact_surface_full_local':lambda:surface_response@sf,
                'scalar_hinted_fast_transfer':scalar_transfer,
                'batched_witness_fast_transfer':batch_transfer,
                'batched_transfer_plus_lu_full_local':lambda:stress@model.solve(batch_transfer()),
                'batched_transfer_plus_surface_full_local':batch_surface_full,
                'batched_transfer_plus_surface_peak':lambda:np.max(_local_vm(batch_surface_full()),axis=0),
                'reused_lu_full_global_tensor':lambda:_to_global_shell_field(stress@model.solve(f),tensor_basis),
                'compact_surface_full_global_tensor':lambda:_to_global_shell_field(surface_response@sf,tensor_basis),
            }
            for fun in methods.values():fun()
            timing={name:[] for name in methods}
            order=[]
            inner=5 if batch==1 else 1
            for repeat in range(7):
                names=list(methods)
                if repeat%2:names.reverse()
                order.append(names)
                for name in names:
                    start=time.perf_counter()
                    for _ in range(inner):methods[name]()
                    timing[name].append(1000*(time.perf_counter()-start)/inner)
            direct=stress@model.solve(f)
            compact=surface_response@sf
            fast,info=batched_witness_transfer(transfer,cases)
            fast_field=stress@model.solve(fast)
            error={
                'compact_relative_field_error':float(np.linalg.norm(compact-direct)/np.linalg.norm(direct)),
                'batched_relative_load_error':float(np.linalg.norm(fast-f)/np.linalg.norm(f)),
                'batched_relative_field_error':float(np.linalg.norm(fast_field-direct)/np.linalg.norm(direct)),
                'compact_relative_peak_error':float(np.linalg.norm(np.max(_local_vm(compact),axis=0)-np.max(_local_vm(direct),axis=0))/np.linalg.norm(np.max(_local_vm(direct),axis=0))),
            }
            row={'batch':batch,'interleaved_orders':order,'inner':inner,'repeats':7,
                 'transfer_info':info,'errors':error,'passed':max(error.values())<1e-8,
                 'methods':{name:{'median_ms':float(np.median(times)),'range_ms':[min(times),max(times)],'repeats_ms':times}
                            for name,times in timing.items()}}
            rows.append(row)
        report={
            'scope':'Bounded single-thread CPU recovery confirmation, not Genesis/GPU/RL throughput; full local output means top/bottom in-plane tensors at four points per element',
            'reason':'Original broad scan overlapped other heavy CPU tasks; this confirmation interleaves reverse orders after those tasks finished',
            'cpu':next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),platform.processor()),
            'ndof':model.ndof,'quads':len(quads),'young_pa':model.E,'poisson':model.nu,'thickness_m':model.h,
            'numpy':np.__version__,'scipy':scipy.__version__,'threads':1,'threadpools':threadpool_info(),
            'source_sha256':hashes,'source_unchanged_during_run':all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[p.name] for p in source_paths),
            'compact_response_build_ms':response_build_ms,'compact_response_bytes':surface_response.nbytes,
            'all_checks_passed':all(row['passed'] for row in rows),'rows':rows,
        }
        output=args.output
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,indent=2))
        for row in rows:
            print('B',row['batch'],'errors',row['errors'])
            for name,item in row['methods'].items():print(name,round(item['median_ms'],6),[round(v,6) for v in item['range_ms']])


if __name__=='__main__':main()
