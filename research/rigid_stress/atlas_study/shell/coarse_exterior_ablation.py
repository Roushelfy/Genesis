"""Bounded one-case check: improve shared coarse exterior, keep patch step."""
from pathlib import Path
import json,time,hashlib
import numpy as np
from threadpoolctl import threadpool_limits
from plate_atlas import Plate,refined_knots,knots,measure,select,ANCHORS,HERE

def main():
    root=HERE/'results';d=json.loads((root/'results.json').read_text());field=np.load(root/'fields.npz')
    row=next(r for r in d['rows'] if r['case']=='two_far');cs=row['contacts'];probes=field['probes'];ref=field['two_far_reference']
    coarse=Plate('coarse_2p5mm',knots(-.02,.02,.0025),knots(-.02,.02,.0025),probes)
    f=sum(coarse.load(c)[0] for c in cs);base,_=coarse.field(f)
    atlas=np.zeros_like(ref);models=[coarse.metadata()];gates=[]
    for i in sorted(set(select(c) for c in cs)):
        m=Plate(f'atlas{i}_exterior_2p5mm',refined_knots(ANCHORS[i],coarse=.0025),refined_knots(0,.006,coarse=.0025),probes)
        loads=[]
        for c in cs:
            if select(c)==i:
                load,info=m.load(c);loads.append(load);gates.append(info)
        s,_=m.field(sum(loads));atlas+=s;models.append(m.metadata())
    result=dict(scope='One bounded shared-coarse-exterior ablation, no new reference. Exact same two-far sources/probes as primary shell plate experiment; patch step remains1mm.',
       old_primary=row['metrics'],models=models,gates=gates,coarse_2p5mm=measure(base,ref),atlas_exterior_2p5mm=measure(atlas,ref),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/'coarse_exterior_ablation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
