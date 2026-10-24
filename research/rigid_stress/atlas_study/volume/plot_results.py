from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .model import vm
ROOT=Path(__file__).resolve().parent

def run():
    p=ROOT/'results_bounded';complete=ROOT/'refinement_complete.json'
    d=json.loads((complete if complete.exists() else ROOT/'refined_reference/results.json').read_text());rows=d['rows'];x=np.arange(len(rows))
    fig,ax=plt.subplots(figsize=(12,4.5),layout='constrained')
    for key,label in [('coarse','coarse'),('hard_atlas','feature-mesh sum'),('blended_atlas','blended feature-mesh sum'),('union','joint local refinement')]:
        ax.plot(x,[100*r[key]['full_sampled_tensor_relative'] for r in rows],'o-',label=label,ms=4)
    ax.set(xticks=x,xticklabels=[r['case'] for r in rows],ylabel='common tensor-sample relative difference (%)',title=f'3D P2 finite-pressure control: {len(rows)} comparisons vs h=0.75 mm union mesh')
    ax.tick_params(axis='x',labelrotation=55,labelsize=8);ax.grid(alpha=.2);ax.legend();fig.savefig(ROOT/'volume_accuracy.png',dpi=160);plt.close(fig)
    a=np.load(p/'two_far.npz');n=847
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    finer=np.load(ROOT/'refined_reference/two_far.npz')
    data=[vm(a[k][6*n:]).reshape(35,17,5)[:,:,0] for k in ['coarse','hard_atlas']]+[vm(finer['reference'][6*n:]).reshape(35,17,5)[:,:,0]]
    vmax=max(z.max() for z in data)/1000
    for ax,z,label in zip(axes,data,['Coarse','Feature-mesh sum','Finer union reference']):
        im=ax.imshow(z.T/1000,origin='lower',extent=[-5,5,-3,3],vmin=0,vmax=vmax,cmap='inferno',aspect='equal');ax.set(title=label,xlabel='x (mm)',ylabel='y (mm)')
    fig.colorbar(im,ax=axes,shrink=.7,label='sampled VM (kPa)');fig.suptitle('Two asymmetric loads, depth 50 micrometres; reference remains finite-grid')
    fig.savefig(ROOT/'volume_fields.png',dpi=160);plt.close(fig)
if __name__=='__main__':run()
