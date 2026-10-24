"""Plot the explicitly scoped CPU recovery cost experiment."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(path):
    result=json.loads(path.read_text())
    rows=result['rows']
    groups=[
        ('Recovery only: common tensor probes',[
            ('coarse_reused_lu_full_samples','coarse LU'),
            ('union_reused_lu_full_samples','joint-refined LU'),
            ('union_response_full_samples','joint-refined response'),
            ('union_response_selected_full_samples','joint-refined selected response'),
            ('atlas_reused_lu_full_samples','atlas LU'),
            ('atlas_response_full_samples','atlas response'),
            ('atlas_response_selected_full_samples','atlas selected response')]),
        ('Current load transfer + recovery',[
            ('union_transfer_plus_lu_full_samples','joint-refined LU'),
            ('union_transfer_plus_response_full_samples','joint-refined response'),
            ('union_transfer_plus_response_selected_full_samples','joint-refined selected response'),
            ('atlas_transfer_plus_lu_full_samples','atlas LU'),
            ('atlas_transfer_plus_response_full_samples','atlas response'),
            ('atlas_transfer_plus_response_selected_full_samples','atlas selected response'),
            ('atlas_transfer_plus_response_vm_max','atlas response + VM max')])]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,(title,methods) in zip(axes,groups):
        for key,label in methods:
            x=np.array([r['batch'] for r in rows])
            y=np.array([r['methods'][key]['median_ms'] for r in rows])
            lo=np.array([r['methods'][key]['range_ms'][0] for r in rows])
            hi=np.array([r['methods'][key]['range_ms'][1] for r in rows])
            ax.errorbar(x,y,yerr=[y-lo,hi-y],marker='o',capsize=3,label=label)
        ax.set(xscale='log',yscale='log',xlabel='batch environments',ylabel='ms per batch',title=title)
        ax.set_xticks([r['batch'] for r in rows],[str(r['batch']) for r in rows])
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Single-thread CPU 3D P2 feature-mesh atlas\n'
                 'Same sampled tensor outputs; different FEM discretizations, not equal-accuracy speedup',fontsize=12)
    target=path.with_name('cpu_cost.png')
    fig.savefig(target,dpi=165)
    plt.close(fig)
    print(target)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('json',type=Path,nargs='?',default=Path(__file__).with_name('results.json'))
    main(p.parse_args().json)
