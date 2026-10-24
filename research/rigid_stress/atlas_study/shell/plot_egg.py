"""Render retained closed-egg supplementary differences, without new solves."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent/'egg_results';d=json.loads((root/'results.json').read_text())
labels=[r['case'].replace('angle_','theta=').replace('two_balanced_pairs','two pairs').replace('auto_tiny','auto 45um') for r in d['rows']]
x=np.arange(len(labels));fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
for method in ('coarse','atlas','joint'):
    rr=[r['atlas'] if method=='atlas' else r['metrics'][method] for r in d['rows']]
    axes[0].plot(x,[r['tensor_relative']*100 for r in rr],'o-',label=method)
    axes[1].plot(x,[r['peak_relative']*100 for r in rr],'o-',label=method)
for ax in axes:
    ax.set_xticks(x,labels,rotation=35,ha='right');ax.grid(alpha=.2)
axes[0].set(ylabel='common sampled tensor difference (%)',title='Closed egg: feature atlas is not yet accurate')
axes[1].set(ylabel='sampled VM peak difference (%)',title='Joint-to-finer change is still material')
axes[0].legend();fig.suptitle('Shell-only balanced-normal supplement; finer shell is NOT contact ground truth')
fig.savefig(root/'accuracy.png',dpi=155);plt.close(fig)
