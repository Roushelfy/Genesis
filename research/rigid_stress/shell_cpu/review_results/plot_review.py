"""Render the independently evaluated analytical convergence controls."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
d=json.loads((root/'review.json').read_text())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(10.8,8.0),layout='constrained')
ax=axes[0,0]
for h in (.01,.001):
 r=[x for x in d['navier']['rows'] if x['h_over_L']==h]
 ax.loglog([x['n'] for x in r],[100*x['relative_KL_difference'] for x in r],'o-',label=f'h/L = {h:g}')
ax.set(xlabel='Plate elements per side',ylabel='Center displacement difference (%)',title='A  Simply supported plate vs Navier series')
ax.legend();ax.grid(alpha=.25,which='both')
ax=axes[0,1];r=d['sphere']['rows']
ax.loglog([x['dofs'] for x in r],[100*x['relative_radial_mean'] for x in r],'o-',label='Mean radial displacement')
ax.loglog([x['dofs'] for x in r],[100*x['relative_membrane_stress_L2'] for x in r],'s-',label='Membrane stress L2')
ax.set(xlabel='Shell scalar DOFs',ylabel='Difference from thin-sphere solution (%)',title='B  Uniformly pressurized sphere')
ax.legend();ax.grid(alpha=.25,which='both')
ax=axes[1,0]
for h in (.05,.02,.005):
 r=[x for x in d['cylinder_ovalization']['rows'] if x['h_over_R']==h]
 ax.loglog([x['n'] for x in r],[100*x['relative_displacement_error'] for x in r],'o-',label=f'h/R = {h:g}')
ax.set(xlabel='Circumferential elements',ylabel='Ovalization amplitude difference (%)',title='C  Curved bending: sinusoidally loaded cylinder')
ax.legend();ax.grid(alpha=.25,which='both')
ax=axes[1,1]
r=[x for x in d['contacts']['rows'] if x['side']==1 and x['center']==[.047,.054] and 'error' not in x]
ax.semilogy([x['order'] for x in r],[x['independent_nodal_load_error'] for x in r],'o-',label='Nodal load vs analytic Cartesian moments')
ax.semilogy([x['order'] for x in r],[max(x['moment_error'],1e-16) for x in r],'s-',label='Normalized wrench error')
ax.set(xlabel='Clipped-patch quadrature order',ylabel='Relative difference',title='D  Asymmetric finite Hertz disk across four faces')
ax.legend(fontsize=8);ax.grid(alpha=.25,which='both')
fig.suptitle('Independent CPU shell controls',fontsize=15)
fig.savefig(root/'shell_validation.png',dpi=160)
print(root/'shell_validation.png')
