"""Render saved common sampled fields without altering solver/results evidence."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from atlas_study.nonconvex.study import vm
OUT=Path(__file__).resolve().parent

def main():
    d=np.load(OUT/'fields.npz');p=d['points'];xs=np.unique(p[0]);ys=np.unique(p[1]);vmax=vm(d['reference']).max()
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for ax,key,title in [(axes[0,0],'reference','Finer reference: sampled VM'),(axes[0,1],'coarse','Coarse: sampled VM'),(axes[1,0],'contact_only_hard','Contact-only atlas'),(axes[1,1],'contact_and_neck_hard','Contact + fixed-neck atlas')]:
        image=np.full((len(ys),len(xs)),np.nan)
        image[np.searchsorted(ys,p[1]),np.searchsorted(xs,p[0])]=vm(d[key])
        im=ax.pcolormesh(xs,ys,image,shading='nearest',cmap='inferno',vmin=0,vmax=vmax)
        ax.set_aspect('equal');ax.set_title(title);ax.set(xlim=(-.05,4.05),ylim=(-1.05,1.05),xlabel='x',ylabel='y')
        fig.colorbar(im,ax=ax,shrink=.72,label='2D VM / unit thickness')
    fig.suptitle('Nonconvex dogbone: common sampled fields, finite edge tractions\nSharp reentrant-corner exclusion is explicit; no continuum maximum claimed')
    fig.savefig(OUT/'stress_fields.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
