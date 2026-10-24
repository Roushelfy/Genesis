"""Bounded 3D suite with explicit setup-stage progress diagnostics."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import time
from . import model
from .study import run
from threadpoolctl import threadpool_limits
original=model.splu

def factor(a,*args,**kwargs):
 print('factor_start',a.shape,a.nnz,flush=True);t=time.perf_counter();q=original(a,*args,**kwargs);print('factor_done',time.perf_counter()-t,q.L.nnz+q.U.nnz,flush=True);return q
model.splu=factor
origmap=model.BoxModel.stress_map

def stress(self,points):
 print('stress_map_start',points.shape,self.N,flush=True);t=time.perf_counter();q=origmap(self,points);print('stress_map_done',time.perf_counter()-t,flush=True);return q
model.BoxModel.stress_map=stress
if __name__=='__main__':
 with threadpool_limits(limits=1):run('atlas_study/volume/results_bounded',fine_h=.0009)
