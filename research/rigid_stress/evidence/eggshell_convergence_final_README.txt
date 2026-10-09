P2 eggshell stress recovery: crossed mesh convergence and online benchmark

CPU MEASUREMENTS
AMD EPYC 9V74; one BLAS thread; FP64. No CUDA GPU is available here.
The capability report records absent nvidia-smi/device nodes and unavailable CuPy.
GPU code has NOT been executed on a CUDA device; no GPU speed is claimed.

MODEL
Egg outer map on the unit sphere: x=.022*(1-.18*u_z)*u_x,
y=.022*(1-.18*u_z)*u_y, z=.030*u_z (meters).
Normal-offset wall thickness .5 mm, E=10 GPa, nu=.3, density=2000 kg/m^3.
Opposite smooth finite-area x loads at z=0, 4 N per finger.
Pressure profile exp(-d^2/(2*(.002)^2))*max(0,1-d^2/(.005)^2)^2,
d^2=y^2+z^2; surface integration uses a 10-by-10 Duffy/Gauss rule.
Quarter model with physical symmetry boundary conditions and one vertical gauge.
All node/tet counts below are QUARTER counts; compliance is multiplied by four.
This is prescribed linear-elastic traction recovery, not coupled contact,
Genesis integration, fracture, or measured material calibration.

ALGORITHM AND VALIDATION
10-node P2 displacement on straight-sided tets. Exact degree-two stiffness
quadrature; analytic consistent mass. Unaveraged stress is affine per tet;
the convex von Mises norm reaches its element maximum at a corner.
Large systems use PCG with symmetric pre/post Jacobi and an exact coarse
P2 L5/T2 solve. Spherical/prism P2 prolongation defines the preconditioner;
the fine equations and fine load remain unchanged. The fine residual and
stress sensitivity to tightening tolerance 1e-8 -> 1e-10 are checked.
Direct-vs-PCG peak relative difference: 1.77e-12.
Largest tolerance-induced relative peak change: 3.97e-11.
Largest final true relative free-equilibrium residual: 1.32e-10.
The previous suite also checks rigid/affine/pure-bending patches, legal gauge
changes, full-versus-quarter symmetry and independent P1 matrix assembly.

Model              Nodes     Tets     Peak(MPa) Full compliance(micro N m)
QP2-L5-T2             21125    12288 7.43881373 92.47859251
QP2-L6-T1             49923    24576 7.70361685 93.09733978
QP2-L6-T2             83205    49152 7.63283224 93.22503702
QP2-L6-T2-A1         117075    69372 7.63410312 93.31064671
QP2-L6-T2-A2         250675   149328 7.62700102 93.32245594
QP2-L6-T4            149769    98304 7.63204504 93.24736411
QP2-L7-T2            330245   196608 7.63402664 93.31205233

Latest refinement steps:
global surface : peak 0.015646%, compliance 0.093252%
local contact  : peak 0.093118%, compliance 0.012654%
wall thickness : peak 0.010314%, compliance 0.023944%
All satisfy <1% adjacent peak change and <.5% adjacent compliance change.
Engineering central-load reference: about 7.63 MPa. This criterion is not a
rigorous bound on error to the continuum solution. Axes cross around L6;
L7/T4 was not tested. Global surface refinement also changes faceted geometry.
The direct-LU checkpoints in complete_convergence.json used physical-node
METIS ordering. Current source uses surface-column ordering for its small
coarse solve. Large direct LU exceeded this runtime's 8 GiB memory limit;
the successful fine PCG solves are in iterative_convergence.json.

ONLINE QUERY: MEASURED CPU
L7/T2 response table: 786,432 stress samples, rank 3,
108.000 MiB FP64 response coefficients.
Exported coefficients are unit N per finger at z=-10,0,+12 mm with opposite
normal forces and quarter symmetry. Only CENTRAL load has mesh convergence.
Full scan p50/p95: 15.6655/18.4708 ms.
Previous-frame bounds p50/p95: 0.9996/1.3187 ms.
Mean evaluated sample fraction (including initial scans): 0.023469.
Maximum temporal error relative to full FP64 scan: 0.
128 smoothly varying frames, three repetitions, paths timed separately.
The temporal workload slightly mixes the three height coefficients and does
not certify stress accuracy for noncentral loading. It is a rank-3 workload,
so timing is not directly comparable with the prior rank-19 full-body test.
For a FIXED central pressure shape, stress maximum is simply proportional
to force: sigma_max(F)=sigma_max(4N)*abs(F)/4; no full scan or GPU is needed.

REPRODUCE
Unzip all files together. Python dependencies:
  pip install numpy scipy matplotlib threadpoolctl pymetis
Convergence including response export (several minutes; <=8 GiB here):
  python eggshell_p2_iterative.py --jobs 6:2:0 6:2:2 6:4:0 7:2:0 --export
The prior direct checkpoints and original results are bundled for comparison.
Regenerate figures and combined table:
  python eggshell_convergence_summary.py
CPU query timing:
  python eggshell_gpu_benchmark.py --device cpu --batches 1 4 --repeats 3 --warmup 2 --output eggshell_converged_cpu_query.json

CUDA TEST ON YOUR GPU
Install a CuPy wheel matching the installed CUDA runtime/driver; see
https://docs.cupy.dev/en/stable/install.html
  python eggshell_gpu_benchmark.py --device cuda --response eggshell_converged_response.npz --batches 1 8 32 128 --output eggshell_gpu_benchmark_results.json
Reports actual device name, CuPy/CUDA versions, FP32/FP64 error vs FP64 host
oracle, CUDA-event device time and synchronized host time, p50/p95, batches
sharing one response table, and q-upload/result-download time separately.
Preloading the response is timed once, outside query timing. CUDA warm-up
excludes first-call compilation/context costs. Large-batch OOM is recorded
as failure, never a timing. This is a GEMM/norm/max baseline, not a fused
kernel or GPU temporal implementation. Contact/rigid stepping/rendering
are excluded. The bounded-memory CPU oracle checks each GPU result.

FIGURES
convergence_final.png: independently varied surface/contact/thickness axes.
converged_stress.png: outer-surface corner maxima projected along x, mirrored
about y=0; includes full outer shell and contact zoom. Colors represent actual
FEM response values, not generated illustrations or smoothed nodal stresses.
