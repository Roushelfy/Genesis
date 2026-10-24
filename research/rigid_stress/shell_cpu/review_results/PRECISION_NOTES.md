# Bounded CPU precision and gauge experiments

`precision.json` records fixed legacy-gauge experiments on asymmetric egg
meshes with 588 and 2,316 DOFs, E=10 GPa, nu=0.3, h=0.5 mm and eight current
normal-plus-friction load cases. Geometry, Hertz/DPR input loads, mass projection
and stiffness assembly are FP64; “FP32” means the linear factor/solve and
stress-map application are rounded to FP32. These are not FP32 contact/geometry
or GPU TF32 results.

The reported stress difference is the relative Euclidean norm of all stacked
top/bottom local coefficients `[sigma_xx, sigma_yy, tau_xy]` at every in-plane
Gauss point. It is a complete sampled in-plane field comparison; it is not
the 3D Frobenius metric (which would weight the shear coefficient twice).

At the legacy gauge, raw FP32 stress errors reach 3.29% / 60.38%; diagonal
equilibration improves them to 0.139% / 16.09%. Stationary FP64 residual
refinement using the FP32 factor converges slowly at 588 DOFs and diverges at
2,316 DOFs. The original failure is retained in
`precision_initial_stationary.json`. GMRES with an FP32 factor and FP64
operator eventually converges, but is much slower than FP64 direct solving.

This is NOT evidence that thin shells intrinsically require FP64 everywhere.
`precision_condition.json` exposes a legacy-gauge conditioning problem: after
diagonal equilibration, estimated condition numbers are about 7.8e8 / 1.5e10.
The legacy QR pins three rotations at one node. Restricting a normalized,
rank-revealing QR to translation rows selects distributed translational gauge
DOFs and reduces the estimates to about 1.3e4 / 3.5e4.

The optional `ShellModel(..., gauge="translation")` leaves stiffness, mass
projection, all loads, material and physics unchanged. Legacy remains the
default to preserve primary performance provenance. `gauge_checks.json`
validates invariance, membrane/bending energies, uniform-pressure sphere and
the eight friction-load cases. FP64 stress differences versus the legacy gauge
are 3.1e-11 / 3.5e-10. One FP64 residual correction with an equilibrated FP32
factor gives stress errors <=1.03e-10; two give <=2.49e-11.

The CPU speed gain is principally scaling and LU fill reduction, not FP32.
For eight RHS at 2,316 DOFs, FP64 in original units costs about 11.97 ms for
solve+stress; equilibrated FP64 costs 3.14 ms, while one/two FP32-factor residual
corrections cost 5.55 / 8.46 ms. LU nonzeros fall from 3.68M to 0.664M. Timings
are bounded CPU diagnostics, not Genesis rollout or GPU throughput.

FP32 storage of a precomputed FP64 stress response is a distinct experiment.
At 588 DOFs, FP32 product error is 1.14e-7; selecting columns and accumulating
in FP64 reduces it to 2.19e-8. It halves response storage, but a response-only
field calculation does not independently certify displacement/FEM residuals.

The byte-exact original core at primary-run SHA256
`43d425e52af6d095535fbfcbdd4a8695060345fa3ee24648e019011f6f4fa8fc`
is preserved at `source_at_primary_runs/core.py`. The new optional-gauge core
and focused tests have their own hashes in `gauge_checks.json`.
