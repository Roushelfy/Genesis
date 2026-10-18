# Common observables and qualification rules

The primary three-way stress observable is the compatible tangential Cauchy
tensor on the same1280 material face-centroid wall columns, sampled every10ms.
Each method uses its own actual mass-weighted corotation. The integration uses
the union of reference/native tet intersections, with two Gauss points per
segment. It integrates the discrete P1/P2 tensor norms and membrane/bending
resultants exactly along those columns; in-plane centroid sampling remains
finite. It is not an exact full-volume integral or a global extremum search.
Inner/outer samples use the exact boundary point and the cell entered from
inside the wall. World, material and tangent frames stay explicit.

For the complete history, use one space/time norm for all three methods:

\[
e_{\Omega,T}^2 =
\frac{\sum_t\Delta t\,\|\sigma_T(t)-\sigma_{T,ref}(t)\|_{\Omega}^2}
     {\sum_t\Delta t\,\|\sigma_{T,ref}(t)\|_{\Omega}^2}.
\]

Time quadrature uses the declared end-of-interval observations, including every
low-load/release/recontact observation. No per-time relative-error mask, fitted
absolute threshold or selected-phase cut enters this norm. A complete score
requires completed source/candidate processes, finite fields and complete common
observation coverage. Otherwise retain the prefix/failure and leave the complete
score undefined. The norm is an integration of one observable over time; it is
not a sum of elastic, reconstruction and feedback scalar error norms.

Retain per-time relative/absolute errors, phase summaries, raw global peaks,
hotspot cells/fibers and outcomes alongside this complete-history norm. A small
history norm does not certify the worst transient or peak. Per-time relative
errors are undefined below1Pa reference tensor RMS,5e-4N/m membrane RMS or
2.5e-7N bending RMS; absolute discrepancies remain defined. Complete-history
integration uses the absolute norms from every finite observation, including
those with an undefined per-time relative ratio. The same floors only decide
whether the final whole-history denominator is meaningful.

Solid-only full3D tensor discrepancies on these columns supplement the common
tangent metric. Shell omits transverse stress, so it does not enter a differently
defined full3D ranking. Raw solid VM/principal maxima use full tensors over all
discrete elements/corners, while Shell maxima cover its actual face/fiber model.
The shared plotting grid does not define these global maxima.

Reaction qualification uses actual physical per-link gradient reactions and
moments about each link COM. Finger vector norms keep opposing loads separate;
a small net egg wrench error cannot qualify a large pinch reaction. Report both
complete-history reaction norms and per-time/per-link differences, with declared
near-zero floors1e-3N/1e-7Nm. Physical impulse moments accumulate at each substep
application point; never multiply a final point by an accumulated impulse.

Keep the goal's initial engineering targets visible: approximately2% reaction,
5% common field and10% raw peak changes. These are useful acceptance targets,
not proofs. Qualification is by protocol and measured quantity and also requires
input matching, actual residuals/caps, timestep/solve/contact/friction controls,
energy/work and lifecycle evidence. A highest-resolution or finite complete run
alone is not qualified. Singular/unresolved peaks, uncertain transitions and
unstable settings stay in the tables. Numerical throughput validity is separate
from physical-reference accuracy; no fastest-in-band claim follows from a
diagnostic discrepancy curve.

These complete-history summaries extend the earlier per-time diagnostic
figures. The existing figures/receipts remain intact. Final surface3/wall4 and
checkpointed capped uncertainty runs were still active when this metric contract
was recorded; no threshold is fitted to make those results pass.
