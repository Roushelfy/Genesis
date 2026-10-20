# Next goals: physical contact regions, robust recovery and throughput

User direction, 2026-10-20: finish the existing comparison, investigate automatic
contact regions on CPU, address the distinct SCR failures, and optimize native
throughput. This extends the [active comparison goal](QUALIFIED_STRESS_RANKING_GOAL.md)
and [accuracy/throughput protocol](ACCURACY_AND_THROUGHPUT_PROTOCOL.md).
The implementation agent should choose the simplest useful approach from measured
results, rather than build another framework or expose many tuning options.

## 1. Finish the current comparison independently

Reuse the [published fixed-radius comparison](QUALIFIED_FRICTION_RESULTS_20261020.md):
its declared0.1N normal/friction pad reference,54 dimensional errors,18 rankings,
inspected figures and matched research costs are now complete. Preserve this
baseline and close only remaining scope/configuration consolidation; do not
rerun the completed matrix to adopt the new error-budget study. Prioritize new
scientific results over expanding generic scheduling, recovery or audit tools.
Do not hot-edit submitted sources or restart valid work.

Retain fixed-radius/manual configurations and original/repaired Shell variants
as named baselines. Automatic contact regions are a separate comparison and
must not delay the current results. Performance work proceeds independently
of expensive reference qualification.

## 2. Separate and address current SCR failures

Earlier balanced-load normalization and numerical-fit repairs remain accepted
within their scope. The current failures are different:

| Case | Established result | Next objective |
| --- | --- | --- |
| Diverse banks: B128 step767/env9 and B512 step645/env330 | Independent continuous-wrench certificates prove incompatibility within the declared surface regions and Coulomb cones. Actual first-contact radii are about5.47/5.62mm, mu about0.439/0.713. | Attribute geometry/normal, support-region and force/moment-transfer contributions; repair input/model inconsistencies where justified. |
| Pad every-step SCR observation38 | Zero normal load but about1.76e-20N total residual tangential load conflicts with the instantaneous unilateral input contract. DPR has a separate observation37 local-source rejection. | Reconcile friction time levels and justified roundoff treatment, then verify the actual failing every-step path. |

Reuse the [native diagnosis](NATIVE_CONTACT_CONSISTENCY_20261019.md) and
[pad input analysis](FRICTION_KEYFRAME_COST_REUSE_20261019.md).
Use compact controlled experiments, such as force-only versus full-wrench
feasibility and matched geometry/normal/support variants, to identify causes.
Continuous infeasibility is not an iteration-limit or quadrature-resolution
diagnosis. It proves a conflict between the declared model and its inputs,
not impossibility of real physical contact.

Align source generation and recovery with a consistent declared friction law.
If that changes the old law, label the variant and preserve its baseline.
Keep exact-model infeasibility diagnostics and the original strict path.
Separately investigate the error-budgeted approximation authorized below:
strict infeasibility alone does not decide its engineering usefulness.
A larger patch or diagnostic DPR fallback does not itself admit physical accuracy.

## Error judgments and practical acceptance budgets

User clarification, 2026-10-20: distinguish mathematical exactness from useful
stress accuracy. A1e-8 load-fitting requirement can be unnecessarily expensive
or reject useful observations when the application tolerates substantially
larger stress errors. This is a hypothesis to measure, not a chosen new default.
Keep solver convergence, permitted load-model discrepancy and physical stress
error as three separate quantities.

**Tiny detachment residues:** scaling an allowance by the residual itself made
the earlier approximately1.76e-20N shear conflict with an approximately1e-34N
roundoff allowance. This is an inappropriate engineering scale for a0.1N task.
Use declared physical force/moment scales and justified absolute-plus-relative
budgets, including meaningful zero-load residual/stress checks. A diagnostic
relative elastic residual near1e17 at a zero/tiny RHS does not alone establish
a bad stress field, and finiteness alone does not establish a good one.

The [new bounded-input results](RESOLVED_PAD_AND_SOURCE_INPUTS_20261020.md)
already supply positive controls: source reconciliation changes the captured
inputs by approximately3.8e-21N/1.8e-20N and its independent linear P2 stress
perturbations are below8e-15Pa. Both labelled research variants complete1000
every-step observations. Reuse these results and original failures; do not
repeat unchanged controls. These correction-response results do not bound
arbitrary continuum placement, prove the general SCR model, or establish
production batching/throughput.

**Certified diverse-bank conflicts:** exact continuous force/wrench infeasibility
does not imply that every nearby admissible load has unacceptable stress error.
Conversely, a small required wrench change does not establish a small stress
change. The earlier proportional-budget estimates of about0.135% and0.000371%
only make one historical certificate inconclusive; they are not measured minimum
load corrections, guaranteed sufficient budgets or recommended tolerances.
Later env330 force-only evidence is stronger than its earlier wrench witness.
Measured source/anchor normal differences are about16–18degrees, a geometry/model
mismatch that must not be dismissed as roundoff.
Retain the complete diagnoses and do not select acceptance from those estimates.

Prioritize a compact CPU study on the retained first-failure states:

- Find a nearby surface traction distribution satisfying the declared unilateral
  and friction cones. Measure changes to individual sources, total force and
  moment, using physically scaled force/moment arms and explicit locality.
- Compare a small set of justified approximation budgets against the strict
  baseline; solve the approximation problem to its own numerical convergence.
  Do not treat a failed optimizer or exhausted iteration cap as an accepted
  engineering approximation.
- Compute the stress response of the actual load correction with the same
  declared transfer, geometry/material and free-link balance/inertial treatment.
  Include any equilibrium correction. Examine complete weighted stress tensors,
  membrane/bending and raw tensile/von-Mises maxima/locations, using absolute
  error near zero and relative error at meaningful scales.
- Preserve source priors and local compression. A budget based only on tiny net
  force or aggregate wrench can hide substantial self-equilibrated stress.
- Check representative changing/asymmetric normal/friction inputs and measure
  cost. Choose practical budgets from stress sensitivity, applicable physical
  reference error and the intended RL accuracy, rather than retuning until a
  failed example passes.

A useful explicitly labelled engineering approximation is allowed to adjust
observation loads within a demonstrated error budget. Preserve original loads,
certificates and tests, expose correction magnitude/quality, and keep this
separate from strict/raw acceptance. Reject corrections that exceed the admitted
budget or remain physically/numerically unresolved. Never relabel the original
exactly infeasible input as feasible, change rigid dynamics through an observer,
or silently enlarge radius/friction to conceal the conflict.

**DPR load fidelity:** prioritize correcting the observed approximately42%
inner-boundary transfer in the declared external-pad case. Global force/moment
conservation did not prevent a wrong pressure distribution. Already resolved
pressure samples should use conservative work-conjugate transfer to their actual
contacted boundary, without a second finite disk. Generalize by boundary/source
provenance, not a universal outer-surface rule that forbids legitimate inner or
nonconvex contacts. Sparse collider points still need a physical footprint model.

The measured8.70% complete-stress change and increased peaks after conservative
transfer are differences between load models on one mesh, not an8.70% accuracy
improvement or proof that larger/smaller stress is more correct. CPU/native
same-load agreement verifies implementation only. Compare against qualified
same-problem reference fields before assigning physical error/rank, and do not
subtract scalar error norms as an attribution. Keep source sampling, pressure
reconstruction, elasticity discretization and model reduction contributions
distinct, with signed cross terms where an actual decomposition is available.

This clarification takes precedence over blanket wording in earlier goals or
reports that every finite incompatibility must remain a fatal engineering
failure. Original strict tests and immutable results stay unchanged. Do not
globally relax their assertions; develop and validate the separately identified
approximation. New physical/load settings require their own matched error and
cost evidence, while pure implementation optimizations need focused consistency
checks rather than another expensive convergence campaign.

## 3. CPU study: determine contact support and pressure from physics

Distinguish physical pressure-bearing support, reconstruction-kernel width
and numerical search region. Predict support and pressure together; inserting
a radius into the current Gaussian kernel is not a physical validation.

Compare a compact set of candidates: current fixed radius, cheap
curvature/material/load predictions, and precomputed elastic compliance with
a small contact solve when a local formula is inadequate.

For smooth isotropic nonadhesive half-space contact, use Hertz only in its
applicable regime:

```text
a = (3 Fn R* / (4 E*))^(1/3)
1/E* = (1-nu1^2)/E1 + (1-nu2^2)/E2
1/R* = 1/R1 + 1/R2                 # two convex spherical surfaces
```

For the declared independent-column pad p=k_A*[delta-g]+, a quadratic spherical
gap g(r)=r^2/(2R), and negligible link deformation, direct integration gives:

```text
Fn = pi*k_A*a^4/(4R)
a = (4 Fn R/(pi*k_A))^(1/4)
```

These are initial models, not universal shell laws. Account for two principal
curvatures and orientation, finite flat lands/edges, pad thickness/backing,
shell thickness and global compliance as needed. Nonconvex contact can have
separate regions. Do not apply the spherical formula at zero/negative relative
gap curvature or substitute numerical penalty stiffness/penetration for physical
material properties/indentation without validation.

Group samples into physical contact regions before predicting force-controlled
size; collision points are not independent physical patches. Splitting one
region into more sources must not arbitrarily change its support. For already
distributed pressure samples, test consistent conservative transfer instead
of spreading each sample over a new finite disk. Quantify possible double
spreading in the current all-centroid path.

For a compliance-based model, reuse fixed elastic/pad operators, retain necessary
coupling between regions and consistent free-link balance/inertia, and solve
nonnegative pressure with separation/compression consistency. Do not pin the
free shell to invert its stiffness. Expand search regions when needed without
imposing them as the physical support. Reuse previous pressure/active sets as
initial guesses during smooth contact motion.

First compare area/axes/connectivity, pressure shape/centroid, force/moment and
force-indentation response; then compare complete stress and peaks. Reuse
independent references and a small discriminating set of convex/nonconvex,
asymmetric, changing-position/load, thickness/stiffness and friction/unloading
cases. Separate source sampling, load-model and elastic-discretization errors.
Do not fit radii to evaluation stress; retain useful negative results and limits.
Normal support does not uniquely determine tangential stress: include the
friction/slip history or coupling required by the chosen physical model.

## 4. Native integration and measured throughput optimization

Prioritize DPR for the first useful integration given its current measured
speed and broader passing condition banks. Continue SCR optimization on feasible
inputs and preserve its diverse-condition failures. Production remains a feature
of the existing Genesis rigid solver, entirely in Quadrants. Only rest
geometry/material is fixed: changing convex/nonconvex and asymmetric contacts,
normal/friction loads, max/full output, batching, isolation and resets remain
required. CPU prototypes stay validation tools.

Integrate the useful CPU-validated model with a small public configuration:
automatic physical-region selection plus the explicit manual baseline and
physically meaningful material/pad inputs. Keep missing data or unsupported
regimes visible. Exact API and algorithm choices are left to the agent.

Precompute curvature/compliance data and interpolate at current contacts.
Exploit temporal coherence with current-frame acceptance and correct invalidation.
A footprint below mesh resolution needs suitable integration/resolution or an
unresolved result, not a larger artificial physical radius.

Profile source grouping/geometry, region prediction/contact solve, traction
fitting, scatter, elastic recovery, max/full output, history work, dispatch,
copies and synchronization. Follow measured bottlenecks, including the known
SCR fitting/scatter/readback costs. Consider batching, changed-environment
compaction, cheaper history updates, fusion and lower precision where matched
numerical checks and measured benefit justify them.

Use low-resolution moving Franka/egg and diverse contact histories. Measure
paired pure-rigid/max/full modes, useful batch sizes, cadence, preprocessing
and memory, including all new prediction work. Keep module replay and actual
scene throughput separate. Do not drop hard environments or time only passing
prefixes. Sparse updates do not establish every-step peak reliability.
Pure scheduling/copy optimizations need focused consistency checks, not another
fine-reference sweep; physical load-model changes need appropriate small-case
physical validation.

## Delivery

Publish the fixed-radius comparison, the automatic-versus-manual CPU study,
the useful native implementation and measured tradeoffs. Provide contact and
stress maps, errors/uncertainty, stability/failure classification, separate
module/scene throughput and practical configuration recommendations. Join
accuracy and speed only for matching actual configurations.

Submitted jobs, a prototype, a certificate or another report alone do not
complete the work. State unresolved physical regimes and the next concrete
experiment. Reuse concise entrypoints and existing tables, figures and evidence;
keep provenance proportional to changes instead of adding sprawling frameworks.

## Mechanics references

- [Abaqus Hertz contact relations](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEANLRefMap/simaanl-c-demanalysis.htm).
- [Current pad declaration](COMPLIANT_PAD_DECLARATION.md): the fourth-root estimate
  above is a direct derivation under its stated quadratic-gap/rigid-link assumptions.
- [Sun and Paulose, shell indentation](https://arxiv.org/abs/2103.06307):
  geometry and pressure affect shell response.
- [Willert, tangential contact histories](https://arxiv.org/abs/2104.02436):
  history-dependent tangential reconstruction under Hertz-Mindlin assumptions.
